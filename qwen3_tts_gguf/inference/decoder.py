"""
decoder.py - 状态化解码器封装 (Decoder)
提供友好的对外接口，内部自动管理 KV Cache 和历史状态。

使用方法:
    decoder = StatefulDecoder("model/qwen3_tts_decoder_stateful.onnx")
    
    # 流式调用
    for codes in code_stream:
        is_final = (codes is last_chunk)
        audio = decoder.decode(codes, is_final=is_final)
        play(audio)
    
    # 初始化状态
    decoder.create_state()
"""
import os
os.environ["OMP_NUM_THREADS"] = "4"
import time
from typing import Optional, Union
import numpy as np
from . import logger

class StatefulDecoder:
    """
    状态化 ONNX 解码器封装。
    
    特点:
    - 内部自动管理 KV Cache 和历史状态
    - 支持流式和一次性解码
    - 外部只需传递 audio_codes 和 is_final
    - 自动处理 DirectML 加速（如果可用）
    """
    
    # 常量配置
    NUM_LAYERS = 8
    NUM_HEADS = 16
    HEAD_DIM = 64
    PRE_CONV_WINDOW = 2
    CONV_HISTORY_WINDOW = 4
    LOOKAHEAD_FRAMES = 4
    KV_CACHE_WINDOW = 72
    SAMPLES_PER_FRAME = 1920
    
    def __init__(self, onnx_path: str, onnx_provider: str = 'CPU', chunk_size: int = 12):
        """
        初始化解码器。
        
        Args:
            onnx_path: 状态化 ONNX 模型路径
            onnx_provider: ONNX Runtime 提供者 (CPU, CUDA, DML, TRT)
            chunk_size: 解码批大小，防止 VRAM 溢出
        """
        import onnxruntime as ort
        from pathlib import Path
        
        self.chunk_size = chunk_size
        self.onnx_provider = onnx_provider.upper()
        
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX 模型不存在: {onnx_path}")
        
        # 选择 Provider
        available_providers = ort.get_available_providers()
        providers = ['CPUExecutionProvider']
        
        if self.onnx_provider == 'GPU':
            # GPU = 自动挑当前 onnxruntime 包可用的加速后端
            for prov in ('CUDAExecutionProvider', 'DmlExecutionProvider'):
                if prov in available_providers:
                    providers.insert(0, prov)
                    break
        elif self.onnx_provider in ('TRT', 'TENSORRT') and 'TensorrtExecutionProvider' in available_providers:
            providers.insert(0, ('TensorrtExecutionProvider', {
                'trt_fp16_enable': True,
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': Path(onnx_path).parent / 'trt_cache',
            }))
        elif self.onnx_provider == 'DML' and 'DmlExecutionProvider' in available_providers:
            providers.insert(0, 'DmlExecutionProvider')
        elif self.onnx_provider == 'CUDA' and 'CUDAExecutionProvider' in available_providers:
            providers.insert(0, 'CUDAExecutionProvider')

        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3
        sess_opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        sess_opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.sess = ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=providers)
        
        # 动态检测输入类型 (跳过第一个整数类型的 audio_codes)
        float_input = next(i for i in self.sess.get_inputs() if "float" in i.type)
        self.dtype = np.float16 if "float16" in float_input.type else np.float32
        
        self.output_names = [out.name for out in self.sess.get_outputs()]
        
        # 获取实际使用的 provider
        self.active_provider = self.sess.get_providers()[0]
        
        # 用虚假的历史和code预热
        warmup_codes = np.zeros((self.chunk_size, 16), dtype=np.int64)
        self.decode(warmup_codes, state=self.create_state(72), is_final=True)
        logger.info(f"✅ [Decoder] 已就绪 ({self.active_provider}), 精度: {self.dtype.__name__}")
        
    
    def create_state(self, hostory_num = 0):
        """
        创建一个解码器状态。可指定历史记忆数量。
        
        Returns:
            DecoderState: 包含初始化的状态对象
        """
        from .schema.protocol import DecoderState
        
        if hostory_num != 0:
            pre_conv_history = np.zeros((1, 512, 2), dtype=self.dtype)
            latent_buffer = np.zeros((1, 1024, 4), dtype=self.dtype)
            conv_history = np.zeros((1, 1024, 4), dtype=self.dtype)
        else:
            pre_conv_history = np.zeros((1, 512, 0), dtype=self.dtype)
            latent_buffer = np.zeros((1, 1024, 0), dtype=self.dtype)
            conv_history = np.zeros((1, 1024, 0), dtype=self.dtype)
        
        # KV Cache: [NUM_LAYERS] 个 Key + [NUM_LAYERS] 个 Value
        kv_cache = []
        for _ in range(self.NUM_LAYERS):
            # Key
            kv_cache.append(np.zeros((1, self.NUM_HEADS, max(0,hostory_num), self.HEAD_DIM), dtype=self.dtype))
            # Value 
            kv_cache.append(np.zeros((1, self.NUM_HEADS, max(0,hostory_num), self.HEAD_DIM), dtype=self.dtype))
            
        return DecoderState(
            pre_conv_history=pre_conv_history,
            latent_buffer=latent_buffer,
            conv_history=conv_history,
            kv_cache=kv_cache,
            skip_samples=0,
            latent_audio=None
        )
    
    def decode(self, audio_codes: np.ndarray, state: "DecoderState" = None, is_final: bool = False):
        """
        [对外封装] 对多于 self.chunk_size 帧的 codes 分批解码，防止 VRAM 溢出。
        """
        # 输入规范化获取长度
        if audio_codes.ndim == 2:
            n_frames = audio_codes.shape[0]
        elif audio_codes.ndim == 3:
            n_frames = audio_codes.shape[1]
        else:
            n_frames = len(audio_codes)

        if n_frames <= self.chunk_size:
            return self._decode(audio_codes, state=state, is_final=is_final)
            
        # 批量解码逻辑
        chunk_size = self.chunk_size
        full_audio = []
        curr_state = state
        
        for i in range(0, n_frames, chunk_size):
            chunk = audio_codes[i:i+chunk_size]
            is_last_chunk = (i + chunk_size >= n_frames)
            
            # 只有最后一个分片才接受外部的 is_final 状态
            chunk_is_final = is_final if is_last_chunk else False
            
            chunk_audio, curr_state = self._decode(chunk, state=curr_state, is_final=chunk_is_final)
            full_audio.append(chunk_audio)
            
        return np.concatenate(full_audio), curr_state

    def _decode(self, audio_codes: np.ndarray, state: "DecoderState" = None, is_final: bool = False):
        """
        底层原子解码 (Stateless Decode)。一次最多建议 self.chunk_size 帧。
        
        Args:
            audio_codes: 音频码 [N, 16]
            state: 上下文状态 (DecoderState)。如果为 None，则从零开始。
            is_final: 是否结束
            
        Returns:
            (audio, new_state): 
                audio: 生成的波形
                new_state: 更新后的状态对象
        """
        if state is None:
            state = self.create_state()
            
        skip_counter = state.skip_samples
            
        # 输入规范化
        if audio_codes.ndim == 1:
             audio_codes = audio_codes.reshape(-1, 16)
        if audio_codes.ndim == 2:
            audio_codes = audio_codes[np.newaxis, ...]  # [1, N, 16]
        
        audio_codes = audio_codes.astype(np.int64)
        n_frames = audio_codes.shape[1]
        
        if n_frames == 0:
            if is_final and state.latent_audio is not None:
                audio = state.latent_audio
                state.latent_audio = None # 提取后清空
                return audio, state
            return np.array([], dtype=np.float32), state
        
        # 构建输入 feed dict
        feed = {
            "audio_codes": audio_codes,
            "is_last": np.array([1.0 if is_final else 0.0], dtype=self.dtype),
            "pre_conv_history": state.pre_conv_history,
            "latent_buffer": state.latent_buffer,
            "conv_history": state.conv_history,
        }
        
        # KV Cache 解包: 列表顺序假定为 k0, v0, k1, v1 ...
        for i in range(self.NUM_LAYERS):
            feed[f"past_key_{i}"] = state.kv_cache[2*i]
            feed[f"past_value_{i}"] = state.kv_cache[2*i + 1]
        
        # 执行推理
        outputs = self.sess.run(self.output_names, feed)
        
        # 解包输出
        final_wav = outputs[0]        # [1, num_samples]
        valid_samples = int(outputs[1][0])  # 有效样本数
        
        # 构建新状态
        new_state = self._build_state_from_outputs(outputs)
        
        # 提取音频与尾部处理
        if is_final:
            audio = final_wav[0]  # 当结束时，全量提取
            new_state.latent_audio = None
        else:
            # 正常流式块：取有效部分作为当前输出，残留部分存入状态
            audio = final_wav[0, :valid_samples] if valid_samples > 0 else np.array([], dtype=np.float32)
            new_state.latent_audio = final_wav[0, valid_samples:]
        
        # 处理历史采样点的抵消 (用于过滤注入状态时的初始残留音频)
        if skip_counter > 0 and len(audio) > 0:
            if len(audio) <= skip_counter:
                skip_counter -= len(audio)
                audio = np.array([], dtype=np.float32)
            else:
                audio = audio[skip_counter:]
                skip_counter = 0
                
        # 如果是任务结束，标记该状态在下次使用时需要跳过 4 帧的残留音频=
        new_state.skip_samples = 4 * 1920 if is_final else skip_counter
        
        return audio.astype(np.float32), new_state

    def _build_state_from_outputs(self, outputs) -> "DecoderState":
        """从 ONNX 输出构建 DecoderState 对象"""
        from .schema.protocol import DecoderState
        
        new_state = DecoderState(
            pre_conv_history=outputs[2],
            latent_buffer=outputs[3],
            conv_history=outputs[4],
            kv_cache=[],
            skip_samples=0, # 默认不携带跳过计数，由外部手动设置
            latent_audio=None
        )
        
        # 收集 KV Cache
        # 输出顺序: Key 块 (5 ~ 5+N) 和 Value 块 (5+N ~ 5+2N)
        # 我们按交替顺序存入 list: k0, v0, k1, v1 ... 以匹配 feed 循环
        new_kv = []
        base_idx = 5
        num_layers = self.NUM_LAYERS
        
        for i in range(num_layers):
            k = outputs[base_idx + i]
            v = outputs[base_idx + num_layers + i]
            new_kv.append(k)
            new_kv.append(v)
            
        new_state.kv_cache = new_kv
        return new_state
    
    def decode_full(self, audio_codes: np.ndarray) -> np.ndarray:
        """
        一次性解码所有音频码（非流式场景）。仅返回音频，自动丢弃状态。
        """
        audio, _ = self.decode(audio_codes, state=None, is_final=True)
        return audio
    
    @property
    def info(self) -> dict:
        """返回解码器状态信息"""
        return {
            "provider": self.active_provider,
            "total_frames": self.total_frames_processed,
            "total_samples": self.total_samples_output,
            "kv_cache_len": self.past_keys[0].shape[2] if self.past_keys else 0,
        }


class LocalDecoder:
    """
    进程内解码器：与 DecoderProxy.decode() 同签名的同步实现，服务离线场景 (GUI/批量)。
    无队列无监听线程，无播放相关接口 (pause/raw_play 等不支持)。
    差异: 流式中间包本实现返回含该块音频的 DecodeResult (DecoderProxy 同分支返回空数组)，
    离线场景调用方不消费该返回值。
    """
    def __init__(self, onnx_path: str, onnx_provider: str = 'CPU', chunk_size: int = 12):
        self._dec = StatefulDecoder(onnx_path, onnx_provider=onnx_provider, chunk_size=chunk_size)
        self.sessions = {}  # task_id -> DecoderSession (流式多次调用间保持状态)
        self._bank = {}    # task_id -> 已累积的 AUDIO 响应 (储蓄罐语义，终态时一并打包)
        self.ready_states = {"decoder": True, "speaker": False}  # engine 打印用

    def wait_until_ready(self, timeout=10):
        return True

    def decode(self, input: Union[np.ndarray, "TTSResult"], task_id="default", is_final: bool = False,
               stream: bool = False, state: Optional["DecoderState"] = None) -> "DecodeResult":
        """
        同步解码。参数语义与 DecoderProxy.decode 一致：
        TTSResult 输入时先预解码 ref_codes 对齐记忆，完成后回写 audio/final_state/耗时。
        """
        from .schema.protocol import DecoderSession, DecoderResponse
        from .schema.result import TTSResult as _TTSResult, DecodeResult as _DecodeResult

        if isinstance(input, _TTSResult):
            if input.ref_codes is not None and len(input.ref_codes) > 0 and input.final_state is None:
                res_ref = self.decode(input.ref_codes, task_id=f"{task_id}_ref_init", is_final=True)
                input.final_state = res_ref.final_state
            state = state or input.final_state
            codes = input.codes
            is_final = True
        else:
            codes = input

        t_start = time.time()
        codes_arr = np.asarray(codes, dtype=np.int64)
        if codes_arr.ndim == 1:
            codes_arr = codes_arr.reshape(-1, 16)

        session = self.sessions.get(task_id)
        curr_state = session.state if session is not None else state
        is_task_final = is_final or not stream
        audio, new_state = self._dec.decode(codes_arr, state=curr_state, is_final=is_task_final)

        audio_resp = DecoderResponse(
            task_id=task_id, msg_type="AUDIO",
            audio=audio.copy() if len(audio) > 0 else np.array([], dtype=np.float32),
            compute_time=time.time() - t_start, recv_time=time.time())
        if is_task_final:
            # 储蓄罐语义：终态时把整个任务累积的所有 AUDIO 响应一并打包返回
            responses = self._bank.pop(task_id, []) + [audio_resp]
            responses.append(DecoderResponse(task_id=task_id, msg_type="FINISH", state=new_state))
            self.sessions.pop(task_id, None)
        else:
            self.sessions[task_id] = DecoderSession(state=new_state)
            self._bank.setdefault(task_id, []).append(audio_resp)
            responses = [audio_resp]

        result = _DecodeResult(responses=responses)
        if isinstance(input, _TTSResult):
            input.audio = result.audio
            input.final_state = result.final_state
            if input.stats:
                input.stats.decoder_compute_times = result.chunk_compute_times
        return result

    def stop(self, task_id="default"):
        """中止任务：丢弃会话与累积响应。

        进程内模式无播放线路，丢弃累积音频，返回 None。
        """
        self.sessions.pop(task_id, None)
        self._bank.pop(task_id, None)

    def join_decoder(self, timeout=None):
        """同步执行，无等待语义。"""
        pass

    def join_speaker(self, timeout=None):
        """同步执行，无等待语义。"""
        pass

    def shutdown(self):
        self.sessions.clear()
        self._bank.clear()


