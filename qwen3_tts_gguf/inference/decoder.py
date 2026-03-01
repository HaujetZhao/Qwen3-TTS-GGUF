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
    
    def __init__(self, onnx_path: str, use_dml: bool = True):
        """
        初始化解码器。
        
        Args:
            onnx_path: 状态化 ONNX 模型路径
            use_dml: 是否尝试使用 DirectML 加速 (Windows GPU)
        """
        import onnxruntime as ort
        
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX 模型不存在: {onnx_path}")
        
        # 选择 Provider
        providers = ['CPUExecutionProvider']
        if use_dml:
            available = ort.get_available_providers()
            if 'DmlExecutionProvider' in available:
                providers = ['DmlExecutionProvider', 'CPUExecutionProvider']

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
        
        logger.info(f"✅ [Decoder] 已就绪 ({self.active_provider}), 精度: {self.dtype.__name__}")
        
        # 初始化状态 (工厂调用)
        self.create_state()
    
    def create_state(self):
        """
        创建一个全新的、空的解码器状态。
        
        Returns:
            DecoderState: 包含初始化的状态对象
        """
        from .protocol import DecoderState
        
        pre_conv_history = np.zeros((1, 512, 0), dtype=self.dtype)
        latent_buffer = np.zeros((1, 1024, 0), dtype=self.dtype)
        conv_history = np.zeros((1, 1024, 0), dtype=self.dtype)
        
        # KV Cache: [NUM_LAYERS] 个 Key + [NUM_LAYERS] 个 Value
        kv_cache = []
        for _ in range(self.NUM_LAYERS):
            # Key
            kv_cache.append(np.zeros((1, self.NUM_HEADS, 0, self.HEAD_DIM), dtype=self.dtype))
            # Value 
            kv_cache.append(np.zeros((1, self.NUM_HEADS, 0, self.HEAD_DIM), dtype=self.dtype))
            
        return DecoderState(
            pre_conv_history=pre_conv_history,
            latent_buffer=latent_buffer,
            conv_history=conv_history,
            kv_cache=kv_cache
        )
    
    def decode(self, audio_codes: np.ndarray, state: "DecoderState" = None, is_final: bool = False):
        """
        [对外封装] 对多于 45 帧的 codes 分批解码，防止 VRAM 溢出。
        """
        # 输入规范化获取长度
        if audio_codes.ndim == 2:
            n_frames = audio_codes.shape[0]
        elif audio_codes.ndim == 3:
            n_frames = audio_codes.shape[1]
        else:
            n_frames = len(audio_codes)

        if n_frames <= 45:
            return self._decode(audio_codes, state=state, is_final=is_final)
            
        # 批量解码逻辑
        chunk_size = 45
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
        底层原子解码 (Stateless Decode)。一次最多建议 45 帧。
        
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
            
        # 输入规范化
        if audio_codes.ndim == 1:
             audio_codes = audio_codes.reshape(-1, 16)
        if audio_codes.ndim == 2:
            audio_codes = audio_codes[np.newaxis, ...]  # [1, N, 16]
        
        audio_codes = audio_codes.astype(np.int64)
        n_frames = audio_codes.shape[1]
        
        if n_frames == 0:
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
        
        # 提取有效音频
        audio = final_wav[0, :valid_samples] if valid_samples > 0 else np.array([], dtype=np.float32)
        
        return audio.astype(np.float32), new_state

    def _build_state_from_outputs(self, outputs) -> "DecoderState":
        """从 ONNX 输出构建 DecoderState 对象"""
        from .protocol import DecoderState
        
        new_state = DecoderState(
            pre_conv_history=outputs[2],
            latent_buffer=outputs[3],
            conv_history=outputs[4],
            kv_cache=[]
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



