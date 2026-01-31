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
    
    # 重置状态（新句子）
    decoder.reset()
"""
import os
os.environ["OMP_NUM_THREADS"] = "4"
import numpy as np

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
        
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.output_names = [out.name for out in self.sess.get_outputs()]
        
        # 获取实际使用的 provider
        self.active_provider = self.sess.get_providers()[0]
        
        # 初始化状态
        self.reset()
    
    def reset(self):
        """
        创建一个全新的、空的解码器状态。
        
        Returns:
            DecoderState: 包含全 0 初始化的状态对象
        """
        from .protocol import DecoderState
        
        pre_conv_history = np.zeros((1, 512, 0), dtype=np.float32)
        latent_buffer = np.zeros((1, 1024, 0), dtype=np.float32)
        conv_history = np.zeros((1, 1024, 0), dtype=np.float32)
        
        # KV Cache: [NUM_LAYERS] 个 Key + [NUM_LAYERS] 个 Value
        # 展平成一个列表: [k0, k1... v0, v1...] 或者交替？onnx inputs 是 past_key_0, past_value_0...
        # 我们的 protocol 定义 kv_cache 是 List[np.ndarray]，顺序需严格匹配
        kv_cache = []
        for _ in range(self.NUM_LAYERS):
            # Key
            kv_cache.append(np.zeros((1, self.NUM_HEADS, 0, self.HEAD_DIM), dtype=np.float32))
            # Value 
            kv_cache.append(np.zeros((1, self.NUM_HEADS, 0, self.HEAD_DIM), dtype=np.float32))
            
        return DecoderState(
            pre_conv_history=pre_conv_history,
            latent_buffer=latent_buffer,
            conv_history=conv_history,
            kv_cache=kv_cache
        )
    
    def decode(self, audio_codes: np.ndarray, state: "DecoderState" = None, is_final: bool = False):
        """
        无副作用解码 (Stateless Decode)。
        
        Args:
            audio_codes: 音频码 [N, 16]
            state: 上下文状态 (DecoderState)。如果为 None，则从零开始。
            is_final: 是否结束
            
        Returns:
            (audio, new_state): 
                audio: 生成的波形
                new_state: 更新后的状态对象 (如果 is_final=True，则可能返回空状态或脏状态，视具体需求而定，但 caller 应该销毁它)
        """
        if state is None:
            state = self.reset()
            
        # 输入规范化
        if audio_codes.ndim == 2:
            audio_codes = audio_codes[np.newaxis, ...]  # [1, N, 16]
        
        audio_codes = audio_codes.astype(np.int64)
        n_frames = audio_codes.shape[1]
        
        if n_frames == 0:
            return np.array([], dtype=np.float32), state
        
        # 构建输入 feed dict
        feed = {
            "audio_codes": audio_codes,
            "is_last": np.array([1.0 if is_final else 0.0], dtype=np.float32),
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
        from .protocol import DecoderState
        new_state = DecoderState(
            pre_conv_history = outputs[2],
            latent_buffer = outputs[3],
            conv_history = outputs[4],
            kv_cache = []
        )
        
        # 收集 KV Cache: outputs[5...]
        # 输出顺序: past_key_0, past_value_0, past_key_1, past_value_1 ... 
        # (注意：onnx 导出时的输出顺序通常是 k0, v0, k1, v1... 需确认，或者这里代码假设是 k0..k7, v0..v7)
        # 
        # 让我们回顾一下之前的代码：
        # for i in range(self.NUM_LAYERS):
        #     self.past_keys[i] = outputs[5 + i]
        #     self.past_values[i] = outputs[5 + self.NUM_LAYERS + i]
        # 
        # 之前的代码明确显示输出是主要分为两块：Key 块 (5 ~ 5+N) 和 Value 块 (5+N ~ 5+2N)。
        # 因此我们需要按照这个顺序收集，但存入 DecoderState.kv_cache 时建议保持 k, v, k, v 的交替顺序（方便 feed）或者 split 顺序。
        # 为了与 feed 循环 (L120) 保持一致 "state.kv_cache[2*i]"，我们这里应该按交替顺序存入 list。
        
        new_kv = []
        base_idx = 5
        num_layers = self.NUM_LAYERS
        
        for i in range(num_layers):
            k = outputs[base_idx + i]
            v = outputs[base_idx + num_layers + i]
            new_kv.append(k)
            new_kv.append(v)
            
        new_state.kv_cache = new_kv
        
        # 提取有效音频
        audio = final_wav[0, :valid_samples] if valid_samples > 0 else np.array([], dtype=np.float32)
        
        return audio.astype(np.float32), new_state
    
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



# 兼容旧版 API 的工厂函数
def create_decoder(onnx_path: str, use_dml: bool = True, multiprocessing: bool = True):
    """创建解码器实例 (decoder)"""
    if multiprocessing:
        from .proxy import DecoderProxy
        return DecoderProxy(onnx_path, use_dml)
    return StatefulDecoder(onnx_path, use_dml)
