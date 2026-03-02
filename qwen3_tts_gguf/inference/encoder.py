import os
from typing import Union
import numpy as np
import onnxruntime as ort
from . import logger
from .schema.result import TTSResult
from .utils.mel import MelExtractor
from .utils.audio import preprocess_audio

class CodecEncoder:
    """音频码本编码器：将原始音频转换为 Codec IDs。"""
    def __init__(self, onnx_path: str):
        self.onnx_path = onnx_path
        providers = ['CPUExecutionProvider']

        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3
        sess_opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        sess_opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        logger.info(f"[CodecEncoder] 正在初始化 ONNX 会话...")
        self.sess = ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=providers)
        
        # 动态检测输入类型
        input_info = self.sess.get_inputs()[0]
        self.dtype = np.float16 if "float16" in input_info.type else np.float32
        
        self.active_provider = self.sess.get_providers()[0]
        logger.info(f"✅ [CodecEncoder] 已就绪 ({self.active_provider}), 精度: {self.dtype.__name__}")

    def encode(self, wav: np.ndarray) -> np.ndarray:
        """
        输入: wav [samples] (float32)
        返回: codes [T, 16] (int64)
        """
        c_out = self.sess.run(
            ['audio_codes'], 
            {'input_values': wav.reshape(1, -1).astype(self.dtype)}
        )
        return c_out[0][0] # [T, 16]


class SpeakerEncoder:
    """说话人特征编码器：将原始音频转换为 Speaker Embedding。"""
    def __init__(self, onnx_path: str):
        self.onnx_path = onnx_path
        providers = ['CPUExecutionProvider']

        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3
        sess_opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        sess_opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        logger.info(f"[SpeakerEncoder] 正在初始化 ONNX 会话...")
        self.sess = ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=providers)
        
        # 动态检测输入类型
        input_info = self.sess.get_inputs()[0]
        self.dtype = np.float16 if "float16" in input_info.type else np.float32

        # 初始化 Mel 提取器 (纯 NumPy/SciPy 对齐版)
        self.mel_extractor = MelExtractor(
            sr=24000, n_fft=1024, hop_length=256, n_mels=128, fmin=0.0, fmax=12000.0
        )
        self.active_provider = self.sess.get_providers()[0]
        logger.info(f"✅ [SpeakerEncoder] 已就绪 ({self.active_provider}), 精度: {self.dtype.__name__}")

    def encode_audio(self, wav: np.ndarray) -> np.ndarray:
        """从音频提取 Speaker Embedding"""
        mels = self.mel_extractor.extract(wav)
        mels_input = mels[np.newaxis, ...].astype(self.dtype) # [1, T, 128]
        
        s_out = self.sess.run(
            ['spk_emb'], 
            {'mels': mels_input}
        )
        return s_out[0][0] # [2048]

    def encode(self, input: Union[np.ndarray, TTSResult]) -> np.ndarray:
        """
        输入: 
            wav: [samples] (float32) 或 TTSResult 对象
        返回: 
            spk_emb [2048] (float32)
        """

        if isinstance(input, TTSResult):
            wav = input.audio
            if wav is None:
                logger.warning("⚠️ TTSResult 中不包含音频数据，跳过特征提取过程。")
                return input.spk_emb
            spk_emb = self.encode_audio(wav)
            input.spk_emb = spk_emb
        else:
            wav = input
            spk_emb = self.encode_audio(wav)
            
            
        return spk_emb

