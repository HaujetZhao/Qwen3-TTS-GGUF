"""
encoder.py - 音频特征提取器 (Codec & Speaker Encoder)
封装官方 Mel 谱图算法与 ONNX 编码器。
"""
import os
import numpy as np
import onnxruntime as ort
import librosa
from .. import logger

class EncoderPredictor:
    """
    音频编码预测器：将原始音频转换为 Codec IDs 和 Speaker Embedding。
    支持 24kHz 采样，对齐官方 CustomVoice (12Hz) 算法。
    """
    def __init__(self, codec_onnx_path: str, spk_onnx_path: str, use_dml: bool = False):
        self.codec_path = codec_onnx_path
        self.spk_path = spk_onnx_path
        
        # 编码器通常较轻量，且 DML 在某些 Reshape 节点上存在稳定性问题，推荐 CPU
        providers = ['CPUExecutionProvider']
        
        logger.info(f"[Encoder] 正在初始化编码器 ONNX 会话...")
        self.codec_sess = ort.InferenceSession(codec_onnx_path, providers=providers)
        self.spk_sess = ort.InferenceSession(spk_onnx_path, providers=providers)
        
        self.active_codec_provider = self.codec_sess.get_providers()[0]
        self.active_spk_provider = self.spk_sess.get_providers()[0]
        logger.info(f"✅ [Encoder] 已就绪 (Codec: {self.active_codec_provider}, Spk: {self.active_spk_provider})")

    def _extract_mel(self, wav: np.ndarray) -> np.ndarray:
        """
        官方对齐的 Mel 谱图提取算法。
        参数: 24kHz, n_fft=1024, hop=256, n_mels=128, fmin=0, fmax=12000
        """
        # 1. 计算 Slaney-style Mel Filterbank
        mel_basis = librosa.filters.mel(
            sr=24000, n_fft=1024, n_mels=128, fmin=0.0, fmax=12000.0
        )
        
        # 2. 手动 Padding (对齐官方: (n_fft - hop_size) // 2 = 384)
        padding = (1024 - 256) // 2
        wav_padded = np.pad(wav, (padding, padding), mode='reflect')
        
        # 3. STFT (禁用 center 自动 Padding)
        stft = librosa.stft(
            wav_padded, n_fft=1024, hop_length=256, win_length=1024, 
            window='hann', center=False
        )
        
        # 4. 计算幅度并叠加 1e-9 防止数值溢出
        magnitudes = np.sqrt(np.abs(stft)**2 + 1e-9)
        
        # 5. Mel 映射
        mel_spec = np.dot(mel_basis, magnitudes)
        
        # 6. 动态范围压缩
        log_mel = np.log(np.maximum(mel_spec, 1e-5))
        
        return log_mel.T

    def encode(self, wav_path: str) -> tuple[np.ndarray, np.ndarray]:
        """
        全量编码接口。
        返回: (codes [T, 16], spk_emb [2048])
        """
        # 1. 加载音频 (强制 24kHz, 单声道)
        wav, _ = librosa.load(wav_path, sr=24000)
        
        # 2. Codec 编码
        # 输入形状 [1, T], 输出 ['audio_codes'] (1, T, 16)
        c_out = self.codec_sess.run(
            ['audio_codes'], 
            {'input_values': wav.reshape(1, -1).astype(np.float32)}
        )
        codes = c_out[0][0] # [T, 16]
        
        # 3. Speaker 编码
        mels = self._extract_mel(wav)
        mels_input = mels[np.newaxis, ...].astype(np.float32) # [1, T, 128]
        
        s_out = self.spk_sess.run(
            ['spk_emb'], 
            {'mels': mels_input}
        )
        spk_emb = s_out[0][0] # [2048]
        
        return codes, spk_emb
