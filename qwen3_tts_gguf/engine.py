"""
engine.py - Qwen3-TTS 核心引擎
负责资源管理、模型初始化、音频特征提取及渲染。
"""
import os
import ctypes
import numpy as np
from transformers import AutoTokenizer

from . import llama, logger
from .assets import AssetsManager
from .stream import TTSStream
from .sampler import sample
from .mouth_decoder import StatefulMouthDecoder
from .predictors.encoder import EncoderPredictor

class TTSEngine:
    """
    Qwen3-TTS 引擎：资源池与 Stream 工厂。
    """
    def __init__(self, model_dir="model", tokenizer_path="Qwen3-TTS-12Hz-1.7B-CustomVoice"):
        self.project_root = os.getcwd()
        self.model_dir = os.path.join(self.project_root, model_dir)
        self.tokenizer_path = os.path.join(self.project_root, tokenizer_path)
        
        # 路径定义
        self.paths = {
            "master_gguf": os.path.join(self.model_dir, "qwen3_tts_talker.gguf"),
            "craftsman_gguf": os.path.join(self.model_dir, "qwen3_tts_craftsman.gguf"),
            "mouth_onnx": os.path.join(self.model_dir, "qwen3_tts_decoder_stateful.onnx"),
            "codec_enc_onnx": os.path.join(self.model_dir, "qwen3_tts_codec_encoder.onnx"),
            "spk_enc_onnx": os.path.join(self.model_dir, "qwen3_tts_speaker_encoder.onnx"),
        }
        
        # 1. 资产加载
        self.assets = AssetsManager(self.model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True, fix_mistral_regex=True)
        
        # 2. 模型引擎初始化
        self._init_llama_engines()
        
        # 3. 口腔解码器渲染模块 (同步模式)
        self.mouth = StatefulMouthDecoder(self.paths["mouth_onnx"], use_dml=True)
        
        # 4. 音频及说话人编码器 (可选，仅用于克隆)
        self.encoder = None
        if os.path.exists(self.paths["codec_enc_onnx"]) and os.path.exists(self.paths["spk_enc_onnx"]):
            self.encoder = EncoderPredictor(self.paths["codec_enc_onnx"], self.paths["spk_enc_onnx"], use_dml=False)
            
        logger.info(f"✅ [Engine] 引擎已就绪 (Mouth: {self.mouth.active_provider})")

    def _init_llama_engines(self):
        """初始化 GGUF 模型（仅加载模型，不创建 Context）"""
        logger.info("[Engine] 正在加载 GGUF 模型...")
        
        m_path = os.path.relpath(self.paths["master_gguf"], os.getcwd())
        c_path = os.path.relpath(self.paths["craftsman_gguf"], os.getcwd())
        
        self.m_model = llama.load_model(m_path, n_gpu_layers=-1)
        self.c_model = llama.load_model(c_path, n_gpu_layers=-1)
        
        if not self.m_model or not self.c_model:
            raise RuntimeError("GGUF 模型加载失败，请检查路径。")
            
        logger.info("✅ [Engine] GGUF 模型加载完成。")

    def create_stream(self, n_ctx=4096) -> TTSStream:
        """工厂方法：创建语音流"""
        return TTSStream(self, n_ctx=n_ctx)

    def _do_sample(self, logits, temperature):
        """引擎内部采样辅助 (供 Stream 调用)"""
        return sample(logits, temperature=temperature, top_p=1.0, top_k=50)

    def shutdown(self):
        """释放资源"""
        logger.info("[Engine] 正在关闭引擎...")
        try:
            llama.llama_model_free(self.m_model)
            llama.llama_model_free(self.c_model)
        except: pass
        logger.info("✅ [Engine] 引擎持有的模型资源已释放。")

    def __del__(self):
        self.shutdown()
