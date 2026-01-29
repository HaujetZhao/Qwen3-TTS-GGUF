"""
engine.py - Qwen3-TTS 核心引擎
负责资源管理、模型初始化和异步流水线调度。
"""
import os
import ctypes
import numpy as np
from multiprocessing import Process, Queue
from transformers import AutoTokenizer

from . import llama, logger
from .assets import AssetsManager
from .stream import TTSStream
from .sampler import sample
from .mouth_decoder import StatefulMouthDecoder

class TTSEngine:
    """
    Qwen3-TTS 引擎：资源池与 Stream 工厂。
    """
    def __init__(self, model_dir="model", tokenizer_path="Qwen3-TTS-12Hz-1.7B-CustomVoice", streaming=False):
        self.project_root = os.getcwd()
        self.model_dir = os.path.join(self.project_root, model_dir)
        self.tokenizer_path = os.path.join(self.project_root, tokenizer_path)
        self.streaming = streaming
        
        # 路径定义
        self.paths = {
            "master_gguf": os.path.join(self.model_dir, "qwen3_tts_talker.gguf"),
            "craftsman_gguf": os.path.join(self.model_dir, "qwen3_tts_craftsman.gguf"),
            "mouth_onnx": os.path.join(self.model_dir, "qwen3_tts_decoder_stateful.onnx")
        }
        
        # 1. 资产加载
        self.assets = AssetsManager(self.model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True, fix_mistral_regex=True)
        
        # 2. 模型引擎初始化
        self._init_llama_engines()
        
        # 3. 口腔解码器渲染模块
        if streaming:
            self._init_streaming_pipeline()
        else:
            # 非流式模式下，mouth 解码器常驻内存
            self.mouth = StatefulMouthDecoder(self.paths["mouth_onnx"], use_dml=True)
            logger.info(f"✅ [Engine] 非流式引擎已就绪 (Mouth Provider: {self.mouth.active_provider})")

    def _init_llama_engines(self):
        """初始化 GGUF 推理核心"""
        logger.info("[Engine] 正在加载 GGUF 模型...")
        
        # 使用相对路径，遵循用户建议
        m_path = os.path.relpath(self.paths["master_gguf"], os.getcwd())
        c_path = os.path.relpath(self.paths["craftsman_gguf"], os.getcwd())
        
        self.m_model = llama.load_model(m_path, n_gpu_layers=-1)
        self.c_model = llama.load_model(c_path, n_gpu_layers=-1)
        
        if not self.m_model or not self.c_model:
            raise RuntimeError("GGUF 模型加载失败，请检查路径。")
            
        # 大师模型 Context: 4096
        m_params = llama.llama_context_default_params()
        m_params.n_ctx = 4096
        m_params.embeddings = True
        self.m_ctx = llama.llama_init_from_model(self.m_model, m_params)
        
        # 工匠模型 Context: 512 (防止显存溢出，因为它 metadata 默认 32k)
        c_params = llama.llama_context_default_params()
        c_params.n_ctx = 512
        c_params.embeddings = False
        self.c_ctx = llama.llama_init_from_model(self.c_model, c_params)
        
        if not self.m_ctx or not self.c_ctx:
            raise RuntimeError("llama Context 初始化失败 (检查显存/路径)")
        
        # 批次缓冲区
        self.m_batch = llama.llama_batch_init(4096, 2048, 1)
        self.c_batch = llama.llama_batch_init(32, 1024, 1)
        logger.info("✅ [Engine] GGUF 核心加载完成。")

    def _init_streaming_pipeline(self):
        """启动多进程播放流水线"""
        from .workers import decoder_worker_proc, speaker_worker_proc, wav_writer_proc
        
        self.codes_q = Queue()
        self.pcm_q = Queue()
        self.record_q = Queue()
        
        # 解码进程
        self.dec_p = Process(target=decoder_worker_proc, 
                            args=(self.codes_q, self.pcm_q, self.paths["mouth_onnx"], self.record_q))
        # 播放进程
        self.spk_p = Process(target=speaker_worker_proc, args=(self.pcm_q,))
        # 录制/验证进程
        debug_wav = os.path.join(self.project_root, "output/refactored_stream_debug.wav")
        self.wav_p = Process(target=wav_writer_proc, args=(self.record_q, debug_wav))
        
        self.dec_p.daemon = self.spk_p.daemon = self.wav_p.daemon = True
        self.dec_p.start()
        self.spk_p.start()
        self.wav_p.start()
        
        logger.info("🚀 [Engine] 流式异步流水线已启动。")

    def create_stream(self, speaker_id="vivian", language="chinese") -> TTSStream:
        """工厂方法：创建语音流"""
        return TTSStream(self, speaker_id, language)

    def _do_sample(self, logits, temperature):
        """引擎内部采样辅助 (供 Stream 调用)"""
        return sample(logits, temperature=temperature, top_p=1.0, top_k=50)

    def shutdown(self):
        """优雅关闭进程与释放资源"""
        logger.info("[Engine] 正在关闭引擎...")
        if self.streaming:
            try:
                self.codes_q.put(None)
                self.dec_p.terminate()
                self.spk_p.terminate()
                self.wav_p.terminate()
            except: pass
            
        try:
            llama.llama_batch_free(self.m_batch)
            llama.llama_batch_free(self.c_batch)
            llama.llama_free(self.m_ctx)
            llama.llama_free(self.c_ctx)
            llama.llama_model_free(self.m_model)
            llama.llama_model_free(self.c_model)
        except: pass
        logger.info("✅ [Engine] 引擎已清理。")

    def __del__(self):
        self.shutdown()
