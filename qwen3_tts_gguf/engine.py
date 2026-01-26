import os
import ctypes
import numpy as np
from . import nano_llama, logger

class GGUFEngine:
    """Qwen3-TTS GGUF 推理引擎"""
    
    def __init__(self, model_path, n_ctx=2048, n_gpu_layers=0):
        self.model_path = os.path.abspath(model_path)
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        
        self.model = None
        self.ctx = None
        self.vocab = None
        self.initialized = False

    def initialize(self):
        """初始化库、加载模型与创建上下文"""
        if self.initialized:
            return True
        
        try:
            # 1. 初始化 DLL 库
            nano_llama.init_llama_lib()
            nano_llama.llama_backend_init()
            
            # 2. 设置模型参数
            m_params = nano_llama.llama_model_default_params()
            m_params.n_gpu_layers = self.n_gpu_layers
            
            # 3. 加载模型
            self.model = nano_llama.llama_model_load_from_file(
                self.model_path.encode('utf-8'), 
                m_params
            )
            if not self.model:
                raise RuntimeError(f"Failed to load model from {self.model_path}")
            
            self.vocab = nano_llama.llama_model_get_vocab(self.model)
            
            # 4. 创建上下文
            c_params = nano_llama.llama_context_default_params()
            c_params.n_ctx = self.n_ctx
            c_params.n_batch = self.n_ctx
            c_params.embeddings = True # 需要支持 embedding 输入
            
            self.ctx = nano_llama.llama_init_from_model(self.model, c_params)
            if not self.ctx:
                raise RuntimeError("Failed to initialize context")
                
            self.initialized = True
            logger.info("GGUF 引擎初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"GGUF 引擎初始化失败: {e}")
            return False

    def get_logits_from_embeddings(self, embeddings: np.ndarray):
        """
        注入 Embeddings 并获取最后一个位置的 Logits
        
        Args:
            embeddings (np.ndarray): 形状为 (seq_len, hidden_size) 的 float32 数组
            
        Returns:
            np.ndarray: 最后一个 token 的全量 Logits
        """
        if not self.initialized:
            raise RuntimeError("Engine not initialized")
            
        n_tokens = embeddings.shape[0]
        hidden_size = embeddings.shape[1]
        
        # 1. 初始化 Batch
        batch = nano_llama.llama_batch_init(n_tokens, hidden_size, 1)
        batch.n_tokens = n_tokens
        
        # 2. 注入 Embedding 数据
        # 确保数组是连续的并转换为 float32
        if not embeddings.flags['C_CONTIGUOUS']:
            embeddings = np.ascontiguousarray(embeddings)
        embeddings = embeddings.astype(np.float32)
        
        ctypes.memmove(batch.embd, embeddings.ctypes.data, embeddings.nbytes)
        
        # 3. 配置 Batch 元数据
        for i in range(n_tokens):
            batch.pos[i] = i
            batch.n_seq_id[i] = 1
            batch.seq_id[i][0] = 0
            # 只有最后一位需要 logits
            batch.logits[i] = 1 if i == n_tokens - 1 else 0
        
        # 将 token 指针置空，强制使用 embd
        batch.token = ctypes.cast(None, ctypes.POINTER(nano_llama.llama_token))
        
        # 4. 执行推理
        ret = nano_llama.llama_decode(self.ctx, batch)
        if ret != 0:
            nano_llama.llama_batch_free(batch)
            raise RuntimeError(f"llama_decode 失败 (ret={ret})")
            
        # 5. 提取 Logits
        vocab_size = nano_llama.llama_vocab_n_tokens(self.vocab)
        logits_ptr = nano_llama.llama_get_logits(self.ctx)
        
        # llama_get_logits 返回的是 batch 中最后一个设置了 logits=True 的位置的 logits
        logits_arr = np.ctypeslib.as_array(logits_ptr, shape=(vocab_size,)).copy()
        
        # 清理
        nano_llama.llama_batch_free(batch)
        
        return logits_arr

    def cleanup(self):
        """释放资源"""
        if self.ctx:
            nano_llama.llama_free(self.ctx)
            self.ctx = None
        if self.model:
            nano_llama.llama_model_free(self.model)
            self.model = None
        nano_llama.llama_backend_free()
        self.initialized = False
        logger.info("GGUF 引擎资源已释放")
