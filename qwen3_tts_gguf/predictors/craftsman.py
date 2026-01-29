"""
predictors/craftsman.py - 工匠预测器
负责单帧 Codec 详细码 (Q1-Q15) 的阶梯式推理。
"""
import ctypes
import numpy as np
from .. import llama
from ..sampler import sample

class CraftsmanPredictor:
    """
    封装工匠模型 (Craftsman/Subtalker) 的推理行为。
    特点：每步完全重置 Context，本质上处理单帧的局部相关性。
    """
    def __init__(self, model, context, batch, assets):
        self.model = model
        self.ctx = context
        self.batch = batch
        self.assets = assets
        
    def predict_frame(self, master_hidden: np.ndarray, code_0: int, 
                      do_sample=True, temperature=0.9, top_p=1.0, top_k=50) -> tuple:
        """
        为给定的 Master 隐层状态和第一个码，生成完整的 16 个码。
        
        Args:
            master_hidden: [2048] 大师隐层
            code_0: 由大师直接预测出的第一个量化器码
            ...采样参数...
            
        Returns:
            (codes, embeds) -> (16项整数列表, 16项2048维向量列表)
        """
        # 1. 准备输入：Master Hidden (投影后) + Code_0 Embedding (投影后)
        # NumPy 投影: 2048 -> 1024 (矩阵乘法)
        proj = self.assets.proj
        m_h_1024 = master_hidden @ proj["weight"].T + proj["bias"]
        
        # Code_0 已经由 Master 确定，我们直接取其 2048 嵌入存起来供 Master 反馈
        step_codes = [code_0]
        step_embeds_2048 = [self.assets.get_codec_embedding(0, code_0).copy()]
        
        # 构造工匠模型的初始输入：[Master_H, Code0_E] -> 形状 [2, 1024]
        c_in = np.stack([m_h_1024, self.assets.get_codec_embedding_1024(0, code_0)], axis=0)
        
        # 2. 清理工匠的记忆（由于工匠是跨帧无状态的）
        llama.llama_memory_clear(llama.llama_get_memory(self.ctx), True)
        
        # 3. Prefill 工匠：输入两个 Token，预测第二个位置
        self.batch.n_tokens = 2
        ctypes.memmove(self.batch.embd, c_in.ctypes.data, c_in.nbytes)
        for j in range(2):
            self.batch.pos[j] = j
            self.batch.n_seq_id[j] = 1
            self.batch.seq_id[j][0] = 0
            self.batch.logits[j] = (1 if j == 1 else 0) # 只对第二个位置求 Logits
            
        llama.llama_decode(self.ctx, self.batch)
        
        # 获取 15 级分步码的 Logits 基础指针 (形状 [15, 30720])
        # 注意：这里 Qwen3-TTS 的工艺是把所有分步 Logits 平铺在同一个 Vocab 空间内
        all_stage_logits = np.ctypeslib.as_array(llama.llama_get_logits(self.ctx), shape=(1, 30720))[0]
        
        # 4. 阶梯式生成 Q1 -> Q15
        for cs in range(1, 16):
            # 提取当前等级的 Logits 块 [2048]
            # 偏移量算法：(cs-1) * 2048
            logits_slice = all_stage_logits[(cs-1)*2048 : cs*2048]
            
            # 采样
            if do_sample:
                c = sample(logits_slice, temperature, top_p, top_k)
            else:
                c = int(np.argmax(logits_slice))
                
            step_codes.append(c)
            step_embeds_2048.append(self.assets.get_codec_embedding(cs, c).copy())
            
            # 如果不是最后一级，将当前预测反馈给工匠以预测下一级
            if cs < 15:
                # 反馈当前码的 1024 维嵌入
                emb_1024 = self.assets.get_codec_embedding_1024(cs, c)
                
                self.batch.n_tokens = 1
                self.batch.pos[0] = cs + 1 # 位置递增
                self.batch.logits[0] = 1
                ctypes.memmove(self.batch.embd, emb_1024.ctypes.data, 4096)
                
                llama.llama_decode(self.ctx, self.batch)
                
                # 更新 Logits 指针以获取下一级
                all_stage_logits = np.ctypeslib.as_array(llama.llama_get_logits(self.ctx), shape=(30720,))
                
        return step_codes, step_embeds_2048
