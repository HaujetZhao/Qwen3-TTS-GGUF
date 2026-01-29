"""
predictors/master.py - 大师预测器
负责 Qwen3-TTS 的主体 LLM 推理。
管理 llama.cpp 上下文、KV Cache 和步数。
"""
import ctypes
import numpy as np
import time
from .. import llama
from ..sampler import sample
from ..constants import PROTOCOL

class MasterPredictor:
    """
    封装大师模型 (Talker) 的推理行为。
    """
    def __init__(self, model, context, batch, assets):
        self.model = model
        self.ctx = context
        self.batch = batch
        self.assets = assets
        
        self.n_ctx = 4096 # 默认上下文大小
        self.cur_pos = 0
        
    def clear_memory(self):
        """完全清空 KV Cache"""
        llama.llama_memory_clear(llama.llama_get_memory(self.ctx), True)
        self.cur_pos = 0

    def prefill(self, prompt_embeds: np.ndarray) -> tuple:
        """
        全量推入初始 Prompt 或新文本批次。
        Args:
            prompt_embeds: [Batch=1, Seq, Hidden=2048]
        Returns:
            (hidden, logits)
        """
        n_p = prompt_embeds.shape[1]
        self.batch.n_tokens = n_p
        
        # 拷贝数据到批次
        ctypes.memmove(self.batch.embd, 
                       np.ascontiguousarray(prompt_embeds[0]).ctypes.data, 
                       prompt_embeds[0].nbytes)
        
        # 设置位置与 Logits 标记
        for i in range(n_p):
            # Qwen3 需要三个位置编码分量对齐 (Verified from 41-Inference.py)
            pos_val = self.cur_pos + i
            self.batch.pos[i] = self.batch.pos[n_p + i] = self.batch.pos[2*n_p + i] = pos_val
            self.batch.pos[3*n_p + i] = 0
            self.batch.n_seq_id[i] = 1
            self.batch.seq_id[i][0] = 0
            self.batch.logits[i] = 1 # 我们需要最后一个位置的 logits
            
        llama_status = llama.llama_decode(self.ctx, self.batch)
        if llama_status != 0:
            raise RuntimeError(f"Master Prefill Decode failed with status {llama_status} at pos {self.cur_pos}")

        # 提取最后一个位置的输出
        hidden = np.ctypeslib.as_array(llama.llama_get_embeddings(self.ctx), shape=(n_p, 2048))[-1].copy()
        logits = np.ctypeslib.as_array(llama.llama_get_logits(self.ctx), shape=(n_p, 3072))[-1].copy()
        
        self.cur_pos += n_p
        return hidden, logits

    def decode_step(self, feedback_embed: np.ndarray) -> tuple:
        """
        单步预测下一帧。
        Args:
            feedback_embed: [Hidden=2048] 汇总后的上一帧反馈向量
        Returns:
            (hidden, logits)
        """
        # 检查上下文是否已满
        if self.cur_pos >= self.n_ctx - 1:
            raise IndexError(f"Master context overflow: {self.cur_pos} >= {self.n_ctx}")
            
        self.batch.n_tokens = 1
        ctypes.memmove(self.batch.embd, 
                       feedback_embed.ctypes.data, 
                       feedback_embed.nbytes)
        
        self.batch.pos[0] = self.batch.pos[1] = self.batch.pos[2] = self.cur_pos
        self.batch.pos[3] = 0
        self.batch.n_seq_id[0] = 1
        self.batch.seq_id[0][0] = 0
        self.batch.logits[0] = 1
        
        llama_status = llama.llama_decode(self.ctx, self.batch)
        if llama_status != 0:
            raise RuntimeError(f"Master Step Decode failed at pos {self.cur_pos}")

        # 提取预测结果
        hidden = np.ctypeslib.as_array(llama.llama_get_embeddings(self.ctx), shape=(1, 2048))[0].copy()
        logits = np.ctypeslib.as_array(llama.llama_get_logits(self.ctx), shape=(1, 3072))[0].copy()
        
        self.cur_pos += 1
        return hidden, logits
