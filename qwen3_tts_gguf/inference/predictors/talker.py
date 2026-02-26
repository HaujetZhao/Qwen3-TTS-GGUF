"""
predictors/talker.py - 大师预测器 (Talker)
负责 Qwen3-TTS 的主体 LLM 推理。
管理 llama.cpp 上下文、KV Cache 和步数。
"""
import ctypes
import numpy as np
import time
from .. import llama

from ..constants import PROTOCOL

class TalkerPredictor:
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
        
        # 预分配单步位置 Buffer (3 pos + 1 zero)
        self.pos_step_buffer = np.zeros(4, dtype=np.int32)
        
        # 步进式生成状态
        self.trailing_text_pool = None
        self.step_idx = 0
        
    def clear_memory(self):
        """完全清空 KV Cache"""
        self.ctx.clear_kv_cache()
        self.cur_pos = 0
        self.trailing_text_pool = None
        self.step_idx = 0

    def prefill(self, pdata, seq_id: int = 0) -> np.ndarray:
        """
        全量推入初始 Prompt。
        Args:
            pdata: PromptData 对象，包含 embd 和 trailing_text_embd
            seq_id: 序列 ID
        Returns:
            hidden: 最后一个位置的隐层输出
        """
        prompt_embeds = pdata.embd
        n_p = prompt_embeds.shape[1]
        
        # 初始化步进文本池
        if pdata.trailing_text_embd is not None:
            self.trailing_text_pool = pdata.trailing_text_embd[0]
        else:
            self.trailing_text_pool = None
        self.step_idx = 0
        
        # 构造 Qwen3 专用的位置编码 (3层Pos + 1层Zero)
        pos_base = np.arange(self.cur_pos, self.cur_pos + n_p, dtype=np.int32)
        pos_arr = np.concatenate([pos_base, pos_base, pos_base, np.zeros(n_p, dtype=np.int32)])
        
        # 注入数据
        self.batch.set_embd(prompt_embeds[0], pos=pos_arr, seq_id=seq_id)
            
        llama_status = self.ctx.decode(self.batch)
        if llama_status != 0:
            raise RuntimeError(f"Talker Prefill Decode failed with status {llama_status} at pos {self.cur_pos}")

        hidden_ptr = self.ctx.get_embeddings()
        hidden = np.ctypeslib.as_array(hidden_ptr, shape=(n_p, 2048))[-1].copy()
        
        self.cur_pos += n_p
        return hidden

    def decode_step(self, audio_embed: np.ndarray, seq_id: int = 0) -> np.ndarray:
        """
        单步预测：执行 [音频特征 + 文本特征] 的融合投喂。
        Args:
            audio_embed: [Hidden=2048] 纯音频特征（16层叠加后的）
        Returns:
            hidden: [Hidden=2048]
        """
        if self.cur_pos >= self.n_ctx - 1:
            raise IndexError(f"Talker context overflow: {self.cur_pos} >= {self.n_ctx}")
            
        # 1. 动态特征融合 (Fusion)
        if self.trailing_text_pool is not None and self.step_idx < len(self.trailing_text_pool):
            text_vec = self.trailing_text_pool[self.step_idx]
        else:
            # 文本耗尽，使用 Pad 填充背景
            text_vec = self.assets.tts_pad
            
        fused_embed = audio_embed + text_vec
        self.step_idx += 1
            
        # 2. 投喂融合后的特征
        if fused_embed.ndim == 1:
            fused_embed = fused_embed.reshape(1, -1)
            
        # 构造单步位置编码
        self.pos_step_buffer[0:3] = self.cur_pos
        self.batch.set_embd(fused_embed, pos=self.pos_step_buffer, seq_id=seq_id)
        
        llama_status = self.ctx.decode(self.batch)
        if llama_status != 0:
            raise RuntimeError(f"Talker Step Decode failed at pos {self.cur_pos}")

        hidden_ptr = self.ctx.get_embeddings()
        hidden = np.ctypeslib.as_array(hidden_ptr, shape=(1, 2048))[0].copy()
        
        self.cur_pos += 1
        return hidden
