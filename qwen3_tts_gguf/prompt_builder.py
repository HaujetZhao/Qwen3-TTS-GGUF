"""
prompt_builder.py - Prompt 构造工厂
提供静态方法用于构建 Master 模型所需的嵌入序列。
支持：1. 原生生成模式 2. 身份克隆引导模式
"""
import time
import numpy as np
from . import logger
from dataclasses import dataclass
from typing import List, Tuple, Optional
from .constants import PROTOCOL

@dataclass
class PromptData:
    """Prompt 构造结果封装"""
    embd: np.ndarray      # 最终输入 Master 的嵌入序列 (1, L, 2048)
    text: str            # 当前任务文本
    text_ids: List[int]  # 文本 Token IDs
    spk_emb: np.ndarray  # 本次使用的全局音色向量 (2048)
    codes: Optional[np.ndarray] = None # 如果是克隆模式，包含参考音频的 codes
    compile_time: float = 0.0          # Prompt 构建耗时

class PromptBuilder:
    @staticmethod
    def build_native_prompt(text: str, tokenizer, assets, lang_id: int, spk_id: int) -> PromptData:
        """
        构建原生生成模式的 Prompt (Header + Text)
        """
        t_start = time.time()
        ids = tokenizer.encode(text).ids
        p = PROTOCOL
        
        # 1. 构造 Token 序列
        seq = [
            (151644, 0), (77091, 0), (198, 0), # <|im_start|>system\n
            (151671, p["THINK"]), 
            (151671, p["THINK_BOS"]), 
            (151671, lang_id), 
            (151671, p["THINK_EOS"]), 
            (151671, spk_id), 
            (p["BOS_TOKEN"], p["PAD"]),
        ]
        
        for tid in ids: seq.append((tid, p["PAD"]))
        seq.append((p["EOS_TOKEN"], p["PAD"]))
        seq.append((151671, p["BOS"])) # 2149 / 激活标志
        
        # 2. 投影为 Embedding
        embeds = []
        for tid, aid in seq:
            v = assets.text_table[tid].copy()
            if aid != 0:
                v += assets.emb_tables[0][aid]
            embeds.append(v)
            
        embd_np = np.array(embeds).reshape(1, len(seq), 2048).astype(np.float32)
        
        # 获取 spk_emb 向量，供后续调用者直接使用，无需查表
        spk_emb_vec = assets.emb_tables[0][spk_id].copy()
        
        return PromptData(
            embd=embd_np,
            text=text,
            text_ids=ids,
            spk_emb=spk_emb_vec,
            codes=None,
            compile_time=time.time() - t_start
        )

    @staticmethod
    def build_clone_prompt(text: str, anchor: 'TTSResult', tokenizer, assets, lang_id: int) -> PromptData:
        """
        构建身份克隆引导模式的 Prompt
        """
        t_start = time.time()
        if not anchor.is_valid_anchor:
            raise ValueError("Anchor (TTSResult) is not a valid anchor for cloning.")
            
        # [NEW] 特征按需动态重构: 如果没有 summed_embeds，则从 codes 重建
        if anchor.summed_embeds is None or len(anchor.summed_embeds) == 0:
            logger.info("⚡ Voice summed_embeds is missing. Reconstructing from codes...")
            codes = anchor.codes
            reconstructed = []
            for t in range(codes.shape[0]):
                frame_codes = codes[t]
                # 叠加 16 个分步码表的特征 + PAD 偏置
                frame_vec = np.zeros(2048, dtype=np.float32)
                for q in range(len(frame_codes)):
                    frame_vec += assets.emb_tables[q][frame_codes[q]]
                frame_vec += assets.tts_pad
                reconstructed.append(frame_vec)
            anchor.summed_embeds = reconstructed
            
        ids = tokenizer.encode(text).ids
        p = PROTOCOL
        
        # 1. Header 部分
        header = [
            (151644, 0), (77091, 0), (198, 0),
            (151671, p["THINK"]), 
            (151671, p["THINK_BOS"]), 
            (151671, lang_id), 
            (151671, p["THINK_EOS"]), 
            (151671, 0), 
        ]
        
        embeds = []
        for tid, aid in header:
            v = assets.text_table[tid].copy()
            if aid != 0:
                v += assets.emb_tables[0][aid]
            embeds.append(v)
            
        # 注入全局音色
        embeds[-1] += anchor.spk_emb
        
        # 2. Identity Overlay 区域
        n_frames = anchor.codes.shape[0]
        ref_ids = anchor.text_ids
        if len(ref_ids) < n_frames:
            ref_ids = ref_ids + [151671] * (n_frames - len(ref_ids))
        else:
            ref_ids = ref_ids[:n_frames]
            
        for t in range(n_frames):
            v = assets.text_table[ref_ids[t]].copy()
            v += anchor.summed_embeds[t]
            embeds.append(v)
            
        # 3. 新文本任务区域
        embeds.append(assets.text_table[p["BOS_TOKEN"]] + assets.emb_tables[0][p["PAD"]])
        for tid in ids: 
            embeds.append(assets.text_table[tid] + assets.emb_tables[0][p["PAD"]])
        embeds.append(assets.text_table[p["EOS_TOKEN"]] + assets.emb_tables[0][p["PAD"]])
        embeds.append(assets.text_table[151671] + assets.emb_tables[0][p["BOS"]])
        
        embd_np = np.array(embeds).reshape(1, len(embeds), 2048).astype(np.float32)
        
        return PromptData(
            embd=embd_np,
            text=text,
            text_ids=ids,
            spk_emb=anchor.spk_emb.copy(),
            codes=anchor.codes.copy(),
            compile_time=time.time() - t_start
        )
