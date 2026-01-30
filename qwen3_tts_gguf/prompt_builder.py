"""
prompt_builder.py - Qwen3-TTS 提示词构造器 (极致模块化重构版)
"""
import time
import numpy as np
from typing import Optional, List, Union
from .constants import PROTOCOL

class PromptData:
    """包装构建好的 Prompt Embedding 数据"""
    def __init__(self, embd: np.ndarray, text: str, text_ids: List[int], spk_emb: np.ndarray, codes: Optional[np.ndarray] = None, compile_time: float = 0):
        self.embd = embd # (1, seq, 2048)
        self.text = text
        self.text_ids = text_ids
        self.spk_emb = spk_emb
        self.codes = codes
        self.compile_time = compile_time

class PromptBuilder:
    
    @staticmethod
    def build_custom_prompt(text: str, tokenizer, assets, spk_id: int, lang_id: Optional[int] = None, instruct: Optional[str] = None) -> PromptData:
        """[精品音色入口]"""
        return PromptBuilder._build_core(text, tokenizer, assets, lang_id=lang_id, spk_id=spk_id, instruct=instruct)

    @staticmethod
    def build_design_prompt(text: str, tokenizer, assets, instruct: str, lang_id: Optional[int] = None) -> PromptData:
        """[音色设计入口]"""
        return PromptBuilder._build_core(text, tokenizer, assets, lang_id=lang_id, spk_id=None, instruct=instruct)

    @staticmethod
    def build_clone_prompt(text: str, tokenizer, assets, anchor, lang_id: int) -> PromptData:
        """[声音克隆入口] 通过 mid_embeds 注入 ICL 序列"""
        p = PROTOCOL
        mid_embeds = []
        
        # 1. 注入身份覆盖区文本 (Bos -> ID -> Eos)
        ref_ids = [p["BOS_TOKEN"]] + list(anchor.text_ids) + [p["EOS_TOKEN"]]
        for tid in ref_ids:
            mid_embeds.append(assets.text_table[tid] + assets.emb_tables[0][p["PAD"]])
            
        # 2. 注入音频特征码 (Codec BOS -> Codes -> Pad)
        codes = anchor.codes
        if codes is not None:
            # 硬性规定：我们自己的管道中，codes 必须是 (Steps, 16) 维度
            assert codes.ndim == 2 and codes.shape[1] == 16, f"音频特征码维度错误，必须为 (Steps, 16)，当前为 {codes.shape}"
            
            mid_embeds.append(assets.text_table[151671] + assets.emb_tables[0][2160]) # Codec BOS
            for step in range(codes.shape[0]): # T
                # 叠加 16 组码表特征
                summed_c = np.zeros(2048, dtype=np.float32)
                for q in range(16):
                    summed_c += assets.emb_tables[q][codes[step, q]]
                mid_embeds.append(assets.text_table[151671] + summed_c)
            mid_embeds.append(assets.text_table[151671] + assets.emb_tables[0][p["PAD"]])

        return PromptBuilder._build_core(
            text, tokenizer, assets, 
            lang_id=lang_id, 
            spk_emb=anchor.spk_emb, # 注入向量
            mid_embeds=mid_embeds
        )

    @staticmethod
    def _build_core(text: str, 
                    tokenizer, 
                    assets, 
                    lang_id: Optional[int], 
                    spk_id: Optional[int] = None, 
                    spk_emb: Optional[np.ndarray] = None,
                    instruct: Optional[str] = None,
                    mid_embeds: Optional[List[np.ndarray]] = None) -> PromptData:
        """
        [通用的极致核心] 
        承担所有模式的拼装职责：[Instruct] -> [Role] -> [Control] -> [Mid/ICL] -> [Task] -> [Activation]
        """
        t_start = time.time()
        p = PROTOCOL
        embeds = []
        
        # 1. 指令块 (ChatML User)
        if instruct:
            ins_ids = [151644, 872, 198] # <|im_start|>user\n
            ins_ids.extend(tokenizer.encode(instruct).ids)
            ins_ids.extend([151645, 198]) # <|im_end|>\n
            for tid in ins_ids: embeds.append(assets.text_table[tid].copy())
        
        # 2. 角色块 (ChatML Assistant)
        for tid in [151644, 77091, 198]: embeds.append(assets.text_table[tid].copy())
            
        # 3. 控制块 (Think/Lang/Spk)
        if lang_id is not None:
            c_tags = [(p["THINK"], p["THINK_BOS"], lang_id, p["THINK_EOS"])]
            for tid_offset in c_tags[0]:
                embeds.append(assets.text_table[151671] + assets.emb_tables[0][tid_offset])
        else:
            for tid_offset in [p["NOTHINK"], p["THINK_BOS"], p["THINK_EOS"]]:
                embeds.append(assets.text_table[151671] + assets.emb_tables[0][tid_offset])
            
        if spk_id is not None:
            embeds.append(assets.text_table[151671] + assets.emb_tables[0][spk_id])
        elif spk_emb is not None:
            embeds.append(assets.text_table[151671] + spk_emb)

        # 4. 中间注入块 (ICL / Identity Overlay)
        if mid_embeds:
            embeds.extend(mid_embeds)

        # 5. 任务文本块
        ids = tokenizer.encode(text).ids
        embeds.append(assets.text_table[p["BOS_TOKEN"]] + assets.emb_tables[0][p["PAD"]])
        for tid in ids: 
            embeds.append(assets.text_table[tid] + assets.emb_tables[0][p["PAD"]])
        embeds.append(assets.text_table[p["EOS_TOKEN"]] + assets.emb_tables[0][p["PAD"]])
        
        # 6. 最终激活块 (BOS 2149)
        embeds.append(assets.text_table[151671] + assets.emb_tables[0][p["BOS"]])
        
        embd_np = np.array(embeds).reshape(1, len(embeds), 2048).astype(np.float32)
        
        # 确定输出的 spk_emb 用于后处理记录
        out_spk_emb = spk_emb if spk_emb is not None else (assets.emb_tables[0][spk_id].copy() if spk_id is not None else np.zeros(2048, dtype=np.float32))
        
        return PromptData(embd=embd_np, text=text, text_ids=ids, spk_emb=out_spk_emb, compile_time=time.time() - t_start)
