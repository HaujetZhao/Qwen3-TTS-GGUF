"""
prompt_builder.py - Qwen3-TTS 提示词构造器 (极致模块化重构版)
"""
import time
import numpy as np
from typing import Optional, List, Union
from .constants import PROTOCOL

class PromptData:
    """包装构建好的 Prompt Embedding 数据"""
    def __init__(self, embd: np.ndarray, text: str, text_ids: List[int], spk_emb: np.ndarray, 
                 trailing_text_embd: Optional[np.ndarray] = None, compile_time: float = 0):
        self.embd = embd # (1, seq, 2048) - 这是进入 Talker 的初始 Prompt
        self.text = text
        self.text_ids = text_ids # 目标文本的 ID
        self.spk_emb = spk_emb
        self.trailing_text_embd = trailing_text_embd # (1, T_rem, 2048) - 待步内注入的文本池
        self.compile_time = compile_time

class PromptBuilder:
    @staticmethod
    def _get_ids(tokenizer, text: str) -> List[int]:
        """兼容 transformers 和 tokenizers 的 encode 返回值"""
        res = tokenizer.encode(text)
        if hasattr(res, "ids"):
            return res.ids
        return res

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
        """[声音克隆入口] 采用特征叠加 (Fusion) 协议"""
        t_start = time.time()
        p = PROTOCOL
        
        # 1. 构造“文本池” (Ref_Text + Target_Text + EOS)
        ref_ids = list(anchor.text_ids)
        target_ids = PromptBuilder._get_ids(tokenizer, text)
        full_text_ids = ref_ids + target_ids + [p["EOS_TOKEN"]]
        
        # 转换为已投影的文本向量
        text_pool = assets.text_table[full_text_ids] # (60, 2048)
        
        # 2. 构造“音频池” (Codec_BOS + Codes_Sum)
        codes = anchor.codes # (T, 16)
        audio_vectors = []
        # Codec BOS (2149)
        audio_vectors.append(assets.emb_tables[0][2149])
        # Audio steps sum
        for t in range(codes.shape[0]):
            step_sum = np.zeros(2048, dtype=np.float32)
            for q in range(16):
                step_sum += assets.emb_tables[q][codes[t, q]]
            audio_vectors.append(step_sum)
        audio_pool = np.array(audio_vectors) # (53, 2048)
        
        # 3. 执行叠加对齐 (Fuse)
        t_len = len(text_pool)
        a_len = len(audio_pool)
        
        # 如果音频更长，文本补 Pad
        if a_len > t_len:
            pad_seq = np.tile(assets.tts_pad, (a_len - t_len, 1))
            text_pool_padded = np.vstack([text_pool, pad_seq])
            icl_fused = text_pool_padded + audio_pool
            trailing_text = None
        else:
            # 如果文本更长，截取前 a_len 进行叠加，剩下的作为 trailing
            icl_fused = text_pool[:a_len] + audio_pool
            trailing_text = text_pool[a_len:]
            
        # 4. 构建前缀 (Prefix: Role + Control + Speaker + BOS)
        # 固定 9 个 Token 协议
        prefix = []
        # Role: <|im_start|>, assistant, \n
        for tid in [151644, 77091, 198]:
            prefix.append(assets.text_table[tid])
        
        # Control: [think, think_bos, lang, think_eos, spk, codec_pad] + 背景
        # 叠加背景 Pad (151671) 或 BOS (151672)
        pad = assets.tts_pad
        bos = assets.text_table[151672]
        
        control_ids = [2154, 2156, lang_id, 2157]
        for tid in control_ids:
            prefix.append(pad + assets.emb_tables[0][tid])
        
        # Speaker
        prefix.append(pad + anchor.spk_emb)
        
        # Codec Pad (叠加在 BOS 上)
        prefix.append(bos + assets.emb_tables[0][2148])
        
        # 5. 组装初始 Prompt
        initial_prompt = np.vstack([np.array(prefix), icl_fused])
        
        # 包装 Trailing Text
        trailing_text_np = None
        if trailing_text is not None and len(trailing_text) > 0:
            trailing_text_np = trailing_text.reshape(1, len(trailing_text), 2048).astype(np.float32)

        return PromptData(
            embd=initial_prompt.reshape(1, len(initial_prompt), 2048).astype(np.float32),
            text=text,
            text_ids=target_ids,
            spk_emb=anchor.spk_emb,
            trailing_text_embd=trailing_text_np,
            compile_time=time.time() - t_start
        )

    @staticmethod
    def _build_core(text: str, tokenizer, assets, lang_id: Optional[int], spk_id: Optional[int] = None, 
                    spk_emb: Optional[np.ndarray] = None, instruct: Optional[str] = None) -> PromptData:
        """[非克隆模式的核心构造器] 同样需要对齐首字叠加逻辑"""
        t_start = time.time()
        p = PROTOCOL
        embeds = []
        
        # 1. 指令块 (ChatML User)
        if instruct:
            ins_ids = [151644, 872, 198]
            ins_ids.extend(PromptBuilder._get_ids(tokenizer, instruct))
            ins_ids.extend([151645, 198])
            for tid in ins_ids: embeds.append(assets.text_table[tid])
        
        # 2. 角色块 + 控制块 (同克隆模式)
        for tid in [151644, 77091, 198]: embeds.append(assets.text_table[tid])
        
        pad = assets.tts_pad
        bos = assets.text_table[151672]
        control_ids = [2154, 2156, lang_id or 2055, 2157]
        for tid in control_ids:
            embeds.append(pad + assets.emb_tables[0][tid])
            
        cur_spk_emb = spk_emb if spk_emb is not None else assets.emb_tables[0][spk_id or p["SPK"]]
        embeds.append(pad + cur_spk_emb)
        
        # 3. 核心动作：首字叠加 (Fusion start)
        # 官方逻辑：最后一个预填充位是 BOS_EMB + CODEC_PAD
        embeds.append(bos + assets.emb_tables[0][2148])
        
        # 4. 文本处理：首字之后全进 Trailing
        target_ids = PromptBuilder._get_ids(tokenizer, text)
        # full_ids = target_ids + [p["EOS_TOKEN"]]
        full_ids = target_ids 
        
        # 取出第一个字叠加在最后一步 (Codec_BOS 2149) 上
        # 注意：官方 generate 里的 logic 是在第 0 步开始加第 0 个 token
        # 所以进入 generate 的 Prompt 应该只到 BOS 结束。
        
        text_pool = assets.text_table[full_ids]
        trailing_text_np = text_pool.reshape(1, len(text_pool), 2048).astype(np.float32)

        return PromptData(
            embd=np.array(embeds).reshape(1, len(embeds), 2048).astype(np.float32),
            text=text,
            text_ids=target_ids,
            spk_emb=cur_spk_emb,
            trailing_text_embd=trailing_text_np,
            compile_time=time.time() - t_start
        )
