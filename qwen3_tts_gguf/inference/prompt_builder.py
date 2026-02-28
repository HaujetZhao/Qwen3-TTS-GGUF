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
    def _wrap_ref(text: str) -> str:
        """官方 Ref 包装: <|im_start|>assistant\n{text}<|im_end|>\n"""
        return f"<|im_start|>assistant\n{text}<|im_end|>\n"

    @staticmethod
    def _wrap_target(text: str) -> str:
        """官方 Target 包装: <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"""
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    
    @staticmethod
    def build_design_prompt(text: str, tokenizer, assets, instruct: str, lang_id: Optional[int] = None) -> PromptData:
        """[音色设计入口]"""
        return PromptBuilder._build_core(text, tokenizer, assets, lang_id=lang_id, spk_id=None, instruct=instruct)
    
    @staticmethod
    def build_custom_prompt(text: str, tokenizer, assets, spk_id: int, lang_id: Optional[int] = None, instruct: Optional[str] = None) -> PromptData:
        """[精品音色入口]"""
        return PromptBuilder._build_core(text, tokenizer, assets, lang_id=lang_id, spk_id=spk_id, instruct=instruct)

    @staticmethod
    def build_clone_prompt(text: str, tokenizer, assets, voice, lang_id: int = None) -> PromptData:
        """[声音克隆入口] 采用特征叠加 (Fusion) 协议 - 完美对齐官方逻辑"""
        t_start = time.time()
        
        p = PROTOCOL
        # 1. 构造官方切片的 Text ID 序列
        ref_full_ids = PromptBuilder._get_ids(tokenizer, PromptBuilder._wrap_ref(voice.text))
        ref_id_slice = ref_full_ids[3:-2]
        
        target_full_ids = PromptBuilder._get_ids(tokenizer, PromptBuilder._wrap_target(text))
        target_id_slice = target_full_ids[3:-5]
        
        # 最终文本池 = Ref Slice + Target Slice + EOS
        full_text_ids = ref_id_slice + target_id_slice + [p['EOS_TOKEN']] 
        text_pool = assets.text_table[full_text_ids]
        
        # 2. 构造音频池 (Codec_BOS + Codes_Sum)
        codes = voice.codes
        audio_vectors = []
        audio_vectors.append(assets.emb_tables[0][2149]) # Codec BOS
        for t in range(codes.shape[0]):
            step_sum = np.zeros(2048, dtype=np.float32)
            for q in range(16):
                step_sum += assets.emb_tables[q][codes[t, q]]
            audio_vectors.append(step_sum)
        audio_pool = np.array(audio_vectors) # (T2, 2048)

        # 3. 文本和音频融合
        t_len = len(text_pool)
        a_len = len(audio_pool)
        
        if t_len > a_len:
            # 文本更长：融合前 a_len，剩下的作为 trailing
            icl_fused = text_pool[:a_len] + audio_pool
            trailing_text = text_pool[a_len:]
        else:
            # 音频更长：文本补 Pad
            pad_seq = np.tile(assets.tts_pad, (a_len - t_len, 1))
            text_pool_padded = np.vstack([text_pool, pad_seq])
            icl_fused = text_pool_padded + audio_pool
            trailing_text = None

        # 4. 构建前缀
        prefix = []

        # Role: <|im_start|>, assistant, \n 
        for tid in target_full_ids[:3]:
            prefix.append(assets.text_table[tid])
        
        # Language
        tts_pad = assets.tts_pad
        if lang_id and lang_id in range(2048, 2147): 
            prefill_ids = [p['THINK'], p['THINK_BOS'], lang_id, p['THINK_EOS']] 
        else: 
            prefill_ids = [p['THINK'], p['THINK_BOS'], p['THINK_EOS']]
        for tid in prefill_ids:
            prefix.append(tts_pad + assets.emb_tables[0][tid])
        
        # Speaker
        prefix.append(tts_pad + voice.spk_emb)
        
        # BOS
        bos_text = assets.text_table[p['BOS_TOKEN']]
        prefix.append(bos_text + assets.emb_tables[0][p['PAD']])
        

        # 5. 组装
        initial_prompt = np.vstack([np.array(prefix), icl_fused])
        initial_prompt = initial_prompt.reshape(1, len(initial_prompt), 2048).astype(np.float32)
        
        trailing_text_np = None
        if trailing_text is not None and len(trailing_text) > 0:
            trailing_text_np = trailing_text.reshape(1, len(trailing_text), 2048).astype(np.float32)

        return PromptData(
            embd=initial_prompt,
            text=text,
            text_ids=target_id_slice,
            spk_emb=voice.spk_emb,
            trailing_text_embd=trailing_text_np,
            compile_time=time.time() - t_start
        )

    @staticmethod
    def _build_core(text: str, tokenizer, assets, lang_id: Optional[int], spk_id: Optional[int] = None, 
                    spk_emb: Optional[np.ndarray] = None, instruct: Optional[str] = None) -> PromptData:
        """[基础生成构造器]"""
        t_start = time.time()
        p = PROTOCOL
        prefix = []
        
        # 1. 指令块 (User Role)
        if instruct:
            ins_full_ids = PromptBuilder._get_ids(tokenizer, f"<|im_start|>user\n{instruct}<|im_end|>\n")
            for tid in ins_full_ids: prefix.append(assets.text_table[tid])
        
        # 2. 角色块
        target_full_ids = PromptBuilder._get_ids(tokenizer, PromptBuilder._wrap_target(text))
        for tid in target_full_ids[:3]:
            prefix.append(assets.text_table[tid])
            
        # 3. 控制块
        pad = assets.tts_pad
        prefill_ids = [2154, 2156, lang_id or 2055, 2157]
        for tid in prefill_ids:
            prefix.append(pad + assets.emb_tables[0][tid])
            
        cur_spk_emb = spk_emb if spk_emb is not None else assets.emb_tables[0][spk_id or p["SPK"]]
        prefix.append(pad + cur_spk_emb)
        
        # 4. Prefill 结束位与首字融合
        bos_text = assets.text_table[151672]
        # 基础模式下，首部直接连 text
        # 官方逻辑 L2201: text_projection(hidden(input_id[:,3:4])) + codec_bos
        prefix.append(bos_text + assets.emb_tables[0][2148])
        
        # 5. 文本池处理
        # 剩下部分入 Trailing。切片为 [3:-5] 后接 EOS
        target_id_slice = target_full_ids[3:-5]
        full_ids = target_id_slice + [151643]
        
        text_pool = assets.text_table[full_ids]
        trailing_text_np = text_pool.reshape(1, len(text_pool), 2048).astype(np.float32)

        return PromptData(
            embd=np.array(prefix).reshape(1, len(prefix), 2048).astype(np.float32),
            text=text,
            text_ids=target_id_slice,
            spk_emb=cur_spk_emb,
            trailing_text_embd=trailing_text_np,
            compile_time=time.time() - t_start
        )

