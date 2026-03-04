"""
prompt_builder.py - Qwen3-TTS 统一提示词构造器

通过参数组合控制三种生成模式：
  1. 非流式 (codes=None, icl=False): CustomVoice / VoiceDesign
  2. 流式   (codes=None, icl=True):  流式文本注入
  3. ICL    (codes=array):           VoiceClone (文本+音频按位融合)
"""
import time
import numpy as np
from typing import Optional, List, Union
from .schema.constants import PROTOCOL, map_speaker
from . import logger

class PromptData:
    """包装构建好的 Prompt Embedding 数据"""
    def __init__(self, embd: np.ndarray, text: str, text_ids: List[int], spk_emb: Optional[np.ndarray], 
                 trailing_text_embd: Optional[np.ndarray] = None, compile_time: float = 0):
        self.embd = embd # (1, seq, D) - 进入 Talker 的初始 Prompt
        self.text = text
        self.text_ids = text_ids # 目标文本的 ID
        self.spk_emb = spk_emb
        self.trailing_text_embd = trailing_text_embd # (1, T_rem, D) - 待步内注入的文本池
        self.compile_time = compile_time

class PromptBuilder:
    def __init__(self, tokenizer, assets):
        self.tokenizer = tokenizer
        self.assets = assets
        self.p = PROTOCOL

    def _get_ids(self, text: str) -> List[int]:
        """内聚 tokenizer 调度"""
        res = self.tokenizer.encode(text)
        if hasattr(res, "ids"):
            return res.ids
        return res

    # ===================================================================
    # 公开入口方法 (去掉了 tokenizer 和 assets 参数)
    # ===================================================================

    def build_design_prompt(self, text: str, instruct: str, lang_id: Optional[int] = None) -> PromptData:
        """[音色设计入口]"""
        return self._build_core(text, lang_id=lang_id, instruct=instruct)
    
    def build_custom_prompt(self, text: str, speaker: Union[str, int, np.ndarray], 
                            lang_id: Optional[int] = None, instruct: Optional[str] = None) -> PromptData:
        """[精品音色入口]"""
        return self._build_core(text, lang_id=lang_id, speaker=speaker, instruct=instruct)

    def build_clone_prompt(self, text: str, voice, lang_id: int = None) -> PromptData:
        """[声音克隆入口]"""
        return self._build_core(
            text,
            lang_id=lang_id,
            speaker=voice.spk_emb,
            ref_text=voice.text,
            codes=voice.codes,
        )

    # ===================================================================
    # 统一核心构造器
    # ===================================================================

    def _build_core(self, text: str,
                    lang_id: Optional[int] = None,
                    speaker: Union[str, int, np.ndarray, None] = None,
                    instruct: Optional[str] = None,
                    ref_text: Optional[str] = None,
                    ref_ids: Optional[List[int]] = None,
                    codes: Optional[np.ndarray] = None,
                    icl: bool = False) -> PromptData:
        r"""
        [统一 Prompt 构造器]

        通过参数组合控制三种模式:

        ┌─────────────────────────────────────────────────────────────────┐
        │ 模式 A: 非流式 (codes=None, icl=False)                         │
        │   CustomVoice / VoiceDesign                                    │
        │   所有文本 + codec_pad 一次性放入 Prompt, trailing=None         │
        │                                                                │
        │ 模式 B: 流式 (codes=None, icl=True)                            │
        │   仅首个文本 token 进入 Prompt, 其余 + tts_eos 进 trailing      │
        │                                                                │
        │ 模式 C: ICL (codes=array)                                      │
        │   VoiceClone: 文本与参考音频按位融合, 超出部分进 trailing        │
        └─────────────────────────────────────────────────────────────────┘

        通用 Prefix 结构 (所有模式共享):
            [指令块]              (可选)
            <|im_start|>assistant\n
            tts_pad + think/nothink
            tts_pad + think_bos
           (tts_pad + lang_id)    (可选)
            tts_pad + think_eos
           (tts_pad + spk_emb)    (可选)
            tts_bos + codec_pad
        """

        t_start = time.time()
        p = self.p
        _ids = self._get_ids

        # 1. 文本 ID 构造
        if ref_text is not None:
            text_ids = _ids(ref_text + text)
        elif ref_ids is not None:
            text_ids = list(ref_ids) + _ids(text)
        else:
            text_ids = _ids(text)

        target_text_ids = _ids(text)

        # 2. 前缀构造
        prefix = []
        tts_pad = self.assets.tts_pad
        hidden_dim = tts_pad.shape[0]

        # 2a. 指令块
        if instruct:
            ins_ids = _ids(f"<|im_start|>user\n{instruct}<|im_end|>\n")
            for tid in ins_ids:
                prefix.append(self.assets.text_table[tid])

        # 2b. 角色头
        role_ids = _ids("<|im_start|>assistant\n")
        for tid in role_ids:
            prefix.append(self.assets.text_table[tid])

        # 2c. 语言
        if lang_id is not None and lang_id in range(2048, 2147):
            prefill_ids = [p['THINK'], p['THINK_BOS'], lang_id, p['THINK_EOS']]
        else:
            prefill_ids = [p['NOTHINK'], p['THINK_BOS'], p['THINK_EOS']]
        for tid in prefill_ids:
            prefix.append(tts_pad + self.assets.emb_tables[0][tid])

        # 2d. 说话人
        cur_spk_emb = None
        if isinstance(speaker, np.ndarray):
            cur_spk_emb = speaker
        elif speaker is not None:
            spk_id = map_speaker(speaker)
            if spk_id is not None:
                cur_spk_emb = self.assets.emb_tables[0][spk_id]
        
        if cur_spk_emb is not None:
            prefix.append(tts_pad + cur_spk_emb)

        # 2e. TTS_BOS + codec_pad
        prefix.append(self.assets.text_table[p['TTS_BOS']] + self.assets.emb_tables[0][p['PAD']])

        # 3. Body 构造
        if codes is not None:
            # 模式 C: ICL
            text_pool_ids = text_ids + [p['TTS_EOS']]
            text_pool = self.assets.text_table[text_pool_ids]
            audio_vectors = [self.assets.emb_tables[0][p['BOS']]]
            for t in range(codes.shape[0]):
                step_sum = np.zeros(hidden_dim, dtype=np.float32)
                for q in range(16):
                    step_sum += self.assets.emb_tables[q][codes[t, q]]
                audio_vectors.append(step_sum)
            audio_pool = np.array(audio_vectors)
            t_len, a_len = len(text_pool), len(audio_pool)
            if t_len > a_len:
                body = text_pool[:a_len] + audio_pool
                trailing = text_pool[a_len:]
            else:
                pad_seq = np.tile(tts_pad, (a_len - t_len, 1))
                text_padded = np.vstack([text_pool, pad_seq])
                body = text_padded + audio_pool
                trailing = None
        elif icl:
            # 模式 B: 流式，没有参考音频，首个文本 token + codec_bos 进入 prompt
            first_text_emb = self.assets.text_table[text_ids[0]]
            codec_bos_emb = self.assets.emb_tables[0][p['BOS']]
            body = (first_text_emb + codec_bos_emb).reshape(1, -1)

            # trailing: 剩余文本 + tts_eos (纯文本嵌入, decode_step 中与音频融合)
            tts_eos_emb = self.assets.text_table[p['TTS_EOS']].reshape(1, -1)
            if len(text_ids) > 1:
                remaining = self.assets.text_table[text_ids[1:]]
                trailing = np.vstack([remaining, tts_eos_emb])
            else:
                trailing = tts_eos_emb
        else:
            # 模式 A: 非流式，全量入 Prompt
            codec_pad_emb = self.assets.emb_tables[0][p['PAD']]
            text_pool = self.assets.text_table[text_ids]
            if len(text_pool) > 0:
                text_fused = text_pool + codec_pad_emb
            else:
                text_fused = np.empty((0, hidden_dim), dtype=np.float32)
            eos_fused = (self.assets.text_table[p['TTS_EOS']] + codec_pad_emb).reshape(1, -1)
            bos_codec = (tts_pad + self.assets.emb_tables[0][p['BOS']]).reshape(1, -1)
            body = np.vstack([text_fused, eos_fused, bos_codec]) if len(text_fused) > 0 \
                else np.vstack([eos_fused, bos_codec])
            trailing = None

        # 4. 组装
        initial_prompt = np.vstack([np.array(prefix), body])
        initial_prompt = initial_prompt.reshape(1, len(initial_prompt), hidden_dim).astype(np.float32)

        trailing_text_np = None
        if trailing is not None and len(trailing) > 0:
            trailing_text_np = trailing.reshape(1, len(trailing), hidden_dim).astype(np.float32)

        return PromptData(
            embd=initial_prompt,
            text=text,
            text_ids=target_text_ids,
            spk_emb=cur_spk_emb,
            trailing_text_embd=trailing_text_np,
            compile_time=time.time() - t_start
        )
