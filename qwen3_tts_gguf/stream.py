"""
stream.py - TTS 语音流
核心逻辑所在，管理单次会话的上下文，支持流式和非流式合成。
"""
import time
import numpy as np
from typing import Optional, List
from .constants import PROTOCOL, SPEAKER_MAP, LANGUAGE_MAP
from .result import SynthesisResult
from .predictors.master import MasterPredictor
from .predictors.craftsman import CraftsmanPredictor

class TTSStream:
    """
    保存大师、工匠、嘴巴记忆的语音流。
    """
    def __init__(self, engine, speaker_id, language):
        self.engine = engine
        self.assets = engine.assets
        self.tokenizer = engine.tokenizer
        
        # 映射 ID
        self.spk_id = self._map_speaker(speaker_id)
        self.lang_id = self._map_language(language)
        
        # 内部状态
        # 注意：为了让同一条流能持续合成，这两个 predictor 的状态是持久的
        self.master = MasterPredictor(engine.m_model, engine.m_ctx, engine.m_batch, self.assets)
        self.craftsman = CraftsmanPredictor(engine.c_model, engine.c_ctx, engine.c_batch, self.assets)
        self.mouth = getattr(engine, 'mouth', None)
        
        self.is_first_sentence = True

    def synthesize(self, text: str, 
                   chunk_size: int = 25,
                   play: bool = False,
                   save_path: Optional[str] = None,
                   temperature: float = 0.9,
                   max_steps: int = 600,
                   verbose: bool = True) -> SynthesisResult:
        """
        合成文本。
        """
        start_time = time.time()
        
        # 1. 准备 Prompt
        t_p_start = time.time()
        # 如果是第一句，需要推入完整的 Prompt 模板（包含 Header 和音色标签）
        prompt_embeds = self._construct_prompt(text, self.is_first_sentence)
        prompt_time = time.time() - t_p_start
        
        # 2. 推理执行 (大师 + 工匠)
        all_codes = []
        all_audio_chunks = []
        
        stats = {
            "prefill": 0.0,
            "master": 0.0,
            "craftsman": 0.0,
            "mouth": 0.0
        }

        # Prefill / Initial Decode
        t_pre_s = time.time()
        m_hidden, m_logits = self.master.prefill(prompt_embeds)
        stats["prefill"] = time.time() - t_pre_s
        
        loop_start = time.time()
        cur_codes_batch = []
        
        for step_idx in range(max_steps):
            # (A) 大师选点
            code_0 = self.engine._do_sample(m_logits, temperature)
            if code_0 == PROTOCOL["EOS"]:
                if verbose: print(f"  └─ 步数 {step_idx}: 获得 EOS 信号，结束生成。")
                break
            
            # (B) 工匠补全 16 级码
            t_c_s = time.time()
            step_codes, step_embeds_2048 = self.craftsman.predict_frame(m_hidden, code_0, temperature=temperature)
            stats["craftsman"] += (time.time() - t_c_s)
            
            all_codes.append(step_codes)
            cur_codes_batch.append(step_codes)
            
            # (C) 流式分级（如果是流式引擎）
            if self.engine.streaming and len(cur_codes_batch) >= chunk_size:
                self.engine.codes_q.put((list(cur_codes_batch), False))
                cur_codes_batch = []
                if verbose: print(f"  └─ 流式分片推送 (帧 {len(all_codes)-chunk_size}-{len(all_codes)})")

            # (D) 大师反馈 (Feedback)
            t_m_s = time.time()
            summed_embed = np.sum(step_embeds_2048, axis=0) + self.assets.tts_pad.flatten()
            m_hidden, m_logits = self.master.decode_step(summed_embed)
            stats["master"] += (time.time() - t_m_s)
            
        else:
            if verbose: print(f"  ⚠️ 推理达到步数上限 {max_steps}。")

        # 3. 渲染音频 (Mouth)
        # 推送最后余量
        if self.engine.streaming:
            t_r_s = time.time()
            self.engine.codes_q.put((cur_codes_batch, True)) # is_final=True
            # 流式引擎下，音频数据由 pcm_q 异步传回，或者直接放任 worker 处理
            # 为了返回 SynthesisResult，我们需要从 record_q 或 pcm_q 收集（如果用户需要同步结果）
            # 简化起见：流式模式下 synthesize 返回一个空的 audio，用户通过 play 参数或 output 目录查看
            audio_out = np.array([], dtype=np.float32)
            stats["mouth"] = 0 # 统计由 worker 承担
        else:
            t_r_s = time.time()
            audio_out = self.mouth.decode_full(np.array(all_codes))
            stats["mouth"] = time.time() - t_r_s

        # 4. 封装结果
        self.is_first_sentence = False
        
        res = SynthesisResult(
            audio=audio_out,
            text=text,
            prompt_time=prompt_time,
            prefill_time=stats["prefill"],
            master_loop_time=stats["master"],
            craftsman_loop_time=stats["craftsman"],
            mouth_render_time=stats["mouth"],
            total_steps=len(all_codes)
        )
        
        if save_path:
            import soundfile as sf
            sf.write(save_path, audio_out, 24000)
            
        if play and not self.engine.streaming:
            import sounddevice as sd
            sd.play(audio_out, 24000)
            sd.wait()
            
        return res

    def reset(self):
        """完全重置，开启全新对话"""
        self.master.clear_memory()
        self.is_first_sentence = True
        # 注意: 口腔解码器也需要重置
        if self.engine.streaming:
            self.engine.codes_q.put("CLEAR")
        else:
            self.mouth.reset()

    def _construct_prompt(self, text, include_header=True):
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        p = PROTOCOL
        
        seq = []
        if include_header:
            # 标准全量 Header
            seq.extend([
                (151644, 0), (77091, 0), (198, 0), # Header <|im_start|>system\n
                (151671, p["THINK"]), 
                (151671, p["THINK_BOS"]), 
                (151671, self.lang_id), 
                (151671, p["THINK_EOS"]), 
                (151671, self.spk_id), 
                (p["BOS_TOKEN"], p["PAD"]) 
            ])
            
        # 文本正文
        for tid in ids: seq.append((tid, p["PAD"]))
        seq.append((p["EOS_TOKEN"], p["PAD"]))
        
        # 激活 Codec 生成的标志
        seq.append((151671, p["BOS"])) # 2149
        
        # 转换为 Embeddings
        embeds = []
        for tid, cid in seq:
            v = self.assets.text_table[tid] + (self.assets.emb_tables[0][cid] if cid != 0 else 0)
            embeds.append(v)
            
        return np.array(embeds).reshape(1, len(seq), 2048).astype(np.float32)

    def _map_speaker(self, spk):
        if isinstance(spk, int): return spk
        return SPEAKER_MAP.get(str(spk).lower(), 3065)

    def _map_language(self, lang):
        if isinstance(lang, int): return lang
        return LANGUAGE_MAP.get(str(lang).lower(), 2055)
