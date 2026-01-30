"""
stream.py - TTS 语音流
核心逻辑所在，管理单次会话的上下文，支持流式和非流式合成。
"""
import time
import numpy as np
from typing import Optional, List, Tuple, Union
from .constants import PROTOCOL, map_speaker, map_language
from .result import TTSResult, Timing, LoopOutput, TTSConfig
from .predictors.master import MasterPredictor
from .predictors.craftsman import CraftsmanPredictor

from . import llama, logger
from .prompt_builder import PromptBuilder, PromptData

class TTSStream:
    """
    保存大师、工匠、嘴巴记忆的语音流。
    """
    def __init__(self, engine, n_ctx=4096):
        self.engine = engine
        self.assets = engine.assets
        self.tokenizer = engine.tokenizer
        self.n_ctx = n_ctx
        
        # 1. 初始化流独立的 Context 和 Batch
        self._init_contexts()
        
        # 2. 初始化推理核心
        self.master = MasterPredictor(engine.m_model, self.m_ctx, self.m_batch, self.assets)
        self.craftsman = CraftsmanPredictor(engine.c_model, self.c_ctx, self.c_batch, self.assets)
        
        # 3. 身份锚点 (直接使用 TTSResult)
        self.identity: Optional[TTSResult] = None
        self.mouth = getattr(engine, 'mouth', None)

    def _init_contexts(self):
        """初始化此语音流专属的推理环境"""
        logger.info(f"[Stream] 正在初始化独立 Context (n_ctx={self.n_ctx})...")
        m_params = llama.llama_context_default_params()
        m_params.n_ctx = self.n_ctx
        m_params.embeddings = True
        self.m_ctx = llama.llama_init_from_model(self.engine.m_model, m_params)
        
        c_params = llama.llama_context_default_params()
        c_params.n_ctx = 512
        self.c_ctx = llama.llama_init_from_model(self.engine.c_model, c_params)
        
        self.m_batch = llama.llama_batch_init(self.n_ctx, 2048, 1)
        self.c_batch = llama.llama_batch_init(32, 1024, 1)

    def tts(self, 
            text: str, 
            language: str = "chinese",
            config: Optional[TTSConfig] = None) -> TTSResult:
        """
        同步合成接口（要求 Identity 已设置）。
        """
        if self.identity is None:
            raise RuntimeError("Identity is not set. Please call `set_identity` first.")

        cfg = config or TTSConfig()

        # 1. 构建 Prompt (由 PromptBuilder 统计耗时)
        pdata, timing = self._build_prompt_data(text, language, is_clone=True)

        # 2. 驱动内核进行生成 (儿子负责统计推理时间)
        lout = self._run_engine_loop(pdata, timing, cfg)

        # 3. 后处理 (渲染与封装)
        res = self._post_process(text, pdata, lout)
        
        return res

    def _build_prompt_data(self, text: str, language: str, is_clone: bool, speaker_id: Optional[str] = None) -> Tuple[PromptData, Timing]:
        """准备 Prompt 并初始化 Timing 对象"""
        lang_id = map_language(language)
        
        if is_clone:
            pdata = PromptBuilder.build_clone_prompt(text, self.identity, self.tokenizer, self.assets, lang_id)
        else:
            spk_id = map_speaker(speaker_id)
            pdata = PromptBuilder.build_native_prompt(text, self.tokenizer, self.assets, lang_id, spk_id)
            
        timing = Timing()
        timing.prompt_time = pdata.compile_time
        return pdata, timing

    def _run_engine_loop(self, 
                       pdata: PromptData,
                       timing: Timing,
                       cfg: TTSConfig) -> LoopOutput:
        """
        内核层：负责 Master 与 Craftsman 的逐帧推理循环。
        """
        self.master.clear_memory()
        self.mouth.reset()
        
        all_codes = []
        turn_summed_embeds = []

        # 大师 Prefill
        t_pre_s = time.time()
        m_hidden, m_logits = self.master.prefill(pdata.embd, seq_id=0)
        timing.prefill_time = time.time() - t_pre_s
        
        for step_idx in range(cfg.max_steps):
            code_0 = self.engine._do_sample(
                m_logits, 
                do_sample=cfg.do_sample, 
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k
            )
            if code_0 == PROTOCOL["EOS"]:
                m_hidden, m_logits = self.master.decode_step(
                    self.assets.emb_tables[0][PROTOCOL["EOS"]].flatten() + self.assets.tts_pad.flatten(),
                    seq_id=0
                )
                break
            
            # 工匠补全
            t_c_s = time.time()
            step_codes, step_embeds_2048 = self.craftsman.predict_frame(
                m_hidden, 
                code_0, 
                do_sample=cfg.sub_do_sample,
                temperature=cfg.sub_temperature,
                top_p=cfg.sub_top_p,
                top_k=cfg.sub_top_k
            )
            timing.craftsman_loop_time += (time.time() - t_c_s)
            
            all_codes.append(step_codes)
            
            # 大师反馈
            t_m_s = time.time()
            summed = np.sum(step_embeds_2048, axis=0) + self.assets.tts_pad.flatten()
            turn_summed_embeds.append(summed.copy())
            m_hidden, m_logits = self.master.decode_step(summed, seq_id=0)
            timing.master_loop_time += (time.time() - t_m_s)
            
        timing.total_steps = len(all_codes)
        return LoopOutput(all_codes=all_codes, summed_embeds=turn_summed_embeds, timing=timing)

    def _post_process(self, 
                     text: str, 
                     pdata: PromptData, 
                     lout: LoopOutput) -> TTSResult:
        """
        渲染音频并封装 TTSResult。
        """
        t_r_s = time.time()
        audio_out = self.mouth.decode_full(np.array(lout.all_codes))
        lout.timing.mouth_render_time = time.time() - t_r_s

        return TTSResult(
            audio=audio_out,
            text=text,
            text_ids=pdata.text_ids,
            spk_emb=pdata.spk_emb,
            codes=np.array(lout.all_codes),
            summed_embeds=lout.summed_embeds,
            stats=lout.timing
        )

    def save_audio(self, audio: np.ndarray, path: str):
        """兼容层：保存音频文件"""
        import soundfile as sf
        sf.write(path, audio, 24000)

    def play_audio(self, audio: np.ndarray):
        """兼容层：播放音频"""
        import sounddevice as sd
        sd.play(audio, 24000)
        sd.wait()

    def reset(self):
        """完全重置身份和状态"""
        self.identity = None
        self.master.clear_memory()
        self.mouth.reset()

    def shutdown(self):
        """释放占用的资源"""
        try:
            llama.llama_batch_free(self.m_batch)
            llama.llama_batch_free(self.c_batch)
            llama.llama_free(self.m_ctx)
            llama.llama_free(self.c_ctx)
        except: pass

    def __del__(self):
        self.shutdown()

    def set_identity(self, res: TTSResult):
        """直接设置身份锚点"""
        if not res.is_valid_anchor:
            raise ValueError("Provided TTSResult is not a valid anchor.")
        self.identity = res
        logger.info(f"🔒 Identity locked to text: '{res.text}'")

    def set_identity_from_speaker(self, speaker_id: str, text: str, language: str = "chinese", config: Optional[TTSConfig] = None) -> TTSResult:
        """原生定调：从指定说话人生成一个身份锚点结果"""
        logger.info(f"📍 Setting Identity from Speaker: {speaker_id}, language: {language}")
        
        cfg = config or TTSConfig()
        
        # 1. 编译 Prompt
        pdata, timing = self._build_prompt_data(text, language, is_clone=False, speaker_id=speaker_id)
        
        # 2. 推理循环
        lout = self._run_engine_loop(pdata, timing, cfg)
        
        # 3. 生成结果并设为锚点
        res = self._post_process(text, pdata, lout)
        self.set_identity(res)
        return res

    def set_identity_from_clone(self, wav_path: str, text: str, language: str = "chinese") -> Union[TTSResult, bool]:
        """克隆定调：从外部 WAV 文件提取特征并设为身份锚点"""
        if self.engine.encoder is None:
            logger.info("⚠️ [Stream] 编码器模型未就绪，音色克隆功能不可用。")
            return False
            
        logger.info(f"🎤 Setting Identity from Clone: {wav_path}")
        
        # 1. 提取特征
        codes, spk_emb = self.engine.encoder.encode(wav_path)
        
        # 2. 构造 TTSResult 作为锚点
        # 注意: 克隆锚点不需要 audio 和 summed_embeds (后者会自动重构)
        res = TTSResult(
            text=text,
            text_ids=[], # 暂时留空，PromptBuilder 会处理文本
            spk_emb=spk_emb,
            codes=codes
        )
        
        # 3. 设置为当前身份
        self.set_identity(res)
        return res
