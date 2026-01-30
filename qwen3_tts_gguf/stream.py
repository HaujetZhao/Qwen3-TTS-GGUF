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
    def __init__(self, engine, n_ctx=4096, voice_path: Optional[str] = None):
        self.engine = engine
        self.assets = engine.assets
        self.tokenizer = engine.tokenizer
        self.n_ctx = n_ctx
        
        # 1. 初始化流独立的 Context 和 Batch
        self._init_contexts()
        
        # 2. 初始化推理核心
        self.master = MasterPredictor(engine.m_model, self.m_ctx, self.m_batch, self.assets)
        self.craftsman = CraftsmanPredictor(engine.c_model, self.c_ctx, self.c_batch, self.assets)
        
        # 3. 音色锚点 (Voice)
        self.voice: Optional[TTSResult] = None
        if voice_path:
            self.set_voice_from_json(voice_path)
            
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

    # =========================================================================
    # 核心推理 API (Adapt to Base / CustomVoice / VoiceDesign)
    # =========================================================================

    def clone(self, 
              text: str, 
              language: str = "chinese",
              config: Optional[TTSConfig] = None,
              verbose: bool = True) -> TTSResult:
        """
        [克隆模式] 对应 Base 模型能力。
        基于已设定的音色锚点（通过 set_voice 设定）合成新文本。
        """
        if self.voice is None:
            raise RuntimeError("⚠️ 请先调用 set_voice() 设定音色锚点，才能进行 clone。")
            
        cfg = config or TTSConfig()
        self.master.clear_memory() # 确保记忆纯净
        
        # 克隆模式入口
        lang_id = map_language(language)
        pdata = PromptBuilder.build_clone_prompt(text, self.tokenizer, self.assets, self.voice, lang_id)
        
        timing = Timing()
        timing.prompt_time = pdata.compile_time
        
        lout = self._run_engine_loop(pdata, timing, cfg, verbose=verbose)
        return self._post_process(text, pdata, lout, cfg=cfg)

    def custom(self,
               text: str,
               speaker: str,
               language: str = "chinese",
               instruct: Optional[str] = None,
               config: Optional[TTSConfig] = None,
               verbose: bool = True) -> TTSResult:
        """
        [精品音色模式] 对应 CustomVoice 模型能力。
        """
        cfg = config or TTSConfig()
        self.master.clear_memory()
        
        # 精品音色入口
        spk_id = map_speaker(speaker)
        lang_id = map_language(language) if language.lower() != "auto" else None
        pdata = PromptBuilder.build_custom_prompt(text, self.tokenizer, self.assets, spk_id, lang_id, instruct)
        
        timing = Timing()
        timing.prompt_time = pdata.compile_time
        
        lout = self._run_engine_loop(pdata, timing, cfg, verbose=verbose)
        return self._post_process(text, pdata, lout, cfg=cfg)

    def design(self,
               text: str,
               instruct: str,
               language: str = "chinese",
               config: Optional[TTSConfig] = None,
               verbose: bool = True) -> TTSResult:
        """
        [音色设计模式] 对应 VoiceDesign 模型能力。
        """
        cfg = config or TTSConfig()
        self.master.clear_memory()
        
        # 设计模式入口
        lang_id = map_language(language) if language.lower() != "auto" else None
        pdata = PromptBuilder.build_design_prompt(text, self.tokenizer, self.assets, instruct, lang_id)
        
        timing = Timing()
        timing.prompt_time = pdata.compile_time
        
        lout = self._run_engine_loop(pdata, timing, cfg, verbose=verbose)
        return self._post_process(text, pdata, lout, cfg=cfg)

    def tts(self, *args, **kwargs):
        """兼容性包装：现默认指向 clone 逻辑"""
        return self.clone(*args, **kwargs)


    def _run_engine_loop(self, pdata: PromptData, timing: Timing, cfg: TTSConfig, verbose: bool = False) -> LoopOutput:
        """核心推理循环：支持生成全量 Codes 并可选进行流式推送"""
        all_codes = []
        turn_summed_embeds = []
        chunk_buffer = []
        pushed_count = 0
        
        # 流式播放初始化
        if cfg.stream_play and self.mouth:
            self.mouth.reset()
            if verbose: logger.info(f"🚀 [Loop] 开始流式推送任务 (TaskID: {self.mouth.active_task_id})")
            
        for step_codes, summed_vec in self._run_engine_loop_gen(pdata, cfg, timing):
            all_codes.append(step_codes)
            turn_summed_embeds.append(summed_vec)
            
            # 流式分块推送
            if cfg.stream_play and self.mouth:
                chunk_buffer.append(step_codes)
                if len(chunk_buffer) >= cfg.mouth_chunk_size:
                    self.mouth.decode(np.array(chunk_buffer), is_final=False, stream=True)
                    pushed_count += len(chunk_buffer)
                    if verbose: print(f"\r   📡 已推送: {pushed_count:3} 帧 Codec...", end="", flush=True)
                    chunk_buffer = []

        # 结束流式推送
        if cfg.stream_play and self.mouth:
            self.mouth.decode(np.array(chunk_buffer) if chunk_buffer else np.zeros((0, 16)), is_final=True, stream=True)
            pushed_count += len(chunk_buffer)
            if verbose: print(f"\n   ✅ 推送完毕, 共计: {pushed_count} 帧。")

        return LoopOutput(all_codes=all_codes, summed_embeds=turn_summed_embeds, timing=timing)

    def _run_engine_loop_gen(self, pdata: PromptData, cfg: TTSConfig, timing: Timing):
        """
        [New] 生成器版推理循环。
        逐帧产出 (codes, summed_embeds)。
        """
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
                # 发送最后一个 EOS 信号（可选，模型通常会自动结束）
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
            
            # 大师反馈 (由外部驱动结果消费)
            t_m_s = time.time()
            summed = np.sum(step_embeds_2048, axis=0) + self.assets.tts_pad.flatten()
            m_hidden, m_logits = self.master.decode_step(summed, seq_id=0)
            timing.master_loop_time += (time.time() - t_m_s)
            
            yield step_codes, summed
            
        else:
            # for 循环正常结束（即达到 max_steps 而未 break）
            print(f"\n⚠️  [Stream] 推理达到了最大步数限制 ({cfg.max_steps}) 仍未检出 EOS。这可能是模型进入了无限静音或重复模式。")
            
        timing.total_steps = len(pdata.embd) + step_idx # 粗略统计

    def _post_process(self, 
                     text: str, 
                     pdata: PromptData, 
                     lout: LoopOutput,
                     cfg: Optional[TTSConfig] = None) -> TTSResult:
        """
        后处理：生成音频波形并封装。
        如果 stream_play=True，则波形生成已在子进程处理，这里仅做同步。
        """
        audio = None
        if self.mouth:
            # 无论是否流式，最终都需要获取完整音频用于 TTSResult (可选)
            # 对于流式模式，离线解码依然可以并行或直接从子进程收集
            t0 = time.time()
            audio = self.mouth.decode(np.array(lout.all_codes), is_final=True, stream=False)
            
            # [RTF Logic] 如果是非流式，算在 mouth_render_time
            # 如果是流式，此时 mouth.decode 应该非常快（因为大部分已在后台算完）
            lout.timing.mouth_render_time = time.time() - t0

        return TTSResult(
            audio=audio,
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
        """重置流：清除记忆与音色设置"""
        self.m_ctx.kv_cache_clear()
        self.c_ctx.kv_cache_clear()
        self.voice = None
        logger.info("🧹 Stream memory and voice cleared.")

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

    # =========================================================================
    # 音色设置 API (Voice Management)
    # =========================================================================

    def set_voice(self, source: Union[TTSResult, str], text: Optional[str] = None):
        """
        统一设置当前流的音色锚点。
        参数：
          - source: TTSResult 对象，或 .json 路径，或 .wav 路径
          - text: 如果 source 为音频路径，需提供对应的参考文本（用于 ICL 引导）
        """
        if isinstance(source, TTSResult):
            self._set_voice_from_result(source)
        elif isinstance(source, str):
            low_source = source.lower()
            if low_source.endswith(".json"):
                self._set_voice_from_json(source)
            elif low_source.endswith((".wav", ".mp3", ".flac")):
                if text is None:
                    logger.warning("📍 设置克隆音色时建议提供参考文本 (text)，否则可能导致合成效果不稳定。")
                self._set_voice_from_wav(source, text or "")
            else:
                # 尝试当作内置说话人名称处理 (生成一小段作为锚点)
                self.set_voice_from_speaker(source, text or "你好")
        else:
            raise TypeError(f"Unsupported voice source type: {type(source)}")

    def _set_voice_from_result(self, res: TTSResult):
        """命令式设置：直接将一个 TTSResult 设为当前流的音色锚点。"""
        if not res.is_valid_anchor:
            raise ValueError("Provided TTSResult is not a valid anchor.")
        self.voice = res
        logger.info(f"🎭 Voice switched to: {res.text[:20]}...")

    def _set_voice_from_json(self, path: str):
        """从 JSON 文件恢复音色锚点"""
        res = TTSResult.from_json(path)
        self._set_voice_from_result(res)

    def _set_voice_from_wav(self, wav_path: str, text: str):
        """克隆音色：从外部文件提取特征并设为音色锚点"""
        if self.engine.encoder is None:
            raise RuntimeError("⚠️ 编码器模块未加载，无法执行音色克隆。")
            
        logger.info(f"🎤 Extracting features from: {wav_path}")
        codes, spk_emb = self.engine.encoder.encode(wav_path)
        
        res = TTSResult(text=text, text_ids=self.tokenizer.encode(text).ids, spk_emb=spk_emb, codes=codes)
        self._set_voice_from_result(res)

    def set_voice_from_speaker(self, speaker_id: str, text: str, language: str = "chinese", config: Optional[TTSConfig] = None, verbose: bool = False) -> TTSResult:
        """从指定内置说话人生成一个音色锚点并激活 (支持流式预览)"""
        logger.info(f"📍 Initializing Voice from Speaker: {speaker_id}")
        
        cfg = config or TTSConfig()
        self.master.clear_memory()
        
        lang_id = map_language(language)
        spk_id = map_speaker(speaker_id)
        pdata = PromptBuilder.build_custom_prompt(text, self.tokenizer, self.assets, spk_id, lang_id)
        
        timing = Timing()
        timing.prompt_time = pdata.compile_time
        
        lout = self._run_engine_loop(pdata, timing, cfg, verbose=verbose)
        res = self._post_process(text, pdata, lout)
        self._set_voice_from_result(res)
        return res
