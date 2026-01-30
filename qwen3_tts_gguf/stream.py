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

    def tts(self, 
            text: str, 
            language: str = "chinese",
            config: Optional[TTSConfig] = None,
            verbose: bool = True) -> TTSResult:
        """
        同步合成接口。
        """
        # 0. 检查 Voice 是否已设置
        if self.voice is None:
            msg = (
                "\n❌ Voice is not set! 你必须先设置音色才能进行合成。\n"
                "你可以尝试以下方法之一：\n"
                "  1. stream.set_voice_from_speaker('vivian')  <- 使用内置音色\n"
                "  2. stream.set_voice_from_clone('path.wav')  <- 从外部音频克隆\n"
                "  3. stream.set_voice_from_json('path.json')  <- 载入持久化音色\n"
                "  4. engine.create_stream(voice_path='...')   <- 在创建流时指定"
            )
            logger.error(msg)
            raise RuntimeError("Voice not set. Please follow the instructions in the log.")

        cfg = config or TTSConfig()
        
        # 0. 准备工作：清空推理记忆（确保每轮合成状态独立）
        self.master.clear_memory()
        
        # 1. 准备文本 Prompt 数据
        pdata, timing = self._build_prompt_data(text, language, is_clone=cfg.voice_clone_mode)
        
        # 2. 推理循环 (流式产出)
        all_codes = []
        turn_summed_embeds = []
        chunk_buffer = []
        pushed_count = 0
        
        # 如果开启流式播放，先重置嘴巴状态
        if cfg.stream_play and self.mouth:
            self.mouth.reset()
            if verbose: logger.info(f"🚀 [TTS] 开始流式推送任务 (TaskID: {self.mouth.active_task_id})")
            
        # 迭代生成
        for step_codes, summed_vec in self._run_engine_loop_gen(pdata, cfg, timing):
            all_codes.append(step_codes)
            turn_summed_embeds.append(summed_vec)
            
            # 流式分块逻辑
            if cfg.stream_play and self.mouth:
                chunk_buffer.append(step_codes)
                if len(chunk_buffer) >= cfg.mouth_chunk_size:
                    # 异步丢给子进程，不阻塞主进程推理
                    self.mouth.decode(np.array(chunk_buffer), is_final=False, stream=True)
                    pushed_count += len(chunk_buffer)
                    if verbose: print(f"\r   📡 已推送: {pushed_count:3} 帧 Codec...", end="", flush=True)
                    chunk_buffer = []
        
        # 最后一段处理 (Final)
        if cfg.stream_play and self.mouth:
            # 发送剩余的 buffer，并标记 final 以触发最后一段音频输出
            self.mouth.decode(np.array(chunk_buffer) if chunk_buffer else np.zeros((0, 16)), is_final=True, stream=True)
            pushed_count += len(chunk_buffer)
            if verbose: print(f"\n   ✅ 推送完毕, 共计: {pushed_count} 帧。")
        
        lout = LoopOutput(all_codes=all_codes, summed_embeds=turn_summed_embeds, timing=timing)
        
        # 3. 后处理：生成波形并封装结果
        res = self._post_process(text, pdata, lout, cfg=cfg)
        res.timing = timing
        return res

    def _build_prompt_data(self, text: str, language: str, is_clone: bool, speaker_id: Optional[str] = None) -> Tuple[PromptData, Timing]:
        """准备 Prompt 并初始化 Timing 对象"""
        lang_id = map_language(language)
        
        if is_clone:
            pdata = PromptBuilder.build_clone_prompt(text, self.voice, self.tokenizer, self.assets, lang_id)
        else:
            spk_id = map_speaker(speaker_id)
            pdata = PromptBuilder.build_native_prompt(text, self.tokenizer, self.assets, lang_id, spk_id)
            
        timing = Timing()
        timing.prompt_time = pdata.compile_time
        return pdata, timing

    def _run_engine_loop(self, pdata: PromptData, timing: Timing, cfg: TTSConfig) -> LoopOutput:
        """同步包装器：消耗生成器并返回完整结果"""
        all_codes = []
        turn_summed_embeds = []
        for step_codes, summed_vec in self._run_engine_loop_gen(pdata, cfg, timing):
            all_codes.append(step_codes)
            turn_summed_embeds.append(summed_vec)
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

    def set_voice(self, res: TTSResult):
        """
        命令式设置：直接将一个 TTSResult 设为当前流的音色锚点。
        """
        if not res.is_valid_anchor:
            raise ValueError("Provided TTSResult is not a valid anchor.")
        self.voice = res
        logger.info(f"🎭 Voice switched to: {res.text[:20]}...")

    def set_voice_from_speaker(self, speaker_id: str, text: str, language: str = "chinese", config: Optional[TTSConfig] = None) -> TTSResult:
        """从指定内置说话人生成一个音色锚点结果"""
        logger.info(f"📍 Setting Voice from Speaker: {speaker_id}, language: {language}")
        
        cfg = config or TTSConfig()
        
        # 1. 编译 Prompt (原生模式，不使用 self.voice)
        pdata, timing = self._build_prompt_data(text, language, is_clone=False, speaker_id=speaker_id)
        
        # 2. 推理循环
        lout = self._run_engine_loop(pdata, timing, cfg)
        
        # 3. 生成结果并设为锚点
        res = self._post_process(text, pdata, lout)
        self.set_voice(res)
        return res

    def set_voice_from_clone(self, wav_path: str, text: str, language: str = "chinese") -> Union[TTSResult, bool]:
        """克隆音色：从外部 WAV 文件提取特征并设为音色锚点"""
        if self.engine.encoder is None:
            logger.info("⚠️ [Stream] 编码器模型未就绪，音色克隆功能不可用。")
            return False
            
        logger.info(f"🎤 Setting Voice from Clone: {wav_path}")
        
        # 1. 提取特征
        codes, spk_emb = self.engine.encoder.encode(wav_path)
        
        # 2. 构造 TTSResult 作为锚点
        res = TTSResult(
            text=text,
            text_ids=[], 
            spk_emb=spk_emb,
            codes=codes
        )
        
        # 3. 设置为当前音色
        self.set_voice(res)
        return res

    def set_voice_from_json(self, path: str):
        """从 JSON 文件恢复音色锚点并设为当前音色"""
        res = TTSResult.from_json(path)
        self.set_voice(res)
        return res
