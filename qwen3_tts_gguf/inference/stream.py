"""
stream.py - TTS 语音流
核心逻辑所在，管理单次会话的上下文，支持流式和非流式合成。
"""
import time
import os
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Union
from .schema.constants import PROTOCOL, map_speaker, map_language
from .schema.result import TTSResult, Timing, LoopOutput, DecodeResult
from .config import TTSConfig
from .talker import TalkerPredictor
from .predictor import Predictor

from . import llama, logger
from .prompt_builder import PromptBuilder, PromptData
from .utils.audio import preprocess_audio, save_temp_wav

class TTSStream:
    """
    保存 Talker, Predictor, Decoder 记忆的语音流。
    """
    def __init__(self, engine, n_ctx=2048, voice_path: Optional[str] = None):
        self.engine = engine
        self.assets = engine.assets
        self.tokenizer = engine.tokenizer
        self.n_ctx = n_ctx
        
        # 1. 初始化流独立的 Context 和 Batch
        self._init_contexts()
        
        # 2. 初始化推理核心 (Talker, Predictor & PromptBuilder)
        self.talker = TalkerPredictor(engine.talker_model, self.talker_ctx, self.talker_batch, self.assets)
        self.predictor = Predictor(engine.predictor_model, self.predictor_ctx, self.predictor_batch, self.assets)
        self.prompt_builder = PromptBuilder(self.tokenizer, self.assets)
        
        # 3. 音色锚点 (Voice)
        self.voice: Optional[TTSResult] = None
        if voice_path:
            self.set_voice(voice_path)
            
        self.decoder = getattr(engine, 'decoder', None)
        self.task_counter = 0 # 任务计数器，用于生成唯一 task_id

    def _init_contexts(self):
        """初始化此语音流专属的推理环境"""
        # engine.talker_model 是 LlamaModel 对象
        self.talker_ctx = llama.LlamaContext(self.engine.talker_model, n_ctx=self.n_ctx, embeddings=True)
        self.predictor_ctx = llama.LlamaContext(self.engine.predictor_model, n_ctx=64, embeddings=False)
        
        # 使用模型自带的 n_embd，兼容 0.6B(1024) 和 1.7B(2048)
        logger.info(f"[Stream] Talker Dim: {self.engine.talker_model.n_embd} | Predictor Dim: {self.engine.predictor_model.n_embd}")
        self.talker_batch = llama.LlamaBatch(self.n_ctx, embd_dim=self.engine.talker_model.n_embd)
        self.predictor_batch = llama.LlamaBatch(2, embd_dim=self.engine.predictor_model.n_embd)

    # =========================================================================
    # 核心推理 API
    # =========================================================================

    def clone(self, 
              text: str, 
              language: str = "chinese",
              config: Optional[TTSConfig] = None,
              streaming: bool = False,
              chunk_size: int = 8,
              verbose: bool = True) -> Optional[TTSResult]:
        """
        [克隆模式] 使用当前流中已设定的音色锚点（Voice Anchor）进行语音合成。

        Args:
            text: 待合成的目标文本。
            language: 目标语言。可选:
                - 'chinese' (默认), 'english', 'japanese', 'korean'
                - 'german', 'spanish', 'french', 'russian', 'italian', 'portuguese'
                - 'beijing_dialect' (北京话), 'sichuan_dialect' (四川话)
            config: 推理配置对象 (TTSConfig)，可控制 Temperature, Top-P 等采样参数。
            streaming: 是否启用流式推理。若为 True，则边推理边向播放器推送数据。
            chunk_size: 流式推理时，每积压多少帧特征码即送去解码播放一次。越小延迟越低，但每次解码会有8帧的额外计算。
            verbose: 是否输出详细推理进度和时延统计。

        Returns:
            TTSResult 对象，包含完整音频、特征码及性能统计。
        """
        if self.voice is None:
            logger.error("❌ 尚未设定音色锚点，请先调用 set_voice()。")
            return None
            
        cfg = config or TTSConfig()
        self.talker.clear_memory()
        
        try:
            lang_id = map_language(language)
            pdata = self.prompt_builder.build_clone_prompt(text, self.voice, lang_id)
            
            timing = Timing()
            timing.prompt_time = pdata.compile_time
            
            lout = self._run_engine_loop(pdata, timing, cfg, streaming=streaming, chunk_size=chunk_size, verbose=verbose)
            
            return self._post_process(text, pdata, lout)
        except Exception as e:
            logger.error(f"❌ Clone 推理失败: {e}", exc_info=True)
            print(f"❌ Clone 推理失败: {e}")
            return None

    def custom(self,
               text: str,
               speaker: str,
               language: str = "chinese",
               instruct: Optional[str] = None,
               config: Optional[TTSConfig] = None,
               streaming: bool = False,
               chunk_size: int = 8,
               verbose: bool = True) -> Optional[TTSResult]:
        """
        [精品音色模式] 使用官方内置的精品预设音色进行合成，支持自然语言渲染指令。

        Args:
            text: 待合成的目标文本。
            speaker: 精品音色名称。可选:
                - 女性: 'vivian', 'serena', 'ono_anna', 'sohee'
                - 男性: 'ryan', 'aiden', 'uncle_fu'
                - 方言专用: 'eric' (四川话男声), 'dylan' (北京话男声)
            language: 目标语言 (见 clone 方法)。
            instruct: 渲染指令，如 "用温柔的语气说" 或 "充满活力的播报"。
            config: 推理配置对象 (TTSConfig)。
            streaming: 是否启用流式推理。
            chunk_size: 流式推理块大小。
            verbose: 是否输出详细进度。

        Returns:
            TTSResult 对象。
        """
        cfg = config or TTSConfig()
        self.talker.clear_memory()
        
        try:
            spk_id = map_speaker(speaker)
            lang_id = map_language(language) if language.lower() != "auto" else None
            pdata = self.prompt_builder.build_custom_prompt(text, spk_id, lang_id, instruct)
            
            timing = Timing()
            timing.prompt_time = pdata.compile_time
            
            lout = self._run_engine_loop(pdata, timing, cfg, streaming=streaming, chunk_size=chunk_size, verbose=verbose)
            return self._post_process(text, pdata, lout)
        except Exception as e:
            logger.error(f"❌ Custom 推理失败: {e}", exc_info=True)
            return None

    def design(self,
               text: str,
               instruct: str,
               language: str = "chinese",
               config: Optional[TTSConfig] = None,
               streaming: bool = False,
               chunk_size: int = 8,
               verbose: bool = True) -> Optional[TTSResult]:
        """
        [音色设计模式] 完全通过自然语言描述来设计并生成一个全新的音色。

        Args:
            text: 待合成的目标文本。
            instruct: 音色设计描述。例如："体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显。"
            language: 目标语言 (见 clone 方法)。
            config: 推理配置对象 (TTSConfig)。
            streaming: 是否启用流式推理。
            chunk_size: 流式推理块大小。
            verbose: 是否输出详细进度。

        Returns:
            TTSResult 对象。
        """
        cfg = config or TTSConfig()
        self.talker.clear_memory()
        
        try:
            lang_id = map_language(language) if language.lower() != "auto" else None
            pdata = self.prompt_builder.build_design_prompt(text, instruct, lang_id)
            
            timing = Timing()
            timing.prompt_time = pdata.compile_time
            
            lout = self._run_engine_loop(pdata, timing, cfg, streaming=streaming, chunk_size=chunk_size, verbose=verbose)
            return self._post_process(text, pdata, lout)
        except Exception as e:
            logger.error(f"❌ Design 推理失败: {e}", exc_info=True)
            return None

    def tts(self, *args, **kwargs):
        return self.clone(*args, **kwargs)

    def _run_engine_loop(self, pdata: PromptData, timing: Timing, cfg: TTSConfig, 
                         streaming: bool = False, chunk_size: int = 8, verbose: bool = False) -> LoopOutput:
        all_codes = []
        turn_summed_embeds = []
        chunk_buffer = []
        
        logger.info(f"[Stream] 启动推理循环: max_steps={cfg.max_steps}, streaming={streaming}")
        
        if self.decoder:
            logger.info("[Stream] 检测到解码器，准备流式输出")
            
        step_count = 0
        
        # 每次推理生成一个唯一的 task_id
        current_task_id = f"task_{self.task_counter}"
        self.task_counter += 1
        last_chunk_time = time.time()
        
        for step_codes, summed_vec in self._run_engine_loop_gen(pdata, cfg, timing):
            step_count += 1
            all_codes.append(step_codes) # 保持 numpy 状态，供 decoder 使用
            turn_summed_embeds.append(summed_vec)
            
            if step_count % 50 == 0:
                logger.info(f"[Stream] 正在生成... 已完成 {step_count} 步")
            
            # 累积到 chunk_buffer
            chunk_buffer.append(step_codes)

            if not streaming or len(chunk_buffer) < chunk_size:
                continue

            # 记录该 chunk 的生成耗时 
            timing.chunk_gen_times.append(time.time() - last_chunk_time); last_chunk_time = time.time()
            
            # 解码 chunk
            self.decoder.decode(np.array(chunk_buffer), task_id=current_task_id, is_final=False, stream=streaming)
            
            # 清空 chunk_buffer 以积累下一批
            chunk_buffer = []

        logger.info(f"[Stream] 推理循环结束，共 {step_count} 步")

        # 最后一个 chunk，此时 decode 会返回 DecodeResult
        timing.chunk_gen_times.append(time.time() - last_chunk_time)
        decode_result = self.decoder.decode(
            np.array(chunk_buffer) if chunk_buffer else np.zeros((0, 16)), 
            task_id=current_task_id, is_final=True, stream=streaming
        )
        # 记录 decoder 的每一个 chunk 的耗时
        timing.decoder_compute_times = decode_result.chunk_compute_times

        return LoopOutput(
            all_codes=all_codes, 
            summed_embeds=turn_summed_embeds, 
            timing=timing,
            decode_result=decode_result
        )

    def _create_sampler(self, do_sample: bool, temperature: float, top_p: float, top_k: int, 
                        min_p: float = 0.0, repeat_penalty: float = 1.0, 
                        frequency_penalty: float = 0.0, presence_penalty: float = 0.0,
                        penalty_last_n: int = 128, seed: Optional[int] = None) -> llama.LlamaSampler:
        """创建原生采样器实例"""
        return llama.LlamaSampler(
            temperature=temperature if do_sample else 0.0,
            top_p=top_p if do_sample else 1.0,
            top_k=top_k if do_sample else 0,
            min_p=min_p if do_sample else 0.0,
            repeat_penalty=repeat_penalty,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            penalty_last_n=penalty_last_n,
            seed=seed
        )

    def _run_engine_loop_gen(self, pdata: PromptData, cfg: TTSConfig, timing: Timing):
        t_pre_s = time.time()
        # Talker 内部现在会处理 pdata 中的 embd 和 trailing_text_pool
        m_hidden = self.talker.prefill(pdata, seq_id=0)
        timing.prefill_time = time.time() - t_pre_s
        
        # 1. 初始化两级原生采样器
        # 大师阶段 (Talker): 需要全套采样增强 (惩罚项 + Min-P)
        talker_sampler = self._create_sampler(
            cfg.do_sample, cfg.temperature, cfg.top_p, cfg.top_k, 
            min_p=cfg.min_p, 
            repeat_penalty=cfg.repeat_penalty,
            frequency_penalty=cfg.frequency_penalty,
            presence_penalty=cfg.presence_penalty,
            penalty_last_n=cfg.penalty_last_n,
            seed=cfg.seed
        )
        # 工匠阶段 (Predictor): 通常使用简洁采样，不应用惩罚项以保持声音稳定
        predictor_sampler = self._create_sampler(
            cfg.sub_do_sample, 
            cfg.sub_temperature, 
            cfg.sub_top_p, 
            cfg.sub_top_k, 
            seed=cfg.sub_seed
        )
        
        # 惩罚项豁免名单：不希望因为生成过 EOS/BOS 而降低结尾概率
        allow_tokens = {PROTOCOL["EOS"], PROTOCOL["PAD"], PROTOCOL["BOS"]}
            
        step_idx = 0
        try:
            for step_idx in range(cfg.max_steps):
                # 采样获取第 0 层码本 (大师决策)
                code_0 = talker_sampler.sample(
                    self.talker.ctx, idx=-1, 
                    limit_start=0, limit_end=2048, 
                    allow_tokens=allow_tokens # 官方豁免权：允许采到 EOS
                )
                
                # 更新历史记录
                talker_sampler.accept(code_0)
                
                if code_0 == PROTOCOL["EOS"]:
                    break
                
                t_c_s = time.time()
                
                # ---------------- Predictor Stage ----------------
                # 根据第 0 层码本和 Talker 隐层，预测完整的 16 层码本
                t_c_s = time.time()
                step_codes, step_embeds_2048 = self.predictor.predict_frame(
                    m_hidden, 
                    code_0, 
                    sampler=predictor_sampler
                )
                timing.predictor_loop_times.append(time.time() - t_c_s)
                
                # ---------------- Feedback Stage ----------------
                t_m_s = time.time()
                # 汇总 16 层音频 Embedding
                audio_summed = np.sum(step_embeds_2048, axis=0) 
                # 反馈给 Talker。Talker 内部会自动执行 [Audio + Text] 融合
                m_hidden = self.talker.decode_step(audio_summed, seq_id=0)
                timing.talker_loop_times.append(time.time() - t_m_s)
                
                yield step_codes, audio_summed
                
        finally:
            talker_sampler.free()
            predictor_sampler.free()
            
        timing.total_steps = len(pdata.embd) + step_idx

    def _post_process(self, 
                     text: str, 
                     pdata: PromptData, 
                     lout: LoopOutput) -> TTSResult:
        return TTSResult(
            audio=lout.decode_result.audio if lout.decode_result else None,
            text=text,
            text_ids=pdata.text_ids,
            spk_emb=pdata.spk_emb,
            codes=np.array(lout.all_codes),
            summed_embeds=lout.summed_embeds,
            stats=lout.timing
        )

    def reset(self):
        self.talker_ctx.clear_kv_cache()
        self.predictor_ctx.clear_kv_cache()
        self.voice = None
        logger.info("扫 [Stream] 记忆与音色已清除。")

    def join(self, timeout: Optional[float] = None):
        """阻塞直至当前流所有音频（解码+播报）全部完毕"""
        if self.decoder:
            # 1. 先等解码器把活干完 (Bitstream -> PCM)
            self.decoder.join_decoder(timeout)
            # 2. 再等播放器把声音放完 (PCM -> 声卡)
            self.decoder.join_speaker(timeout)

    def shutdown(self):
        # 让 Python GC 处理内存释放 (_del_ 会调用 free)
        self.talker_batch = None
        self.talker_ctx = None
        self.predictor_batch = None
        self.predictor_ctx = None

    # =========================================================================
    # 音色设置 API (Voice Management)
    # =========================================================================

    def set_voice(self, source: Union[TTSResult, str, Path], text: Optional[str] = None, **kwargs) -> Union[bool, TTSResult]:
        """统一设置当前流的音色锚点。返回生成的 TTSResult 或 False。"""
        try:
            success = False
            if isinstance(source, TTSResult):
                success = self._set_voice_from_result(source)
            else:
                source_p = Path(source)
                if source_p.suffix.lower() == ".json":
                    success = self._set_voice_from_json(source_p)
                elif source_p.suffix.lower() in [".wav", ".mp3", ".flac", ".m4a", ".opus"]:
                    success = self._set_voice_from_audio(source_p, text or "", **kwargs)
                else:
                    # 尝试作为内置说话人处理
                    return self.set_voice_from_speaker(str(source), text or "你好", **kwargs)
            
            return self.voice if success else False
        except Exception as e:
            logger.error(f"❌ 设置音色时出现无法预料的异常: {e}")
            print(f"❌ 设置音色时出现无法预料的异常: {e}")
            return False

    def _set_voice_from_result(self, res: TTSResult) -> bool:
        if not res.is_valid_anchor:
            return False
            
        # 如果维度不匹配，执行重编码转换
        if res.spk_emb.shape[-1] != self.engine.talker_model.n_embd:
            logger.info(f"🔄 [Stream] 维度不匹配 ({res.spk_emb.shape[-1]}->{self.engine.talker_model.n_embd})，正在转换...")
            if res.audio is None:
                self.engine.decode(res)
            self.engine.encode(res)

        self.voice = res
        logger.info(f"🎭 音色已切换为: {res.text[:20]}...")
        return True

    def _set_voice_from_json(self, path: Path) -> bool:
        """从 JSON 文件恢复音色锚点"""
        if not path.exists():
            logger.error(f"❌ 未找到音色 JSON 文件: {path}")
            return False
        try:
            res = TTSResult.from_json(str(path))
            return self._set_voice_from_result(res)
        except Exception as e:
            logger.error(f"❌ 解析音色 JSON 失败 ({path.name}): {e}")
            return False

    def _set_voice_from_audio(self, wav_path: Path, text: str) -> bool:
        """从音频文件克隆音色：使用 pydub 标准化输入"""
        if self.engine.codec_encoder is None or self.engine.speaker_encoder is None:
            logger.error("⚠️ 编码器模块未加载，无法执行音色克隆。")
            return False
            
        logger.info(f"🎤 正在从音频提取音色特征: {wav_path.name}")
        
        # 1. 万能格式转换与预处理 (24kHz, Mono, float32)
        samples = preprocess_audio(wav_path)
        if samples is None:
            return False
            
        # 2. 推理提取特征 (直接传入内存 samples，不再依赖临时文件)
        try:
            codes = self.engine.codec_encoder.encode(samples)
            spk_emb = self.engine.speaker_encoder.encode(samples)
            
            res = TTSResult(text=text, text_ids=self.tokenizer.encode(text).ids, spk_emb=spk_emb, codes=codes)
            return self._set_voice_from_result(res)
        except Exception as e:
            logger.error(f"❌ 音声特征提取失败: {e}")
            return False

    def set_voice_from_speaker(self, speaker_id: str, text: str, **kwargs) -> Optional[TTSResult]:
        """从内置说话人生成音色锚点并设置"""
        try:
            logger.info(f"📍 正在从内置说话人初始化音色核心: {speaker_id}")
            # kwargs 包含 language, streaming, verbose 等
            res = self.custom(text, speaker_id, **kwargs)
            if res:
                self._set_voice_from_result(res)
                return res
            return None
        except Exception as e:
            logger.error(f"❌ 内置音色初始化失败: {e}")
            return None
