"""
stream.py - TTS 语音流
核心逻辑所在，管理单次会话的上下文，支持流式和非流式合成。
"""
import time
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Union
from .constants import PROTOCOL, map_speaker, map_language
from .result import TTSResult, Timing, LoopOutput, TTSConfig
from .predictors.talker import TalkerPredictor
from .predictors.predictor import Predictor

from . import llama, logger
from .prompt_builder import PromptBuilder, PromptData
from .utils.audio import preprocess_audio, save_temp_wav

class TTSStream:
    """
    保存 Talker, Predictor, Decoder 记忆的语音流。
    """
    def __init__(self, engine, n_ctx=4096, voice_path: Optional[str] = None):
        self.engine = engine
        self.assets = engine.assets
        self.tokenizer = engine.tokenizer
        self.n_ctx = n_ctx
        
        # 1. 初始化流独立的 Context 和 Batch
        self._init_contexts()
        
        # 2. 初始化推理核心 (Talker & Predictor)
        self.talker = TalkerPredictor(engine.talker_model, self.talker_ctx, self.talker_batch, self.assets)
        self.predictor = Predictor(engine.predictor_model, self.predictor_ctx, self.predictor_batch, self.assets)
        
        # 3. 音色锚点 (Voice)
        self.voice: Optional[TTSResult] = None
        if voice_path:
            self.set_voice(voice_path)
            
        self.decoder = getattr(engine, 'decoder', None)

    def _init_contexts(self):
        """初始化此语音流专属的推理环境"""
        # 此时 engine.talker_model 是 Path 对象还是指针？在 engine.py 中它是指针。
        self.talker_ctx = llama.create_context(self.engine.talker_model, n_ctx=self.n_ctx, embeddings=True)
        self.predictor_ctx = llama.create_context(self.engine.predictor_model, n_ctx=512, embeddings=False)
        
        self.talker_batch = llama.create_batch(self.n_ctx, embd_dim=2048)
        self.predictor_batch = llama.create_batch(32, embd_dim=1024)

    # =========================================================================
    # 核心推理 API
    # =========================================================================

    def clone(self, 
              text: str, 
              language: str = "chinese",
              config: Optional[TTSConfig] = None,
              verbose: bool = True) -> Optional[TTSResult]:
        """[克隆模式]"""
        if self.voice is None:
            logger.error("❌ 尚未设定音色锚点，请先调用 set_voice()。")
            return None
            
        cfg = config or TTSConfig()
        self.talker.clear_memory()
        
        try:
            lang_id = map_language(language)
            pdata = PromptBuilder.build_clone_prompt(text, self.tokenizer, self.assets, self.voice, lang_id)
            
            timing = Timing()
            timing.prompt_time = pdata.compile_time
            
            lout = self._run_engine_loop(pdata, timing, cfg, verbose=verbose)
            return self._post_process(text, pdata, lout, cfg=cfg)
        except Exception as e:
            logger.error(f"❌ Clone 推理失败: {e}")
            return None

    def custom(self,
               text: str,
               speaker: str,
               language: str = "chinese",
               instruct: Optional[str] = None,
               config: Optional[TTSConfig] = None,
               verbose: bool = True) -> Optional[TTSResult]:
        """[精品音色模式]"""
        cfg = config or TTSConfig()
        self.talker.clear_memory()
        
        try:
            spk_id = map_speaker(speaker)
            lang_id = map_language(language) if language.lower() != "auto" else None
            pdata = PromptBuilder.build_custom_prompt(text, self.tokenizer, self.assets, spk_id, lang_id, instruct)
            
            timing = Timing()
            timing.prompt_time = pdata.compile_time
            
            lout = self._run_engine_loop(pdata, timing, cfg, verbose=verbose)
            return self._post_process(text, pdata, lout, cfg=cfg)
        except Exception as e:
            logger.error(f"❌ Custom 推理失败: {e}")
            return None

    def design(self,
               text: str,
               instruct: str,
               language: str = "chinese",
               config: Optional[TTSConfig] = None,
               verbose: bool = True) -> Optional[TTSResult]:
        """[音色设计模式]"""
        cfg = config or TTSConfig()
        self.talker.clear_memory()
        
        try:
            lang_id = map_language(language) if language.lower() != "auto" else None
            pdata = PromptBuilder.build_design_prompt(text, self.tokenizer, self.assets, instruct, lang_id)
            
            timing = Timing()
            timing.prompt_time = pdata.compile_time
            
            lout = self._run_engine_loop(pdata, timing, cfg, verbose=verbose)
            return self._post_process(text, pdata, lout, cfg=cfg)
        except Exception as e:
            logger.error(f"❌ Design 推理失败: {e}")
            return None

    def tts(self, *args, **kwargs):
        return self.clone(*args, **kwargs)

    def _run_engine_loop(self, pdata: PromptData, timing: Timing, cfg: TTSConfig, verbose: bool = False) -> LoopOutput:
        all_codes = []
        turn_summed_embeds = []
        chunk_buffer = []
        pushed_count = 0
        
        if self.decoder:
            self.decoder.reset() # 强制重置，消除上一句的残留状态
            
        for step_codes, summed_vec in self._run_engine_loop_gen(pdata, cfg, timing):
            all_codes.append(step_codes) # 保持 numpy 状态，供 decoder 使用
            turn_summed_embeds.append(summed_vec)
            
            if cfg.stream_play and self.decoder:
                chunk_buffer.append(step_codes)
                if len(chunk_buffer) >= cfg.decoder_chunk_size:
                    self.decoder.decode(np.array(chunk_buffer), is_final=False, stream=True)
                    pushed_count += len(chunk_buffer)
                    chunk_buffer = []

        if cfg.stream_play and self.decoder:
            self.decoder.decode(np.array(chunk_buffer) if chunk_buffer else np.zeros((0, 16)), is_final=True, stream=True)

        return LoopOutput(all_codes=all_codes, summed_embeds=turn_summed_embeds, timing=timing)

    def _run_engine_loop_gen(self, pdata: PromptData, cfg: TTSConfig, timing: Timing):
        t_pre_s = time.time()
        m_hidden, m_logits = self.talker.prefill(pdata.embd, seq_id=0)
        timing.prefill_time = time.time() - t_pre_s
        
        step_idx = 0
        for step_idx in range(cfg.max_steps):
            # 1. 显式传递 Talker 参数 (扁平化)
            code_0 = self.engine._do_sample(
                m_logits, 
                do_sample=cfg.do_sample, 
                temperature=cfg.temperature, 
                top_p=cfg.top_p, 
                top_k=cfg.top_k
            )
            if code_0 == PROTOCOL["EOS"]:
                break
            
            t_c_s = time.time()
            # 2. 显式传递 Predictor 参数 (扁平化)
            step_codes, step_embeds_2048 = self.predictor.predict_frame(
                m_hidden, 
                code_0, 
                do_sample=cfg.sub_do_sample, 
                temperature=cfg.sub_temperature, 
                top_p=cfg.sub_top_p, 
                top_k=cfg.sub_top_k
            )
            timing.predictor_loop_time += (time.time() - t_c_s)
            
            t_m_s = time.time()
            summed = np.sum(step_embeds_2048, axis=0) + self.assets.tts_pad.flatten()
            m_hidden, m_logits = self.talker.decode_step(summed, seq_id=0)
            timing.talker_loop_time += (time.time() - t_m_s)
            
            yield step_codes, summed
            
        timing.total_steps = len(pdata.embd) + step_idx

    def _post_process(self, 
                     text: str, 
                     pdata: PromptData, 
                     lout: LoopOutput,
                     cfg: Optional[TTSConfig] = None) -> TTSResult:
        audio = None
        if self.decoder:
            self.decoder.reset() # 离线渲染前也必须重置
            t0 = time.time()
            audio = self.decoder.decode(np.array(lout.all_codes), is_final=True, stream=False)
            lout.timing.decoder_render_time = time.time() - t0

        return TTSResult(
            audio=audio,
            text=text,
            text_ids=pdata.text_ids,
            spk_emb=pdata.spk_emb,
            codes=np.array(lout.all_codes),
            summed_embeds=lout.summed_embeds,
            stats=lout.timing
        )

    def reset(self):
        self.talker_ctx.kv_cache_clear()
        self.predictor_ctx.kv_cache_clear()
        self.voice = None
        logger.info("扫 [Stream] 记忆与音色已清除。")

    def shutdown(self):
        try:
            llama.llama_batch_free(self.talker_batch)
            llama.llama_batch_free(self.predictor_batch)
            llama.llama_free(self.talker_ctx)
            llama.llama_free(self.predictor_ctx)
        except: pass

    # =========================================================================
    # 音色设置 API (Voice Management)
    # =========================================================================

    def set_voice(self, source: Union[TTSResult, str, Path], text: Optional[str] = None) -> bool:
        """统一设置当前流的音色锚点。返回是否成功。"""
        try:
            if isinstance(source, TTSResult):
                return self._set_voice_from_result(source)
            
            source_p = Path(source)
            if source_p.suffix.lower() == ".json":
                return self._set_voice_from_json(source_p)
            elif source_p.suffix.lower() in [".wav", ".mp3", ".flac", ".m4a", ".opus"]:
                return self._set_voice_from_audio(source_p, text or "")
            else:
                # 尝试作为内置说话人处理
                return self.set_voice_from_speaker(str(source), text or "你好")
        except Exception as e:
            logger.error(f"❌ 设置音色时出现无法预料的异常: {e}")
            return False

    def _set_voice_from_result(self, res: TTSResult) -> bool:
        if not res.is_valid_anchor:
            logger.error("❌ 提供的 TTSResult 不是有效的音色锚点 (缺少 codes 或 spk_emb)。")
            return False
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
        if self.engine.encoder is None:
            logger.error("⚠️ 编码器模块未加载，无法执行音色克隆。")
            return False
            
        logger.info(f"🎤 正在从音频提取音色特征: {wav_path.name}")
        
        # 1. 万能格式转换与预处理
        samples = preprocess_audio(wav_path)
        if samples is None:
            return False
            
        # 2. 交互过渡: 编码器目前只接受文件路径，因此我们需要一个临时 wav
        temp_wav = save_temp_wav(samples)
        try:
            codes, spk_emb = self.engine.encoder.encode(temp_wav)
            res = TTSResult(text=text, text_ids=self.tokenizer.encode(text).ids, spk_emb=spk_emb, codes=codes.tolist())
            return self._set_voice_from_result(res)
        except Exception as e:
            logger.error(f"❌ 音声特征提取失败: {e}")
            return False
        finally:
            if os.path.exists(temp_wav):
                try: os.remove(temp_wav)
                except: pass

    def set_voice_from_speaker(self, speaker_id: str, text: str, language: str = "chinese") -> bool:
        """从内置说话人生成音色锚点"""
        try:
            logger.info(f"📍 正在从内置说话人初始化音色: {speaker_id}")
            res = self.custom(text, speaker_id, language=language, verbose=False)
            if res:
                return self._set_voice_from_result(res)
            return False
        except Exception as e:
            logger.error(f"❌ 内置音色初始化失败: {e}")
            return False
