"""
voice.py - 音色锚点准备链
json / 音频文件 / TTSResult -> 规范化的 TTSResult (补 spk_emb、补 final_state)。
自 TTSStream._set_voice_from_* 迁移，GUI 与 Stream 共用。
"""
from pathlib import Path
from typing import Optional, Union

from . import logger
from .schema.result import TTSResult
from .utils.audio import load_audio

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".opus"}


def prepare_voice(engine, tokenizer, source: Union[TTSResult, str, Path],
                  text: Optional[str] = None) -> Optional[TTSResult]:
    """
    构造并规范化音色锚点。失败返回 None (原因见日志)。
    source 为音频文件时 text 用作锚点文本 (可传空串)。
    """
    try:
        if isinstance(source, TTSResult):
            res = source
        else:
            p = Path(source)
            if p.suffix.lower() == ".json":
                if not p.exists():
                    logger.error(f"❌ 未找到音色 JSON 文件: {p}")
                    return None
                res = TTSResult.from_json(str(p))
            elif p.suffix.lower() in AUDIO_EXTS:
                res = _from_audio(engine, tokenizer, p, text or "")
            else:
                logger.error(f"❌ 不支持的克隆源类型: {p.suffix}")
                return None

        if res is None or not _normalize(engine, res):
            return None
        return res
    except Exception as e:
        logger.error(f"❌ 准备音色锚点时出现无法预料的异常: {e}")
        return None


def _from_audio(engine, tokenizer, wav_path: Path, text: str) -> Optional[TTSResult]:
    """从音频文件提取音色特征 (codes + spk_emb)"""
    if engine.codec_encoder is None or engine.speaker_encoder is None:
        logger.error("⚠️ 编码器模块未加载，无法执行音色克隆。")
        return None

    logger.info(f"🎤 正在从音频提取音色特征: {wav_path.name}")
    samples = load_audio(wav_path)
    if samples is None:
        return None

    try:
        codes = engine.codec_encoder.encode(samples)
        spk_emb = engine.speaker_encoder.encode(samples)
        return TTSResult(text=text, text_ids=tokenizer.encode(text).ids,
                         spk_emb=spk_emb, codes=codes)
    except Exception as e:
        logger.error(f"❌ 音声特征提取失败: {e}")
        return None


def _normalize(engine, res: TTSResult) -> bool:
    """规范化锚点：缺 spk_emb 则解码补齐；维度不匹配则重编码；缺 final_state 则预解码对齐"""
    if not res.is_valid_anchor:
        return False

    if res.spk_emb is None and len(res.codes) > 0:
        logger.info("🎤 音色向量缺失，正在从 codes 解码音频并提取...")
        if res.audio is None:
            engine.decode(res)
        engine.encode(res)

    if res.spk_emb is not None and res.spk_emb.shape[-1] != engine.talker_model.n_embd:
        logger.info(f"🔄 维度不匹配 ({res.spk_emb.shape[-1]}->{engine.talker_model.n_embd})，正在转换...")
        if res.audio is None:
            engine.decode(res)
        engine.encode(res)

    if engine.decoder and res.final_state is None and len(res.codes) > 0:
        logger.info("🧠 缺少解码器上下文记忆 (final_state)，正在执行预解码以对齐记忆...")
        engine.decode(res)

    return True
