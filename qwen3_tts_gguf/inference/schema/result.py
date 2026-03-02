"""
result.py - 合成结果与身份锚点统一类
核心职责：
1. 承载 TTS 合成结果 (音频、元数据、性能统计)。
2. 作为 Voice Identity 锚点提供克隆所需的特征。
3. 提供音频播放、保存以及 JSON 持久化能力。
"""
import base64
import json
import os

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from .constants import SAMPLE_RATE
from . import logger


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class Timing:
    """推理各阶段耗时统计"""
    # 基础耗时 (支持列表记录，用于分析分段耗时)
    prompt_time: float = 0.0                            # 仅单次
    prefill_time: float = 0.0                           # 仅单次
    talker_loop_times:      List[float] = field(default_factory=list)
    predictor_loop_times:   List[float] = field(default_factory=list)
    chunk_gen_times:        List[float] = field(default_factory=list)
    decoder_compute_times:  List[float] = field(default_factory=list)
    
    total_steps: int = 0

    @property
    def first_chunk_decode_time(self) -> float:
        """从第一块音频消息发给子进程到渲染完成的时间"""
        return self.decoder_compute_times[0] if self.decoder_compute_times else 0.0

    @property
    def total_first_audio_delay(self) -> float:
        """全链路首音延迟 = 攒码完成耗时 + 首包解码渲染耗时"""
        if self.chunk_gen_times and self.decoder_compute_times:
            return self.time_to_first_chunk + self.first_chunk_decode_time
        return 0.0

    @property
    def time_to_first_chunk(self) -> float:
        """即 chunk_gen_times[0]，从推理开始到第一个 chunk 攒码完成的绝对耗时"""
        return self.chunk_gen_times[0] if self.chunk_gen_times else 0.0

    @property
    def total_talker_time(self) -> float:
        return sum(self.talker_loop_times)

    @property
    def total_predictor_time(self) -> float:
        return sum(self.predictor_loop_times)

    @property
    def total_decoder_time(self) -> float:
        return sum(self.decoder_compute_times)

    @property
    def total_inference_time(self) -> float:
        """全链路总耗时（含解码渲染）"""
        return (self.prompt_time + self.prefill_time +
                self.total_talker_time + self.total_predictor_time +
                self.total_decoder_time)

    @property
    def inference_only_time(self) -> float:
        """核心推理耗时（不含解码渲染）"""
        return (self.prompt_time + self.prefill_time +
                self.total_talker_time + self.total_predictor_time)


@dataclass
class LoopOutput:
    """推理内核循环的输出封装"""
    all_codes: List[List[int]]       # 所有生成的 Codec IDs
    summed_embeds: List[np.ndarray]  # 叠加特征序列
    timing: Timing                   # 性能统计对象


# ---------------------------------------------------------------------------
# 合成结果 / 音色锚点
# ---------------------------------------------------------------------------

@dataclass
class TTSResult:
    """TTS 合成结果（同时也是音色锚点）"""

    # 核心特征（锚点要素）
    text: str                                          # 文字内容
    text_ids: List[int]                                # 文本 Token IDs
    codes: np.ndarray                                  # 音频 Codec IDs，形状 (T, 16)
    spk_emb: np.ndarray                                # 全局音色向量，形状 (2048,)

    # 选填
    info: str = ""                                     # 备注信息（如音色描述）
    summed_embeds: Optional[List[np.ndarray]] = None   # 音频叠加特征序列，形状 (T, 2048)

    # 产出附件（可选）
    audio: Optional[np.ndarray] = None                 # 音频波形，PCM float32
    stats: Optional[Timing] = None                     # 性能统计

    # --- 工厂方法 ---

    @classmethod
    def empty(cls) -> "TTSResult":
        """构造一个空的、无效的 TTSResult（用于加载失败时的降级返回）"""
        return cls(
            text="",
            text_ids=[],
            codes=np.empty((0, 16), dtype=np.int64),
            spk_emb=np.zeros(2048, dtype=np.float32),
        )

    # --- 属性 ---

    @property
    def is_valid_anchor(self) -> bool:
        """是否具有作为 Voice 锚点的必要特征"""
        return len(self.codes) > 0 and self.spk_emb is not None

    @property
    def duration(self) -> float:
        """音频时长（秒）"""
        if self.audio is not None:
            return len(self.audio) / SAMPLE_RATE
        return 0.0

    @property
    def rtf(self) -> float:
        """实时因子（Real-Time Factor），基于核心推理耗时计算"""
        if self.duration == 0 or self.stats is None:
            return 0.0
        return self.stats.inference_only_time / self.duration

    # --- 播放 ---

    def play(self, blocking: bool = True):
        """播放音频结果"""
        if self.audio is None or len(self.audio) == 0:
            if len(self.codes) > 0:
                logger.warning("⚠️ 此结果当前无音频数据，但检测到 Codec 特征。请先调用 .decode(engine.decoder) 进行解码渲染。")
            else:
                logger.warning("⚠️ 此结果不包含音频数据，且无可用特征。")
            return
        import sounddevice as sd
        sd.play(self.audio, samplerate=SAMPLE_RATE, blocking=blocking)

    # --- 保存 ---

    def save(self, path: str, **kwargs):
        """统一保存方法，根据后缀名自动选择 wav 或 json"""
        ext = os.path.splitext(path)[1].lower()
        if ext == ".wav":
            self.save_wav(path)
        elif ext == ".json":
            self.save_json(path, **kwargs)
        else:
            logger.error(f"❌ 不支持的保存格式: {ext}。请使用 .wav 或 .json")

    def save_wav(self, path: str):
        """保存为 WAV 文件"""
        if self.audio is None:
            logger.error("❌ 无音频数据，无法保存为 WAV。")
            return
        import soundfile as sf
        os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
        sf.write(path, self.audio, SAMPLE_RATE)
        logger.info(f"💾 音频已保存: {path}")

    def save_json(self, path: str, include_audio: bool = False, include_embeds: bool = False,
                  light: bool = False, info: Optional[str] = None):
        """将特征锚点序列化为 JSON 文件"""
        if info is not None:
            self.info = info

        if not self.is_valid_anchor:
            logger.warning("⚠️ 当前结果不完整，无法保存为锚点。")
            return

        # spk_emb 采用 fp16 + Base64 存储以节省空间
        spk_b64 = base64.b64encode(self.spk_emb.astype(np.float16).tobytes()).decode('ascii')

        data = {
            "info": self.info,
            "text": self.text,
            "text_ids": self.text_ids,
            "codes": self.codes.tolist(),
            "spk_emb": spk_b64,
        }

        if include_embeds and self.summed_embeds is not None:
            data["summed_embeds"] = [e.tolist() for e in self.summed_embeds]

        if include_audio and self.audio is not None:
            data["audio"] = self.audio.tolist()

        os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=None if light else 2)

    # --- 加载 ---

    @staticmethod
    def _is_valid_json(path: str) -> bool:
        """检测路径指向的 JSON 是否符合锚点格式，只返回 True/False，不抛异常"""
        # 1. 文件存在性
        if not os.path.exists(path):
            logger.warning(f"⚠️ 找不到文件: {path}")
            return False

        # 2. JSON 可解析性
        with open(path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️ 不是有效的 JSON 文件: {e} at {path}")
                return False

        # 3. 必要字段存在性
        for key in ("text", "codes", "spk_emb", "text_ids"):
            if key not in data:
                logger.warning(f"⚠️ 缺少必要字段 '{key}' at {path}")
                return False

        # 4. 类型与数值深度校验
        if not isinstance(data["text"], str):
            logger.warning(f"⚠️ 'text' 必须是字符串 at {path}")
            return False

        if not isinstance(data["text_ids"], list) or not all(isinstance(x, int) for x in data["text_ids"]):
            logger.warning(f"⚠️ 'text_ids' 必须是整数列表 at {path}")
            return False

        codes = data["codes"]
        if not isinstance(codes, list) or len(codes) == 0:
            logger.warning(f"⚠️ 'codes' 必须是非空列表 at {path}")
            return False
        if not isinstance(codes[0], list) or len(codes[0]) != 16 or not all(isinstance(x, int) for x in codes[0]):
            logger.warning(f"⚠️ 'codes' 的每一帧必须包含 16 个整数 at {path}")
            return False

        spk_data = data["spk_emb"]
        if isinstance(spk_data, list):
            if len(spk_data) not in [1024, 2048]:
                logger.warning(f"⚠️ 'spk_emb' 列表维度必须为 2048，实际为 {len(spk_data)} at {path}")
                return False
        elif isinstance(spk_data, str):
            # Base64(fp16 bytes) → 解码后校验真实维度
            try:
                spk_arr = np.frombuffer(base64.b64decode(spk_data), dtype=np.float16)
            except Exception as e:
                logger.warning(f"⚠️ 'spk_emb' Base64 解码失败: {e} at {path}")
                return False
            if len(spk_arr) not in [1024, 2048]:
                logger.warning(f"⚠️ 'spk_emb' 解码后维度错误: {len(spk_arr)}，期望 2048 at {path}")
                return False
        else:
            logger.warning(f"⚠️ 'spk_emb' 格式应为 list 或 Base64 字符串 at {path}")
            return False

        return True

    @classmethod
    def from_json(cls, path: str) -> "TTSResult":
        """从 JSON 加载锚点（支持 Base64 和旧列表格式）；文件无效时记录日志并返回空结果"""
        if not cls._is_valid_json(path):
            logger.warning(f"⚠️ 无法加载锚点，返回空 TTSResult: {path}")
            return cls.empty()

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        spk_data = data["spk_emb"]
        if isinstance(spk_data, str):
            # 新版：Base64(fp16) → fp32
            spk_emb = np.frombuffer(base64.b64decode(spk_data), dtype=np.float16).astype(np.float32)
        else:
            # 旧版：直接数组列表
            spk_emb = np.array(spk_data, dtype=np.float32)

        res = cls(
            text=data["text"],
            text_ids=data["text_ids"],
            codes=np.array(data["codes"], dtype=np.int64),
            spk_emb=spk_emb,
            info=data.get("info", ""),
        )

        if "summed_embeds" in data:
            res.summed_embeds = [np.array(e, dtype=np.float32) for e in data["summed_embeds"]]

        if "audio" in data:
            res.audio = np.array(data["audio"], dtype=np.float32)

        return res

    # --- 统计报告 ---

    def print_stats(self):
        """打印性能报告"""
        if self.stats is None:
            logger.warning("⚠️ 当前结果无性能统计信息。")
            return

        s = self.stats
        print("-" * 40)
        print(f"性能分析报告 (音频长度: {self.duration:.2f}s | 文本长度: {len(self.text)})")
        print(f"  1. Prompt 编译:    {s.prompt_time:.4f}s")
        print(f"  2. Talker Prefill: {s.prefill_time:.4f}s")
        print(f"  3. 自回环总计:     {s.total_talker_time + s.total_predictor_time:.4f}s")
        print(f"     └─ 大师 (Talker):     {s.total_talker_time:.4f}s")
        print(f"     └─ 工匠 (Predictor):  {s.total_predictor_time:.4f}s")
        print(f"  4. 解码渲染:       {s.total_decoder_time:.4f}s")
        
        # 补充流式延迟统计
        if s.chunk_gen_times:
            print(f"  5. 流式分段统计 (Chunk Metrics):")
            print(f"     └─ 生成耗时序列: {', '.join([f'{x:.3f}s' for x in s.chunk_gen_times])}")
            if s.total_first_audio_delay > 0:
                print(f"     └─ 首包解码渲染: {s.first_chunk_decode_time:.4f}s")
                print(f"     └─ 全链路 FBL:   {s.total_first_audio_delay:.4f}s")
            
        print("-" * 40)
        print(f"核心推理耗时: {s.inference_only_time:.2f}s | RTF (Core): {self.rtf:.2f}")
        print(f"全链路总响应: {s.total_inference_time:.2f}s")
