"""
result.py - 合成结果与身份锚点统一类
核心职责：
1. 承载 TTS 合成结果 (音频、元数据、性能统计)。
2. 作为 Voice Identity 锚点提供克隆所需的特征。
3. 提供音频播放、保存以及 JSON 持久化能力。
"""
import json
import os
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Union
from .constants import SAMPLE_RATE, map_speaker, map_language
from . import logger

@dataclass
class Timing:
    """性能耗时统计"""
    prompt_time: float = 0.0
    prefill_time: float = 0.0
    master_loop_time: float = 0.0
    craftsman_loop_time: float = 0.0
    mouth_render_time: float = 0.0
    total_steps: int = 0

    @property
    def total_inference_time(self) -> float:
        return (self.prompt_time + self.prefill_time + 
                self.master_loop_time + self.craftsman_loop_time + 
                self.mouth_render_time)

@dataclass
class LoopOutput:
    """推理内核循环的输出封装"""
    all_codes: List[List[int]]     # 所有生成的 Codec IDs
    summed_embeds: List[np.ndarray] # 叠加特征序列
    timing: Timing                  # 性能统计对象

@dataclass
class TTSConfig:
    """推理控制参数封装"""
    # 大师控制 (Master/Talker/Talker)
    do_sample: bool = True
    temperature: float = 0.8
    top_p: float = 1.0
    top_k: int = 50
    
    # 工匠控制 (Craftsman/Predictor/Subtalker)
    sub_do_sample: bool = False
    sub_temperature: float = 0.5
    sub_top_p: float = 1.0
    sub_top_k: int = 50
    
    max_steps: int = 300
    voice_clone_mode: bool = True
    
    # 流式发声控制
    stream_play: bool = False       # 是否开启流式边推边播 (需要 sounddevice)
    mouth_chunk_size: int = 25      # 流式播放时，每累积多少帧 Codec 发送给嘴巴 (默认 25 帧)

@dataclass
class TTSResult:
    """TTS 合成结果 (同时也是音色锚点)"""
    # 核心特征 (锚点要素)
    text: str                               # 文字内容
    spk_emb: np.ndarray                     # 全局音色向量 (2048)
    text_ids: List[int]                     # 文本 Token IDs
    codes: np.ndarray                       # 音频 Codec IDs (T, 16)
    summed_embeds: Optional[List[np.ndarray]] = None # 音频叠加特征 (T, 2048) - 可选
    info: str = ""                          # 备注信息 (如音色描述)
    
    # 产出附件 (可选)
    audio: Optional[np.ndarray] = None      # 音频波形 (PCM float32)
    stats: Optional[Timing] = None          # 性能统计信息

    @property
    def is_valid_anchor(self) -> bool:
        """是否具有作为 Voice 锚点的必要特征"""
        return self.codes is not None and self.spk_emb is not None

    @property
    def duration(self) -> float:
        """音频时长 (s)"""
        if self.audio is not None:
            return len(self.audio) / SAMPLE_RATE
        return 0.0
    
    @property
    def rtf(self) -> float:
        """实时因子 (Real-Time Factor)"""
        if self.duration == 0 or self.stats is None: return 0.0
        return self.stats.total_inference_time / self.duration

    # --- IO 能力 ---

    def play(self, blocking: bool = True):
        """播放音频结果"""
        if self.audio is None or len(self.audio) == 0:
            logger.warning("⚠️ No audio data available in this result to play.")
            return
        import sounddevice as sd
        sd.play(self.audio, samplerate=24000, blocking=blocking)

    def save_wav(self, path: str):
        """保存为 WAV 文件"""
        if self.audio is None:
            logger.error("❌ No audio data to save.")
            return
        import soundfile as sf
        os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
        sf.write(path, self.audio, SAMPLE_RATE)
        logger.info(f"💾 Audio saved to: {path}")

    # --- 持久化能力 ---

    def save_json(self, path: str, include_audio: bool = False, include_embeds: bool = False, light: bool = False, info: Optional[str] = None):
        """将特征锚点保存到 JSON"""
        if info is not None:
            self.info = info
            
        if not self.is_valid_anchor:
            logger.warning("⚠️ Result is incomplete, cannot save as anchor.")
            return
        
        data = {
            "text": self.text,
            "info": self.info,
            "text_ids": self.text_ids,
            "codes": self.codes.tolist(),
            "spk_emb": self.spk_emb.tolist(),
        }

        if include_embeds and self.summed_embeds is not None:
            data["summed_embeds"] = [e.tolist() for e in self.summed_embeds]
        
        if include_audio and self.audio is not None:
            data["audio"] = self.audio.tolist()
        
        os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=None if light else 2)
        
        logger.info(f"💾 Voice JSON saved to: {path} (Light: {light})")

    @classmethod
    def from_json(cls, path: str) -> 'TTSResult':
        """从 JSON 文件恢复 Voice 锚点"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Identity file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        res = cls(
            text=data.get("text", ""),
            info=data.get("info", ""),
            text_ids=data["text_ids"],
            spk_emb=np.array(data["spk_emb"], dtype=np.float32),
            codes=np.array(data["codes"], dtype=np.int64),
        )

        if "summed_embeds" in data:
            res.summed_embeds = [np.array(e, dtype=np.float32) for e in data["summed_embeds"]]
        
        if "audio" in data:
            res.audio = np.array(data["audio"], dtype=np.float32)
            
        return res

    def print_stats(self):
        """打印性能报告"""
        if self.stats is None:
            print("No performance stats available for this result.")
            return
            
        s = self.stats
        print("-" * 40)
        print(f"性能分析报告 (音频长度: {self.duration:.2f}s | 文本长度: {len(self.text)})")
        print(f"  1. Prompt 编译:   {s.prompt_time:.4f}s")
        print(f"  2. 大师 Prefill:  {s.prefill_time:.4f}s")
        print(f"  3. 自回环总计:    {s.master_loop_time + s.craftsman_loop_time:.4f}s")
        print(f"     └─ 大师 (Master):    {s.master_loop_time:.4f}s")
        print(f"     └─ 工匠 (Craftsman): {s.craftsman_loop_time:.4f}s")
        print(f"  4. 嘴巴渲染 (Mouth): {s.mouth_render_time:.4f}s")
        print("-" * 40)
        print(f"总耗时: {s.total_inference_time:.2f}s | RTF: {self.rtf:.2f}")
