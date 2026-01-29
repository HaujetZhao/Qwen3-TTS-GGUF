"""
result.py - 合成结果封装类
包含音频波形和详细的性能统计信息。
"""
import numpy as np
from dataclasses import dataclass, field
from .constants import SAMPLE_RATE

@dataclass
class SynthesisResult:
    """TTS 合成结果与性能统计"""
    audio: np.ndarray           # 音频数据 (PCM float32)
    text: str = ""              # 输入文本
    
    # 内部统计 (秒)
    prompt_time: float = 0.0
    prefill_time: float = 0.0
    master_loop_time: float = 0.0
    craftsman_loop_time: float = 0.0
    mouth_render_time: float = 0.0
    
    # 步数统计
    total_steps: int = 0
    
    @property
    def duration(self) -> float:
        """音频时长 (s)"""
        return len(self.audio) / SAMPLE_RATE if len(self.audio) > 0 else 0
    
    @property
    def total_inference_time(self) -> float:
        """推理阶段总耗时 (s)"""
        return (self.prompt_time + self.prefill_time + 
                self.master_loop_time + self.craftsman_loop_time + 
                self.mouth_render_time)

    @property
    def rtf(self) -> float:
        """实时因子 (Real-Time Factor)"""
        if self.duration == 0: return 0
        return self.total_inference_time / self.duration

    def print_stats(self):
        """打印类似 41 号脚本的性能报告"""
        print("-" * 40)
        print(f"性能分析报告 (音频长度: {self.duration:.2f}s | 文本长度: {len(self.text)})")
        print(f"  1. Prompt 编译:   {self.prompt_time:.4f}s")
        print(f"  2. 大师 Prefill:  {self.prefill_time:.4f}s")
        print(f"  3. 自回环总计:    {self.master_loop_time + self.craftsman_loop_time:.4f}s")
        print(f"     └─ 大师 (Master):    {self.master_loop_time:.4f}s")
        print(f"     └─ 工匠 (Craftsman): {self.craftsman_loop_time:.4f}s")
        print(f"  4. 嘴巴渲染 (Mouth): {self.mouth_render_time:.4f}s")
        print("-" * 40)
        print(f"总耗时: {self.total_inference_time:.2f}s | RTF: {self.rtf:.2f}")
