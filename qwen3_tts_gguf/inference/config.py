from dataclasses import dataclass

@dataclass
class TTSConfig:
    """
    TTS 推理全链路控制参数封装。
    包含 Talker (生成语义特征) 和 Predictor (生成声学码) 两个阶段的独立采样控制。
    """
    # --- 大师控制 (Talker / Semantic Stage) ---
    do_sample: bool = True           # 是否开启随机采样。False 则使用 Greedy Search，结果稳定但机械。
    temperature: float = 0.9         # 采样温度。值越大越随机(情感起伏大)，过大可能崩字；值越小越严谨。
    top_p: float = 1.0               # 核采样阈值。只从累积概率达到 p 的 Token 中采样。
    top_k: int = 50                  # 候选集大小。采样时只考虑概率最高的前 k 个 Token。
    
    # --- 工匠控制 (Predictor / Acoustic Stage) ---
    sub_do_sample: bool = True      # 工匠阶段通常建议 False，使用确定性生成或低温度生成以保证音频稳定。
    sub_temperature: float = 0.9     # 工匠阶段的温度。调低可以减少语速抖动和电音感。
    sub_top_p: float = 1.0           # 工匠阶段的 Top-P。
    sub_top_k: int = 50              # 工匠阶段的 Top-K。
    
    # --- 采样增强控制 (增强语音稳定性与灵活性) ---
    min_p: float = 0.0               # Min-P 采样阈值。建议 0.05。过滤低概率噪声，减少电音。
    repeat_penalty: float = 1.05     # 重复惩罚。官方默认值 1.05。防止复读机，增加语气变化。
    frequency_penalty: float = 0.0   # 频率惩罚。防止特定词汇过度出现。
    presence_penalty: float = 0.0    # 存在惩罚。增加词汇多样性。
    penalty_last_n: int = 128        # 惩罚项回溯的历史 Token 长度。
    
    # --- 全局生成控制 ---
    max_steps: int = 300             # 最大生成步数。决定了单次合成最长的持续时间。
    seed: int = None                 # 随机种子。设置为固定数字（如 42）可以使生成结果在同环境下完全一致。
