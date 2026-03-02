"""
constants.py - Qwen3-TTS 常量定义
包含说话人映射、语言映射和官方协议标签。
"""

# 说话人 ID 映射 (Verified from official config.json)
SPEAKER_MAP = {
    "vivian": 3065, 
    "serena": 3066, 
    "uncle_fu": 3010, 
    "ryan": 3061,
    "aiden": 2861, 
    "ono_anna": 2873, 
    "sohee": 2864, 
    "eric": 2875, 
    "dylan": 2878
}

# 语言 ID 映射
LANGUAGE_MAP = {
    "english": 2050, 
    "german": 2053, 
    "spanish": 2054, 
    "chinese": 2055, 
    "japanese": 2058, 
    "french": 2061, 
    "sichuan_dialect": 2062, 
    "korean": 2064,
    "russian": 2069,
    "italian": 2070,
    "portuguese": 2071,
    "beijing_dialect": 2074, 
}
# 官方流程协议标签
PROTOCOL = {
    "PAD": 2148, 
    "BOS": 2149, 
    "EOS": 2150, 
    "TTS_BOS": 151672, 
    "TTS_EOS": 151673,
    "THINK": 2154, 
    "NOTHINK": 2155, 
    "THINK_BOS": 2156, 
    "THINK_EOS": 2157
}

# 默认分步码表数量
NUM_QUANTIZERS = 16

# 采样率
SAMPLE_RATE = 24000

def map_speaker(spk) -> int:
    """将说话人名称或 ID 映射为官方数值 ID"""
    if isinstance(spk, int): return spk
    return SPEAKER_MAP.get(str(spk).lower(), 3065)

def map_language(lang) -> int:
    """将语言名称或 ID 映射为官方数值 ID"""
    if isinstance(lang, int): return lang
    return LANGUAGE_MAP.get(str(lang).lower(), None)
