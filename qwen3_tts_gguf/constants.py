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
    "chinese": 2055, 
    "english": 2050, 
    "japanese": 2058, 
    "korean": 2064,
    "german": 2053, 
    "spanish": 2054, 
    "french": 2061, 
    "russian": 2069,
    "beijing_dialect": 2074, 
    "sichuan_dialect": 2062, 
    "auto": 2055 # 默认跟随中文
}

# 官方流程协议标签
PROTOCOL = {
    "PAD": 2148, 
    "BOS": 2149, 
    "EOS": 2150, 
    "BOS_TOKEN": 151672, 
    "EOS_TOKEN": 151673,
    "THINK": 2154, 
    "NOTHINK": 2155, 
    "THINK_BOS": 2156, 
    "THINK_EOS": 2157
}

# 默认分步码表数量
NUM_QUANTIZERS = 16

# 采样率
SAMPLE_RATE = 24000
