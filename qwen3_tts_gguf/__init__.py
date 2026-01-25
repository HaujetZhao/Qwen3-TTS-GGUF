# Qwen3-TTS GGUF 核心逻辑包
# 遵循低耦合、高内聚原则

import logging
import os
import sys
from datetime import datetime

# 1. 确定路径
# 当前包目录
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
# 项目根目录 (假设包在根目录下的一级目录)
PROJECT_ROOT = os.path.dirname(PACKAGE_DIR)
# 日志目录
LOG_DIR = os.path.join(PROJECT_ROOT, "log")

# 确保日志目录存在
os.makedirs(LOG_DIR, exist_ok=True)

# 2. 配置统一的 Logger
# 使用固定名称以保证全局单例性
logger = logging.getLogger("qwen3_tts_gguf")
logger.setLevel(logging.INFO)
logger.propagate = False  # 防止向上层 Logger (如 root) 传播导致重复

# 3. 添加 Handlers (仅在没有 Handler 时添加，防止重复)
if not logger.handlers:
    # 格式器
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # A. 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # B. 文件输出 (覆盖模式)
    log_file_path = os.path.join(LOG_DIR, "latest.log")
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # 文件记录更详细的信息
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"日志系统初始化完成。日志文件: {log_file_path}")

# 导出 logger 供包内模块使用
__all__ = ['logger']
