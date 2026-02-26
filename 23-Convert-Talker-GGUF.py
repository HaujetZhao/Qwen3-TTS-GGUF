"""
[83] Convert Master to GGUF with Monkey Patching
使用 Python Monkey Patch 技术绕过 convert_hf_to_gguf.py 的哈希校验。
强制将当前迷你 Tokenizer 识别为 "qwen2" 类型。
"""
import sys
import os
import shutil
import logging
from unittest.mock import patch
import json
from pathlib import Path

# 1. 确保能导入 qwen3_tts_gguf 目录下的模块
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONVERT_LIB_DIR = os.path.join(PROJECT_ROOT, "qwen3_tts_gguf", 'export')

if CONVERT_LIB_DIR not in sys.path:
    # 插入到最前面以确保优先加载
    sys.path.insert(0, CONVERT_LIB_DIR)

# 2. 导入目标模块
try:
    import convert_hf_to_gguf
    # get_vocab_base_pre 是 TextModel 的方法
    from convert_hf_to_gguf import TextModel
    import gguf
except ImportError as e:
    print(f"❌ Error importing convert_hf_to_gguf: {e}")
    sys.exit(1)


# 3. 定义补丁函数
def patched_get_vocab_base_pre(self, tokenizer) -> str:
    """
    Monkey Patch 替代函数。
    不进行任何哈希计算，直接返回 'qwen2'。
    """
    print(f"💉 [补丁] 拦截到 get_vocab_base_pre 调用。")
    print(f"💉 [补丁] 绕过哈希检查，强制返回 'qwen2'。")
    return "qwen2"

def patched_load_hparams(dir_model: Path, is_mistral_format: bool):
    """
    Monkey Patch 替代函数。
    强制从 config.json 加载参数，绕过 AutoConfig 的潜在远程代码加载和解析干扰。
    """
    print(f"💉 [补丁] 拦截到 load_hparams 调用。")
    print(f"💉 [补丁] 强制从 {dir_model / 'config.json'} 加载配置。")
    
    if is_mistral_format:
        with open(dir_model / "params.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        return config

    try:
        with open(dir_model / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        print(f"❌ 加载配置文件失败: {e}")
        raise

    if "llm_config" in config:
        config["text_config"] = config["llm_config"]
    if "lm_config" in config:
        config["text_config"] = config["lm_config"]
    if "thinker_config" in config:
        config["text_config"] = config["thinker_config"]["text_config"]
    if "lfm" in config:
        config["text_config"] = config["lfm"]
        
    return config

from export_config import EXPORT_DIR

# 4. 转换主逻辑
def main():
    TALKER_MODEL_DIR = os.path.join(EXPORT_DIR, "hf")
    GGUF_OUT = os.path.join(EXPORT_DIR, "qwen3_tts_talker.gguf")

    print(f"--- 正在执行 GGUF 转换 (Talker) ---")
    print(f"源目录: {TALKER_MODEL_DIR}")
    print(f"输出文件: {GGUF_OUT}")

    # [步骤 A] 检查必要文件
    required_files = ["model.safetensors", "config.json", "tokenizer.json"]
    for f in required_files:
        if not os.path.exists(os.path.join(TALKER_MODEL_DIR, f)):
            print(f"❌ 错误: 缺少必要文件 {f}，请制作模型后再运行。")
            sys.exit(1)

    # [步骤 B] 应用 Monkey Patch
    print("[1/2] 正在应用虚拟加载补丁 (动态映射权重键)...")
    
    # Patch 1: TextModel.get_vocab_base_pre (绕过分词器哈希检查)
    TextModel.get_vocab_base_pre = patched_get_vocab_base_pre
    
    # Patch 2: ModelBase.load_hparams (支持从 config.json 加载参数)
    from convert_hf_to_gguf import ModelBase
    ModelBase.load_hparams = staticmethod(patched_load_hparams)
    
    # Patch 3: ModelBase.index_tensors (核心：虚拟映射权重键)
    # GGUF 转换器探测到 qwen2 架构时，要求 backbone 权重带 model. 前缀
    # 我们在读取时动态加上前缀，从而避免在磁盘上复制和重命名文件
    original_index_tensors = ModelBase.index_tensors

    def patched_index_tensors(self, *args, **kwargs):
        # 调用原始索引逻辑获取所有 Tensor 生成器
        tensors = original_index_tensors(self, *args, **kwargs)
        new_tensors = {}
        for name, data_gen in tensors.items():
            # lm_head 不需要前缀；model. 开头的说明已经有前缀了
            if name.startswith("lm_head") or name.startswith("model."):
                new_tensors[name] = data_gen
            else:
                # 给骨干网络动态加上 model. 前缀 (如 layers.0... -> model.layers.0...)
                new_tensors[f"model.{name}"] = data_gen
        return new_tensors

    ModelBase.index_tensors = patched_index_tensors

    # [步骤 C] 调用转换器主函数
    print(f"[2/2] 正在调用 convert_hf_to_gguf.main()...")
    
    # 模拟命令行参数
    sys.argv = [
        "convert_hf_to_gguf.py",
        TALKER_MODEL_DIR,
        "--outfile", GGUF_OUT,
        "--outtype", "f16"
    ]
    
    try:
        convert_hf_to_gguf.main()
        print(f"\n✅ Talker GGUF 转换成功!")
        print(f"输出路径: {GGUF_OUT}")
    except Exception as e:
        print(f"\n❌ 转换过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
