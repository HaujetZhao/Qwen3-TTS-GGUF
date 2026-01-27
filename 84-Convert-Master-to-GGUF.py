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

# 1. 确保能导入 qwen3_tts_gguf 目录下的模块
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONVERT_LIB_DIR = os.path.join(PROJECT_ROOT, "qwen3_tts_gguf")

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
    print(f"💉 [Monkey Patch] Intercepted get_vocab_base_pre call.")
    print(f"💉 [Monkey Patch] Bypassing hash check, forcing return 'qwen2'.")
    return "qwen2"

# 4. 转换主逻辑
def main():
    MASTER_MODEL_DIR = os.path.join(PROJECT_ROOT, "Standalone-Bare-Master")
    GGUF_OUT = os.path.join(PROJECT_ROOT, "master-codec-only-3072-f16.gguf")
    TEMP_DIR = os.path.join(PROJECT_ROOT, "83_temp_hf_format")

    # [步骤 A] 准备临时权重 (转置 lm_head 并统一命名)
    # 这部分逻辑从之前的脚本继承，因为 python 调用 main() 也会读取磁盘文件
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)

    print("[1/3] Preparing weights for conversion...")
    from safetensors import safe_open
    from safetensors.torch import save_file
    
    # 读取原始文件
    src_weights = os.path.join(MASTER_MODEL_DIR, "model.safetensors")
    master_weights = {}
    
    with safe_open(src_weights, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            if key == "lm_head":
                if tensor.shape == (2048, 3072):
                    # print(f"  → Transposing lm_head to {tensor.shape} -> [3072, 2048]")
                    tensor = tensor.t().contiguous()
                master_weights["lm_head.weight"] = tensor
            elif key == "embed_tokens":
                master_weights["model.embed_tokens.weight"] = tensor
            else:
                new_key = key if key.startswith("model.") else f"model.{key}"
                master_weights[new_key] = tensor

    save_file(master_weights, os.path.join(TEMP_DIR, "model.safetensors"))

    # 复制配置文件
    for f in ["config.json", "vocab.json", "merges.txt", "tokenizer_config.json", "tokenizer.json"]:
        src = os.path.join(MASTER_MODEL_DIR, f)
        dst = os.path.join(TEMP_DIR, f)
        if os.path.exists(src):
            shutil.copy(src, dst)

    # [步骤 B] 应用 Monkey Patch
    print("[2/3] Applying Monkey Patch to convert_hf_to_gguf.TextModel...")
    
    # Patch TextModel 类
    original_method = TextModel.get_vocab_base_pre
    TextModel.get_vocab_base_pre = patched_get_vocab_base_pre
    
    # [步骤 C] 调用转换器主函数
    print(f"[3/3] executing convert_hf_to_gguf.main()...")
    
    # 构造 sys.argv
    sys.argv = [
        "convert_hf_to_gguf.py",
        TEMP_DIR,
        "--outfile", GGUF_OUT,
        "--outtype", "f16"
    ]
    
    try:
        convert_hf_to_gguf.main()
        print(f"\n✅ GGUF conversion successful!")
        print(f"Output: {GGUF_OUT}")
    except Exception as e:
        print(f"\n❌ Conversion failed with python exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理
        print("[Cleanup] Removing temp directory...")
        # shutil.rmtree(TEMP_DIR) # Keep for debugging if needed

if __name__ == "__main__":
    main()
