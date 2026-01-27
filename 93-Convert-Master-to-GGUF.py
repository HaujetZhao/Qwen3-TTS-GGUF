"""
将 Standalone-Bare-Master-Merged-Vocab 转换为 GGUF 格式

方案：
- 使用 21 方案的 tokenizer 文件（从原始 Qwen3-TTS 模型复制）
- llama.cpp 要求 lm_head 是 [2048, 151936]（转置后的格式）
- 需要先转置 lm_head，然后转换
"""
import os
import subprocess
import sys
import shutil

# Configuration
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SOURCE_MODEL_DIR = os.path.join(PROJECT_ROOT, "Qwen3-TTS-12Hz-1.7B-CustomVoice")
MASTER_MODEL_DIR = os.path.join(PROJECT_ROOT, "Standalone-Bare-Master-Merged-Vocab")
OUTPUT_HF_DIR = os.path.join(PROJECT_ROOT, "Master-Merged-Vocab-HF-For-GGUF")
CONVERT_SCRIPT = os.path.join(PROJECT_ROOT, "qwen3_tts_gguf", "convert_hf_to_gguf.py")
GGUF_OUT = os.path.join(PROJECT_ROOT, "master-merged-vocab-f16.gguf")

def prepare_for_gguf():
    """准备 GGUF 转换：转置 lm_head 并复制 tokenizer"""

    print(f"[1/4] Preparing Master model for GGUF conversion...")

    # 1. 确保输出目录存在
    os.makedirs(OUTPUT_HF_DIR, exist_ok=True)

    # 2. 复制 config.json
    print(f"[2/4] Copying config.json...")
    src_config = os.path.join(MASTER_MODEL_DIR, "config.json")
    dst_config = os.path.join(OUTPUT_HF_DIR, "config.json")
    shutil.copy(src_config, dst_config)
    print(f"  Copied config.json")

    # 3. 处理权重：将 lm_head 转回 HF 标准格式 [151936, 2048]
    # 因为 llama.cpp 转换脚本会自动转置 lm_head.weight
    print(f"[3/4] Processing weights for GGUF conversion...")
    from safetensors import safe_open
    from safetensors.torch import save_file
    import torch

    master_weights = {}
    src_weights = os.path.join(MASTER_MODEL_DIR, "model.safetensors")

    with safe_open(src_weights, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)

            # lm_head 在 91 脚本中是 [2048, 151936]（转置后的格式）
            # 但 HF 标准格式应该是 [151936, 2048]
            # llama.cpp 转换脚本会自动将 [151936, 2048] 转置成 [2048, 151936]
            if key == "lm_head":
                print(f"  lm_head shape in 91 output: {tensor.shape}")
                if tensor.shape == (2048, 151936):
                    print(f"  → Transposing back to HF format [151936, 2048]")
                    print(f"     (llama.cpp conversion script will transpose it back to [2048, 151936])")
                    # 转置回 HF 标准格式
                    lm_head_hf = tensor.t().contiguous()  # [2048, 151936] -> [151936, 2048]
                    print(f"  ✓ lm_head HF format shape: {lm_head_hf.shape}")
                    master_weights["lm_head.weight"] = lm_head_hf
                else:
                    print(f"  ⚠ lm_head has unexpected shape: {tensor.shape}")
                    print(f"  Expected: [2048, 151936]")
                    raise ValueError(f"Invalid lm_head shape: {tensor.shape}")

            # embed_tokens 应该是 [151936, 2048]（无需修改）
            elif key == "embed_tokens":
                print(f"  embed_tokens shape: {tensor.shape}")
                if tensor.shape == (151936, 2048):
                    print(f"  ✓ embed_tokens is correct [151936, 2048]")
                    master_weights["model.embed_tokens.weight"] = tensor
                else:
                    print(f"  ⚠ embed_tokens has unexpected shape: {tensor.shape}")
                    raise ValueError(f"Invalid embed_tokens shape: {tensor.shape}")

            # 其他权重：添加 model. 前缀（如果还没有）
            else:
                if not key.startswith("model."):
                    new_key = f"model.{key}"
                else:
                    new_key = key
                master_weights[new_key] = tensor

    # 保存处理后的权重
    dst_weights = os.path.join(OUTPUT_HF_DIR, "model.safetensors")
    save_file(master_weights, dst_weights)
    print(f"  Saved processed weights to: {dst_weights}")

    # 4. 复制 tokenizer 文件（从原始 Qwen3-TTS 模型）
    print(f"[4/4] Copying tokenizer files from original model...")
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "generation_config.json"
    ]

    for filename in tokenizer_files:
        src = os.path.join(SOURCE_MODEL_DIR, filename)
        dst = os.path.join(OUTPUT_HF_DIR, filename)
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"  ✓ Copied {filename}")
        else:
            print(f"  ⚠ Skipped {filename} (not found)")

    # 复制 special_tokens_map.json（如果存在）
    src_special = os.path.join(SOURCE_MODEL_DIR, "special_tokens_map.json")
    if os.path.exists(src_special):
        shutil.copy(src_special, os.path.join(OUTPUT_HF_DIR, "special_tokens_map.json"))
        print(f"  ✓ Copied special_tokens_map.json")

    print(f"\n✅ Preparation complete!")
    print(f"Output directory: {OUTPUT_HF_DIR}")
    print(f"  - config.json")
    print(f"  - model.safetensors (with lm_head.weight [2048, 151936])")
    print(f"  - tokenizer files")

    return OUTPUT_HF_DIR

def convert_to_gguf():
    """转换为 GGUF 格式"""

    print(f"\n--- Converting to GGUF ---")

    # 1. 准备 HF 格式的模型
    hf_model_dir = prepare_for_gguf()

    # 2. 检查转换脚本
    if not os.path.exists(CONVERT_SCRIPT):
        print(f"❌ Error: Could not find llama.cpp conversion script at {CONVERT_SCRIPT}")
        return False

    # 3. 构建转换命令
    cmd = [
        sys.executable,
        CONVERT_SCRIPT,
        hf_model_dir,
        "--outfile", GGUF_OUT,
        "--outtype", "f16"
    ]

    # 可以添加 --verbose 查看详细信息
    # cmd.append("--verbose")

    print(f"Running command:")
    print(f"  {cmd[0]} {cmd[1]} {cmd[2]} --outfile {cmd[4]} --outtype {cmd[6]}")
    print()

    try:
        # 运行转换
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        print("✅ Conversion successful!")
        print(f"\nGGUF model saved to: {GGUF_OUT}")
        print(f"File size: {os.path.getsize(GGUF_OUT) / 1024**3:.2f} GB")

        # 显示转换输出（最后 50 行）
        if result.stdout:
            print("\n--- Conversion Output (last 50 lines) ---")
            lines = result.stdout.strip().split('\n')
            for line in lines[-50:]:
                print(line)

        return True

    except subprocess.CalledProcessError as e:
        print("\n❌ Conversion failed!")
        print("\n--- Error Output ---")
        print(e.stderr)
        if e.stdout:
            print("\n--- Standard Output ---")
            print(e.stdout)
        return False

if __name__ == "__main__":
    try:
        success = convert_to_gguf()
        sys.exit(0 if success else 1)
    except Exception as e:
        import traceback
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
        sys.exit(1)
