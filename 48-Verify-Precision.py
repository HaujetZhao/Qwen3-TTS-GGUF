import os
from safetensors import safe_open

# 抽离后的模型路径
STANDALONE_MODEL_PATH = r"C:\Users\Haujet\Desktop\qwen3-tts\Qwen3-TTS-Talker-Standalone\model.safetensors"

def main():
    if not os.path.exists(STANDALONE_MODEL_PATH):
        print(f"Error: {STANDALONE_MODEL_PATH} not found!")
        return

    print(f"Checking precision of: {STANDALONE_MODEL_PATH}")
    
    # 检查文件大小
    size_gb = os.path.getsize(STANDALONE_MODEL_PATH) / (1024**3)
    print(f"File size: {size_gb:.2f} GB")

    # 检查 tensor 精度
    with safe_open(STANDALONE_MODEL_PATH, framework="pt", device="cpu") as f:
        # 随便选一个层检查
        test_key = "model.layers.0.self_attn.q_proj.weight"
        if test_key in f.keys():
            tensor = f.get_tensor(test_key)
            print(f"Tensor '{test_key}' dtype: {tensor.dtype}")
        else:
            print(f"Key '{test_key}' not found in safetensors!")
            # 打印前 5 个 key 看看
            print("Available keys (first 5):", list(f.keys())[:5])

if __name__ == "__main__":
    main()
