import os
import sys
from safetensors import safe_open

# 路径
STANDALONE_MODEL_PATH = r"C:\Users\Haujet\Desktop\qwen3-tts\Qwen3-TTS-Talker-Standalone\model-00001-of-00002.safetensors"

def main():
    if not os.path.exists(STANDALONE_MODEL_PATH):
        print("Error: model.safetensors not found!")
        return

    print(f"Inspecting weights in {STANDALONE_MODEL_PATH}...")
    with safe_open(STANDALONE_MODEL_PATH, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        keys.sort()
        
        print(f"Total keys: {len(keys)}")
        print("\nFirst 20 keys:")
        for k in keys[:20]:
            print(f"  {k}")
            
        print("\nLast 10 keys:")
        for k in keys[-10:]:
            print(f"  {k}")

        # 检查是否有 talker. 前缀
        has_prefix = any(k.startswith("talker.") for k in keys)
        if has_prefix:
            print("\n⚠️ Warning: Keys still have 'talker.' prefix!")
        else:
            print("\n✅ Success: Keys do not have 'talker.' prefix.")

if __name__ == "__main__":
    main()
