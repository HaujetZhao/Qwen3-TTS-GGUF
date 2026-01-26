import os
import sys
import torch
from qwen_tts import Qwen3TTSModel

# 确保导入本地源码
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(PROJECT_ROOT, "Qwen3-TTS")
sys.path.insert(0, SOURCE_DIR)

def main():
    MODEL_PATH = os.path.abspath("Qwen3-TTS-12Hz-1.7B-CustomVoice")
    OUTPUT_FILE = "qwen3_tts_layer_structure.txt"
    
    print(f"Loading official model from {MODEL_PATH}...")
    # 只需要结构，可以用 cpu 加载，无需精度
    tts = Qwen3TTSModel.from_pretrained(MODEL_PATH, device_map="cpu", torch_dtype=torch.float32)
    
    model = tts.model
    
    print(f"Writing layer structure to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"Qwen3-TTS Model Structure\n")
        f.write(f"="*50 + "\n\n")
        
        # 递归遍历所有模块
        for name, module in model.named_modules():
            # 获取模块的参数信息
            params = list(module.named_parameters(recurse=False))
            
            if params:
                f.write(f"Layer: {name}\n")
                f.write(f"Type:  {type(module).__name__}\n")
                for p_name, p in params:
                    f.write(f"  - Param: {p_name} | Shape: {list(p.shape)} | Dtype: {p.dtype}\n")
                f.write("-" * 30 + "\n")
            elif len(list(module.children())) == 0:
                # 即使没有参数，如果是叶子节点也记录一下（如 Identity, ReLU 等）
                f.write(f"Module (Leaf): {name}\n")
                f.write(f"Type:  {type(module).__name__}\n")
                f.write("-" * 30 + "\n")

    print(f"✅ Done! Layer structure saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
