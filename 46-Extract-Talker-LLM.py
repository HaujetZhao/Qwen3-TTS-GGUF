import os
import sys
import torch
import json
import shutil

# 确保导入本地源码
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(PROJECT_ROOT, "Qwen3-TTS")
sys.path.insert(0, SOURCE_DIR)

from qwen_tts import Qwen3TTSModel

def main():
    MODEL_PATH = os.path.abspath("Qwen3-TTS-12Hz-1.7B-CustomVoice")
    OUTPUT_PATH = os.path.abspath("Qwen3-TTS-Talker-Standalone")

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    print(f"Loading official model from {MODEL_PATH}...")
    # 使用 bfloat16 加载以保持精度并节省内存/磁盘
    tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map="cpu",
        torch_dtype=torch.bfloat16
    )
    
    # 原始 model 是 Qwen3TTSForConditionalGeneration
    full_model = tts.model
    
    print("Extracting Talker component...")
    # talker 是 Qwen3TTSTalkerForConditionalGeneration
    talker = full_model.talker

    print(f"Saving Talker weights and config to {OUTPUT_PATH}...")
    # save_pretrained 会保存 model.safetensors 和 config.json
    talker.save_pretrained(OUTPUT_PATH)

    # 复制分词器和预处理器相关文件，这些是 Talker 推理也需要的
    files_to_copy = [
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "preprocessor_config.json",
        "tokenizer.json"
    ]

    for f in files_to_copy:
        src = os.path.join(MODEL_PATH, f)
        if os.path.exists(src):
            shutil.copy(src, OUTPUT_PATH)
            print(f"Copied {f}")
        else:
            print(f"Skipped {f} (not found)")

    # 修正 saved config.json 中的元数据
    # 有时 save_pretrained 可能不会包含我们想要的 architecture
    config_file = os.path.join(OUTPUT_PATH, "config.json")
    if os.path.exists(config_file):
        with open(config_file, "r", encoding="utf-8") as f:
            config_data = json.load(f)
        
        # 将 architecture 改为 Talker 专用的类名
        config_data["architectures"] = ["Qwen3TTSTalkerForConditionalGeneration"]
        # 确保 model_type 设为 talker
        config_data["model_type"] = "qwen3_tts_talker"
        
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)
        print("Updated config.json for standalone use.")

    print(f"\n\u2705 Standalone Talker LLM has been saved to:\n{OUTPUT_PATH}")

if __name__ == "__main__":
    main()
