"""
从原始的 Qwen3-TTS 模型中提取"大师"(LLM Backbone)的权重
将大师权重保存为独立的 safetensors 文件
"""
import os
import json
from safetensors import safe_open
from safetensors.torch import save_file

def extract_master_weights():
    """提取大师权重到独立文件"""

    # 路径配置
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    ORIGINAL_MODEL_PATH = os.path.join(PROJECT_ROOT, "Qwen3-TTS-12Hz-1.7B-CustomVoice")
    ORIGINAL_WEIGHTS = os.path.join(ORIGINAL_MODEL_PATH, "model.safetensors")
    ORIGINAL_CONFIG = os.path.join(ORIGINAL_MODEL_PATH, "config.json")

    # 输出路径
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Standalone-Bare-Master")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    OUTPUT_WEIGHTS = os.path.join(OUTPUT_DIR, "model.safetensors")
    OUTPUT_CONFIG = os.path.join(OUTPUT_DIR, "config.json")

    print(f"--- Extracting Master Weights ---")
    print(f"Loading original weights from: {ORIGINAL_WEIGHTS}")

    # 1. 读取原始配置
    with open(ORIGINAL_CONFIG, "r", encoding="utf-8") as f:
        original_config = json.load(f)

    # 提取 talker_config 作为大师的配置
    master_config = original_config['talker_config'].copy()
    # 添加必要的元数据
    master_config['_original_model_type'] = 'qwen3_tts_talker'
    master_config['_extracted_from'] = 'Qwen3-TTS-12Hz-1.7B-CustomVoice'

    # 2. 提取大师权重 (talker.model.* -> 无前缀)
    master_weights = {}
    codec_head_weight = None

    with safe_open(ORIGINAL_WEIGHTS, framework="pt", device="cpu") as f:
        total_keys = len(list(f.keys()))
        print(f"Total keys in original model: {total_keys}")

        for key in f.keys():
            # 大师 backbone 权重: talker.model.*
            if key.startswith("talker.model."):
                # 移除前缀,变成无前缀的格式
                new_key = key.replace("talker.model.", "")
                tensor = f.get_tensor(key)
                master_weights[new_key] = tensor
                print(f"  Extracted: {key} -> {new_key}, shape: {tensor.shape}")

            # Codec head 权重: talker.codec_head.weight
            elif key == "talker.codec_head.weight":
                codec_head_weight = f.get_tensor(key)
                print(f"  Separated: {key}, shape: {codec_head_weight.shape}")

            # 跳过其他权重 (code_predictor, speaker_encoder 等)
            else:
                print(f"  Skipped: {key}")

    print(f"\nExtracted {len(master_weights)} master weights")

    # 3. 保存 codec_head 权重到单独文件 (如果需要)
    if codec_head_weight is not None:
        codec_head_path = os.path.join(OUTPUT_DIR, "codec_head.safetensors")
        save_file({"weight": codec_head_weight}, codec_head_path)
        print(f"Saved codec head to: {codec_head_path}")

    # 4. 保存大师权重
    save_file(master_weights, OUTPUT_WEIGHTS)
    print(f"Saved master weights to: {OUTPUT_WEIGHTS}")

    # 5. 保存大师配置
    with open(OUTPUT_CONFIG, "w", encoding="utf-8") as f:
        json.dump(master_config, f, indent=2, ensure_ascii=False)
    print(f"Saved master config to: {OUTPUT_CONFIG}")

    # 6. 生成模型信息
    metadata = {
        "model_type": "Qwen3TTSTalkerModel",
        "architecture": "Master-only (LLM Backbone)",
        "source_model": "Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "extraction_date": "2026-01-26",
        "components": {
            "master": "LLM Backbone (28 layers, 2048 hidden)",
            "codec_head": "Separated to codec_head.safetensors"
        },
        "usage": {
            "load_with": "from bare_master.modeling import Qwen3TTSTalkerModel",
            "config_class": "from bare_master.configuration import Qwen3TTSTalkerConfig"
        }
    }

    metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Saved metadata to: {metadata_path}")

    print("\n✅ Extraction complete!")
    print(f"Master model saved to: {OUTPUT_DIR}")
    print(f"  - model.safetensors ({os.path.getsize(OUTPUT_WEIGHTS) / 1024**3:.2f} GB)")
    print(f"  - config.json")
    print(f"  - codec_head.safetensors")
    print(f"  - metadata.json")

if __name__ == "__main__":
    extract_master_weights()
