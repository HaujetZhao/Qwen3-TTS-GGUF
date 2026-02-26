"""
从原始的 Qwen3-TTS 模型中提取 "Talker" (LLM Backbone) 的权重
将 Talker 权重保存为独立的 safetensors 文件
"""
import os
import json
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from export_config import MODEL_DIR, EXPORT_DIR

def extract_talker_weights():
    """提取 Talker 权重到独立文件"""

    # 路径配置
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    ORIGINAL_MODEL_PATH = MODEL_DIR
    ORIGINAL_WEIGHTS = os.path.join(ORIGINAL_MODEL_PATH, "model.safetensors")
    ORIGINAL_CONFIG = os.path.join(ORIGINAL_MODEL_PATH, "config.json")

    # 输出路径
    OUTPUT_DIR = os.path.join(EXPORT_DIR, "hf")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    OUTPUT_WEIGHTS = os.path.join(OUTPUT_DIR, "model.safetensors")
    OUTPUT_CONFIG = os.path.join(OUTPUT_DIR, "config.json")

    print(f"--- 正在提取 Talker 模型权重 ---")
    print(f"正在从以下路径加载原始权重: {ORIGINAL_WEIGHTS}")

    # 1. 读取原始配置
    with open(ORIGINAL_CONFIG, "r", encoding="utf-8") as f:
        original_config = json.load(f)

    # 提取 talker_config 作为 Talker 的配置
    talker_config = original_config['talker_config'].copy()

    # 纯音频词表方案：只用 codec_embedding
    talker_config['vocab_size'] = 3072  # codec_vocab_size
    talker_config['_original_model_type'] = 'qwen3_tts_talker'
    talker_config['_extracted_from'] = 'Qwen3-TTS-12Hz-1.7B-CustomVoice'
    talker_config['_actual_model_type'] = 'qwen3_tts_talker'
    talker_config['_codec_only_vocab'] = True  # 标记为纯音频词表

    # 伪装成 Qwen3-VL 架构
    talker_config['architectures'] = ['Qwen3VLForConditionalGeneration']
    talker_config['model_type'] = 'qwen3_vl'

    # 2. 提取 Talker 权重 (talker.model.* -> 无前缀)
    talker_weights = {}
    codec_head_weight = None

    with safe_open(ORIGINAL_WEIGHTS, framework="pt", device="cpu") as f:
        total_keys = len(list(f.keys()))
        print(f"原始模型中的总权重键数: {total_keys}")

        for key in f.keys():
            # Talker backbone 权重: talker.model.*
            if key.startswith("talker.model."):
                new_key = key.replace("talker.model.", "")
                tensor = f.get_tensor(key).float() # 强制转成 float32

                if key == "talker.model.codec_embedding.weight":
                    print(f"  已提取 codec embedding: {key}, 形状: {tensor.shape}")
                    talker_weights["embed_tokens.weight"] = tensor
                    continue

                if key == "talker.model.text_embedding.weight":
                    continue

                talker_weights[new_key] = tensor
                print(f"  已提取: {key} -> {new_key}, 形状: {tensor.shape}")

            elif key == "talker.codec_head.weight":
                codec_head_weight = f.get_tensor(key).float()
                print(f"  已提取: {key}, 形状: {codec_head_weight.shape}")

            else:
                print(f"  已跳过: {key}")

    print(f"\n成功提取 {len(talker_weights)} 个 Talker 模型权重")

    # 3. 创建 lm_head
    if codec_head_weight is not None:
        print(f"\n--- 正在创建 lm_head ---")
        lm_head = codec_head_weight.contiguous()
        talker_weights["lm_head.weight"] = lm_head
        print(f"已添加 lm_head.weight")

    # 4. 保存 Talker 权重
    save_file(talker_weights, OUTPUT_WEIGHTS)
    print(f"Talker 模型权重已保存至: {OUTPUT_WEIGHTS}")

    # 5. 保存 Talker 配置
    with open(OUTPUT_CONFIG, "w", encoding="utf-8") as f:
        json.dump(talker_config, f, indent=2, ensure_ascii=False)
    print(f"Talker 模型配置已保存至: {OUTPUT_CONFIG}")

    # 6. 生成模型信息
    metadata = {
        "model_type": "Qwen3TTSTalkerModel",
        "architecture": "Talker-only (LLM Backbone)",
        "source_model": "Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "extraction_date": "2026-01-28",
        "vocab_strategy": "codec_only",
        "components": {
            "talker": "LLM Backbone (28 layers, 2048 hidden)",
            "vocab_size": 3072,
        }
    }

    metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\n✅ 提取完成!")
    print(f"Talker 模型已保存至: {OUTPUT_DIR}")

if __name__ == "__main__":
    extract_talker_weights()
