"""
从原始的 Qwen3-TTS 模型中提取"大师"(LLM Backbone)的权重
将大师权重保存为独立的 safetensors 文件
合并 text_embedding 和 codec_embedding 为单一表（llama.cpp 兼容）

text_embedding: [151936, 2048]  # 1.18 GB
codec_embedding: [3072, 2048]   # 24 MB

Text tokens:      0 ~ 151,935   (原 text_embedding)
Codec tokens:     151,936 ~ 155,007 (原 codec_embedding + offset)


"""
import os
import json
import torch
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
    master_config['_actual_model_type'] = 'qwen3_tts_talker'  # 保留真实类型

    # 伪装成 Qwen3-VL 架构（llama.cpp 已支持，无需修改）
    # 原因：Qwen3TTS 使用 IMRoPE (mrope_section: [24, 20, 20])
    #       llama.cpp 中 QWEN3VL 使用 LLAMA_ROPE_TYPE_IMROPE
    #       而 QWEN3 使用标准 RoPE，不兼容！
    master_config['architectures'] = ['Qwen3VLForConditionalGeneration']
    master_config['model_type'] = 'qwen3_vl'

    # 2. 提取大师权重 (talker.model.* -> 无前缀)
    master_weights = {}
    codec_head_weight = None
    text_emb_weight = None
    codec_emb_weight = None

    with safe_open(ORIGINAL_WEIGHTS, framework="pt", device="cpu") as f:
        total_keys = len(list(f.keys()))
        print(f"Total keys in original model: {total_keys}")

        for key in f.keys():
            # 大师 backbone 权重: talker.model.*
            if key.startswith("talker.model."):
                # 移除前缀,变成无前缀的格式
                new_key = key.replace("talker.model.", "")
                tensor = f.get_tensor(key)

                # 特殊处理：提取两个 embedding 表
                if key == "talker.model.embed_tokens":
                    text_emb_weight = tensor
                    print(f"  Extracted text embedding: {key}, shape: {tensor.shape}")
                    # 不直接添加，稍后合并
                    continue
                elif key == "talker.model.codec_embedding":
                    codec_emb_weight = tensor
                    print(f"  Extracted codec embedding: {key}, shape: {tensor.shape}")
                    # 不直接添加，稍后合并
                    continue

                master_weights[new_key] = tensor
                print(f"  Extracted: {key} -> {new_key}, shape: {tensor.shape}")

            # Codec head 权重: talker.codec_head.weight
            elif key == "talker.codec_head.weight":
                codec_head_weight = f.get_tensor(key)
                print(f"  Separated: {key}, shape: {codec_head_weight.shape}")

            # 跳过其他权重 (code_predictor, speaker_encoder 等)
            else:
                print(f"  Skipped: {key}")

    # 3. 合并两个 embedding 表为一个
    if text_emb_weight is not None and codec_emb_weight is not None:
        print(f"\n--- Merging Embedding Tables ---")
        print(f"Text embedding shape: {text_emb_weight.shape}")  # [151936, 2048]
        print(f"Codec embedding shape: {codec_emb_weight.shape}")  # [3072, 2048]

        # 拼接：text embedding 在前，codec embedding 在后
        # 合并后形状：[151936 + 3072, 2048] = [155008, 2048]
        merged_embedding = torch.cat([text_emb_weight, codec_emb_weight], dim=0)
        print(f"Merged embedding shape: {merged_embedding.shape}")

        # 更新配置中的 vocab_size
        master_config["vocab_size"] = merged_embedding.shape[0]
        master_config["text_vocab_size"] = text_emb_weight.shape[0]
        master_config["codec_vocab_size"] = codec_emb_weight.shape[0]
        print(f"Updated vocab_size in config: {merged_embedding.shape[0]}")

        # 保存为 embed_tokens (llama.cpp 标准命名)
        master_weights["embed_tokens"] = merged_embedding
        print(f"Added merged embedding as 'embed_tokens'")

        # 保存原始 embedding 到单独文件（用于验证）
        emb_output_path = os.path.join(OUTPUT_DIR, "original_embeddings.safetensors")
        save_file({
            "text_embedding": text_emb_weight,
            "codec_embedding": codec_emb_weight
        }, emb_output_path)
        print(f"Saved original embeddings to: {emb_output_path}")
    else:
        print("WARNING: Failed to extract embedding tables!")
        if text_emb_weight is None:
            print("  - text_embedding not found")
        if codec_emb_weight is None:
            print("  - codec_embedding not found")

    print(f"\nExtracted {len(master_weights)} master weights")

    # 3.5. 创建完整的 lm_head (用于 GGUF benchmark)
    if text_emb_weight is not None and codec_head_weight is not None:
        print(f"\n--- Creating Complete lm_head for GGUF ---")
        print(f"Text embedding: {text_emb_weight.shape}")  # [151936, 2048]
        print(f"Codec head: {codec_head_weight.shape}")     # [3072, 2048]

        # 创建完整的 lm_head: [2048, 155008]
        # Text 部分 (0-151935): 全为零（text tokens 不会在 TTS 中生成）
        # Codec 部分 (151936-155007): 使用 codec_head 的转置
        lm_head_text = torch.zeros(2048, text_emb_weight.shape[0])  # [2048, 151936] 全0
        lm_head_codec = codec_head_weight.t()  # [2048, 3072] 真实值

        # 合并：[2048, 151936] + [2048, 3072] = [2048, 155008]
        full_lm_head = torch.cat([lm_head_text, lm_head_codec], dim=1)
        print(f"Full lm_head shape: {full_lm_head.shape}")
        print(f"  - Text part [0:151936]: ZEROS (will never be sampled)")
        print(f"  - Codec part [151936:155008]: ACTIVE (codec tokens only)")

        # 保存为 lm_head (HF 标准命名)
        master_weights["lm_head"] = full_lm_head
        print(f"Added full lm_head for GGUF compatibility (codec-only)")

        # 同时保存原始 codec_head (用于验证和手动使用)
        codec_head_path = os.path.join(OUTPUT_DIR, "codec_head.safetensors")
        save_file({"weight": codec_head_weight}, codec_head_path)
        print(f"Saved original codec_head to: {codec_head_path}")
    elif codec_head_weight is not None:
        # 只有 codec_head，没有 text_emb 的情况
        print(f"\nWARNING: Creating lm_head from codec_head only")
        lm_head_codec = codec_head_weight.t()  # [2048, 3072]
        # 用零填充 text 部分
        lm_head_text = torch.zeros(2048, text_emb_weight.shape[0] if text_emb_weight is not None else 151936)
        full_lm_head = torch.cat([lm_head_text, lm_head_codec], dim=1)
        master_weights["lm_head"] = full_lm_head
        print(f"Added partial lm_head: {full_lm_head.shape}")

        codec_head_path = os.path.join(OUTPUT_DIR, "codec_head.safetensors")
        save_file({"weight": codec_head_weight}, codec_head_path)
        print(f"Saved original codec_head to: {codec_head_path}")

    # 4. 保存大师权重
    save_file(master_weights, OUTPUT_WEIGHTS)
    print(f"Saved master weights to: {OUTPUT_WEIGHTS}")

    # 5. 保存大师配置
    with open(OUTPUT_CONFIG, "w", encoding="utf-8") as f:
        json.dump(master_config, f, indent=2, ensure_ascii=False)
    print(f"Saved master config to: {OUTPUT_CONFIG}")

    # 6. 复制 tokenizer 文件 (GGUF 必需)
    print(f"\n--- Copying Tokenizer Files ---")
    tokenizer_files = [
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json"
    ]

    copied_count = 0
    for fname in tokenizer_files:
        src_path = os.path.join(ORIGINAL_MODEL_PATH, fname)
        dst_path = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(src_path):
            import shutil
            shutil.copy2(src_path, dst_path)
            size_mb = os.path.getsize(dst_path) / 1024**2
            print(f"  Copied: {fname} ({size_mb:.2f} MB)")
            copied_count += 1
        else:
            print(f"  Skipped (not found): {fname}")

    if copied_count == 0:
        print("WARNING: No tokenizer files found!")
        print("  GGUF conversion may fail without tokenizer files.")

    # 7. 生成模型信息
    metadata = {
        "model_type": "Qwen3TTSTalkerModel",
        "architecture": "Master-only (LLM Backbone) with Merged Embeddings + Full lm_head",
        "source_model": "Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "extraction_date": "2026-01-26",
        "components": {
            "master": "LLM Backbone (28 layers, 2048 hidden)",
            "merged_embedding": "Text (151936) + Codec (3072) = 155008 vocab size",
            "lm_head": "Codec-only [2048, 155008] - Text part is ZEROS, only codec tokens [151936:155008] are active",
            "codec_head_original": "Saved to codec_head.safetensors (for verification)",
            "original_embeddings": "Saved to original_embeddings.safetensors (for verification)",
            "tokenizer": "Copied from original model (required for GGUF)"
        },
        "usage": {
            "gguf_conversion": "python convert_hf_to_gguf.py Standalone-Bare-Master --outfile master.gguf",
            "load_with": "from bare_master.modeling import Qwen3TTSTalkerModel",
            "config_class": "from bare_master.configuration import Qwen3TTSTalkerConfig",
            "llamacpp_compatible": "伪装成 Qwen3-VL，包含完整 lm_head，支持 benchmark",
            "note": "Text tokens: 0-151935, Codec tokens: 151936-155007"
        }
    }

    metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Saved metadata to: {metadata_path}")

    print("\n✅ Extraction complete!")
    print(f"Master model saved to: {OUTPUT_DIR}")
    print(f"  - model.safetensors ({os.path.getsize(OUTPUT_WEIGHTS) / 1024**3:.2f} GB)")
    print(f"    └─ embed_tokens [155008, 2048] (merged)")
    print(f"    └─ lm_head [2048, 155008] (complete, for GGUF)")
    print(f"    └─ 28 layers + norm")
    print(f"  - config.json")
    print(f"  - tokenizer files (for GGUF)")
    print(f"  - codec_head.safetensors (original, for manual use)")
    print(f"  - original_embeddings.safetensors (for verification)")
    print(f"  - metadata.json")
    print(f"\nReady for GGUF conversion!")
    print(f"Run: python convert_hf_to_gguf.py {OUTPUT_DIR} --outfile master-qwen3tts.gguf")

if __name__ == "__main__":
    extract_master_weights()
