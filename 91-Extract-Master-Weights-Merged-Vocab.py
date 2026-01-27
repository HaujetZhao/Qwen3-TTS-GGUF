"""
从原始的 Qwen3-TTS 模型中提取"大师"(LLM Backbone)的权重
将大师权重保存为独立的 safetensors 文件

合并词表方案 (Merged Vocab):
- 使用 text_embedding [151936, 2048] 作为 embed_tokens
- lm_head = codec_head.T [2048, 3072] 补零到 [2048, 151936]
- 输入是 text token IDs (0-151935)
- 输出是 codec token IDs (0-3071)，对应 logits 的前 3072 个位置

优势:
- 保持完整的文本词表 (151936)
- 兼容标准的文本 tokenizer
- codec logits 在前 3072 个位置（非零）
- 后面的位置 (3072-151935) 都是零（不会预测这些 token）

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
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Standalone-Bare-Master-Merged-Vocab")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    OUTPUT_WEIGHTS = os.path.join(OUTPUT_DIR, "model.safetensors")
    OUTPUT_CONFIG = os.path.join(OUTPUT_DIR, "config.json")

    print(f"--- Extracting Master Weights (Merged Vocab) ---")
    print(f"Loading original weights from: {ORIGINAL_WEIGHTS}")

    # 1. 读取原始配置
    with open(ORIGINAL_CONFIG, "r", encoding="utf-8") as f:
        original_config = json.load(f)

    # 提取 talker_config 作为大师的配置
    master_config = original_config['talker_config'].copy()

    # 合并词表方案：使用 text_embedding，vocab_size = text_vocab_size
    master_config['vocab_size'] = 151936  # text_vocab_size
    master_config['_original_model_type'] = 'qwen3_tts_talker'
    master_config['_extracted_from'] = 'Qwen3-TTS-12Hz-1.7B-CustomVoice'
    master_config['_actual_model_type'] = 'qwen3_tts_talker'
    master_config['_merged_vocab'] = True  # 标记为合并词表
    master_config['_codec_vocab_size'] = 3072  # 记录 codec 词表大小

    # 伪装成 Qwen3-VL 架构（llama.cpp 已支持 IMRoPE）
    master_config['architectures'] = ['Qwen3VLForConditionalGeneration']
    master_config['model_type'] = 'qwen3_vl'

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

                # 特殊处理：使用 text_embedding (合并词表方案)
                if key == "talker.model.text_embedding.weight":
                    print(f"  Extracted text embedding: {key}, shape: {tensor.shape}")
                    # 保存为 embed_tokens
                    master_weights["embed_tokens"] = tensor
                    print(f"  -> Saved as embed_tokens [151936, 2048]")
                    continue

                # 跳过 codec_embedding (合并词表方案用 text_embedding)
                if key == "talker.model.codec_embedding.weight":
                    print(f"  Skipped codec embedding: {key}, shape: {tensor.shape} (using text_embedding instead)")
                    continue

                master_weights[new_key] = tensor
                print(f"  Extracted: {key} -> {new_key}, shape: {tensor.shape}")

            # Codec head 权重: talker.codec_head.weight
            elif key == "talker.codec_head.weight":
                codec_head_weight = f.get_tensor(key)
                print(f"  Extracted: {key}, shape: {codec_head_weight.shape}")

            # 跳过其他权重 (code_predictor, speaker_encoder 等)
            else:
                print(f"  Skipped: {key}")

    print(f"\nExtracted {len(master_weights)} master weights")

    # 3. 创建 lm_head (合并词表方案：codec_head.T 补零到 151936)
    if codec_head_weight is not None:
        print(f"\n--- Creating lm_head (Merged Vocab) ---")
        print(f"Codec head: {codec_head_weight.shape}")  # [3072, 2048]

        # lm_head 基础部分: codec_head.T [3072, 2048] -> [2048, 3072]
        codec_head_t = codec_head_weight.t().contiguous()
        print(f"codec_head.T shape: {codec_head_t.shape}")

        # 补零到 [2048, 151936]
        lm_head = torch.zeros(2048, 151936, dtype=codec_head_t.dtype)
        lm_head[:, :3072] = codec_head_t  # 前 3072 列是 codec_head.T
        print(f"lm_head shape: {lm_head.shape}")
        print(f"  - Columns [0:3072]: codec_head.T (non-zero)")
        print(f"  - Columns [3072:151936]: zeros (padding)")
        print(f"  - Vocab size: {lm_head.shape[1]} (merged vocab: text + codec)")

        # 保存为 lm_head (HF 标准命名)
        master_weights["lm_head"] = lm_head
        print(f"Added lm_head for merged vocab (codec logits at positions 0-3071)")
    else:
        print("WARNING: codec_head not found, model will not have lm_head!")

    # 4. 保存大师权重
    save_file(master_weights, OUTPUT_WEIGHTS)
    print(f"Saved master weights to: {OUTPUT_WEIGHTS}")

    # 5. 保存大师配置
    with open(OUTPUT_CONFIG, "w", encoding="utf-8") as f:
        json.dump(master_config, f, indent=2, ensure_ascii=False)
    print(f"Saved master config to: {OUTPUT_CONFIG}")

    # 6. 生成模型信息
    print(f"\n--- Creating Metadata ---")
    metadata = {
        "model_type": "Qwen3TTSTalkerModel",
        "architecture": "Master-only (LLM Backbone) with Merged Vocab",
        "source_model": "Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "extraction_date": "2026-01-27",
        "vocab_strategy": "merged_vocab",
        "components": {
            "master": "LLM Backbone (28 layers, 2048 hidden)",
            "embed_tokens": "Text embedding [151936, 2048] (text vocab)",
            "lm_head": "Codec head transpose padded [2048, 151936] (codec at 0-3071, zeros at 3072-151935)",
            "vocab_size": 151936,
            "codec_vocab_size": 3072,
            "note": "Input text token IDs (0-151935), output codec token IDs (0-3071) at logits[0:3072]"
        },
        "usage": {
            "gguf_conversion": "python convert_hf_to_gguf.py Standalone-Bare-Master-Merged-Vocab --outfile master-merged-vocab.gguf",
            "load_with": "from bare_master.modeling import Qwen3TTSTalkerModel",
            "config_class": "from bare_master.configuration import Qwen3TTSTalkerConfig",
            "llamacpp_compatible": "伪装成 Qwen3-VL，合并词表 (text vocab)，lm_head 前 3072 个位置有效",
            "inference": "Input text token IDs, output codec logits at positions 0-3071"
        }
    }

    metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Saved metadata to: {metadata_path}")

    print("\n✅ Extraction complete!")
    print(f"Master model saved to: {OUTPUT_DIR}")
    print(f"  - model.safetensors ({os.path.getsize(OUTPUT_WEIGHTS) / 1024**3:.2f} GB)")
    print(f"    └─ embed_tokens [151936, 2048] (text vocab)")
    print(f"    └─ lm_head [2048, 151936] (codec at 0-3071, zeros at 3072-151935)")
    print(f"    └─ 28 layers + norm")
    print(f"  - config.json (vocab_size=151936)")
    print(f"  - metadata.json")
    print(f"\n📊 Comparison with codec-only vocab:")
    print(f"  Codec-only: embed_tokens [3072, 2048] + lm_head [2048, 3072]")
    print(f"  Merged:    embed_tokens [151936, 2048] + lm_head [2048, 151936]")
    print(f"  Difference: Larger vocab, but compatible with text tokenizer")
    print(f"\n📝 Note:")
    print(f"  - Input: text token IDs (0-151935)")
    print(f"  - Output: codec logits at positions 0-3071 (non-zero), positions 3072-151935 (zeros)")
    print(f"  - Inference: Argmax over logits[0:3072] to get codec token IDs")
    print(f"Ready for GGUF conversion!")
    print(f"Run: python convert_hf_to_gguf.py {OUTPUT_DIR} --outfile master-merged-vocab.gguf")

if __name__ == "__main__":
    extract_master_weights()
