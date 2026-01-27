"""
构造适配 3072 词表的迷你 Tokenizer (Mini-Tokenizer)
以满足 80 系列模型转换 GGUF 的词表一致性要求。

由于推理时采用注入 Embedding 方式，Tokenizer 内容不影响结果，只需满足数量对齐。
"""
import os
import json
import shutil

def prepare_mini_tokenizer():
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    SOURCE_MODEL_DIR = os.path.join(PROJECT_ROOT, "Qwen3-TTS-12Hz-1.7B-CustomVoice")
    TARGET_DIR = os.path.join(PROJECT_ROOT, "Standalone-Bare-Master")
    
    print(f"--- Preparing Mini Tokenizer (3072 Vocab) ---")
    print(f"Source: {SOURCE_MODEL_DIR}")
    print(f"Target: {TARGET_DIR}")

    # 1. 处理 vocab.json，构造极限大小 (100) 的词表
    print("[1/4] Constructing mini_vocab (limit 100)...")
    vocab_path = os.path.join(SOURCE_MODEL_DIR, "vocab.json")
    with open(vocab_path, "r", encoding="utf-8") as f:
        full_vocab = json.load(f)
    
    # 彻底重构词表，只留前 100 个
    items = sorted(full_vocab.items(), key=lambda x: x[1])
    mini_vocab = {}
    
    for i in range(100):
        if i < len(items):
            k, v = items[i]
            mini_vocab[k] = i

    # 强制确保 'a'(97), 'b'(98), 'ab'(99) 存在，用于 merge
    # BPE 要求：如果规则是 a b -> ab，那么 a, b, ab 都必须在词表里
    mini_vocab["a"] = 97
    mini_vocab["b"] = 98
    mini_vocab["ab"] = 99
    
    # 映射特殊 token 到 0
    special_tokens_to_map = [
        "<|endoftext|>", "<|im_start|>", "<|im_end|>", 
        "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>", 
        "<|image_pad|>", "<|video_pad|>"
    ]
    for tok in special_tokens_to_map:
        mini_vocab[tok] = 0

    with open(os.path.join(TARGET_DIR, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(mini_vocab, f, ensure_ascii=False)
    print(f"  ✓ Saved vocab.json (size: {len(mini_vocab)})")

    # 2. 处理 merges.txt
    print(f"[2/4] Creating dummy merges.txt using 'a' and 'b' -> 'ab'...")
    dummy_merge = "a b"
    with open(os.path.join(TARGET_DIR, "merges.txt"), "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
        f.write(f"{dummy_merge}\n")
    print(f"  ✓ Saved merges.txt")

    # 3. 处理 tokenizer_config.json
    print("[3/4] Customizing tokenizer_config.json...")
    config_path = os.path.join(SOURCE_MODEL_DIR, "tokenizer_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        t_config = json.load(f)
    
    t_config["added_tokens_decoder"] = {
        "0": {
            "content": "<|endoftext|>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "special": True
        }
    }
    keys_to_remove = [k for k, v in t_config.items() if isinstance(v, dict) and "content" in v]
    for k in keys_to_remove:
        del t_config[k]
        
    t_config["bos_token"] = "<|endoftext|>"
    t_config["eos_token"] = "<|endoftext|>"
    t_config["pad_token"] = "<|endoftext|>"
    t_config["unk_token"] = "<|endoftext|>"
    t_config["tokenizer_class"] = "Qwen2Tokenizer"
    t_config["model_type"] = "qwen2"
    
    with open(os.path.join(TARGET_DIR, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(t_config, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved tokenizer_config.json")

    # 4. 重新生成 tokenizer.json
    print("[4/4] Creating minimal tokenizer.json...")
    mini_tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {"id": 0, "content": "<|endoftext|>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True}
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
        "post_processor": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": False, "use_regex": True},
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {
            "type": "BPE",
            "vocab": mini_vocab,
            "merges": [dummy_merge]
        }
    }
    with open(os.path.join(TARGET_DIR, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump(mini_tokenizer_json, f, ensure_ascii=False)
    print(f"  ✓ Saved tokenizer.json")

    print("\n✅ Mini Tokenizer preparation complete!")
    print(f"Files ready in: {TARGET_DIR}")

if __name__ == "__main__":
    prepare_mini_tokenizer()
