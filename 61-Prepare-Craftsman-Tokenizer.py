import os
import json

def create_craftsman_tokenizer():
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    TARGET_DIR = os.path.join(PROJECT_ROOT, "model", "craftsman_hf")
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    print(f"--- 正在构造工匠迷你分词器 (2048 词表) ---")
    
    # 关键修改：只生成极少量的 Token (例如 10 个)，最大 ID 远小于 2048
    # 这样 convert_hf_to_gguf 会自动填充剩余的 [PAD]
    vocab = {}
    for i in range(10):
        vocab[f"<t_{i}>"] = i
        
    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
        "post_processor": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": False, "use_regex": True},
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {
            "type": "BPE",
            "vocab": vocab,
            "merges": []
        }
    }
    
    with open(os.path.join(TARGET_DIR, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump(tokenizer_json, f, ensure_ascii=False)
    
    # 还需要个 config 文件来骗过 gguf 转换器
    with open(os.path.join(TARGET_DIR, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump({"tokenizer_class": "Qwen2Tokenizer"}, f)
        
    print(f"✅ 工匠假分词器构造完成: {TARGET_DIR}")

if __name__ == "__main__":
    create_craftsman_tokenizer()
