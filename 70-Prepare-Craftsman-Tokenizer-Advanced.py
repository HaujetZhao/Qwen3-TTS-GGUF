import os
import json

def create_craftsman_tokenizer_advanced():
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    TARGET_DIR = os.path.join(PROJECT_ROOT, "model", "craftsman_advanced_hf")
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    print(f"--- 正在构造高级工匠迷你分词器 (目标覆盖 30720 范围) ---")
    
    vocab = {}
    # BPE 基础
    vocab["a"] = 0
    vocab["b"] = 1
    vocab["ab"] = 2
    
    # 少量填充
    for i in range(3, 10):
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
            "merges": ["a b"]
        }
    }
    
    with open(os.path.join(TARGET_DIR, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump(tokenizer_json, f, ensure_ascii=False)
    
    with open(os.path.join(TARGET_DIR, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump({"tokenizer_class": "Qwen2Tokenizer"}, f)
        
    print(f"✅ 高级工匠假分词器构造完成: {TARGET_DIR}")

if __name__ == "__main__":
    create_craftsman_tokenizer_advanced()
