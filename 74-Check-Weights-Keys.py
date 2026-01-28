import os
from safetensors.torch import load_file

def check_keys():
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(PROJECT_ROOT, "Qwen3-TTS-12Hz-1.7B-CustomVoice", "model.safetensors")
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found")
        return
        
    print(f"正在读取模型权重: {MODEL_PATH}")
    weights = load_file(MODEL_PATH)
    
    # 定义搜索前缀
    search_terms = ["code_predictor.lm_head", "code_predictor.model.norm", "code_predictor.logit_scale"]
    
    print("\n--- 关键权重键名列表 ---")
    for k in sorted(weights.keys()):
        if any(term in k for term in search_terms):
            print(f"{k} -> Shape: {weights[k].shape}")

if __name__ == "__main__":
    check_keys()
