import os
import json
import torch
from safetensors.torch import save_file, load_file

def extract_craftsman_hf_simple():
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(PROJECT_ROOT, "Qwen3-TTS-12Hz-1.7B-CustomVoice", "model.safetensors")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "model", "craftsman_hf")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"--- 正在提取工匠权重 (Simple 版) ---")
    
    # 加载权重
    weights = load_file(MODEL_PATH)
    
    # 提取投影层
    proj_w = weights["talker.code_predictor.small_to_mtp_projection.weight"] # [1024, 2048]
    proj_b = weights["talker.code_predictor.small_to_mtp_projection.bias"]   # [1024]
    
    # 提取第 0 个 Embedding 表并执行预投影
    # 原本是 [2048, 2048], 投影后变成 [2048, 1024]
    emb_w = weights["talker.code_predictor.model.codec_embedding.0.weight"]
    print(f"执行 Embedding 预投影: {emb_w.shape[1]} -> {proj_w.shape[0]}")
    
    # 执行 Y = X @ W.T + b
    projected_emb = torch.nn.functional.linear(emb_w, proj_w, proj_b)
    
    new_weights = {
        "embed_tokens.weight": projected_emb,
        "lm_head.weight": weights["talker.code_predictor.lm_head.0.weight"],
        "norm.weight": weights["talker.code_predictor.model.norm.weight"]
    }
    
    # 提取 5 层 Backbone
    for i in range(5):
        prefix = f"talker.code_predictor.model.layers.{i}."
        for k, v in weights.items():
            if k.startswith(prefix):
                new_key = k.replace(prefix, f"layers.{i}.")
                new_weights[new_key] = v

    # 保存权重
    save_file(new_weights, os.path.join(OUTPUT_DIR, "model.safetensors"))
    
    # 构造配置 - 伪装成 Qwen3ForCausalLM
    config = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_hidden_layers": 5,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "rms_norm_eps": 1e-06,
        "vocab_size": 2048,
        "rope_theta": 1000000.0,
        "use_cache": True,
        "tie_word_embeddings": False,
        "hidden_act": "silu",
        "max_position_embeddings": 32768  # 关键修复: 对应 qwen3.context_length
    }
    
    with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
        
    print(f"✅ 工匠 HF 格式提取完成: {OUTPUT_DIR}")

if __name__ == "__main__":
    extract_craftsman_hf_simple()
