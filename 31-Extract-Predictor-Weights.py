import os
import json
import torch
import numpy as np
from safetensors.torch import save_file, load_file

from export_config import MODEL_DIR, EXPORT_DIR

def extract_predictor_hf():
    MODEL_PATH = os.path.join(MODEL_DIR, "model.safetensors")
    OUTPUT_DIR = os.path.join(EXPORT_DIR, "predictor_hf")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"--- 正在提取 Predictor 权重 ---")
    
    weights = load_file(MODEL_PATH)
    weights = {k: v for k, v in weights.items()}
    
    # 1. 准备投影层
    proj_w = weights["talker.code_predictor.small_to_mtp_projection.weight"] 
    proj_b = weights["talker.code_predictor.small_to_mtp_projection.bias"]   
    
    # 2. 处理 Embedding
    emb_list = []
    for i in range(15):
        key = f"talker.code_predictor.model.codec_embedding.{i}.weight"
        raw_emb = weights[key] 
        proj_emb = torch.nn.functional.linear(raw_emb, proj_w, proj_b) 
        emb_list.append(proj_emb)
        print(f"Embedding {i} processed.")
        
    cat_emb = torch.cat(emb_list, dim=0) 
    print(f"拼接后的 Embedding 形状: {cat_emb.shape}")
    
    # 3. 处理 LM Head
    head_list = []
    for i in range(15):
        key = f"talker.code_predictor.lm_head.{i}.weight"
        raw_head = weights[key] 
        head_list.append(raw_head)
        print(f"LM Head {i} processed.")
    
    cat_head = torch.cat(head_list, dim=0) 
    print(f"拼接后的 LM Head 形状: {cat_head.shape}")
    
    # 4. 组装权重字典
    new_weights = {
        "embed_tokens.weight": cat_emb,
        "lm_head.weight": cat_head,
        "norm.weight": weights["talker.code_predictor.model.norm.weight"]
    }
    
    # 提取 Backbone
    for i in range(5):
        prefix = f"talker.code_predictor.model.layers.{i}."
        for k, v in weights.items():
            if k.startswith(prefix):
                new_key = k.replace(prefix, f"layers.{i}.")
                new_weights[new_key] = v

    # 保存
    save_file(new_weights, os.path.join(OUTPUT_DIR, "model.safetensors"))
    
    # 5. 构造 Config
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
        "vocab_size": 30720, 
        "rope_theta": 1000000.0,
        "use_cache": True,
        "tie_word_embeddings": False,
        "hidden_act": "silu",
        "max_position_embeddings": 32768
    }
    
    with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
        
    # 6. 为推理提供 Numpy 资产
    np.save(os.path.join(EXPORT_DIR, 'embeddings', "proj_weight.npy"), proj_w.float().numpy())
    np.save(os.path.join(EXPORT_DIR, 'embeddings', "proj_bias.npy"), proj_b.float().numpy())
    print(f"📊 额外导出 Numpy 投影层资产至: {EXPORT_DIR}")
        
    print(f"✅ Predictor HF 格式提取完成: {OUTPUT_DIR}")

if __name__ == "__main__":
    extract_predictor_hf()
