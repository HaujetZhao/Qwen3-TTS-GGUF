import os
import torch
import numpy as np
from safetensors.torch import load_file

def verify_lm_head_raw():
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    CAPTURED_DIR = os.path.join(PROJECT_ROOT, "captured_craftsman")
    MODEL_PATH = os.path.join(PROJECT_ROOT, "Qwen3-TTS-12Hz-1.7B-CustomVoice", "model.safetensors")
    
    # 1. 加载官方捕获的数据
    # step_0_output_hidden 是 Transformer Bone 的输出 (通常包含过最后一层 Norm)
    hidden_path = os.path.join(CAPTURED_DIR, "step_0_output_hidden.npy")
    official_id_path = os.path.join(CAPTURED_DIR, "step_0_output_ids.npy")
    
    if not os.path.exists(hidden_path) or not os.path.exists(official_id_path):
        print("Error: Captured data not found.")
        return
        
    hidden = torch.from_numpy(np.load(hidden_path)).float() # [1, 2, 1024]
    official_id = np.load(official_id_path).flatten()[0]
    
    print(f"Hidden Shape: {hidden.shape}, Official ID: {official_id}")
    
    # 2. 加载权重
    print("Loading weights...")
    weights = load_file(MODEL_PATH)
    lm_head_0 = weights["talker.code_predictor.lm_head.0.weight"].float() # [2048, 1024]
    
    # 3. 计算 Logits
    # 只需要最后一个位置的预测
    last_hidden = hidden[:, -1, :] # [1, 1024]
    logits = torch.matmul(last_hidden, lm_head_0.t()) # [1, 2048]
    
    pred_id = torch.argmax(logits, dim=-1).item()
    
    print(f"\n--- Python Raw Calculation Results ---")
    print(f"Predicted ID: {pred_id}")
    print(f"Official ID:  {official_id}")
    
    if pred_id == official_id:
        print("✅ Python 计算结果与官方捕获 ID 一致！")
    else:
        print("❌ Python 计算结果与官方捕获 ID 不一致！")
        print(f"Logit for Official ID ({official_id}): {logits[0, official_id].item()}")
        print(f"Logit for Predicted ID ({pred_id}): {logits[0, pred_id].item()}")

if __name__ == "__main__":
    verify_lm_head_raw()
