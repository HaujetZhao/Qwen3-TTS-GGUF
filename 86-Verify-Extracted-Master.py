"""
使用提取出来的大师模型文件进行推理验证
验证独立的大师模型能否正确推理并得到正确的结果 (token ID: 1995)
"""
import os
import sys
import torch
import numpy as np
import json
from safetensors import safe_open

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 使用独立的大师定义
from bare_master.configuration import Qwen3TTSTalkerConfig
from bare_master.modeling import Qwen3TTSTalkerModel

def verify_extracted_master():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 使用提取出来的大师模型
    EXTRACTED_MODEL_PATH = os.path.abspath("Standalone-Bare-Master")
    WEIGHTS_PATH = os.path.join(EXTRACTED_MODEL_PATH, "model.safetensors")
    CONFIG_PATH = os.path.join(EXTRACTED_MODEL_PATH, "config.json")
    CODEC_HEAD_PATH = os.path.join(EXTRACTED_MODEL_PATH, "codec_head.safetensors")

    print(f"--- Verifying Extracted Master Model ---")
    print(f"Loading config from: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        talker_config_dict = json.load(f)

    # 创建配置对象
    talker_config = Qwen3TTSTalkerConfig(**talker_config_dict)

    print(f"Initializing Master Model (Backbone)...")
    model = Qwen3TTSTalkerModel(talker_config).to(device).to(torch.bfloat16)

    print(f"Loading extracted master weights from: {WEIGHTS_PATH}")
    # 加载提取出来的大师权重 (已经没有前缀了)
    with safe_open(WEIGHTS_PATH, framework="pt", device=device) as f:
        state_dict = {}
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)

        # 加载到模型中
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")

    print(f"Loading codec head from: {CODEC_HEAD_PATH}")
    with safe_open(CODEC_HEAD_PATH, framework="pt", device=device) as f:
        codec_head_weight = f.get_tensor("weight")

    # 准备推理环境
    model.eval()

    print("Loading test data for validation...")
    inputs_embeds = torch.from_numpy(np.load("40_first_step_embeds.npy")).to(device).to(torch.bfloat16)
    expected_logits = torch.from_numpy(np.load("40_first_step_logits.npy")).to(device).to(torch.float32)

    print(f"Input shape: {inputs_embeds.shape}")

    with torch.no_grad():
        # 使用提取出来的大师模型推理
        attention_mask = torch.ones(inputs_embeds.shape[:2], device=device)

        outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # [B, T, Hidden]

        # 取最后一个 token 的隐层,应用 codec head
        next_hidden = last_hidden_state[:, -1, :] # [B, Hidden]

        # 模拟 Linear 层: weight x hidden
        actual_logits = torch.matmul(next_hidden.to(torch.float32), codec_head_weight.to(torch.float32).T)

    # 验证结果
    print("\n--- Results Analysis ---")
    slice_actual = actual_logits[0]
    slice_expected = expected_logits[0]

    diff = torch.abs(slice_actual - slice_expected)
    max_diff = torch.max(diff).item()

    print(f"Max Logit Difference: {max_diff:.6f}")

    actual_id = torch.argmax(actual_logits, dim=-1).item()
    expected_id = torch.argmax(expected_logits, dim=-1).item()

    print(f"Predicted Token ID (Extracted Master): {actual_id}")
    print(f"Expected Token ID                    : {expected_id}")

    if actual_id == expected_id:
        print("\n[SUCCESS] Extracted master model works correctly!")
        print("The master model has been successfully separated and can be used independently.")
        return True
    else:
        print("\n[FAILURE] Extracted master model produces different results!")
        return False

if __name__ == "__main__":
    try:
        success = verify_extracted_master()
        sys.exit(0 if success else 1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
