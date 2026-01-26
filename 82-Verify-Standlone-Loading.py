"""
使用标准的 from_pretrained 方法加载独立的大师模型
验证模型加载和推理的正确性
"""
import os
import sys
import torch
import numpy as np

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 使用独立的大师定义
from bare_master.configuration import Qwen3TTSTalkerConfig
from bare_master.modeling import Qwen3TTSTalkerModel

def verify_standard_loading():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 使用提取出来的大师模型路径
    MODEL_PATH = os.path.abspath("Standalone-Bare-Master")

    print(f"--- Verifying Standard Model Loading ---")
    print(f"Model path: {MODEL_PATH}")

    # 方法1: 使用 from_pretrained (标准方法)
    print("\n[Method 1] Using from_pretrained()...")
    try:
        model = Qwen3TTSTalkerModel.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        print("  Successfully loaded using from_pretrained()!")
    except Exception as e:
        print(f"  Error loading with from_pretrained(): {e}")
        print("\n[Fallback] Trying manual config loading...")
        # 方法2: 手动加载配置和模型
        config = Qwen3TTSTalkerConfig.from_pretrained(MODEL_PATH)
        model = Qwen3TTSTalkerModel(config).to(device).to(torch.bfloat16)
        # 手动加载权重
        model.load_state_dict(torch.load(os.path.join(MODEL_PATH, "model.safetensors"), weights_only=False))
        print("  Successfully loaded using manual method!")

    model.eval()

    # 加载 codec head
    from safetensors import safe_open
    CODEC_HEAD_PATH = os.path.join(MODEL_PATH, "codec_head.safetensors")
    with safe_open(CODEC_HEAD_PATH, framework="pt", device=device) as f:
        codec_head_weight = f.get_tensor("weight")

    # 准备测试数据
    print("\nLoading test data...")
    inputs_embeds = torch.from_numpy(np.load("40_first_step_embeds.npy")).to(device).to(torch.bfloat16)
    expected_logits = torch.from_numpy(np.load("40_first_step_logits.npy")).to(device).to(torch.float32)

    print(f"Input shape: {inputs_embeds.shape}")

    # 推理
    with torch.no_grad():
        attention_mask = torch.ones(inputs_embeds.shape[:2], device=device)
        outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        next_hidden = last_hidden_state[:, -1, :]
        actual_logits = torch.matmul(next_hidden.to(torch.float32), codec_head_weight.to(torch.float32).T)

    # 验证结果
    print("\n--- Results ---")
    actual_id = torch.argmax(actual_logits, dim=-1).item()
    expected_id = torch.argmax(expected_logits, dim=-1).item()

    print(f"Predicted Token ID: {actual_id}")
    print(f"Expected Token ID  : {expected_id}")

    if actual_id == expected_id:
        print("\n[SUCCESS] Standard loading works correctly!")
        print("The master model can be loaded using standard HuggingFace methods.")
        return True
    else:
        print("\n[FAILURE] Prediction mismatch!")
        return False

if __name__ == "__main__":
    try:
        success = verify_standard_loading()
        sys.exit(0 if success else 1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
