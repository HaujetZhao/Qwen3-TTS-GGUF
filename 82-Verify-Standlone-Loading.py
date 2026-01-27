"""
使用标准的 from_pretrained 方法加载独立的大师模型
验证模型加载和推理的正确性

纯音频词表方案 (Codec-Only Vocab):
- vocab_size = 3072
- lm_head 直接输出 codec token IDs (0-3071)
- 无需偏移量
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

    print(f"--- Verifying Standard Model Loading (Codec-Only Vocab) ---")
    print(f"Model path: {MODEL_PATH}")

    # 加载配置
    config = Qwen3TTSTalkerConfig.from_pretrained(MODEL_PATH)
    print(f"\nConfig:")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  _codec_only_vocab: {getattr(config, '_codec_only_vocab', False)}")

    # 方法1: 使用 from_pretrained (标准方法)
    print("\nUsing from_pretrained()...")
    model = Qwen3TTSTalkerModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    print("  Successfully loaded using from_pretrained()!")

    model.eval()

    # 加载 lm_head 从 safetensors (codec-only 方案)
    print("\nLoading lm_head from safetensors...")
    from safetensors import safe_open
    MODEL_WEIGHTS = os.path.join(MODEL_PATH, "model.safetensors")
    with safe_open(MODEL_WEIGHTS, framework="pt", device=device) as f:
        lm_head = f.get_tensor("lm_head")
    print(f"  lm_head shape: {lm_head.shape}")  # [2048, 3072]

    # 准备测试数据
    print("\nLoading test data...")
    inputs_embeds = torch.from_numpy(np.load("40_first_step_embeds.npy")).to(device).to(torch.bfloat16)
    expected_logits = torch.from_numpy(np.load("40_first_step_logits.npy")).to(device).to(torch.float32)

    print(f"Input shape: {inputs_embeds.shape}")

    # 推理 - 直接使用 lm_head
    print("\nRunning inference with lm_head...")
    with torch.no_grad():
        attention_mask = torch.ones(inputs_embeds.shape[:2], device=device)
        outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        next_hidden = last_hidden_state[:, -1, :]

        # 使用加载的 lm_head (codec-only: [2048, 3072])
        actual_logits = torch.matmul(next_hidden.to(torch.float32), lm_head.to(torch.float32))
        print(f"Logits shape: {actual_logits.shape}")  # [1, 3072]

    # 验证结果
    print("\n--- Results ---")
    actual_id = torch.argmax(actual_logits, dim=-1).item()
    expected_id = torch.argmax(expected_logits, dim=-1).item()

    print(f"Predicted Token ID: {actual_id} (codec vocab: 0-3071)")
    print(f"Expected Token ID  : {expected_id}")

    if actual_id == expected_id:
        print("\n[SUCCESS] Codec-only model works correctly!")
        print("  - Direct codec token prediction (no offset needed)")
        print(f"  - Predicted token: {actual_id}")
        return True
    else:
        print("\n[FAILURE] Prediction mismatch!")
        print(f"  Difference: {actual_id - expected_id}")
        return False

if __name__ == "__main__":
    try:
        success = verify_standard_loading()
        sys.exit(0 if success else 1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
