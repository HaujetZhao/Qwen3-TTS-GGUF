"""
验证合并后的 lm_head 是否正确工作
检查使用 lm_head 推理时，token ID 是否有正确的偏移（151936）
"""
import os
import sys
import torch
import numpy as np

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from bare_master.configuration import Qwen3TTSTalkerConfig
from bare_master.modeling import Qwen3TTSTalkerModel
from safetensors import safe_open

def verify_lm_head():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = os.path.abspath("Standalone-Bare-Master")

    print(f"--- Verifying Merged lm_head ---")
    print(f"Model path: {MODEL_PATH}")

    # 1. 加载模型
    print("\n[1] Loading model...")
    model = Qwen3TTSTalkerModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    model.eval()
    print("  [OK] Model loaded")

    # 2. 加载 lm_head
    print("\n[2] Loading lm_head from model.safetensors...")
    MODEL_WEIGHTS = os.path.join(MODEL_PATH, "model.safetensors")
    with safe_open(MODEL_WEIGHTS, framework="pt", device=device) as f:
        lm_head = f.get_tensor("lm_head")
    print(f"  lm_head shape: {lm_head.shape}")
    print(f"  Expected: [2048, 155008]")

    # 3. 验证 lm_head 结构
    print("\n[3] Verifying lm_head structure...")
    text_part = lm_head[:, :151936]
    codec_part = lm_head[:, 151936:]

    print(f"  Text part [0:151936]:")
    print(f"    - Shape: {text_part.shape}")
    print(f"    - Min: {text_part.min().item():.6f}")
    print(f"    - Max: {text_part.max().item():.6f}")
    print(f"    - Mean: {text_part.mean().item():.6f}")
    print(f"    - Is all zeros: {torch.all(text_part == 0).item()}")

    print(f"  Codec part [151936:155008]:")
    print(f"    - Shape: {codec_part.shape}")
    print(f"    - Min: {codec_part.min().item():.6f}")
    print(f"    - Max: {codec_part.max().item():.6f}")
    print(f"    - Mean: {codec_part.mean().item():.6f}")
    print(f"    - Is all zeros: {torch.all(codec_part == 0).item()}")

    # 4. 加载测试数据
    print("\n[4] Loading test data...")
    inputs_embeds = torch.from_numpy(np.load("40_first_step_embeds.npy")).to(device).to(torch.bfloat16)
    expected_logits = torch.from_numpy(np.load("40_first_step_logits.npy")).to(device).to(torch.float32)
    print(f"  Input shape: {inputs_embeds.shape}")

    # 5. 使用 lm_head 推理
    print("\n[5] Running inference with lm_head...")
    with torch.no_grad():
        attention_mask = torch.ones(inputs_embeds.shape[:2], device=device)
        outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        next_hidden = last_hidden_state[:, -1, :]

        # 使用完整的 lm_head
        actual_logits = torch.matmul(next_hidden.to(torch.float32), lm_head)
        # next_hidden: [1, 2048]
        # lm_head: [2048, 155008]
        # actual_logits: [1, 155008]

    # 6. 分析结果
    print("\n[6] Analyzing results...")
    print(f"  Logits shape: {actual_logits.shape}")
    print(f"  Logits range: [{actual_logits.min().item():.2f}, {actual_logits.max().item():.2f}]")

    # 检查 logits 分布
    logits_text_part = actual_logits[0, :151936]
    logits_codec_part = actual_logits[0, 151936:]

    print(f"\n  Text logits [0:151936]:")
    print(f"    - Min: {logits_text_part.min().item():.2f}")
    print(f"    - Max: {logits_text_part.max().item():.2f}")
    print(f"    - Mean: {logits_text_part.mean().item():.2f}")

    print(f"\n  Codec logits [151936:155008]:")
    print(f"    - Min: {logits_codec_part.min().item():.2f}")
    print(f"    - Max: {logits_codec_part.max().item():.2f}")
    print(f"    - Mean: {logits_codec_part.mean().item():.2f}")

    # 7. 获取预测的 token
    actual_id_full = torch.argmax(actual_logits, dim=-1).item()
    actual_id_codec = torch.argmax(actual_logits[0, 151936:], dim=-1).item() + 151936

    print(f"\n[7] Prediction results:")
    print(f"  Full vocab argmax (0-155007): {actual_id_full}")
    print(f"  Codec vocab argmax (151936-155007): {actual_id_codec}")
    print(f"  Expected (from original codec_head): {torch.argmax(expected_logits, dim=-1).item()}")

    # 8. 验证
    expected_id = torch.argmax(expected_logits, dim=-1).item()
    expected_id_with_offset = expected_id + 151936

    print(f"\n[8] Verification:")
    print(f"  Expected token ID (from original): {expected_id}")
    print(f"  Expected token ID (with offset): {expected_id_with_offset}")
    print(f"  Actual token ID (full lm_head): {actual_id_full}")

    if actual_id_full == expected_id_with_offset:
        print("\n  [SUCCESS] lm_head correctly offsets codec tokens!")
        print(f"  The model predicts token {actual_id_full} = {expected_id} (original) + 151936 (offset)")
        return True
    else:
        print("\n  [FAILURE] Token ID mismatch!")
        print(f"  Expected: {expected_id_with_offset}")
        print(f"  Got: {actual_id_full}")
        return False

if __name__ == "__main__":
    try:
        success = verify_lm_head()
        sys.exit(0 if success else 1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
