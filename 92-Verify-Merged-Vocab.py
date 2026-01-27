"""
使用标准的 from_pretrained 方法加载独立的大师模型（合并词表方案）
验证模型加载和推理的正确性

合并词表方案 (Merged Vocab):
- vocab_size = 151936 (text vocab)
- embed_tokens = text_embedding [151936, 2048]
- lm_head = codec_head.T [2048, 3072] padded to [2048, 151936]
- logits[0:3072] 是 codec logits（非零）
- logits[3072:151936] 都是零
- 只需要在前 3072 个位置上做 argmax
"""
import os
import sys
import torch
import numpy as np

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSTalkerConfig
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerModel

def verify_merged_vocab_loading():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 使用提取出来的大师模型路径
    MODEL_PATH = os.path.abspath("Standalone-Bare-Master-Merged-Vocab")

    print(f"--- Verifying Standard Model Loading (Merged Vocab) ---")
    print(f"Model path: {MODEL_PATH}")

    # 加载配置
    config = Qwen3TTSTalkerConfig.from_pretrained(MODEL_PATH)
    print(f"\nConfig:")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  _merged_vocab: {getattr(config, '_merged_vocab', False)}")
    print(f"  _codec_vocab_size: {getattr(config, '_codec_vocab_size', None)}")

    # 方法1: 使用 from_pretrained (标准方法)
    print("\nUsing from_pretrained()...")
    model = Qwen3TTSTalkerModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    print("  Successfully loaded using from_pretrained()!")

    model.eval()

    # 加载 lm_head 从 safetensors (merged vocab 方案)
    print("\nLoading lm_head from safetensors...")
    from safetensors import safe_open
    MODEL_WEIGHTS = os.path.join(MODEL_PATH, "model.safetensors")
    with safe_open(MODEL_WEIGHTS, framework="pt", device=device) as f:
        lm_head = f.get_tensor("lm_head")
    print(f"  lm_head shape: {lm_head.shape}")  # [2048, 151936]

    # 检查 lm_head 的结构
    print(f"\n--- Analyzing lm_head Structure ---")
    # 前 3072 列应该非零
    codec_logits_part = lm_head[:, :3072]
    padding_part = lm_head[:, 3072:]
    print(f"  Codec logits part [0:3072]:")
    print(f"    - Shape: {codec_logits_part.shape}")
    print(f"    - Non-zero elements: {torch.count_nonzero(codec_logits_part).item()}")
    print(f"    - Mean abs value: {torch.abs(codec_logits_part).mean().item():.6f}")
    print(f"  Padding part [3072:151936]:")
    print(f"    - Shape: {padding_part.shape}")
    print(f"    - Non-zero elements: {torch.count_nonzero(padding_part).item()}")
    print(f"    - Mean abs value: {torch.abs(padding_part).mean().item():.6f}")

    # 准备测试数据
    print("\nLoading test data...")
    inputs_embeds = torch.from_numpy(np.load("40_first_step_embeds.npy")).to(device).to(torch.bfloat16)
    expected_logits = torch.from_numpy(np.load("40_first_step_logits.npy")).to(device).to(torch.float32)

    print(f"Input shape: {inputs_embeds.shape}")

    # 推理 - 使用 lm_head（只在前 3072 个位置计算）
    print("\nRunning inference with lm_head (merged vocab)...")
    with torch.no_grad():
        attention_mask = torch.ones(inputs_embeds.shape[:2], device=device)
        outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        next_hidden = last_hidden_state[:, -1, :]

        # 使用加载的 lm_head: [2048, 151936]
        # 但只需要前 3072 列（codec logits）
        actual_logits_full = torch.matmul(next_hidden.to(torch.float32), lm_head.to(torch.float32))
        actual_logits = actual_logits_full[:, :3072]  # 只取前 3072 个
        print(f"Full logits shape: {actual_logits_full.shape}")  # [1, 151936]
        print(f"Codec logits shape: {actual_logits.shape}")  # [1, 3072]

    # 验证结果
    print("\n--- Results ---")
    actual_id = torch.argmax(actual_logits, dim=-1).item()
    expected_id = torch.argmax(expected_logits, dim=-1).item()

    print(f"Predicted Token ID: {actual_id} (codec vocab: 0-3071)")
    print(f"Expected Token ID  : {expected_id}")

    # 检查 logits 位置 3072 之后是否都是零
    print(f"\n--- Checking Logit Distribution ---")
    full_logits_for_last_token = actual_logits_full[0]  # [151936]
    max_logit_value, max_logit_pos = torch.max(full_logits_for_last_token, dim=-1)
    print(f"  Maximum logit value: {max_logit_value.item():.6f} at position {max_logit_pos.item()}")
    print(f"  Is maximum in codec range [0, 3071]? {max_logit_pos.item() < 3072}")

    # 检查 padding 部分的 logits
    padding_logits = full_logits_for_last_token[3072:]
    print(f"  Padding logits range: [{padding_logits.min().item():.6f}, {padding_logits.max().item():.6f}]")
    print(f"  Padding logits are all close to zero: {torch.allclose(padding_logits, torch.zeros_like(padding_logits), atol=1e-6)}")

    if actual_id == expected_id:
        print("\n[SUCCESS] Merged vocab model works correctly!")
        print("  - Codec token prediction at positions 0-3071")
        print(f"  - Predicted token: {actual_id}")
        print("  - Padding logits (3072-151935) are effectively zero")
        return True
    else:
        print("\n[FAILURE] Prediction mismatch!")
        print(f"  Difference: {actual_id - expected_id}")
        return False

if __name__ == "__main__":
    try:
        success = verify_merged_vocab_loading()
        sys.exit(0 if success else 1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
