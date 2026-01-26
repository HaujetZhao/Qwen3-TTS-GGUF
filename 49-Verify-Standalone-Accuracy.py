import os
import sys
import torch
import numpy as np

# 确保导入本地源码
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(PROJECT_ROOT, "Qwen3-TTS")
sys.path.insert(0, SOURCE_DIR)

from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerForConditionalGeneration

def main():
    STANDALONE_MODEL_PATH = os.path.abspath("Qwen3-TTS-Talker-Standalone")
    REF_EMBEDS_PATH = "40_first_step_embeds.npy"
    REF_LOGITS_PATH = "40_first_step_logits.npy"

    if not os.path.exists(STANDALONE_MODEL_PATH):
        print(f"Error: Standalone model not found at {STANDALONE_MODEL_PATH}")
        return
    
    if not os.path.exists(REF_EMBEDS_PATH) or not os.path.exists(REF_LOGITS_PATH):
        print("Error: Reference data (embeds or logits) not found!")
        return

    print("Loading standalone Talker model...")
    # 立足于 standalone 目录加载，精度保持 BF16
    model = Qwen3TTSTalkerForConditionalGeneration.from_pretrained(
        STANDALONE_MODEL_PATH,
        device_map="cpu",
        torch_dtype=torch.bfloat16
    )
    model.eval()

    print("Loading reference data...")
    ref_embeds = np.load(REF_EMBEDS_PATH)
    ref_logits = np.load(REF_LOGITS_PATH)
    
    # 转换为 torch tensor
    # 注意：ref_embeds 保存时是 float32，我们要转回 bfloat16 给模型
    inputs_embeds = torch.from_numpy(ref_embeds).to(torch.bfloat16)
    
    print(f"Inputs shape: {inputs_embeds.shape}")

    print("Running inference on standalone model...")
    with torch.no_grad():
        # 直接调用 forward，只传 inputs_embeds
        # 注意：在官方代码里，如果没有传 position_ids，它会自己生成
        # 但是为了完全一致，我们通常希望捕获当时的 position_ids。
        # 这里先尝试最简化的调用。
        outputs = model(inputs_embeds=inputs_embeds)
        
        # outputs 是 Qwen3TTSTalkerOutputWithPast
        # logits shape: [B, T, Vocab]
        logits = outputs.logits
        
        # 只取最后一个 token 的 logits (对应 prefill 的输出)
        last_logits = logits[:, -1, :].to(torch.float32).numpy()

    print("\nComparing Logits...")
    print(f"Ref Logits shape: {ref_logits.shape}")
    print(f"New Logits shape: {last_logits.shape}")

    # 计算误差
    diff = np.abs(ref_logits - last_logits)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"Max difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")

    # 检查预测的第一个 token ID
    ref_id = np.argmax(ref_logits, axis=-1)[0]
    new_id = np.argmax(last_logits, axis=-1)[0]
    
    print(f"\nOfficial Predict Token ID: {ref_id}")
    print(f"Standalone Predict Token ID: {new_id}")

    if ref_id == new_id:
        print("\n✅ Success: Predicted Token ID matches!")
    else:
        print("\n❌ Failure: Predicted Token ID mismatch!")

    if max_diff < 1e-2:
        print("✅ Success: Logits are consistent (within tolerance for BF16).")
    else:
        print("⚠️ Warning: Significant difference in logits!")

if __name__ == "__main__":
    main()
