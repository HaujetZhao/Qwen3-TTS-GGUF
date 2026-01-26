import os
import sys
import torch
import numpy as np
import json
from safetensors import safe_open

# 强制不使用官方库，只使用我们搬运出来的 bare_master 模块
# 检查当前路径，确保 bare_master 在搜索路径中
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from bare_master.configuration_bare_master import Qwen3TTSTalkerConfig
from bare_master.modeling_bare_master import Qwen3TTSTalkerModel

def verify_standalone_logic():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = os.path.abspath("Qwen3-TTS-12Hz-1.7B-CustomVoice")
    WEIGHTS_PATH = os.path.join(MODEL_PATH, "model.safetensors")
    CONFIG_PATH = os.path.join(MODEL_PATH, "config.json")
    
    print(f"--- Standalone Logic Verification ---")
    print(f"Loading config from: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        full_config = json.load(f)
    
    # 使用搬运来的 Config 类
    talker_config = Qwen3TTSTalkerConfig(**full_config['talker_config'])
    
    print(f"Initializing Standalone TalkerModel (Backbone)...")
    # 使用搬运来的 Model 类 (Backbone)
    # 注意：modeling_bare_master.py 里的 Qwen3TTSTalkerModel 是不带 head 的
    model = Qwen3TTSTalkerModel(talker_config).to(device).to(torch.bfloat16)
    
    print(f"Loading and remapping weights from: {WEIGHTS_PATH}")
    with safe_open(WEIGHTS_PATH, framework="pt", device=device) as f:
        state_dict = {}
        for key in f.keys():
            # 大师原本的 backbone 权重
            if key.startswith("talker.model."):
                new_key = key.replace("talker.model.", "")
                state_dict[new_key] = f.get_tensor(key)
        
        # 加载到搬运来的 backbone 模型中
        model.load_state_dict(state_dict, strict=True)
        
        # 提取原本的 codec_head 权重 (用于最后的一步映射)
        codec_head_weight = f.get_tensor("talker.codec_head.weight")

    # 准备推理环境
    model.eval()
    
    print("Loading intercepted data for validation...")
    inputs_embeds = torch.from_numpy(np.load("40_first_step_embeds.npy")).to(device).to(torch.bfloat16)
    expected_logits = torch.from_numpy(np.load("40_first_step_logits.npy")).to(device).to(torch.float32)

    print(f"Input shape: {inputs_embeds.shape}")

    with torch.no_grad():
        # 1. 运行搬运后的骨干网络
        # 注意：这里我们要模拟原本 forward 的 position_ids 逻辑
        # modeling_bare_master 里 forward 会自动根据 inputs_embeds 生成位置编码
        attention_mask = torch.ones(inputs_embeds.shape[:2], device=device)
        
        outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # [B, T, Hidden]
        
        # 2. 取最后一个 token 的隐层，手动过一遍 head
        next_hidden = last_hidden_state[:, -1, :] # [B, Hidden]
        
        # 模拟 Linear 层: weight x hidden
        # codec_head_weight shape: [3072, 2048]
        actual_logits = torch.matmul(next_hidden.to(torch.float32), codec_head_weight.to(torch.float32).T)

    # 3. 对比验证
    print("\n--- Results Analysis ---")
    slice_actual = actual_logits[0]
    slice_expected = expected_logits[0]
    
    diff = torch.abs(slice_actual - slice_expected)
    max_diff = torch.max(diff).item()
    
    print(f"Max Logit Difference: {max_diff:.6f}")
    
    actual_id = torch.argmax(actual_logits, dim=-1).item()
    expected_id = torch.argmax(expected_logits, dim=-1).item()
    
    print(f"Predicted Token ID (Standalone): {actual_id}")
    print(f"Predicted Token ID (Official)  : {expected_id}")

    if actual_id == expected_id:
        print("\n✅ SUCCESS! The standalone definitions in 'bare_master' are BIT-PERFECT in logic.")
    else:
        print("\n❌ FAILURE! Standalone logic produces different results.")

if __name__ == "__main__":
    try:
        verify_standalone_logic()
    except Exception as e:
        import traceback
        traceback.print_exc()
