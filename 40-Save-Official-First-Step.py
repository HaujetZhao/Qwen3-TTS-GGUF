import os
import sys
import torch
import numpy as np
from qwen_tts import Qwen3TTSModel

# 确保导入本地源码
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(PROJECT_ROOT, "Qwen3-TTS")
sys.path.insert(0, SOURCE_DIR)

# 全局变量用于捕获
captured_data = {}

def talker_model_pre_hook(module, args, kwargs):
    """
    捕获 Master 输入的 Embeddings (Prefill 阶段)
    """
    inputs_embeds = kwargs.get('inputs_embeds')
    # 我们拦截第一步（长度 > 1）
    if inputs_embeds is not None and inputs_embeds.shape[1] > 1 and 'inputs_embeds' not in captured_data:
        print(f"[CAPTURE] Intercepted inputs_embeds shape: {inputs_embeds.shape}")
        captured_data['inputs_embeds'] = inputs_embeds.detach().cpu().to(torch.float32).numpy()
    return None

def codec_head_post_hook(module, input, output):
    """
    捕获 Master 输出的 Logits (第一步生成的 token)
    """
    # output shape: [B, T, Vocab]
    # 我们只关心第一步生成的那个位置的 logits
    if 'logits' not in captured_data:
        print(f"[CAPTURE] Intercepted logits shape: {output.shape}")
        # 在 prefill 阶段，output 包含所有输入 token 的 logits，
        # 我们关心的是最后一个位置，它决定了生成的第一个 code_0
        logits = output[:, -1, :].detach().cpu().to(torch.float32).numpy()
        captured_data['logits'] = logits
    return None

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = os.path.abspath("Qwen3-TTS-12Hz-1.7B-CustomVoice")
    
    print("载入官方模型中...")
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    tts = Qwen3TTSModel.from_pretrained(MODEL_PATH, device_map=device, dtype=dtype)
    
    # 拦截 Embeddings
    handle_in = tts.model.talker.model.register_forward_pre_hook(talker_model_pre_hook, with_kwargs=True)
    # 拦截 Logits
    handle_out = tts.model.talker.codec_head.register_forward_hook(codec_head_post_hook)
    
    # 确定性参数
    deterministic_kwargs = {
        "do_sample": False,
        "subtalker_dosample": False,
        "repetition_penalty": 1.0,
        "temperature": 1.0,
    }
    
    print("用官方模型推理「今天天气好」...")
    
    tts.generate_custom_voice(
        text="今天天气好",
        speaker="Vivian",
        language="Chinese",
        **deterministic_kwargs
    )

    name_embds  = '40-saved-input-embds.npy'
    name_logits = '40-saved-input-logits.npy'
    name_code0  = '40-saved-output-code0.npy'
    
    # --- 保存数据 ---
    if 'inputs_embeds' in captured_data and 'logits' in captured_data:
        # 保存为 npy
        np.save(name_embds, captured_data['inputs_embeds'])
        np.save(name_logits, captured_data['logits'])
        
        # 计算 code_0 并保存
        code_0 = np.argmax(captured_data['logits'], axis=-1)
        np.save(name_code0, code_0)
        
        print("\n--- Success ---")
        print(f"Saved {name_embds} (Shape: {captured_data['inputs_embeds'].shape})")
        print(f"Saved {name_logits} (Shape: {captured_data['logits'].shape})")
        print(f"Master first token (Code 0): {code_0[0]}")
        
    else:
        print("❌ Error: Failed to capture data!")
    
    handle_in.remove()
    handle_out.remove()

if __name__ == "__main__":
    main()
