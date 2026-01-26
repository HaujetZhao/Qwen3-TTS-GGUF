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
    
    print("Loading official model...")
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    tts = Qwen3TTSModel.from_pretrained(MODEL_PATH, device_map=device, dtype=dtype)
    
    # --- 挂载 Hook ---
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
    
    print("Running official inference for '今天天气好'...")
    # 我们拦截并在内部保存数据
    # 为了获取最终的 input_ids，我们拦截 generate 函数的参数
    original_generate = tts.model.talker.generate
    def intercepted_generate(*args, **kwargs):
        # inputs_embeds 在这里可能还没生成，但 input_ids 应该在 kwargs 里
        # 或者在 generate 内部拼凑
        # 实际上我们可以在 talker.model.forward 里拦截，那里有最终的 inputs_embeds
        return original_generate(*args, **kwargs)
    
    # 重新注册一个 hook 来捕获输入的 input_ids (在 talker.generate 内部计算出的)
    # 我们直接拦截 Qwen3TTSTalkerForConditionalGeneration.forward
    def talker_forward_hook(module, args, kwargs):
        input_ids = kwargs.get('input_ids')
        if input_ids is not None and 'input_ids' not in captured_data:
            print(f"[CAPTURE] Intercepted input_ids: {input_ids}")
            captured_data['input_ids'] = input_ids.detach().cpu().numpy()
        return None
    
    handle_ids = tts.model.talker.register_forward_pre_hook(talker_forward_hook, with_kwargs=True)

    tts.generate_custom_voice(
        text="今天天气好",
        speaker="Vivian",
        language="Chinese",
        **deterministic_kwargs
    )
    
    # --- 保存数据 ---
    if 'inputs_embeds' in captured_data and 'logits' in captured_data:
        # 保存为 npy
        np.save("40_first_step_embeds.npy", captured_data['inputs_embeds'])
        np.save("40_first_step_logits.npy", captured_data['logits'])
        if 'input_ids' in captured_data:
            np.save("40_first_step_ids.npy", captured_data['input_ids'])
            print(f"Saved 40_first_step_ids.npy (Shape: {captured_data['input_ids'].shape})")
        
        # 计算 code_0 并保存
        code_0 = np.argmax(captured_data['logits'], axis=-1)
        np.save("40_first_step_code0.npy", code_0)
        
        print("\n--- Success ---")
        print(f"Saved 40_first_step_embeds.npy (Shape: {captured_data['inputs_embeds'].shape})")
        print(f"Saved 40_first_step_logits.npy (Shape: {captured_data['logits'].shape})")
        print(f"Master first token (Code 0): {code_0[0]}")
        
        # 还要把对应的 input_ids 记下来，通常 GGUF 需要 IDs
        # 我们可以通过拦截 generate 的输入来获取
    else:
        print("❌ Error: Failed to capture data!")
    
    handle_in.remove()
    handle_out.remove()

if __name__ == "__main__":
    main()
