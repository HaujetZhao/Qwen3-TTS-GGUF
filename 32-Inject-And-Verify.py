import os
import sys
import torch
import numpy as np
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# 全局变量用于注入
injection_data = {}
injected_flag = False

def talker_injection_hook(module, args, kwargs):
    """
    Hook to modify inputs to Qwen3TTSTalkerModel.forward
    """
    global injected_flag
    
    inputs_embeds = kwargs.get('inputs_embeds')
    
    # 我们只在 Prefill 阶段进行注入
    if inputs_embeds is not None and inputs_embeds.shape[1] > 1:
        print(f"Injecting pre-saved inputs_embeds...")
        # 替换 inputs_embeds
        kwargs['inputs_embeds'] = injection_data['inputs_embeds'].to(inputs_embeds.device).to(inputs_embeds.dtype)
        
        # 同时替换对应的 mask 和 position_ids，确保长度和逻辑一致
        if 'attention_mask' in injection_data and injection_data['attention_mask'] is not None:
             kwargs['attention_mask'] = injection_data['attention_mask'].to(inputs_embeds.device)
        if 'position_ids' in injection_data and injection_data['position_ids'] is not None:
             kwargs['position_ids'] = injection_data['position_ids'].to(inputs_embeds.device)
             
        injected_flag = True
        print("Injected successfully!")
    
    return args, kwargs # 返回修改后的参数

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = os.path.abspath("Qwen3-TTS-12Hz-1.7B-CustomVoice")
    SAVE_FILE = "31_intercepted_embeds.pt"
    
    if not os.path.exists(SAVE_FILE):
        print(f"Error: Could not find saved data file {SAVE_FILE}. Please run 31 first.")
        return

    print(f"Loading injection data from {SAVE_FILE}...")
    global injection_data
    injection_data = torch.load(SAVE_FILE)
    
    print(f"Loading model for injection verification...")
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    tts = Qwen3TTSModel.from_pretrained(MODEL_PATH, device_map=device, dtype=dtype)
    
    # 挂载注入钩子
    target_layer = tts.model.talker.model
    # 注意：这里需要修改 kwargs，所以用 forward_pre_hook
    handle = target_layer.register_forward_pre_hook(talker_injection_hook, with_kwargs=True)
    
    # 即使我们要注入 Embedding，依然需要调用 generate。
    # 我们随便传一些参数，反正第一步会被我们截获并替换成保存好的内容。
    print(f"Running generation with injected embeddings...")
    wavs, sr = tts.generate_custom_voice(
        text=injection_data['text'],
        language="Chinese",
        speaker=injection_data['speaker'],
        instruct="",
    )
    
    # 保存结果
    output_path = "32_injected_verify_audio.wav"
    sf.write(output_path, wavs[0], sr)
    print(f"Verified audio saved to {output_path}")
    
    if injected_flag:
        print("\n✅ Verification SUCCESS: The model used the injected embeddings.")
    else:
        print("\n❌ Verification FAILED: The injection hook was not triggered for prefill.")
        
    handle.remove()

if __name__ == "__main__":
    main()
