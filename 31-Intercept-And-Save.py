import os
import sys
import torch
import numpy as np
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# 存储拦截到的数据
intercepted_data = {}

def talker_model_hook(module, args, kwargs):
    """
    Hook to capture inputs to Qwen3TTSTalkerModel.forward
    """
    # kwargs contains: attention_mask, position_ids, inputs_embeds, etc.
    inputs_embeds = kwargs.get('inputs_embeds')
    attention_mask = kwargs.get('attention_mask')
    position_ids = kwargs.get('position_ids')
    
    # 我们只拦截第一步（Prefill阶段），它的 sequence length 会比较长
    if inputs_embeds is not None and inputs_embeds.shape[1] > 1:
        print(f"Intercepted inputs_embeds shape: {inputs_embeds.shape}")
        intercepted_data['inputs_embeds'] = inputs_embeds.detach().cpu()
        if attention_mask is not None:
            intercepted_data['attention_mask'] = attention_mask.detach().cpu()
        if position_ids is not None:
            intercepted_data['position_ids'] = position_ids.detach().cpu()
    
    return None # 继续执行

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = os.path.abspath("Qwen3-TTS-12Hz-1.7B-CustomVoice")
    
    print(f"Loading model for interception...")
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    tts = Qwen3TTSModel.from_pretrained(MODEL_PATH, device_map=device, dtype=dtype)
    
    # 找到大师的核心模型层进行挂载
    # tts.model 是 Qwen3TTSForConditionalGeneration
    # tts.model.talker 是 Qwen3TTSTalkerForConditionalGeneration
    # tts.model.talker.model 是 Qwen3TTSTalkerModel
    target_layer = tts.model.talker.model
    handle = target_layer.register_forward_pre_hook(talker_model_hook, with_kwargs=True)
    
    text = "今天天气好"
    speaker = "Vivian"
    
    print(f"Generating audio and intercepting...")
    wavs, sr = tts.generate_custom_voice(
        text=text,
        language="Chinese",
        speaker=speaker,
        instruct="",
    )
    
    # 保存参考音频
    ref_audio_path = "31_ref_audio.wav"
    sf.write(ref_audio_path, wavs[0], sr)
    print(f"Reference audio saved to {ref_audio_path}")
    
    # 保存拦截到的数据
    if 'inputs_embeds' in intercepted_data:
        # 保存为 .pt 方便后续载入（保持精度）
        data_to_save = {
            'inputs_embeds': intercepted_data['inputs_embeds'],
            'attention_mask': intercepted_data.get('attention_mask'),
            'position_ids': intercepted_data.get('position_ids'),
            'text': text,
            'speaker': speaker
        }
        torch.save(data_to_save, "31_intercepted_embeds.pt")
        print(f"Successfully saved intercepted data to 31_intercepted_embeds.pt")
    else:
        print("Error: Failed to intercept embeddings!")
    
    handle.remove()

if __name__ == "__main__":
    main()
