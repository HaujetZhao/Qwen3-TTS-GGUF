import os
import sys
import torch
import numpy as np
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# 确保导入的是本地源码
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(PROJECT_ROOT, "Qwen3-TTS")
sys.path.insert(0, SOURCE_DIR)

# 全局变量
manual_embeds = None
injection_done = False

def push_manual_embeds_hook(module, args, kwargs):
    global injection_done
    inputs_embeds = kwargs.get('inputs_embeds')
    
    # 我们只在 Prefill 阶段 (seq_len > 1) 进行注入
    if inputs_embeds is not None and inputs_embeds.shape[1] > 1:
        print(f"Manual Injection: Replacing official embeds (shape {inputs_embeds.shape}) with manual block...")
        kwargs['inputs_embeds'] = manual_embeds.to(inputs_embeds.device).to(inputs_embeds.dtype)
        injection_done = True
        print("Manual block injected!")
    return args, kwargs

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = os.path.abspath("Qwen3-TTS-12Hz-1.7B-CustomVoice")
    NPY_DIR = "model"
    
    print("Loading exported NPY tables...")
    text_table = torch.from_numpy(np.load(os.path.join(NPY_DIR, "text_embedding_projected.npy")))
    codec_table = torch.from_numpy(np.load(os.path.join(NPY_DIR, "codec_embedding_0.npy")))
    
    # --- 14位 Token 手工组装流水线 ---
    print("Constructing 14-token manual block...")
    
    # 准备基础组件
    t_pad = text_table[151671]
    c_pad = codec_table[2148]
    
    # 定义这 14 个位置的 (TextID, CodecID) 映射对
    # 如果 ID 为 None，则表示那一侧不贡献向量（或者你可以理解为贡献了 0，但源码中对于前 3 位是直接查 text 表）
    # 基于 33 脚本的观测结果：
    mapping = [
        (151644, None),   # 0: <|im_start|>
        (77091,  None),   # 1: assistant
        (198,    None),   # 2: \n
        (151671, 2154),   # 3: T_Pad + Think
        (151671, 2156),   # 4: T_Pad + Think_BOS
        (151671, 2055),   # 5: T_Pad + Chinese
        (151671, 2157),   # 6: T_Pad + Think_EOS
        (151671, 3065),   # 7: T_Pad + Vivian
        (151672, 2148),   # 8: TTS_BOS + C_Pad
        (100644, 2148),   # 9: 今天 + C_Pad
        (104307, 2148),   # 10: 天气 + C_Pad
        (52801,  2148),   # 11: 好 + C_Pad
        (151673, 2148),   # 12: TTS_EOS + C_Pad
        (151671, 2149),   # 13: T_Pad + Codec_BOS
    ]
    
    embed_list = []
    for t_id, c_id in mapping:
        # 获取文本侧
        v_t = text_table[t_id] if t_id is not None else torch.zeros(2048)
        # 获取音频侧
        v_c = codec_table[c_id] if c_id is not None else torch.zeros(2048)
        # 融合
        embed_list.append(v_t + v_c)
        
    global manual_embeds
    manual_embeds = torch.stack(embed_list).unsqueeze(0) # [1, 14, 2048]
    
    print(f"Manual block ready, shape: {manual_embeds.shape}")

    # --- 启动官方模型进行验证 ---
    print("Loading official model...")
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    tts = Qwen3TTSModel.from_pretrained(MODEL_PATH, device_map=device, dtype=dtype)
    
    # 挂载注入钩子
    target_layer = tts.model.talker.model
    handle = target_layer.register_forward_pre_hook(push_manual_embeds_hook, with_kwargs=True)
    
    print("Running generation...")
    # 这里传的 text 其实不重要了，因为输入会被我们手工生成的块替换
    # 但传正确的参数能保证 generate 函数内部的逻辑流正常走到 prefill 结束
    wavs, sr = tts.generate_custom_voice(
        text="今天天气好",
        language="Chinese",
        speaker="Vivian",
        instruct="",
    )
    
    output_path = "34_manual_construction_output.wav"
    sf.write(output_path, wavs[0], sr)
    print(f"Manual construction audio saved to {output_path}")
    
    if injection_done:
        print("\n🏆 SUCCESS: We just powered the model using only our exported NPY tables!")
    else:
        print("\n❌ FAILED: Injection was not triggered.")
        
    handle.remove()

if __name__ == "__main__":
    main()
