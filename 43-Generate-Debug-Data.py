import os
import numpy as np
import soundfile as sf
from qwen3_tts_gguf.engine import Qwen3TTSDoubleStreamEngine

def main():
    # 1. 初始化引擎
    # 注意：这里使用 Qwen3TTSDoubleStreamEngine 或者 Qwen3TTS，
    # 既然 41-Inference.py 是 GGUF + ONNX 混合的，我们直接参考其逻辑。
    # 为了简单起见，我们直接修改 41-Inference.py 的逻辑来保存数据。
    
    from qwen3_tts_gguf.engine import Qwen3TTSDoubleStreamEngine
    
    # 创建保存目录
    output_dir = "debug_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # 我们可以直接使用 Engine，但 Engine 是流式的，我们需要捕获完整的 codes。
    # 或者我们直接从 Qwen3TTS (41-Inference.py 中的类) 继承并暴露 codes。
    
    # 为了绝对准确，我们直接在脚本中实现一个简单的同步推理
    from qwen3_tts_gguf.engine import Qwen3TTSDoubleStreamEngine
    
    # 这里我们使用一个技巧：Qwen3TTSDoubleStreamEngine 内部会生成代码并放入队列。
    # 我们创建一个不带多进程的版本，或者直接调用其内部方法。
    
    # 方案：直接使用 41-Inference.py 中的 Qwen3TTS 类逻辑
    from importlib.machinery import SourceFileLoader
    inference_mod = SourceFileLoader("inference", "41-Inference.py").load_module()
    Qwen3TTS = inference_mod.Qwen3TTS
    
    print("🚀 正在初始化 Qwen3TTS 引擎...")
    tts = Qwen3TTS(model_root="model", tokenizer_path="Qwen3-TTS-12Hz-1.7B-CustomVoice")
    
    TARGET_TEXT = "哇哦你好呀！"
    SPEAKER = "vivian"
    LANGUAGE = "chinese"
    
    print(f"🎵 正在合成文本: '{TARGET_TEXT}'...")
    
    # 我们需要获取 all_codes，所以我们手动跑一遍 synthesize 里的逻辑
    real_lang_id = tts.LANGUAGE_MAP.get(LANGUAGE)
    real_spk_id = tts.SPEAKER_MAP.get(SPEAKER)
    
    prompt_embeds = tts._construct_prompt(TARGET_TEXT, real_spk_id, real_lang_id)
    
    sampling_config = {
        "master": {"do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 50},
        "subtalker": {"do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 50}
    }
    
    # 设置随机种子以保证实验可重复性
    np.random.seed(42)
    
    print("  - 执行大师与工匠推理...")
    all_codes, _ = tts._execute_inference(prompt_embeds, max_steps=100, verbose=True, sampling_config=sampling_config)
    
    print("  - 执行嘴巴渲染...")
    audio_data = tts._render_audio(all_codes)
    
    # 保存数据
    codes_path = os.path.join(output_dir, "jintian_codes.npy")
    wav_path = os.path.join(output_dir, "jintian_ref.wav")
    
    np.save(codes_path, np.array(all_codes))
    sf.write(wav_path, audio_data, 24000)
    
    print(f"\n✅ 调试数据已生成:")
    print(f"   - 代码路径: {codes_path} (Shape: {np.array(all_codes).shape})")
    print(f"   - 音频路径: {wav_path}")

if __name__ == "__main__":
    main()
