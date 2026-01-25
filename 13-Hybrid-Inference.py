import os
import sys

# 添加项目根目录
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# 添加 Qwen3-TTS 目录
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Qwen3-TTS"))

from qwen3_tts_gguf.inference import Qwen3HybridSynthesizer
# logging.basicConfig(level=logging.INFO) # Removed

def main():
    MODEL_DIR = r'./Qwen3-TTS-12Hz-1.7B-CustomVoice' # 原始模型目录（包含 config.json 和 tokenizer）
    EXPORT_DIR = r'./model' # 包含 GGUF 和 ONNX 的目录
    
    # 确保 EXPORT_DIR 里有我们需要的文件，如果它们在 MODEL_DIR 里也可以
    # 在 inference.py 里我们假设 GGUF 和 ONNX 在 model_dir 下
    # 我们可以稍微 hack 一下，把 model_dir 指向 EXPORT_DIR，并确保 config.json 在那里
    # 或者修改 synthesizer 接受两个路径。
    # 简单起见，假设用户把 config.json 复制到了 model 目录，或者我们在这里指定原始目录。
    
    # 实例化合成器
    # 注意：这里我们传入 MODEL_DIR 以加载 PyTorch Encoder 和 Tokenizer 配置
    # 但是我们希望它从 EXPORT_DIR 加载 GGUF/ONNX。
    # 实际上 inference.py 需要稍微调整以支持分离的路径。
    # 既然 inference.py 是刚刚写的，我们假设它会在 MODEL_DIR 中查找所有内容。
    # 因此，我们需要把生成的 .gguf 和 .onnx 复制/链接到 MODEL_DIR，或者反过来。
    # 建议：将 MODEL_DIR 视为 "Artifacts Path"，我们需要把 config.json 等复制进去。
    
    # 修正：inference.py 可以接受 load_gguf_from 和 load_onnx_from 参数。
    # 但为了保持简单，我们假设导出脚本已经把所有东西放好了。
    # 这里的 11, 12 脚本把模型放到了 ./model。
    # 所以我们应该把 ./Qwen3-TTS-12Hz.../config.json 等复制到 ./model
    
    # 自动复制配置 (Lazy Setup)
    import shutil
    for f in ["config.json", "tokenizer.json", "vocab.json"]:
        src = os.path.join(MODEL_DIR, f)
        dst = os.path.join(EXPORT_DIR, f)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
            print(f"Copied {f} to {EXPORT_DIR}")
            
    # 使用 ./model 作为模型目录
    tts = Qwen3HybridSynthesizer(model_dir=EXPORT_DIR)
    
    # 测试音频
    REF_AUDIO = r'./ref/example.wav' # 假设存在，或者找一个存在的 wav
    # 搜索一个存在的 wav
    for root, dirs, files in os.walk(r'./ref'):
        for file in files:
            if file.endswith(".wav"):
                REF_AUDIO = os.path.join(root, file)
                break
    
    if not os.path.exists(REF_AUDIO):
        print("Warning: No reference audio found. Synthesis might fail.")
        REF_AUDIO = "dummy.wav" # Will crash if not handled
    
    print(f"Using reference audio: {REF_AUDIO}")
    
    tts.synthesize(
        text="你好，这是 Qwen3-TTS 的混合推理测试。", 
        ref_audio_path=REF_AUDIO,
        output_path=os.path.join(EXPORT_DIR, "hybrid_output.wav")
    )

if __name__ == "__main__":
    main()
