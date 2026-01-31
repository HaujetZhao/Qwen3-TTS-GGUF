"""
41-Inference-Base.py - Qwen3-TTS Base 模型语音克隆脚本 (Engine 版)
调用 TTSEngine 以确保 Prompt 构造协议的标准性。
"""
import os
import soundfile as sf
import sounddevice as sd
from qwen3_tts_gguf.engine import TTSEngine

def main():
    # 1. 配置参数
    TARGET_TEXT = "你好！这是千问3 TTS 的测试。通过调用强大的引擎，我可以精准复刻你的音色。"
    
    REF_AUDIO = "output/sample.json"
    REF_TEXT = "你好，我是千问，你今天过得好吗？"

    # 2. 初始化引擎
    print("🚀 [Base-Clone] 正在初始化 TTS 引擎...")
    engine = TTSEngine(model_dir="model-base")
    stream = engine.create_stream()
    
    # 3. 设置音色锚点 (重要：克隆模式必须先设置)
    stream.set_voice(REF_AUDIO, text=REF_TEXT)
    
    # 3.1 还原并播放参考音频 (如果加载的是不含音频的 JSON)
    if stream.voice:
        print(f"🔊 正在还原参考音频: {REF_AUDIO} ...")
        stream.voice.decode(engine.decoder)
        stream.voice.play(blocking=True)
    
    print(f"🎙️  开始克隆推理...")
    print(f"   参考音频: {REF_AUDIO}")
    print(f"   目标文本: {TARGET_TEXT}")
    
    # 3. 进行推理
    # 从 stream 对象发起克隆
    result = stream.clone(
        text=TARGET_TEXT,
        language="chinese",
        verbose=True
    )
    
    # 4. 播放与保存
    if result.audio is not None:
        print(f"\n✅ 克隆合成成功！ RTF: {result.rtf:.2f}")
        sf.write("output_clone_engine.wav", result.audio, 24000)
        
        print("🔊 正在播放...")
        sd.play(result.audio, 24000)
        sd.wait()
    else:
        print("❌ 合成失败，请检查模型资产是否齐全。")

    engine.shutdown()

if __name__ == "__main__":
    main()
