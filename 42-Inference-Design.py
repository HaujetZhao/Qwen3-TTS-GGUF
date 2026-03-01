"""
43-Inference-Design.py - Qwen3-TTS 音色设计推理脚本 (Engine 版)
"""
from qwen3_tts_gguf.inference.engine import TTSEngine

def main():
    
    # 初始化引擎
    print(f"🚀 [Design-Inference] 正在初始化引擎...")
    # 注意：如果您的设计模型放在 model-design 目录，请指定
    engine = TTSEngine(model_dir="model-design")
    stream = engine.create_stream()

    # 配置参数
    TARGET_TEXT = "哥哥，你回来啦，人家等了你好久好久了，要抱抱！"
    INSTRUCT = "体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显，营造出黏人、做作又刻意卖萌的听觉效果。"

    # 3. 进行推理 (调用 design 模式)
    print(f"🎨 正在设计并合成...")
    result = stream.design(
        text=TARGET_TEXT,
        instruct=INSTRUCT,
        streaming=True,
        verbose=True
    )

    print(f"\n✅ 合成成功！ RTF: {result.rtf:.2f}")
    result.save("./output/design.wav")
    result.save("./output/design.save")

    stream.join()
    engine.shutdown()

if __name__ == "__main__":
    main()
