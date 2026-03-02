"""
Qwen3-TTS Design 模型，用于语音设计

同一个音色，每一次生成，会因随机种子、文本的不同，而略有不同，无法稳定

但是合成的 TTSResult 可保存为 json 或 wav，供 Base 模型用于克隆，可保持稳定的音色
"""
import time
import os
import numpy as np
from qwen3_tts_gguf.inference import TTSEngine, TTSConfig, TTSResult

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
    config = TTSConfig(max_steps=400, temperature=0.6, sub_temperature=0.6, seed=42, sub_seed=45)
    result = stream.design(
        text=TARGET_TEXT,
        instruct=INSTRUCT,
        streaming=True,
        verbose=True, 
        config=config, 
        chunk_size=8,
    )
    result.print_stats()
    stream.join()

    print(f"\n✅ 合成成功！ RTF: {result.rtf:.2f}")
    result.save("./output/design.wav")
    result.save("./output/design.save")

    engine.shutdown()

if __name__ == "__main__":
    main()
