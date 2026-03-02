"""
Qwen3-TTS CustomVoice 模型，内置音色合成

同一个音色，每一次生成，会因随机种子、文本的不同，而略有不同，无法稳定

但是合成的 TTSResult 可保存为 json 或 wav，供 Base 模型用于克隆，可保持稳定的音色
"""
import time
import os
import numpy as np
from qwen3_tts_gguf.inference import TTSEngine, TTSConfig, TTSResult

def main():
    
    # 1. 初始化引擎
    print(f"🚀 [Custom-Inference] 正在初始化引擎")
    # 注意：如果您的定制模型放在 model-custom 目录，请指定
    engine = TTSEngine(model_dir="model-custom")
    stream = engine.create_stream()

    # 2. 配置参数
    # 官方支持音色: Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee
    TARGET_TEXT = "你好，我是千问"
    SPEAKER = "Vivian"
    INSTRUCT = ""

    # 3. 进行推理 (调用 custom 模式)
    print(f"🎭 正在合成...")
    config = TTSConfig(max_steps=400, temperature=0.6, sub_temperature=0.6, seed=42, sub_seed=45)
    result = stream.custom(
        text=TARGET_TEXT,
        speaker=SPEAKER,
        instruct=INSTRUCT,
        language='chinese',
        streaming=True,
        verbose=True, 
        config=config, 
        chunk_size=8,
    )
    result.print_stats()
    stream.join()
    
    print(f"\n✅ 合成成功！ RTF: {result.rtf:.2f}")
    result.save("./output/custom.wav")
    result.save("./output/custom.json")

    engine.shutdown()

if __name__ == "__main__":
    main()
