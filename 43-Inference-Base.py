"""
41-Inference-Base.py - Qwen3-TTS Base 模型语音克隆脚本 (Engine 版)
调用 TTSEngine 以确保 Prompt 构造协议的标准性。
"""
import time
import os
import numpy as np
from qwen3_tts_gguf.inference.engine import TTSEngine
from qwen3_tts_gguf.inference.config import TTSConfig
from qwen3_tts_gguf.inference.prompt_builder import PromptBuilder

# ==================== Vulkan 选项 ====================

# os.environ["VK_ICD_FILENAMES"] = "none"       # 禁止 Vulkan
# os.environ["GGML_VK_VISIBLE_DEVICES"] = "0"   # 禁止 Vulkan 用独显（强制用集显）
# os.environ["GGML_VK_DISABLE_F16"] = "1"       # 禁止 VulkanFP16 计算（Intel集显fp16有溢出问题）


def main():

    # 初始化引擎
    print("🚀 [Base-Clone] 正在初始化 TTS 引擎...")
    engine = TTSEngine(model_dir="model-base-small")
    stream = engine.create_stream()
    
    # 设置音色锚点

    # 读取音频文件，需要编码为 Code，是有损克隆
    # REF_AUDIO = "output/sample.wav"                
    # REF_TEXT = "你好，我是千问，你今天过得好吗？"
    # stream.set_voice(REF_AUDIO, REF_TEXT)

    # 从 json 读取 code，无需从 wav 编码，可以无损克隆
    REF_JSON = "output/sample.json"           
    stream.set_voice(REF_JSON)
    
    
    # if stream.voice:
    #     print(f"🔊 正在还原并播放参考音频: {REF_JSON} ...")
    #     engine.decode(stream.voice)
    #     engine.encode(stream.voice)
    #     stream.voice.play(blocking=True)
    

    # 流式模式下，clone 依然会返回完整 result，但播放是并发进行的
    print(f"\n🎙️  [2/2] 开始流式推理 (边推边播)...")
    target_text = "你今天过得好吗？"
    config = TTSConfig(max_steps=400, temperature=0.6, sub_temperature=0.6, seed=42, sub_seed=45)
    # config = TTSConfig(max_steps=400, do_sample=False, sub_do_sample=False)
    result = stream.clone(
        text=target_text, 
        language='Chinese', 
        streaming=True,
        verbose=True, 
        config=config, 
        chunk_size=8,
    )
    result.print_stats()
    
    print("⏳ 等待流式播放完成 (使用 Event 同步)...")
    time.sleep(2)
    stream.join()

    result.save("./output/clone_result.wav")     # 保存为音频
    result.save("./output/clone_result.json")    # 保存为json，内含无损的音频code

    engine.shutdown()

if __name__ == "__main__":
    main()
