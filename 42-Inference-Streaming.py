import sys
import os
import time 

if __name__ == "__main__":
    from qwen3_tts_gguf.engine import Qwen3TTSDoubleStreamEngine
    
    print("✅ 推理引擎加载完成。")
    engine = Qwen3TTSDoubleStreamEngine()
    print("🚀 引擎已就绪。")
    
    text = "您好，欢迎体验千问3-TTS流式推理。这不仅是在测试模型，更是在测试生产级别的异步消费架构。"
    engine.synthesize(text, speaker_id="vivian", chunk_size=50)
    
    print("🎵 正在播放，请等待...")
    time.sleep(15)
    engine.shutdown()