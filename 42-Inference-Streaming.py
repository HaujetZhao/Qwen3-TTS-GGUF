import sys
import os
import time 

if __name__ == "__main__":
    from qwen3_tts_gguf.engine import Qwen3TTSDoubleStreamEngine
    
    print("✅ 推理引擎加载完成。")
    engine = Qwen3TTSDoubleStreamEngine()
    input("🚀 引擎已就绪。")
    
    text = "你好，很高兴遇见你，你今天过得好吗？"
    engine.synthesize(text, speaker_id="vivian", chunk_size=25)
    
    text = "哇哦！你好棒！"
    engine.synthesize(text, speaker_id="vivian", chunk_size=25)

    
    print("🎵 正在播放，请等待...")
    time.sleep(10)
    engine.shutdown()