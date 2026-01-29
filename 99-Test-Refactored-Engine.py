"""
99-Test-Refactored-Engine.py
综合验证重构后的 TTSEngine。
"""
import time
import os
from qwen3_tts_gguf import TTSEngine

def test_non_streaming():
    print("\n" + "="*50)
    print("🧪 测试 1: 非流式同步合成 (状态保留验证)")
    print("="*50)
    
    engine = TTSEngine(model_dir="model", streaming=False)
    stream = engine.create_stream(speaker_id="vivian", language="chinese")
    
    # 第一句：建立音色和语气
    text1 = "你好，我是重构后的千问语音引擎。"
    print(f"\n[S1] 正在合成: {text1}")
    res1 = stream.synthesize(text1, play=True, verbose=True)
    res1.print_stats()
    
    # 第二句：保留上下文继续
    text2 = "我现在正在测试连续对话的记忆功能，音色应该保持一致。"
    print(f"\n[S2] 正在合成: {text2}")
    res2 = stream.synthesize(text2, play=True, verbose=True, save_path="output/refactor_test_s2.wav")
    res2.print_stats()
    
    engine.shutdown()

def test_streaming():
    print("\n" + "="*50)
    print("🧪 测试 2: 流式异步合成")
    print("="*50)
    
    engine = TTSEngine(model_dir="model", streaming=True)
    stream = engine.create_stream(speaker_id="serena", language="chinese")
    
    text = "流式推理现在已经整合到了统一的引擎类中。通过设置不同参数，我们可以实现极低延迟的语音交互。"
    print(f"\n正在流式合成: {text}")
    # 流式模式下音频会通过 worker 异步播放
    stream.synthesize(text, chunk_size=20, play=True, verbose=True)
    
    print("\n🎵 正在等待播放完成...")
    time.sleep(12)
    engine.shutdown()

if __name__ == "__main__":
    # 确保输出目录存在
    os.makedirs("output", exist_ok=True)
    
    try:
        test_non_streaming()
        test_streaming()
        print("\n✅ 所有测试通过！")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
