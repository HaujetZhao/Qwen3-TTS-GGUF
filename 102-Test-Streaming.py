"""
102-Test-Streaming.py - 验证多进程流式发声
"""
import os
import sys
import time
def test_streaming():
    from qwen3_tts_gguf.engine import TTSEngine
    from qwen3_tts_gguf.result import TTSConfig

    print("\n" + "="*50)
    print("🧪 测试: 多进程流式发声架构")
    print("="*50)

    # 1. 引擎初始化
    engine = TTSEngine()
    
    # 2. 创建流并设置音色 (从现有的 vivian 存档)
    JSON_PATH = "output/vivian.json"
    if not os.path.exists(JSON_PATH):
        print(f"⚠️ 找不到音色存档 {JSON_PATH}，尝试使用内置音色...")
        stream = engine.create_stream()
        res = stream.set_voice_from_speaker("vivian", "你好，我是千问，你今天过得好吗？")
        res.save_json('output/vivian.json')
        res.save_wav('output/vivian.wav')
    else:
        stream = engine.create_stream(voice_path=JSON_PATH)

    # 3. 流式播放配置
    cfg = TTSConfig(
        stream_play=True,      # 开启流式
        mouth_chunk_size=25,    # 每 25 帧发发一次
        max_steps = 200
    )

    print(f"\n🎧 [Mode: Streaming] 正在合成长文本并实时播放...")
    TEXT = "你好，这是一段流式发声测试，通过多进程并发技术实现。"
    
    t0 = time.time()
    res = stream.tts(TEXT, config=cfg)
    total_time = time.time() - t0

    print("\n" + "-"*40)
    print(f"📊 性能分析报告 (音频长度: {res.duration:.2f}s)")
    print(f"  1. Prompt 编译:   {res.stats.prompt_time:.4f}s")
    print(f"  2. 大师 Prefill:  {res.stats.prefill_time:.4f}s")
    print(f"  3. 主进程推理总计: {(res.stats.master_loop_time + res.stats.craftsman_loop_time):.4f}s")
    print(f"  4. 嘴巴同步耗时:   {res.stats.mouth_render_time:.4f}s (流式模式下此值为同步等待时间，通常极短)")
    print("-"*40)
    print(f"总响应耗时: {total_time:.2f}s | RTF: {res.rtf:.2f}")

    # 等待一会确保子进程播完 (因为主进程 tts() 返回时子进程可能还在播最后一块)
    print("\n⏳ 正在等待后台音频播放完毕...")
    time.sleep(3)
    
    # 4. 再次测试非流式播放，确保兼容
    print(f"\n🎧 [Mode: Offline] 正在进行非流式合成 (不自动播放)...")
    cfg_offline = TTSConfig(stream_play=False)
    res_offline = stream.tts("这条语音是非流式合成的，主进程会等待所有计算完成后才返回结果。", config=cfg_offline)
    print(f"✅ 非流式合成成功，音频长度: {res_offline.duration:.2f}s")
    
    # 5. 退出
    engine.shutdown()
    print("\n✅ 测试完成，所有子进程已优雅关闭。")

if __name__ == "__main__":
    test_streaming()
