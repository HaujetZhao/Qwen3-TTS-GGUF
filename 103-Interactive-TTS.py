"""
103-Interactive-TTS.py - 交互式流式语音合成测试
"""
import os
import sys
import time

# 确保能找到 qwen3_tts_gguf 包
sys.path.append(os.getcwd())

from qwen3_tts_gguf.engine import TTSEngine
from qwen3_tts_gguf.result import TTSConfig

def interactive_session():
    print("\n" + "="*60)
    print("🎤 Qwen3-TTS 交互式流式合成终端")
    print("="*60)

    # 1. 引擎初始化 (开启 verbose 以观察启动透明度)
    engine = TTSEngine(verbose=True)
    
    # 2. 默认音色加载
    JSON_PATH = "output/vivian.json"
    if not os.path.exists(JSON_PATH):
        print(f"\n⚠️ 未找到自定义音色存档 {JSON_PATH}，正在创建默认 vivian 音色...")
        stream = engine.create_stream()
        res = stream.set_voice_from_speaker("vivian", "你好")
        # 自动保存一份以便下次快速加载
        os.makedirs("output", exist_ok=True)
        res.save_json(JSON_PATH)
    else:
        print(f"\n✅ 已加载音色存档: {JSON_PATH}")
        stream = engine.create_stream(voice_path=JSON_PATH)

    # 3. 流式配置
    cfg = TTSConfig(
        stream_play=True,      # 强制开启流式播放
        mouth_chunk_size=12,    # 25 帧分块推送
        max_steps=300          # 步数上限保护
    )

    print("\n" + "-"*60)
    print("💡 输入文字并回车即可开始合成（输入 'exit' 或 'q' 退出）")
    print("-"*60)

    try:
        while True:
            text = input("\n👉 请输入文本: ").strip()
            
            if not text:
                continue
            if text.lower() in ['exit', 'q', 'quit', '退出']:
                break
            
            print(f"🚀 正在合成...")
            t_start = time.time()
            
            # 执行合成 (verbose=True 会在控制台打印推送进度)
            res = stream.tts(text, config=cfg, verbose=True)
            
            t_end = time.time()
            print(f"\n✨ 合成完毕! [时长: {res.duration:.2f}s | 响应: {t_end - t_start:.2f}s | RTF: {res.rtf:.2f}]")
            
    except KeyboardInterrupt:
        # 捕捉 Ctrl-C，不需要打印烦人的堆栈信息
        print("\n\n� 检测到中断信号，正在安全退出...")
    except Exception as e:
        print(f"\n❌ [Terminal] 运行异常: {e}")
    finally:
        print("⏳ 正在回收引擎子进程...")
        try:
            engine.shutdown()
        except: pass
        print("✅ 资源已释放，再见！")

if __name__ == "__main__":
    interactive_session()
