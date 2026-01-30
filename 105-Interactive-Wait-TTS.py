"""
105-Interactive-Wait-TTS.py - 交互式“全转完再播”终端 (API 重构版)
"""
import os
import sys
import time

# 确保能找到 qwen3_tts_gguf 包
sys.path.append(os.getcwd())

from qwen3_tts_gguf.engine import TTSEngine
from qwen3_tts_gguf.result import TTSConfig
from qwen3_tts_gguf.constants import SPEAKER_MAP, LANGUAGE_MAP

def print_help():
    print("\n" + "="*50)
    print("🛠️  Qwen3-TTS 等待模式指令:")
    print("  /speakers          列出所有内置说话人")
    print("  /languages         列出支持的语言")
    print("  /voice <人名> <语言> <文本>")
    print("                     合成并【设定】当前音色 (等待完成后播放)")
    print("  /load <路径>       从 JSON 存档载入音色")
    print("  /save <路径>       保存当前音色")
    print("-" * 15)
    print("  /info              查看当前状态")
    print("  /reset             重置状态")
    print("  /q, /exit          退出程序")
    print("="*50)

def interactive_session():
    print("\n🚀 正在启动 Qwen3-TTS 交互式终端 (等待模式)...")
    
    engine = TTSEngine(verbose=False)
    stream = engine.create_stream()

    # 非流式配置: stream_play = False
    cfg = TTSConfig(stream_play=False, max_steps=400)

    print_help()

    try:
        while True:
            raw_input = input("\n[WaitMode] >>> ").strip()
            if not raw_input: continue
            
            if raw_input.startswith('/'):
                parts = raw_input.split(maxsplit=4)
                cmd = parts[0].lower()
                
                if cmd == '/help': print_help()
                elif cmd == '/info':
                    print(f"\n[状态] 温度: {cfg.temperature} | 步数限制: {cfg.max_steps}")
                    print(f"[音色] {stream.voice.info if stream.voice else '未设定'}")
                elif cmd == '/speakers':
                    print("\n🎙️ 内置说话人: " + ", ".join(sorted(SPEAKER_MAP.keys())))
                elif cmd == '/languages':
                    print("\n🌏 支持语言: " + ", ".join(sorted(LANGUAGE_MAP.keys())))
                elif cmd == '/voice':
                    if len(parts) < 4:
                        print("❌ 用法: /voice <人名> <语言> <文本>"); continue
                    print(f"🎬 正在建立音色锚点 [{parts[1]}]...")
                    stream.set_voice(parts[1], text=parts[3])
                    print(f"✅ 音色已锁定。")
                elif cmd == '/load':
                    if len(parts) < 2: print("❌ 用法: /load <路径>"); continue
                    stream.set_voice(parts[1])
                    print(f"✅ 已载入音色。")
                elif cmd == '/save':
                    save_path = parts[1] if len(parts) > 1 else f"output/voice_{int(time.time())}.json"
                    if stream.voice:
                        stream.voice.save_json(save_path)
                        print(f"✅ 已保存至: {save_path}")
                    else: print("❌ 无有效音色。")
                elif cmd == '/reset':
                    stream.reset()
                    print("🧹 已重置。")
                elif cmd in ['/q', '/exit']: break
                continue

            try:
                print("⏳ 正在全力推理并渲染...")
                t_0 = time.time()
                res = stream.clone(raw_input, config=cfg, verbose=False)
                
                print(f"\n✅ 合成完成！耗时: {time.time()-t_0:.2f}s")
                print(f"📊 性能分析:")
                print(f"   - 语音时长: {res.duration:.2f}s")
                print(f"   - 采样速度: {res.stats.steps_per_sec:.2f} step/s")
                print(f"   - 实时率 (RTF): {res.rtf:.2f} (越小越快)")
                
                print("🔊 正在播放...")
                stream.play_audio(res.audio)
            except RuntimeError as e:
                print(f"💡 提示: {e}")
                print("   建议先使用 /voice 指令设定一个音色。")

    except KeyboardInterrupt: print("\n👋 退出。")
    except Exception as e: print(f"\n⚠️ 错误: {e}")
    finally: engine.shutdown()

if __name__ == "__main__":
    interactive_session()
