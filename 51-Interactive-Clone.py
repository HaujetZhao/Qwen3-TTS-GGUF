"""
51-Interactive-Clone.py - 交互式流式语音合成终端 (V3 架构版)
基于最新的 DecoderProxy 和 TTSStream 架构。
"""
import os
import sys
import time
import numpy as np

# 确保能找到 qwen3_tts_gguf 包
sys.path.append(os.getcwd())

from qwen3_tts_gguf.inference import TTSEngine, TTSConfig, TTSResult
from qwen3_tts_gguf.inference.schema.constants import SPEAKER_MAP, LANGUAGE_MAP

def print_help():
    print("\n" + "="*50)
    print("🛠️  Qwen3-TTS 指令系统:")
    print("  /speakers          列出所有内置说话人 (CustomVoice)")
    print("  /languages         列出所有支持的语言标签")
    print("  /voice <人名> <语言> <文本>      合成并【设定】当前音色")
    print("  /load <路径>       从 JSON 存档加载音色锚点")
    print("  /save <路径>       将当前生成的音频/音色保存 (后缀 .wav 或 .json)")
    print("-" * 15)
    print("  /info              查看当前引擎状态与音色信息")
    print("  /temp <值>         调整采样温度 (影响韵律变化)")
    print("  /reset             重置推理记忆 (清除残留音感)")
    print("  /q, /exit          退出程序")
    print("="*50)

def interactive_session():
    print("\n🚀 正在启动 Qwen3-TTS 交互式终端...")
    
    # 1. 引擎初始化
    engine = TTSEngine('model-base', verbose=False)
    stream = engine.create_stream()

    if stream is None:
        print("❌ 引擎初始化失败，请检查模型文件路径是否正确。")
        return

    # 2. 默认配置
    cfg = TTSConfig(temperature=0.6, sub_temperature=0.6, seed=42, sub_seed=45)
    last_result: Optional[TTSResult] = None

    print("\n✨ 引擎就绪！您可以直接输入文本进行合成，或输入 /help 查看指令。")
    print_help()

    try:
        while True:
            try:
                raw_input = input("\n[Qwen3] >>> ").strip()
            except EOFError:
                break
                
            if not raw_input:
                continue
            
            # --- 指令鉴别器 ---
            if raw_input.startswith('/'):
                parts = raw_input.split(maxsplit=4)
                cmd = parts[0].lower()
                
                if cmd == '/help':
                    print_help()
                elif cmd == '/info':
                    print(f"\n[状态] 温度: {cfg.temperature}")
                    print(f"[音色] {stream.voice.info if stream.voice else '未设定'}")
                elif cmd == '/speakers':
                    names = sorted(SPEAKER_MAP.keys())
                    print("\n🎙️ 内置说话人:\n  " + ", ".join(names))
                elif cmd == '/languages':
                    langs = sorted(LANGUAGE_MAP.keys())
                    print("\n🌏 支持语言:\n  " + ", ".join(langs))
                elif cmd == '/voice':
                    if len(parts) < 4:
                        print("❌ 用法: /voice <人名> <语言> <文本>")
                        continue
                    spk, lang, v_text = parts[1], parts[2], parts[3]
                    print(f"🎬 正在根据 [{spk}] 建立音色锚点...")
                    # 内部会自动编码并返回结果
                    last_result = stream.set_voice(spk, text=v_text, language=lang)
                    stream.join()
                    print(f"✅ 音色已锁定。")
                elif cmd == '/load':
                    if len(parts) < 2:
                        print("❌ 用法: /load <路径>")
                        continue
                    last_result = stream.set_voice(parts[1])
                    print(f"✅ 已载入音色。")
                elif cmd == '/save':
                    if len(parts) < 2:
                        print("❌ 用法: /save <文件名>")
                        continue
                    if last_result:
                        last_result.save(parts[1])
                        print(f"✅ 已保存至: {parts[1]}")
                    else:
                        print("❌ 当前没有任何生成结果可保存。")
                elif cmd == '/temp' and len(parts) > 1:
                    cfg.temperature = float(parts[1])
                    print(f"🌡️ 温度调整为: {cfg.temperature}")
                elif cmd == '/design':
                    if len(parts) < 3:
                        print("❌ 用法: /design <文本> <指令>")
                        continue
                    print("🎨 正在设计并生成...")
                    last_result = stream.design(parts[1], instruct=parts[2], config=cfg)
                    stream.join()
                elif cmd == '/custom':
                    if len(parts) < 3:
                        print("❌ 用法: /custom <文本> <人名> [指令]")
                        continue
                    ins = parts[3] if len(parts) > 3 else None
                    print(f"🎭 正在使用精品音色 [{parts[2]}] 合成...")
                    last_result = stream.custom(parts[1], speaker=parts[2], instruct=ins, config=cfg)
                    stream.join()
                elif cmd == '/reset':
                    stream.reset()
                    print("扫 记忆与音色已重置。")
                elif cmd in ['/q', '/exit', '/quit']:
                    break
                else:
                    print(f"❓ 未知指令: {cmd}")
                continue

            # --- 标准合成 (Clone 模式) ---
            try:
                print("🎤 正在流式合成...")
                last_result = stream.clone(raw_input, config=cfg)
                stream.join()
                if last_result:
                    print(f"✨ 完成! [RTF: {last_result.rtf:.2f}]")
            except RuntimeError as e:
                print(f"💡 提示: {e}")
                print("   建议先使用 /voice 或 /load 设定一个音色。")

    except KeyboardInterrupt:
        print("\n👋 退出会话。")
    except Exception as e:
        print(f"\n⚠️ 运行时错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        engine.shutdown()

if __name__ == "__main__":
    interactive_session()
