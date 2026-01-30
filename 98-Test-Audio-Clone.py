"""
98-Test-Audio-Clone.py - 验证音频克隆功能 (set_identity_from_clone)
"""
import os
import sys

# 确保能找到 qwen3_tts_gguf 包
sys.path.append(os.getcwd())

from qwen3_tts_gguf.engine import TTSEngine
from qwen3_tts_gguf.result import TTSConfig, TTSResult

def test_audio_clone():
    print("\n" + "="*50)
    print("🧪 测试: 音频克隆功能 (set_identity_from_clone) 验证")
    print("="*50)

    # 1. 初始化引擎
    engine = TTSEngine(model_dir="model")
    
    # 2. 创建 Stream
    stream = engine.create_stream(n_ctx=4096)
    
    # 3. 准备克隆所用的“参考音频”
    # 我们先用内置的 vivian 生成一段音频作为克隆原件
    REF_WAV = "output/clone_source_vivian.wav"
    REF_TEXT = "你今天过得好吗？"
    
    print(f"\n1️⃣ 正在准备参考音频: {REF_WAV}...")
    # 使用原生定调生成参考音频
    res_ref = stream.set_identity_from_speaker(speaker_id="vivian", text=REF_TEXT)
    res_ref.play()
    res_ref.save_wav(REF_WAV)
    print("   ✅ 参考音频已生成。")
    
    # 4. 重置 Stream，进入“克隆模式”
    print("\n2️⃣ 正在重置 Stream 并切换至 [音频克隆] 模式...")
    stream.reset()
    
    # 5. 使用克隆接口定调
    print(f"\n3️⃣ 正在从文件克隆身份: {REF_WAV}...")
    res_clone_id = stream.set_identity_from_clone(
        wav_path=REF_WAV,
        text=REF_TEXT,
        language="chinese"
    )
    
    if res_clone_id is False:
        print(f"   ℹ️ [Expected] 克隆接口已按预期降级（返回 False）。")
        return

    print(f"   ✅ 克隆锚定成功！")
    print(f"   ├─ 参考文字: '{stream.identity.text}'")
    print(f"   └─ 提取到的 Codec 帧数: {stream.identity.codes.shape[0]}")
    
    # 6. 使用克隆后的身份合成新文字
    TARGET_TEXT = "行的，我知道了"
    print(f"\n4️⃣ 正在使用 [克隆音色] 合成新文本: {TARGET_TEXT}")
    
    res_final = stream.tts(TARGET_TEXT)
    
    print("\n📊 合成性能报告:")
    res_final.print_stats()
    
    # 7. 渲染并验证
    print("\n🔊 正在播放合成结果，请闭耳凝神听音色一致性...")
    res_final.play()
    
    res_final.save_wav("output/clone_final_test.wav")
    print(f"\n🎉 测试完成！结果已存至: output/clone_final_test.wav")

if __name__ == "__main__":
    test_audio_clone()
