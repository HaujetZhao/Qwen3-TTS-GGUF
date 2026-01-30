"""
101-Test-Voice-Info.py - 验证 Voice 备注信息持久化
"""
import os
import sys
import numpy as np

# 确保能找到 qwen3_tts_gguf 包
sys.path.append(os.getcwd())

from qwen3_tts_gguf.result import TTSResult

def test_voice_info():
    print("\n" + "="*50)
    print("🧪 测试: Voice 备注信息 (info) 全链路验证")
    print("="*50)

    # 1. 模拟一个音色结果 (Anchor)
    print("1️⃣ 创建模拟音色并设置 info...")
    res = TTSResult(
        text="测试文本",
        text_ids=[1, 2, 3],
        spk_emb=np.zeros(2048, dtype=np.float32),
        codes=np.zeros((10, 16), dtype=np.int64),
        info="这是一个优雅的中年女性音色，带有一点上海口音"
    )
    print(f"   - 原始 Info: {res.info}")

    # 2. 保存并显式修改 Info
    JSON_PATH = "output/test_voice_info.json"
    print(f"\n2️⃣ 执行 save_json 并重写 info 参数...")
    res.save_json(JSON_PATH, info="修改后的备注：温柔的女声")
    print(f"   - 缓存中的 Info 同时也已更新为: {res.info}")

    # 3. 从 JSON 加载
    print(f"\n3️⃣ 执行 from_json 重新加载...")
    res_loaded = TTSResult.from_json(JSON_PATH)
    print(f"   - 加载后的 Info: {res_loaded.info}")

    # 4. 最终断言
    if res_loaded.info == "修改后的备注：温柔的女声":
        print("\n✅ [Success] Voice 备注信息持久化验证通过！")
    else:
        print("\n❌ [Failed] 备注信息不匹配或未保存。")

if __name__ == "__main__":
    test_voice_info()
