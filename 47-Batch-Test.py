"""
47-Batch-Test.py - 两路批量推理验证

Phase 1 (正确性): greedy 两路同文本同音色 -> 两路 codes 应完全一致，
                  且应与单路基线完全一致 (最强正确性验证，排除 pos/串扰等系统性错误)
Phase 2 (一致性): 采样模式、两路同种子同参数 -> 两路 codes 应完全一致
Phase 3 (听感):   不同文本不同种子 -> 输出 wav，由本人试听判断
"""
import os
import time
import numpy as np
from qwen3_tts_gguf.inference import TTSEngine, TTSConfig
from qwen3_tts_gguf.inference.batch import BatchRunner

REF_JSON = "output/elaborate/Vivian.json"
TEXT = "今天天气真好。"
TEXT2 = "批量推理测试。"


def diff_report(a, b, name_a, name_b):
    """对比两份 codes，返回 True/False 并打印首个差异点"""
    if a.shape != b.shape:
        print(f"  ❌ {name_a} vs {name_b}: 形状不同 {a.shape} vs {b.shape}")
        return False
    neq = a != b
    if not neq.any():
        print(f"  ✅ {name_a} vs {name_b}: {a.shape[0]} 帧完全一致")
        return True
    frames = np.where(neq.any(axis=1))[0]
    print(f"  ❌ {name_a} vs {name_b}: {len(frames)}/{a.shape[0]} 帧存在差异, 首个差异帧 {frames[0]}, "
          f"该帧 {np.sum(neq[frames[0]])}/16 个码不同")
    return False


def main():
    os.makedirs("output/batch", exist_ok=True)
    print("🚀 初始化 TTS 引擎...")
    engine = TTSEngine(model_dir="model-base", onnx_provider="CUDA")

    # ---------- 基线: 单路 greedy ----------
    print("\n========== 基线: 单路 greedy ==========")
    stream = engine.create_stream()
    stream.set_voice(REF_JSON)
    voice = stream.voice
    greedy_cfg = TTSConfig(max_steps=60, do_sample=False, sub_do_sample=False, streaming=False)
    t0 = time.time()
    base = stream.clone(text=TEXT, language='Chinese', zero_shot=False, config=greedy_cfg)
    print(f"  单路 greedy: {base.codes.shape[0]} 帧, 耗时 {time.time()-t0:.2f}s")
    stream.shutdown()  # 释放单路 context，给批量腾显存

    runner = BatchRunner(engine)

    # ---------- Phase 1: 批量 greedy 两路，对照基线 ----------
    print("\n========== Phase 1: 批量 greedy 两路 (对照单路基线) ==========")
    t0 = time.time()
    r1 = runner.clone_batch([
        (TEXT, voice, 'Chinese', False, greedy_cfg),
        (TEXT, voice, 'Chinese', False, greedy_cfg),
    ])
    print(f"  批量 greedy: 帧数 {[r.codes.shape[0] for r in r1]}, 耗时 {time.time()-t0:.2f}s")
    ok1 = diff_report(np.array(r1[0].codes), np.array(r1[1].codes), "批量路0", "批量路1")
    ok2 = diff_report(np.array(r1[0].codes), np.array(base.codes), "批量路0", "单路基线")
    print(f"  => Phase 1 {'✅ 通过' if (ok1 and ok2) else '❌ 未通过'}")

    # ---------- Phase 2: 采样模式，两路同种子同参数 ----------
    print("\n========== Phase 2: 采样模式两路同种子 (应完全一致) ==========")
    same_cfg = TTSConfig(max_steps=60, temperature=0.6, sub_temperature=0.6,
                         seed=42, sub_seed=45, streaming=False)
    t0 = time.time()
    r2 = runner.clone_batch([
        (TEXT, voice, 'Chinese', False, same_cfg),
        (TEXT, voice, 'Chinese', False, same_cfg),
    ])
    print(f"  批量采样: 帧数 {[r.codes.shape[0] for r in r2]}, 耗时 {time.time()-t0:.2f}s")
    ok3 = diff_report(np.array(r2[0].codes), np.array(r2[1].codes), "采样路0", "采样路1")
    print(f"  => Phase 2 {'✅ 通过' if ok3 else '❌ 未通过'}")
    r2[0].save("output/batch/phase2_numpy_sampled.wav")  # numpy 分块采样路径的听感样本

    # ---------- Phase 3: 不同文本不同种子，出音频人工试听 ----------
    print("\n========== Phase 3: 不同文本不同种子 (输出音频供试听) ==========")
    cfg_a = TTSConfig(max_steps=60, temperature=0.6, sub_temperature=0.6, seed=42, sub_seed=45, streaming=False)
    cfg_b = TTSConfig(max_steps=60, temperature=0.8, sub_temperature=0.7, seed=777, sub_seed=888, streaming=False)
    t0 = time.time()
    r3 = runner.clone_batch([
        (TEXT, voice, 'Chinese', False, cfg_a),
        (TEXT2, voice, 'Chinese', False, cfg_b),
    ])
    print(f"  批量异参: 帧数 {[r.codes.shape[0] for r in r3]}, 耗时 {time.time()-t0:.2f}s")
    for i, r in enumerate(r3):
        r.save(f"output/batch/phase3_{i}.wav")
        r.print_stats()
    print(f"  => 音频已保存至 output/batch/phase3_0.wav / phase3_1.wav，请人工试听")

    engine.shutdown()


if __name__ == "__main__":
    main()
