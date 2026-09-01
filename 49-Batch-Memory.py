"""
49-Batch-Memory.py - 批量路数显存/耗时测试
用长短不一 (5~15 字) 的文本，分别以 1/2/4/8 路批量推理，
llama.cpp 的显存分配信息会写入 log/latest.log，事后从 log 中提取。
"""
import time
from qwen3_tts_gguf.inference import TTSEngine, TTSConfig
from qwen3_tts_gguf.inference.batch import BatchRunner
from qwen3_tts_gguf.inference.schema.result import TTSResult

REF_JSON = "output/elaborate/Vivian.json"

# 长短不一: 5 ~ 15 字
TEXTS = [
    "今天天气真好。",           # 7 字
    "晚饭吃什么呢，好纠结啊。",   # 12 字
    "早上好。",                 # 4 字
    "这个电脑的散热好像不太行，烫手。",  # 15 字
    "我明天要去一趟北京。",       # 10 字
    "行，就这样办吧。",           # 8 字
    "别急。",                   # 3 字
    "你看那只猫在晒太阳，好惬意啊。",  # 14 字
]


def main():
    engine = TTSEngine(model_dir="model-base", onnx_provider="CUDA")
    voice = TTSResult.from_json(REF_JSON)
    runner = BatchRunner(engine)
    cfg = TTSConfig(max_steps=100, temperature=0.6, sub_temperature=0.6,
                    seed=42, sub_seed=45, streaming=False)

    for B in [1, 2, 4, 8]:
        tasks = [(TEXTS[i % len(TEXTS)], voice, 'Chinese', False, cfg) for i in range(B)]
        t0 = time.time()
        results = runner.clone_batch(tasks)
        dt = time.time() - t0
        frames = [r.codes.shape[0] for r in results]
        audio_s = sum(f for f in frames) / 12.5
        # 稳态每帧耗时 (去掉 prefill 预热)
        s = results[-1].stats  # 最长路的统计
        t_pred = sum(s.predictor_loop_times) / max(1, len(s.predictor_loop_times))
        t_talk = sum(s.talker_loop_times) / max(1, len(s.talker_loop_times))
        print(f"B={B}: 帧数 {frames}, 壁钟 {dt:.2f}s (prefill {s.prefill_time:.2f}s), "
              f"音频总量 {audio_s:.1f}s, 聚合 RTF {dt/audio_s:.3f}")
        print(f"     稳态每帧: Predictor {t_pred*1000:.1f}ms + Talker {t_talk*1000:.1f}ms")
        print("-" * 60)

    engine.shutdown()


if __name__ == "__main__":
    main()
