"""
51-Batch-Speed.py - 批量路数速度对比: 1/4/8/16/32 路, 每路 ctx 512, 文本 20~30 字
"""
import time
from qwen3_tts_gguf.inference import TTSEngine, TTSConfig
from qwen3_tts_gguf.inference.batch import BatchRunner
from qwen3_tts_gguf.inference.schema.result import TTSResult

REF_JSON = "output/elaborate/Vivian.json"

TEXTS = [
    "今天天气真不错，我们一起去公园走走吧，听说樱花开得正好。",
    "人工智能的发展速度令人惊叹，短短几年就改变了很多行业。",
    "清晨的菜市场总是热闹非凡，吆喝声此起彼伏，烟火气十足。",
    "学习新语言最重要的是动手实践，写出第一个程序就有成就感。",
    "山间的徒步路线比想象中艰难，但登上垭口那一刻都值得了。",
    "城市夜景从高空俯瞰别有风味，道路像发光的血管缓缓涌动。",
    "做菜是一种很治愈的修行，每一步都需要耐心和专注才行。",
    "耳机里放着熟悉的老歌，窗外小雨不紧不慢，适合发呆一下午。",
]


def main():
    engine = TTSEngine(model_dir="model-base", onnx_provider="CUDA", chunk_size=64)
    voice = TTSResult.from_json(REF_JSON)
    runner = BatchRunner(engine, n_ctx_per_seq=512)
    cfg = TTSConfig(max_steps=200, temperature=0.6, sub_temperature=0.6,
                    seed=42, sub_seed=45, streaming=False)

    for B in [1, 16, 32]:
        tasks = [(TEXTS[i % len(TEXTS)], voice, 'Chinese', False, cfg) for i in range(B)]
        t0 = time.time()
        results = runner.clone_batch(tasks)
        dt = time.time() - t0
        s = results[-1].stats
        frames = [r.codes.shape[0] for r in results]
        audio_s = sum(f for f in frames) / 12.5

        prefill_tokens = runner.last_prefill_tokens
        gen_tokens = sum(frames) * 18   # 每帧每路: Talker 1 + Predictor (2 prefill + 15 步)
        t_gen = s.gen_time

        # 解码耗时: clone_batch 内逐路串行解码波形, 从壁钟中剔除
        t_dec = sum(sum(r.stats.decoder_compute_times or []) for r in results)
        t_codes = dt - t_dec

        print(f"B={B:2d}: 路帧数 min/avg/max {min(frames)}/{sum(frames)/B:.0f}/{max(frames)}, "
              f"总音频 {audio_s:.1f}s, 壁钟 {dt:.2f}s "
              f"(prefill {s.prefill_time*1000:.0f}ms + 生成 {t_gen:.2f}s + 解码 {t_dec:.2f}s)")
        print(f"      prefill {prefill_tokens/s.prefill_time:.0f} tok/s | "
              f"生成 {gen_tokens/t_gen:.0f} LLM tok/s | 每路音频均 {audio_s/B:.1f}s")
        print(f"      纯生成 RTF (壁钟-解码) {t_codes/audio_s:.3f} | "
              f"端到端 RTF {dt/audio_s:.3f}")
        print("-" * 60)

    engine.shutdown()


if __name__ == "__main__":
    main()
