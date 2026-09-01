"""
51-Predictor-Sample-Profile.py - Predictor 每步 decode vs 采样耗时剖析 (B=1/4/8/16)

采样方案对比 (同一份 decode logits 上计时):
  A 现行: 每路 get_logits_ith + numpy 全词表掩码 + llama_sampler_sample (原生链)
  B 分块读取 + numpy 逐路采样: get_logits 一次读 (B, n_vocab) 连续块, 切 2048 片, numpy 模拟 top_k/temp/分布采样
计时单位: 每帧 (15 步) 的毫秒数
"""
import time
import numpy as np
from qwen3_tts_gguf.inference import TTSEngine
from qwen3_tts_gguf.inference import llama

N_FRAMES = 30
TOP_K = 50
TEMP = 0.9


def numpy_sample_rows(L_slice: np.ndarray, rngs, temp=TEMP, top_k=TOP_K):
    """对 (B, 2048) 的 logits 切片做逐路 top_k + temp + 多项采样 (模拟 llama.cpp 采样链)"""
    codes = np.empty(L_slice.shape[0], dtype=np.int64)
    for r in range(L_slice.shape[0]):
        row = L_slice[r].astype(np.float32)
        if temp <= 0:
            codes[r] = np.argmax(row)
            continue
        if 0 < top_k < row.shape[0]:
            kth = np.partition(row, -top_k)[-top_k]
            row = np.where(row < kth, -np.inf, row)
        row = row / temp
        row = np.exp(row - row.max())
        row = row / row.sum()
        codes[r] = rngs[r].choice(row.shape[0], p=row)
    return codes


def bench(engine, B):
    assets = engine.assets
    n_embd = engine.predictor_model.n_embd
    n_vocab = llama.llama_vocab_n_tokens(engine.predictor_model.vocab)

    ctx = llama.LlamaContext(engine.predictor_model, n_ctx=max(64, B * 17),
                             n_batch=max(64, B * 17), n_seq_max=B, embeddings=False)
    batch = llama.LlamaBatch(B * 2, embd_dim=n_embd, n_seq_max=B)
    samplers = [llama.LlamaSampler(temperature=TEMP, top_k=TOP_K, top_p=1.0, seed=42 + i)
                for i in range(B)]
    rngs = [np.random.default_rng(42 + i) for i in range(B)]

    emb0 = assets.get_codec_embedding_1024(0, 100)
    hidden = np.zeros(n_embd, dtype=np.float32)

    t_dec = t_sampA = t_sampB = 0.0
    for _ in range(N_FRAMES):
        # prefill
        ctx.clear_kv_cache()
        c_in = np.stack([hidden, emb0], axis=0)
        entries = [(c_in, 0, p) for p in range(B)]
        last = batch.set_embd_multi(entries)
        assert ctx.decode(batch) == 0

        codes = [100] * B
        for cs in range(1, 16):
            emb_in = assets.get_codec_embedding_1024(cs - 1, codes[0]).reshape(1, -1)
            feed = [(emb_in, cs + 1, p) for p in range(B)]

            # decode 计时
            t0 = time.perf_counter()
            last = batch.set_embd_multi(feed)
            assert ctx.decode(batch) == 0
            t_dec += time.perf_counter() - t0

            s_off, e_off = (cs - 1) * 2048, cs * 2048

            # 方案 A: 现行 (掩码 + 原生采样)
            t0 = time.perf_counter()
            codesA = []
            for row in range(B):
                tid = samplers[row].sample(ctx, idx=last[row], limit_start=s_off, limit_end=e_off)
                codesA.append(tid - s_off)
            t_sampA += time.perf_counter() - t0

            # 方案 B: 分块读 + numpy 采样
            t0 = time.perf_counter()
            L = np.ctypeslib.as_array(llama.llama_get_logits(ctx.ptr), shape=(B, n_vocab))
            codesB = numpy_sample_rows(L[:, s_off:e_off], rngs)
            t_sampB += time.perf_counter() - t0

            codes = list(codesB)

    f = N_FRAMES
    del samplers, ctx, batch
    return (t_dec / f * 1000, t_sampA / f * 1000, t_sampB / f * 1000)


def main():
    engine = TTSEngine(model_dir="model-base", onnx_provider="CUDA", verbose=False)
    print(f"{'B':>3} | {'decode/帧':>9} | {'采样A(现行)/帧':>13} | {'采样B(numpy)/帧':>14} | A+B 合计/帧")
    for B in [1, 4, 8, 16]:
        d, a, b = bench(engine, B)
        print(f"{B:>3} | {d:>7.1f}ms | {a:>11.1f}ms | {b:>12.1f}ms | {d+b:>6.1f}ms")
    engine.shutdown()


if __name__ == "__main__":
    main()
