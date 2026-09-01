"""
50-Batch-Profile.py - Talker/Predictor 单步耗时剖析 (B=16)
隔离变量逐一测量:
  1. Talker 纯 decode 每步耗时 (embd 输入, M-RoPE 4平面) vs llama-batched-bench ~9.5ms
  2. KV 总量影响: n_ctx 16384 (16x1024) vs 8192 (16x512)
  3. 采样开销: numpy 掩码写 logits 的耗时
  4. Predictor 纯 decode 每步耗时 vs bench ~3.0ms
"""
import time
import numpy as np
from qwen3_tts_gguf.inference import TTSEngine
from qwen3_tts_gguf.inference import llama

B = 16
N_STEPS = 100


def bench_talker(engine, n_ctx_per_seq):
    n_ctx = B * n_ctx_per_seq
    ctx = llama.LlamaContext(engine.talker_model, n_ctx=n_ctx, n_batch=n_ctx,
                             n_seq_max=B, embeddings=True)
    batch = llama.LlamaBatch(4 * B * 64, embd_dim=engine.talker_model.n_embd, n_seq_max=B)
    n_embd = engine.talker_model.n_embd
    dim = np.zeros((1, n_embd), dtype=np.float32)

    # prefill: B 路 x 40 tokens
    ctx.clear_kv_cache()
    prompts = [(np.zeros((40, n_embd), dtype=np.float32), 0, p) for p in range(B)]
    batch.set_embd_multi(prompts, pos_planes=4)
    assert ctx.decode(batch) == 0
    cur_pos = [40] * B

    # 1. 纯 decode
    t0 = time.perf_counter()
    for _ in range(N_STEPS):
        entries = [(dim, cur_pos[p], p) for p in range(B)]
        batch.set_embd_multi(entries, pos_planes=4)
        assert ctx.decode(batch) == 0
        for p in range(B):
            cur_pos[p] += 1
    t_decode = (time.perf_counter() - t0) / N_STEPS * 1000

    # 2. decode + 取 embeddings
    t0 = time.perf_counter()
    last = None
    for _ in range(N_STEPS):
        entries = [(dim, cur_pos[p], p) for p in range(B)]
        last = batch.set_embd_multi(entries, pos_planes=4)
        assert ctx.decode(batch) == 0
        hs = [np.ctypeslib.as_array(ctx.get_embeddings_ith(i), shape=(n_embd,)).copy() for i in last]
        for p in range(B):
            cur_pos[p] += 1
    t_with_emb = (time.perf_counter() - t0) / N_STEPS * 1000

    # 3. numpy 掩码采样开销 (在一次现成 logits 行上模拟 16 次范围掩码)
    n_vocab = llama.llama_vocab_n_tokens(engine.talker_model.vocab)
    logits = np.random.rand(n_vocab).astype(np.float32)
    t0 = time.perf_counter()
    for _ in range(N_STEPS):
        for _ in range(B):
            mask = np.ones(n_vocab, dtype=bool)
            mask[0:2048] = False
            logits[mask] = -1e10
    t_mask = (time.perf_counter() - t0) / N_STEPS * 1000

    del ctx, batch
    return t_decode, t_with_emb, t_mask


def bench_predictor(engine):
    ctx = llama.LlamaContext(engine.predictor_model, n_ctx=max(64, B * 17),
                             n_batch=max(64, B * 17), n_seq_max=B, embeddings=False)
    batch = llama.LlamaBatch(B * 2, embd_dim=engine.predictor_model.n_embd, n_seq_max=B)
    n_embd = engine.predictor_model.n_embd
    dim = np.zeros((1, n_embd), dtype=np.float32)

    # prefill: B 路 x 2 tokens
    ctx.clear_kv_cache()
    batch.set_embd_multi([(np.zeros((2, n_embd), dtype=np.float32), 0, p) for p in range(B)])
    assert ctx.decode(batch) == 0

    t0 = time.perf_counter()
    for _ in range(N_STEPS * 15):
        entries = [(dim, 10, p) for p in range(B)]
        batch.set_embd_multi(entries)
        assert ctx.decode(batch) == 0
    t_step = (time.perf_counter() - t0) / (N_STEPS * 15) * 1000

    del ctx, batch
    return t_step


def main():
    engine = TTSEngine(model_dir="model-base", onnx_provider="CUDA", verbose=False)

    for ncps in [1024, 512]:
        t_dec, t_emb, t_mask = bench_talker(engine, ncps)
        print(f"[Talker B={B} n_ctx={B*ncps}]  纯decode {t_dec:.1f}ms/步 | "
              f"decode+取embd {t_emb:.1f}ms/步 | 16次numpy掩码 {t_mask:.1f}ms/帧  (bench: ~9.5ms/步)")

    t_p = bench_predictor(engine)
    print(f"[Predictor B={B}]  纯decode {t_p:.2f}ms/步  (bench: ~3.0ms/步)")

    engine.shutdown()


if __name__ == "__main__":
    main()
