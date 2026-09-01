"""
48-Batch-Probe.py - 批量推理污染定位探针

逐步排查:
  Probe A (Talker prefill): 同一 prompt 分别单独进 seq0 / seq1，再一起进 batch，
                            比对三者 hidden，定位 Talker 多 seq 是否互相污染
  Probe B (Predictor):      同一 (hidden, code0) 两路同 batch，比对两路首步 logits
"""
import numpy as np
from qwen3_tts_gguf.inference import TTSEngine
from qwen3_tts_gguf.inference import llama
from qwen3_tts_gguf.inference.prompt_builder import PromptBuilder
from qwen3_tts_gguf.inference.schema.result import TTSResult

REF_JSON = "output/elaborate/Vivian.json"
TEXT = "今天天气真好"


def maxdiff(a, b):
    return float(np.max(np.abs(a - b))) if a.shape == b.shape else float('nan')


def main():
    engine = TTSEngine(model_dir="model-base", onnx_provider="CUDA", verbose=False)
    voice = TTSResult.from_json(REF_JSON)
    pb = PromptBuilder(engine.tokenizer, engine.assets)
    pdata = pb.build_clone_prompt(TEXT, voice, 2055, False)  # 2055 = Chinese
    embd = pdata.embd[0]
    print(f"prompt tokens: {embd.shape[0]}")

    # ---------- Probe A: Talker prefill ----------
    n_embd = engine.talker_model.n_embd
    ctx = llama.LlamaContext(engine.talker_model, n_ctx=4096, n_batch=4096, n_seq_max=2, embeddings=True)
    batch = llama.LlamaBatch(4096, embd_dim=n_embd, n_seq_max=2)

    def prefill_alone(seq_id):
        ctx.clear_kv_cache()
        batch.set_embd_multi([(embd, 0, seq_id)], pos_planes=4)
        assert ctx.decode(batch) == 0
        h = np.ctypeslib.as_array(ctx.get_embeddings_ith(batch.n_tokens - 1), shape=(n_embd,)).copy()
        return h

    ctx.clear_kv_cache()
    last = batch.set_embd_multi([(embd, 0, 0), (embd, 0, 1)], pos_planes=4)
    assert ctx.decode(batch) == 0
    h0 = np.ctypeslib.as_array(ctx.get_embeddings_ith(last[0]), shape=(n_embd,)).copy()
    h1 = np.ctypeslib.as_array(ctx.get_embeddings_ith(last[1]), shape=(n_embd,)).copy()

    ha0 = prefill_alone(0)
    ha1 = prefill_alone(1)
    print(f"[A] 批量 seq0 vs seq1            maxdiff = {maxdiff(h0, h1):.6e}")
    print(f"[A] 批量 seq0 vs 单独 seq0       maxdiff = {maxdiff(h0, ha0):.6e}")
    print(f"[A] 批量 seq1 vs 单独 seq1       maxdiff = {maxdiff(h1, ha1):.6e}")
    print(f"[A] 单独 seq0 vs 单独 seq1       maxdiff = {maxdiff(ha0, ha1):.6e}")

    # ---------- Probe B: Predictor 两路同 batch ----------
    p_n_embd = engine.predictor_model.n_embd
    pctx = llama.LlamaContext(engine.predictor_model, n_ctx=64, n_batch=64, n_seq_max=2, embeddings=False)
    pbatch = llama.LlamaBatch(4, embd_dim=p_n_embd, n_seq_max=2)
    assets = engine.assets
    m_h = (ha0 @ assets.proj["weight"].T + assets.proj["bias"]) if assets.proj is not None else ha0
    c_in = np.stack([m_h, assets.get_codec_embedding_1024(0, 100)], axis=0)

    pctx.clear_kv_cache()
    last = pbatch.set_embd_multi([(c_in, 0, 0), (c_in, 0, 1)])
    assert pctx.decode(pbatch) == 0
    n_vocab = llama.llama_vocab_n_tokens(engine.predictor_model.vocab)
    lg0 = np.ctypeslib.as_array(pctx.get_logits_ith(last[0]), shape=(n_vocab,)).copy()
    lg1 = np.ctypeslib.as_array(pctx.get_logits_ith(last[1]), shape=(n_vocab,)).copy()
    print(f"[B] Predictor 批量两路 logits    maxdiff = {maxdiff(lg0, lg1):.6e}")

    # 对照: 单独单路
    pctx.clear_kv_cache()
    last1 = pbatch.set_embd_multi([(c_in, 0, 0)])
    assert pctx.decode(pbatch) == 0
    lga = np.ctypeslib.as_array(pctx.get_logits_ith(last1[0]), shape=(n_vocab,)).copy()
    print(f"[B] 批量路0 vs 单独路0 logits   maxdiff = {maxdiff(lg0, lga):.6e}")

    del ctx, batch, pctx, pbatch
    engine.shutdown()


if __name__ == "__main__":
    main()
