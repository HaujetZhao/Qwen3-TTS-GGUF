"""
batch.py - 多路批量推理器 (BatchRunner)

将 B 路任务放进同一对 llama context (n_seq_max=B) 逐帧 lockstep 推进:
    每帧: Talker 采样 code_0 (B路) -> Predictor 补全 16 码 (B路) -> Talker 吃回音频反馈 (B路)
各路 prompt 长度可不同 (ragged batch，无需填充)；某路 EOS 后即退出后续帧。
单路径 stream.py / talker.py / predictor.py 保持不动，本模块是独立的批量旁路。
"""
import time
import numpy as np
from typing import List, Optional, Sequence, Tuple

from . import llama, logger
from .config import TTSConfig
from .schema.constants import PROTOCOL, map_language
from .schema.result import TTSResult, Timing
from .prompt_builder import PromptBuilder


class NumpyBlockSampler:
    """
    Predictor 批量采样器: 一次读取 (A, n_vocab) 连续 logits 块，numpy 向量化完成
    top-k / temperature / 多项采样。语义对齐 LlamaSampler 的 temp/top_k/top_p/dist
    无惩罚链；每路独立 RNG 流，seed 语义与原生链一致 (同种子同轨迹)。
    要求全部路共享 sub_* 参数 (批量产数据的典型形态)，异参场景退回原生逐路采样。
    """

    def __init__(self, do_sample: bool, temperature: float, top_k: int, top_p: float,
                 n_vocab: int):
        self.do_sample = do_sample
        self.temp = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.n_vocab = n_vocab

    def sample_block(self, logits_ptr, A: int, s_off: int, e_off: int, rngs) -> np.ndarray:
        """
        读取 (A, n_vocab) 连续 logits 块的 [s_off, e_off) 切片，返回 A 个码。
        rngs: 与行对齐的每路独立 RNG (调用方按任务持有，保证流跨帧连续)。
        """
        L = np.ctypeslib.as_array(logits_ptr, shape=(A, self.n_vocab))[:, s_off:e_off]

        if not self.do_sample or self.temp <= 0:
            return np.argmax(L, axis=1)

        sl = L.astype(np.float32)
        if 0 < self.top_k < sl.shape[1]:
            kth = np.partition(sl, -self.top_k, axis=1)[:, self.top_k - 1]
            sl[sl < kth[:, None]] = -np.inf
        if self.top_p < 1.0:
            for r in range(sl.shape[0]):  # top-p 需排序，逐路处理
                order = np.argsort(-sl[r])
                sorted_l = sl[r][order]
                probs = np.exp(sorted_l - sorted_l.max())
                cdf = np.cumsum(probs / probs.sum())
                cut = int(np.searchsorted(cdf, self.top_p) + 1)
                keep = np.zeros_like(sl[r], dtype=bool)
                keep[order[:cut]] = True
                sl[r][~keep] = -np.inf

        sl /= self.temp
        sl -= sl.max(axis=1, keepdims=True)
        p = np.exp(sl)
        p /= p.sum(axis=1, keepdims=True)
        cum = np.cumsum(p, axis=1)

        codes = np.empty(A, dtype=np.int64)
        for r in range(A):
            u = rngs[r].random()
            codes[r] = np.searchsorted(cum[r], u, side='right')
        return np.clip(codes, 0, sl.shape[1] - 1)


class BatchRunner:
    """
    批量推理器。每次 clone_batch 创建本轮专用的多序列 context，跑完即释放。
    """

    def __init__(self, engine, n_ctx_per_seq: int = 2048):
        self.engine = engine
        self.assets = engine.assets
        self.n_ctx_per_seq = n_ctx_per_seq
        self.prompt_builder = PromptBuilder(engine.tokenizer, engine.assets)
        self.task_counter = 0

    def clone_batch(self, tasks: Sequence[Tuple[str, object, str, bool, TTSConfig]]) -> List[Optional[TTSResult]]:
        """
        批量克隆合成。

        Args:
            tasks: (text, voice: TTSResult, language, zero_shot, config) 元组序列

        Returns:
            与 tasks 等长的 TTSResult 列表
        """
        B = len(tasks)
        n_ctx = B * self.n_ctx_per_seq
        cfgs = [t[4] for t in tasks]

        # 1. 每路独立状态 (先构建 prompt，用于确定 batch 容量)
        n_embd_t = self.engine.talker_model.n_embd
        prompt_datas = []
        for text, voice, language, zero_shot, cfg in tasks:
            lang_id = map_language(language) if language else None
            prompt_datas.append(self.prompt_builder.build_clone_prompt(text, voice, lang_id, zero_shot))

        # 2. 本轮专用的多序列推理环境
        # Talker 是 M-RoPE 模型: 每个 token 需 4 个 pos 平面，pos 缓冲按 token 槽共享，
        # 故 batch 容量按 4 x prefill 总 token 数预留
        n_ctx = B * self.n_ctx_per_seq
        prefill_tokens = sum(pd.embd.shape[1] for pd in prompt_datas)
        talker_ctx = llama.LlamaContext(self.engine.talker_model, n_ctx=n_ctx, n_batch=n_ctx,
                                        n_seq_max=B, embeddings=True)
        pred_ctx = llama.LlamaContext(self.engine.predictor_model, n_ctx=max(64, B * 17),
                                      n_batch=max(64, B * 17), n_seq_max=B, embeddings=False)
        talker_batch = llama.LlamaBatch(4 * prefill_tokens, embd_dim=n_embd_t, n_seq_max=B)
        pred_batch = llama.LlamaBatch(B * 2, embd_dim=self.engine.predictor_model.n_embd, n_seq_max=B)
        talker_ctx.clear_kv_cache()

        hiddens = [None] * B
        cur_pos = [0] * B
        step_idx = [0] * B
        trailing = [pd.trailing_text_embd[0] if pd.trailing_text_embd is not None else None
                    for pd in prompt_datas]
        all_codes = [[] for _ in range(B)]
        summed = [[] for _ in range(B)]
        timings = [Timing(prompt_time=pd.compile_time) for pd in prompt_datas]

        talker_samplers = [self._create_talker_sampler(cfg) for cfg in cfgs]
        pred_samplers = [self._create_predictor_sampler(cfg) for cfg in cfgs]
        allow_tokens = {PROTOCOL["EOS"], PROTOCOL["PAD"], PROTOCOL["BOS"]}

        # 各路 sub 参数一致时启用 numpy 分块采样快路径 (异参退回原生逐路链)
        sub_key = lambda c: (c.sub_do_sample, c.sub_temperature, c.sub_top_k, c.sub_top_p)
        if len({sub_key(c) for c in cfgs}) == 1:
            c0 = cfgs[0]
            block_sampler = NumpyBlockSampler(
                c0.sub_do_sample, c0.sub_temperature, c0.sub_top_k, c0.sub_top_p,
                n_vocab=llama.llama_vocab_n_tokens(self.engine.predictor_model.vocab))
            # 每路独立 RNG 流，跨帧连续 (同种子同轨迹，与原生链语义一致)
            pred_rngs = [np.random.default_rng(
                c.sub_seed if c.sub_seed is not None else int(time.time() * 1e9) + i)
                for i, c in enumerate(cfgs)]
        else:
            block_sampler, pred_rngs = None, None

        # 每帧分段耗时剖析 (累计值，秒)
        prof = {"talker_sample": 0.0, "pred_input": 0.0, "pred_prefill": 0.0,
                "pred_step_fill": 0.0, "pred_step_decode": 0.0, "pred_sample": 0.0,
                "pred_embeds": 0.0, "talker_fill": 0.0, "talker_decode": 0.0, "talker_extract": 0.0}
        self.profile = prof

        # 3. 批量 Prefill (各路 pos 从 0 起，长度可不同；M-RoPE 4 平面)
        t_pre = time.time()
        entries = [(pd.embd[0], 0, p) for p, pd in enumerate(prompt_datas)]
        last_idx = talker_batch.set_embd_multi(entries, pos_planes=4)
        if talker_ctx.decode(talker_batch) != 0:
            raise RuntimeError("Batch Talker Prefill decode failed")
        self.last_prefill_tokens = talker_batch.n_tokens
        batch_idx = dict(zip(range(B), last_idx))
        for p in range(B):
            hiddens[p] = np.ctypeslib.as_array(
                talker_ctx.get_embeddings_ith(batch_idx[p]), shape=(n_embd_t,)).copy()
            cur_pos[p] = prompt_datas[p].embd.shape[1]
            timings[p].prefill_time = time.time() - t_pre
        logger.info(f"[Batch] Prefill 完成 ({B} 路, 共 {talker_batch.n_tokens} tokens)")

        # 4. 逐帧 lockstep 主循环
        active = list(range(B))
        t_gen = time.time()
        for frame in range(max(cfg.max_steps for cfg in cfgs)):
            active = [p for p in active if frame < cfgs[p].max_steps]
            if not active:
                break

            # ---- Stage 1: Talker 采样 code_0 ----
            t0 = time.time()
            code0s = {}
            for p in active:
                code0 = talker_samplers[p].sample(
                    talker_ctx, idx=batch_idx[p],
                    limit_start=0, limit_end=2048, allow_tokens=allow_tokens)
                talker_samplers[p].accept(code0)
                if code0 != PROTOCOL["EOS"]:
                    code0s[p] = code0
            prof["talker_sample"] += time.time() - t0
            active = list(code0s.keys())
            if not active:
                break

            # ---- Stage 2: Predictor 批量补全 16 码 ----
            t_pred = time.time()
            step_codes_list, audio_sum = self._predict_frames(
                pred_ctx, pred_batch,
                [hiddens[p] for p in active], [code0s[p] for p in active],
                [pred_samplers[p] for p in active], prof,
                block_sampler, [pred_rngs[p] for p in active] if block_sampler else None)
            prof["pred_whole"] = prof.get("pred_whole", 0.0) + (time.time() - t_pred)
            for p in active:
                timings[p].predictor_loop_times.append(time.time() - t_pred)

            # ---- Stage 3: Talker 批量吃回音频反馈 ----
            t_talk = time.time()
            t0 = time.time()
            entries = []
            for row, p in enumerate(active):
                audio_summed = audio_sum[row]
                pool = trailing[p]
                if pool is not None and step_idx[p] < len(pool):
                    text_vec = pool[step_idx[p]]
                else:
                    text_vec = self.assets.tts_pad
                entries.append(((audio_summed + text_vec).reshape(1, -1), cur_pos[p], p))

                step_idx[p] += 1
                all_codes[p].append(step_codes_list[row])
                summed[p].append(audio_summed)

                if cur_pos[p] >= self.n_ctx_per_seq - 1:
                    raise IndexError(f"[Batch] Talker context overflow at seq {p}: {cur_pos[p]}")

            last_idx = talker_batch.set_embd_multi(entries, pos_planes=4)
            prof["talker_fill"] += time.time() - t0
            t0 = time.time()
            if talker_ctx.decode(talker_batch) != 0:
                raise RuntimeError(f"[Batch] Talker step decode failed at frame {frame}")
            prof["talker_decode"] += time.time() - t0
            t0 = time.time()
            batch_idx = dict(zip(active, last_idx))
            for p in active:
                hiddens[p] = np.ctypeslib.as_array(
                    talker_ctx.get_embeddings_ith(batch_idx[p]), shape=(n_embd_t,)).copy()
                cur_pos[p] += 1
            prof["talker_extract"] += time.time() - t0
            for p in active:
                timings[p].talker_loop_times.append(time.time() - t_talk)

            if frame % 50 == 0:
                logger.info(f"[Batch] 帧进度 {frame}, 活跃 {len(active)}/{B}")

        logger.info(f"[Batch] 生成结束: 帧数 {[len(c) for c in all_codes]}")
        gen_time = time.time() - t_gen

        # 5. 收尾: 释放推理环境，解码渲染并组装结果
        for sm in talker_samplers + pred_samplers:
            sm.free()
        del talker_batch, pred_batch, talker_ctx, pred_ctx

        results = []
        for p, (text, voice, language, zero_shot, cfg) in enumerate(tasks):
            pd = prompt_datas[p]
            codes = np.array(all_codes[p]) if all_codes[p] else np.zeros((0, 16))
            timings[p].total_steps = len(all_codes[p])
            timings[p].gen_time = gen_time

            dec = None
            if self.engine.decoder:
                state = voice.final_state if voice and voice.final_state else None
                dec = self.engine.decoder.decode(codes, task_id=f"batch_{self.task_counter}_{p}",
                                                 is_final=True, stream=False, state=state)
                timings[p].decoder_compute_times = dec.chunk_compute_times

            results.append(TTSResult(
                audio=dec.audio if dec else None,
                text=text,
                text_ids=pd.text_ids,
                spk_emb=pd.spk_emb,
                codes=codes,
                summed_embeds=summed[p],
                stats=timings[p],
                final_state=dec.final_state if dec else None,
                ref_codes=voice.codes if voice else None,
            ))
        self.task_counter += 1
        return results

    def _predict_frames(self, ctx, batch, hiddens, codes0, samplers, prof=None,
                        block_sampler=None, rngs=None):
        """
        批量版工匠推理: B 路 [m_hidden, code0_emb] prefill 后，15 轮 lockstep 采出 Q1-Q15。
        每帧开始整仓清 KV (B 路同生共死，无需按 seq 清理)。
        传入 block_sampler 时走 numpy 分块采样快路径 (要求各路 sub 参数一致)，否则原生逐路链。
        Returns: (step_codes: 每路 16 码列表, audio_sum: (路数, 2048) 各路 16 码嵌入和)
        """
        t0 = time.time()
        A = len(hiddens)
        # 1. 批量构造输入 (投影合成单个 GEMM) 并 Prefill
        H = np.stack(hiddens)
        if self.assets.proj is not None:
            H = H @ self.assets.proj["weight"].T + self.assets.proj["bias"]
        codes0_arr = np.asarray(codes0, dtype=np.int64)
        c_in = np.empty((A, 2, H.shape[1]), dtype=np.float32)
        c_in[:, 0] = H
        c_in[:, 1] = self.assets.emb_tables_1024[0][codes0_arr]
        if prof is not None: prof["pred_input"] += time.time() - t0

        t0 = time.time()
        ctx.clear_kv_cache()
        prof["pred_pf_clear"] = prof.get("pred_pf_clear", 0.0) + (time.time() - t0)
        t0 = time.time()
        last_idx = batch.set_embd_multi([(c_in[i], 0, i) for i in range(A)])
        prof["pred_pf_fill"] = prof.get("pred_pf_fill", 0.0) + (time.time() - t0)
        t0 = time.time()
        if ctx.decode(batch) != 0:
            raise RuntimeError("Batch Predictor prefill decode failed")
        prof["pred_pf_decode"] = prof.get("pred_pf_decode", 0.0) + (time.time() - t0)

        # 2. 阶梯式 15 轮: 每轮先全路采样，再全路投喂
        step_codes = [[int(c)] for c in codes0]
        # 16 码嵌入和直接整批累加 (花式索引一次取全路行)
        audio_sum = self.assets.emb_tables[0][codes0_arr].astype(np.float32).copy()

        for cs in range(1, 16):
            s_off, e_off = (cs - 1) * 2048, cs * 2048

            t0 = time.time()
            if block_sampler is not None:
                codes_cs = block_sampler.sample_block(
                    llama.llama_get_logits(ctx.ptr), A, s_off, e_off, rngs)
            else:
                codes_cs = np.array([
                    samplers[row].sample(ctx, idx=idx, limit_start=s_off, limit_end=e_off) - s_off
                    for row, idx in enumerate(last_idx)], dtype=np.int64)
            if prof is not None: prof["pred_sample"] += time.time() - t0

            t0 = time.time()
            for row in range(A):
                step_codes[row].append(int(codes_cs[row]))
            audio_sum += self.assets.emb_tables[cs][codes_cs]
            if prof is not None: prof["pred_embeds"] += time.time() - t0

            if cs < 15:
                t0 = time.time()
                feed = self.assets.emb_tables_1024[cs][codes_cs]
                last_idx = batch.set_embd_multi([(feed[i], cs + 1, i) for i in range(A)])
                if prof is not None: prof["pred_step_fill"] += time.time() - t0
                t0 = time.time()
                if ctx.decode(batch) != 0:
                    raise RuntimeError(f"Batch Predictor step decode failed at cs={cs}")
                if prof is not None: prof["pred_step_decode"] += time.time() - t0

        return step_codes, audio_sum

    def _create_talker_sampler(self, cfg: TTSConfig) -> llama.LlamaSampler:
        return llama.LlamaSampler(
            temperature=cfg.temperature if cfg.do_sample else 0.0,
            top_p=cfg.top_p if cfg.do_sample else 1.0,
            top_k=cfg.top_k if cfg.do_sample else 0,
            min_p=cfg.min_p if cfg.do_sample else 0.0,
            repeat_penalty=cfg.repeat_penalty,
            frequency_penalty=cfg.frequency_penalty,
            presence_penalty=cfg.presence_penalty,
            penalty_last_n=cfg.penalty_last_n,
            seed=cfg.seed,
            n_vocab=llama.llama_vocab_n_tokens(self.engine.talker_model.vocab),
        )

    def _create_predictor_sampler(self, cfg: TTSConfig) -> llama.LlamaSampler:
        return llama.LlamaSampler(
            temperature=cfg.sub_temperature if cfg.sub_do_sample else 0.0,
            top_p=cfg.sub_top_p if cfg.sub_do_sample else 1.0,
            top_k=cfg.sub_top_k if cfg.sub_do_sample else 0,
            seed=cfg.sub_seed,
        )
