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

        # 3. 批量 Prefill (各路 pos 从 0 起，长度可不同；M-RoPE 4 平面)
        t_pre = time.time()
        entries = [(pd.embd[0], 0, p) for p, pd in enumerate(prompt_datas)]
        last_idx = talker_batch.set_embd_multi(entries, pos_planes=4)
        if talker_ctx.decode(talker_batch) != 0:
            raise RuntimeError("Batch Talker Prefill decode failed")
        batch_idx = dict(zip(range(B), last_idx))
        for p in range(B):
            hiddens[p] = np.ctypeslib.as_array(
                talker_ctx.get_embeddings_ith(batch_idx[p]), shape=(n_embd_t,)).copy()
            cur_pos[p] = prompt_datas[p].embd.shape[1]
            timings[p].prefill_time = time.time() - t_pre
        logger.info(f"[Batch] Prefill 完成 ({B} 路, 共 {talker_batch.n_tokens} tokens)")

        # 4. 逐帧 lockstep 主循环
        active = list(range(B))
        for frame in range(max(cfg.max_steps for cfg in cfgs)):
            active = [p for p in active if frame < cfgs[p].max_steps]
            if not active:
                break

            # ---- Stage 1: Talker 采样 code_0 ----
            code0s = {}
            for p in active:
                code0 = talker_samplers[p].sample(
                    talker_ctx, idx=batch_idx[p],
                    limit_start=0, limit_end=2048, allow_tokens=allow_tokens)
                talker_samplers[p].accept(code0)
                if code0 != PROTOCOL["EOS"]:
                    code0s[p] = code0
            active = list(code0s.keys())
            if not active:
                break

            # ---- Stage 2: Predictor 批量补全 16 码 ----
            t_pred = time.time()
            frames_out = self._predict_frames(
                pred_ctx, pred_batch,
                [hiddens[p] for p in active], [code0s[p] for p in active],
                [pred_samplers[p] for p in active])
            for p in active:
                timings[p].predictor_loop_times.append(time.time() - t_pred)

            # ---- Stage 3: Talker 批量吃回音频反馈 ----
            t_talk = time.time()
            entries = []
            for p, (step_codes, step_embeds) in zip(active, frames_out):
                audio_summed = np.sum(step_embeds, axis=0)
                pool = trailing[p]
                if pool is not None and step_idx[p] < len(pool):
                    text_vec = pool[step_idx[p]]
                else:
                    text_vec = self.assets.tts_pad
                entries.append(((audio_summed + text_vec).reshape(1, -1), cur_pos[p], p))

                step_idx[p] += 1
                all_codes[p].append(step_codes)
                summed[p].append(audio_summed)

                if cur_pos[p] >= self.n_ctx_per_seq - 1:
                    raise IndexError(f"[Batch] Talker context overflow at seq {p}: {cur_pos[p]}")

            last_idx = talker_batch.set_embd_multi(entries, pos_planes=4)
            if talker_ctx.decode(talker_batch) != 0:
                raise RuntimeError(f"[Batch] Talker step decode failed at frame {frame}")
            batch_idx = dict(zip(active, last_idx))
            for p in active:
                hiddens[p] = np.ctypeslib.as_array(
                    talker_ctx.get_embeddings_ith(batch_idx[p]), shape=(n_embd_t,)).copy()
                cur_pos[p] += 1
                timings[p].talker_loop_times.append(time.time() - t_talk)

            if frame % 50 == 0:
                logger.info(f"[Batch] 帧进度 {frame}, 活跃 {len(active)}/{B}")

        logger.info(f"[Batch] 生成结束: 帧数 {[len(c) for c in all_codes]}")

        # 5. 收尾: 释放推理环境，解码渲染并组装结果
        for sm in talker_samplers + pred_samplers:
            sm.free()
        del talker_batch, pred_batch, talker_ctx, pred_ctx

        results = []
        for p, (text, voice, language, zero_shot, cfg) in enumerate(tasks):
            pd = prompt_datas[p]
            codes = np.array(all_codes[p]) if all_codes[p] else np.zeros((0, 16))
            timings[p].total_steps = len(all_codes[p])

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

    def _predict_frames(self, ctx, batch, hiddens, codes0, samplers):
        """
        批量版工匠推理: B 路 [m_hidden, code0_emb] prefill 后，15 轮 lockstep 采出 Q1-Q15。
        每帧开始整仓清 KV (B 路同生共死，无需按 seq 清理)。
        Returns: [(step_codes[16], step_embeds_raw[16]), ...] 与输入路一一对应
        """
        # 1. 构造各路输入并批量 Prefill
        entries = []
        for h, c0 in zip(hiddens, codes0):
            if self.assets.proj is not None:
                m_h = h @ self.assets.proj["weight"].T + self.assets.proj["bias"]
            else:
                m_h = h
            entries.append(np.stack([m_h, self.assets.get_codec_embedding_1024(0, c0)], axis=0))

        ctx.clear_kv_cache()
        last_idx = batch.set_embd_multi([(e, 0, i) for i, e in enumerate(entries)])
        if ctx.decode(batch) != 0:
            raise RuntimeError("Batch Predictor prefill decode failed")

        # 2. 阶梯式 15 轮: 每轮先全路采样，再全路投喂 (保持单路的逐步采样语义)
        n_seq = len(entries)
        step_codes = [[c0] for c0 in codes0]
        step_embeds = [[self.assets.get_codec_embedding(0, c0).copy()] for c0 in codes0]

        for cs in range(1, 16):
            start_offset, end_offset = (cs - 1) * 2048, cs * 2048

            codes_cs = []
            for row in range(n_seq):
                token_id = samplers[row].sample(ctx, idx=last_idx[row],
                                                limit_start=start_offset, limit_end=end_offset)
                c = token_id - start_offset
                codes_cs.append(c)
                step_codes[row].append(c)
                step_embeds[row].append(self.assets.get_codec_embedding(cs, c).copy())

            if cs < 15:
                feed = [(self.assets.get_codec_embedding_1024(cs, c).reshape(1, -1), cs + 1, row)
                        for row, c in enumerate(codes_cs)]
                last_idx = batch.set_embd_multi(feed)
                if ctx.decode(batch) != 0:
                    raise RuntimeError(f"Batch Predictor step decode failed at cs={cs}")

        return list(zip(step_codes, step_embeds))

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
