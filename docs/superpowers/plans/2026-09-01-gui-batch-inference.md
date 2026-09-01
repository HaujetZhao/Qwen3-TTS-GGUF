# GUI 接入批量推理 + inference 端重构 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 克隆页 GUI 接通真实推理链路：进程内解码器（不走子进程）、随时停止、状态栏 + 日志 tab。

**Architecture:** `TTSEngine` 加 `subprocess_decoder` 参数分流，新增 `LocalDecoder` 与 `DecoderProxy` 同签名（batch/stream 零改动）；voice 准备链从 `TTSStream` 抽到 `voice.py` 供 GUI 直用；`BatchRunner` 加取消检查点与单路容错；GUI 单 worker 线程 + 事件队列 + `after` 轮询。

**Tech Stack:** Python / tkinter + ttkbootstrap / onnxruntime / llama.cpp ctypes。

**约定：**
- 提交信息用中文，不加 Co-Authored-By 署名。
- 跑验证脚本统一用 `D:/anaconda3/envs/fun/python.exe`（工作目录 `d:/repos/qwen3-tts`）。
- 设计文档：`docs/superpowers/specs/2026-09-01-gui-batch-inference-design.md`。

---

### Task 1: LocalDecoder + 引擎双模式分流

**Files:**
- Modify: `qwen3_tts_gguf/inference/decoder.py`（文件末尾追加 `LocalDecoder`）
- Modify: `qwen3_tts_gguf/inference/engine.py:20`（签名）、`engine.py:65-82`（初始化段）
- Test: `tmp/smoke_inline_decoder.py`（新建，跑完即弃）

- [ ] **Step 1: 在 decoder.py 末尾追加 LocalDecoder**

```python
class LocalDecoder:
    """
    进程内解码器：与 DecoderProxy.decode() 同签名的同步实现，服务离线场景 (GUI/批量)。
    无队列无监听线程，无播放相关接口 (pause/raw_play 等不支持)。
    """
    def __init__(self, onnx_path: str, onnx_provider: str = 'CPU', chunk_size: int = 12):
        self._dec = StatefulDecoder(onnx_path, onnx_provider=onnx_provider, chunk_size=chunk_size)
        self.sessions = {}  # task_id -> DecoderSession (流式多次调用间保持状态)
        self.ready_states = {"decoder": True, "speaker": False}  # engine 打印用

    def wait_until_ready(self, timeout=10):
        return True

    def decode(self, input: Union[np.ndarray, TTSResult], task_id="default", is_final: bool = False,
               stream: bool = False, state: Optional["DecoderState"] = None) -> "DecodeResult":
        """
        同步解码。参数语义与 DecoderProxy.decode 一致：
        TTSResult 输入时先预解码 ref_codes 对齐记忆，完成后回写 audio/final_state/耗时。
        """
        from .schema.protocol import DecoderSession, DecoderResponse
        from .schema.result import TTSResult as _TTSResult, DecodeResult as _DecodeResult

        if isinstance(input, _TTSResult):
            if input.ref_codes is not None and len(input.ref_codes) > 0 and input.final_state is None:
                res_ref = self.decode(input.ref_codes, task_id=f"{task_id}_ref_init", is_final=True)
                input.final_state = res_ref.final_state
            state = state or input.final_state
            codes = input.codes
            is_final = True
        else:
            codes = input

        t_start = time.time()
        codes_arr = np.asarray(codes, dtype=np.int64)
        if codes_arr.ndim == 1:
            codes_arr = codes_arr.reshape(-1, 16)

        session = self.sessions.get(task_id)
        curr_state = session.state if session is not None else state
        is_task_final = is_final or not stream
        audio, new_state = self._dec.decode(codes_arr, state=curr_state, is_final=is_task_final)

        responses = [DecoderResponse(
            task_id=task_id, msg_type="AUDIO",
            audio=audio.copy() if len(audio) > 0 else np.array([], dtype=np.float32),
            compute_time=time.time() - t_start)]
        if is_task_final:
            responses.append(DecoderResponse(task_id=task_id, msg_type="FINISH", state=new_state))
            self.sessions.pop(task_id, None)
        else:
            self.sessions[task_id] = DecoderSession(state=new_state)

        result = _DecodeResult(responses=responses)
        if isinstance(input, _TTSResult):
            input.audio = result.audio
            input.final_state = result.final_state
            if input.stats:
                input.stats.decoder_compute_times = result.chunk_compute_times
        return result

    def shutdown(self):
        self.sessions.clear()
```

decoder.py 头部补 import（现有只有 `os/time/numpy/logger`）：

```python
from typing import Optional, Union
```

- [ ] **Step 2: engine.py 加参数分流**

`__init__` 签名（[engine.py:20](qwen3_tts_gguf/inference/engine.py#L20)）改为：

```python
def __init__(self, model_dir="model", onnx_provider="CUDA", llm_use_gpu=True, chunk_size=12, verbose=True, subprocess_decoder=True):
```

初始化段（原 3/4/5 步，[engine.py:65-82](qwen3_tts_gguf/inference/engine.py#L65-L82)）改为：

```python
            # 3. 解码后端: 子进程 (流式播放场景) 或进程内 (GUI/离线批量场景)
            t_parallel = time.time()
            if subprocess_decoder:
                self.decoder = DecoderProxy(str(self.paths["decoder_onnx"]), onnx_provider=onnx_provider, chunk_size=self.chunk_size)
                if verbose: print("⏳ [Engine] 正在拉起子进程解码器...")
            else:
                from .decoder import LocalDecoder
                self.decoder = LocalDecoder(str(self.paths["decoder_onnx"]), onnx_provider=onnx_provider, chunk_size=self.chunk_size)

            # 4. 模型引擎初始化 (并行点 2: GGUF 在主进程加载，Decoder 在子进程同时初始化)
            t_gguf = time.time()
            self._init_llama_engines(llm_use_gpu)
            if verbose: print(f"🧠 [Engine] GGUF 推理后端就绪 (耗时: {time.time()-t_gguf:.2f}s)")

            # 5. 子进程模式同步等待解码器信号；进程内模式天然就绪
            if subprocess_decoder:
                is_decoder_ready = self.decoder.wait_until_ready(timeout=10)
                if not is_decoder_ready:
                    logger.warning("⚠️ [Engine] 解码器就绪超时，渲染功能将不可用。")
                    self.ready = False
                    return
            self.ready = True
            if verbose:
                mode = "子进程" if subprocess_decoder else "进程内"
                print(f"✅ [Engine] 解码器就绪 ({mode}) (总并行初始化耗时: {time.time()-t_parallel:.2f}s)")

            print(f"🚀 [Engine] 引擎全链路初始化完成! 总耗时: {time.time()-t_start:.2f}s")
```

（同时删掉原 78-87 行的 else 分支与重复的 ready 判断。）

- [ ] **Step 3: 写冒烟脚本 tmp/smoke_inline_decoder.py**

```python
"""冒烟: 进程内解码器走通 codes -> 音频"""
from qwen3_tts_gguf.inference import TTSEngine
from qwen3_tts_gguf.inference.schema.result import TTSResult

engine = TTSEngine(model_dir="model-base", onnx_provider="CUDA", subprocess_decoder=False)
assert engine, "引擎未就绪"
voice = TTSResult.from_json("output/elaborate/Vivian.json")
res = engine.decoder.decode(voice.codes, task_id="smoke", is_final=True)
assert res.audio is not None and len(res.audio) > 1000, f"音频异常: {res.audio}"
print(f"inline decoder OK, samples={len(res.audio)}, chunks={len(res.chunk_compute_times)}")
engine.shutdown()
```

- [ ] **Step 4: 运行验证**

Run: `cd /d/repos/qwen3-tts && D:/anaconda3/envs/fun/python.exe tmp/smoke_inline_decoder.py`
Expected: `inline decoder OK, samples=<六位数>, chunks=<块数>`，无异常栈。

- [ ] **Step 5: Commit**

```bash
git add qwen3_tts_gguf/inference/decoder.py qwen3_tts_gguf/inference/engine.py
git commit -m "decoder 双模式：新增进程内 LocalDecoder，引擎按 subprocess_decoder 分流"
```

---

### Task 2: voice 准备链抽到 voice.py

**Files:**
- Create: `qwen3_tts_gguf/inference/voice.py`
- Modify: `qwen3_tts_gguf/inference/stream.py:386-465`（set_voice 委托，删三个 `_set_voice_from_*`）
- Test: `tmp/smoke_prepare_voice.py`

- [ ] **Step 1: 新建 voice.py**（逻辑自 stream.py 原样迁移，`self.engine`→`engine`、`self.tokenizer`→`tokenizer`）

```python
"""
voice.py - 音色锚点准备链
json / 音频文件 / TTSResult -> 规范化的 TTSResult (补 spk_emb、补 final_state)。
自 TTSStream._set_voice_from_* 迁移，GUI 与 Stream 共用。
"""
from pathlib import Path
from typing import Optional, Union

import numpy as np

from . import logger
from .schema.result import TTSResult
from .utils.audio import load_audio

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".opus"}


def prepare_voice(engine, tokenizer, source: Union[TTSResult, str, Path],
                  text: Optional[str] = None) -> Optional[TTSResult]:
    """
    构造并规范化音色锚点。失败返回 None (原因见日志)。
    source 为音频文件时 text 用作锚点文本 (可传空串)。
    """
    try:
        if isinstance(source, TTSResult):
            res = source
        else:
            p = Path(source)
            if p.suffix.lower() == ".json":
                if not p.exists():
                    logger.error(f"❌ 未找到音色 JSON 文件: {p}")
                    return None
                res = TTSResult.from_json(str(p))
            elif p.suffix.lower() in AUDIO_EXTS:
                res = _from_audio(engine, tokenizer, p, text or "")
            else:
                logger.error(f"❌ 不支持的克隆源类型: {p.suffix}")
                return None

        if res is None or not _normalize(engine, res):
            return None
        return res
    except Exception as e:
        logger.error(f"❌ 准备音色锚点时出现无法预料的异常: {e}")
        return None


def _from_audio(engine, tokenizer, wav_path: Path, text: str) -> Optional[TTSResult]:
    """从音频文件提取音色特征 (codes + spk_emb)"""
    if engine.codec_encoder is None or engine.speaker_encoder is None:
        logger.error("⚠️ 编码器模块未加载，无法执行音色克隆。")
        return None

    logger.info(f"🎤 正在从音频提取音色特征: {wav_path.name}")
    samples = load_audio(wav_path)
    if samples is None:
        return None

    try:
        codes = engine.codec_encoder.encode(samples)
        spk_emb = engine.speaker_encoder.encode(samples)
        return TTSResult(text=text, text_ids=tokenizer.encode(text).ids,
                         spk_emb=spk_emb, codes=codes)
    except Exception as e:
        logger.error(f"❌ 音声特征提取失败: {e}")
        return None


def _normalize(engine, res: TTSResult) -> bool:
    """规范化锚点：缺 spk_emb 则解码补齐；维度不匹配则重编码；缺 final_state 则预解码对齐"""
    if not res.is_valid_anchor:
        return False

    if res.spk_emb is None and len(res.codes) > 0:
        logger.info("🎤 音色向量缺失，正在从 codes 解码音频并提取...")
        if res.audio is None:
            engine.decode(res.codes)
        engine.encode(res)

    if res.spk_emb is not None and res.spk_emb.shape[-1] != engine.talker_model.n_embd:
        logger.info(f"🔄 维度不匹配 ({res.spk_emb.shape[-1]}->{engine.talker_model.n_embd})，正在转换...")
        if res.audio is None:
            engine.decode(res)
        engine.encode(res)

    if engine.decoder and res.final_state is None and len(res.codes) > 0:
        logger.info("🧠 缺少解码器上下文记忆 (final_state)，正在执行预解码以对齐记忆...")
        engine.decode(res)

    return True
```

- [ ] **Step 2: stream.py 改为委托**

替换 [stream.py:386-465](qwen3_tts_gguf/inference/stream.py#L386-L465)（`set_voice` / `_set_voice_from_result` / `_set_voice_from_json` / `_set_voice_from_audio` 四个方法）为：

```python
    def set_voice(self, source: Union[TTSResult, str, Path], text: Optional[str] = None, **kwargs) -> Union[bool, TTSResult]:
        """统一设置当前流的音色锚点。返回生成的 TTSResult 或 False。"""
        try:
            from .voice import prepare_voice, AUDIO_EXTS
            if isinstance(source, TTSResult):
                res = prepare_voice(self.engine, self.tokenizer, source)
            else:
                p = Path(source)
                if p.suffix.lower() == ".json" or p.suffix.lower() in AUDIO_EXTS:
                    res = prepare_voice(self.engine, self.tokenizer, p, text)
                else:
                    # 尝试作为内置说话人处理
                    return self.set_voice_from_speaker(str(source), text or "你好", **kwargs)

            if res is None:
                return False
            self.voice = res
            logger.info(f"🎭 音色已切换为: {res.text[:20]}...")
            return res
        except Exception as e:
            logger.error(f"❌ 设置音色时出现无法预料的异常: {e}")
            return False
```

`set_voice_from_speaker`（用 `self.custom` 生成）保持不动。

- [ ] **Step 3: 写冒烟脚本 tmp/smoke_prepare_voice.py**

```python
"""冒烟: prepare_voice 从 json 构造锚点并补齐 final_state"""
from qwen3_tts_gguf.inference import TTSEngine
from qwen3_tts_gguf.inference.voice import prepare_voice

engine = TTSEngine(model_dir="model-base", onnx_provider="CUDA", subprocess_decoder=False)
assert engine
voice = prepare_voice(engine, engine.tokenizer, "output/elaborate/Vivian.json")
assert voice is not None, "prepare_voice 返回 None"
assert voice.final_state is not None, "final_state 未补齐"
print(f"prepare_voice OK: text={voice.text[:20]!r}, codes={voice.codes.shape}, "
      f"spk_emb={None if voice.spk_emb is None else voice.spk_emb.shape}")
engine.shutdown()
```

- [ ] **Step 4: 运行验证**

Run: `D:/anaconda3/envs/fun/python.exe tmp/smoke_prepare_voice.py`
Expected: `prepare_voice OK: ...`，日志中出现"缺少解码器上下文记忆…预解码"。

- [ ] **Step 5: Commit**

```bash
git add qwen3_tts_gguf/inference/voice.py qwen3_tts_gguf/inference/stream.py
git commit -m "voice 准备链抽出为 voice.py，TTSStream.set_voice 委托复用"
```

---

### Task 3: BatchRunner 取消检查点 + 单路容错

**Files:**
- Modify: `qwen3_tts_gguf/inference/batch.py:79`（`__init__`）、`batch.py:217-296`（主循环与清理）
- Test: `tmp/smoke_batch_cancel.py`

- [ ] **Step 1: `__init__` 加 cancel_event**（[batch.py:79](qwen3_tts_gguf/inference/batch.py#L79)）

```python
    def __init__(self, engine, n_ctx_per_seq: int = 2048, cancel_event=None):
        self.engine = engine
        self.assets = engine.assets
        self.n_ctx_per_seq = n_ctx_per_seq
        self.cancel_event = cancel_event  # threading.Event；置位后当前批整批丢弃
        self.prompt_builder = PromptBuilder(engine.tokenizer, engine.assets)
        self.task_counter = 0
```

文件头补 `import threading` 无必要——参数不限定类型，注释说明即可（上方已注明）。

- [ ] **Step 2: 主循环加取消检查 + try/finally 清理**

`_run_batch` 中，从"# 3. 批量 Prefill"到"# 4. 逐帧 lockstep 主循环"结束（[batch.py:202-291](qwen3_tts_gguf/inference/batch.py#L202-L291)）包进 `try:`，原"# 5. 收尾"里的 sampler 释放与 context 删除挪到 `finally:`。结构如下（`...` 为保持原样的代码段）：

```python
        talker_samplers = []
        pred_samplers = []
        cancelled = False
        try:
            # 3. 批量 Prefill (各路 pos 从 0 起，长度可不同；M-RoPE 4 平面)
            ... 原第 3 段代码，其中 talker_samplers/pred_samplers 的创建改为向已有空列表 append 或直接赋值 ...
            talker_samplers = [self._create_talker_sampler(cfg) for cfg in cfgs]
            pred_samplers = [self._create_predictor_sampler(cfg) for cfg in cfgs]
            ...

            # 4. 逐帧 lockstep 主循环
            for frame in range(max(cfg.max_steps for cfg in cfgs)):
                # 取消检查点: 每帧开头检查，置位则本批整批丢弃
                if self.cancel_event is not None and self.cancel_event.is_set():
                    cancelled = True
                    logger.info("[Batch] 收到取消信号，本批丢弃")
                    break

                ... 原帧体 ...
        finally:
            # 释放推理环境 (取消/异常路径同样执行)
            for sm in talker_samplers + pred_samplers:
                sm.free()
            del talker_batch, pred_batch, talker_ctx, pred_ctx

        if cancelled:
            return [None] * B

        gen_time = time.time() - t_gen
        ... 原第 5 段组装 results (去掉其中的 del 清理行) ...
```

注意：`t_gen = time.time()` 在循环前定义；`gen_time` 计算移到 finally 之后、组装之前；原"# 5. 收尾"段开头的 `for sm in ...: sm.free()` 与 `del talker_batch, ...` 两行删除（已入 finally）。

- [ ] **Step 3: 单路溢出降级**（原 [batch.py:268-269](qwen3_tts_gguf/inference/batch.py#L268-L269) 的 `raise IndexError`）

Stage 3 帧体内 `for row, p in enumerate(active):` 循环改为：

```python
            entries = []
            overflowed = set()
            for row, p in enumerate(active):
                if cur_pos[p] >= self.n_ctx_per_seq - 1:
                    logger.warning(f"[Batch] Talker context 溢出，seq {p} 提前退出 (已生成 {len(all_codes[p])} 帧)")
                    overflowed.add(p)
                    continue
                audio_summed = audio_sum[row]
                ... 原 append 逻辑不变 ...
            active = [p for p in active if p not in overflowed]
            if not entries:
                if active:
                    continue
                break
            last_idx = talker_batch.set_embd_multi(entries, pos_planes=4)
```

（`if not entries: continue/break`——entries 空但仍有活跃路说明本帧全溢出，跳到下一帧让剩余路自然走 EOS/步数上限；entries 空且无活跃路则结束。）

- [ ] **Step 4: 写冒烟脚本 tmp/smoke_batch_cancel.py**

```python
"""冒烟: 取消信号 -> 整批 None；正常小批量照常出结果"""
import threading
from qwen3_tts_gguf.inference import TTSEngine, TTSConfig
from qwen3_tts_gguf.inference.batch import BatchRunner
from qwen3_tts_gguf.inference.voice import prepare_voice

engine = TTSEngine(model_dir="model-base", onnx_provider="CUDA", subprocess_decoder=False)
assert engine
voice = prepare_voice(engine, engine.tokenizer, "output/elaborate/Vivian.json")
assert voice is not None
cfg = TTSConfig(max_steps=40, temperature=0.6, sub_temperature=0.6, seed=42, sub_seed=45, streaming=False)
tasks = [("取消测试", voice, "Chinese", False, cfg)] * 2

# 1. 预置取消: 立即返回全 None
ev = threading.Event(); ev.set()
runner = BatchRunner(engine, n_ctx_per_seq=512, cancel_event=ev)
out = runner.clone_batch(tasks)
assert all(r is None for r in out), f"预期全 None: {out}"

# 2. 生成中途取消: 仍整批 None
ev2 = threading.Event()
runner2 = BatchRunner(engine, n_ctx_per_seq=512, cancel_event=ev2)
import threading as _t
_t.Thread(target=lambda: (_t.Event().wait(0.5), ev2.set()), daemon=True).start()
out2 = runner2.clone_batch(tasks)
assert all(r is None for r in out2)

# 3. 不取消: 正常出音频
runner3 = BatchRunner(engine, n_ctx_per_seq=512)
out3 = runner3.clone_batch(tasks)
assert all(r is not None and r.audio is not None and len(r.audio) > 1000 for r in out3)
print(f"batch cancel OK: 取消={len(out)}路None, 中途取消={len(out2)}路None, 正常帧数={[r.codes.shape[0] for r in out3]}")
engine.shutdown()
```

- [ ] **Step 5: 运行验证**

Run: `D:/anaconda3/envs/fun/python.exe tmp/smoke_batch_cancel.py`
Expected: `batch cancel OK: 取消=2路None, 中途取消=2路None, 正常帧数=[n, n]`。

- [ ] **Step 6: Commit**

```bash
git add qwen3_tts_gguf/inference/batch.py
git commit -m "BatchRunner 支持取消检查点与单路溢出降级，清理逻辑入 finally"
```

---

### Task 4: GUI 骨架——状态栏 + 日志 tab + 事件队列

**Files:**
- Modify: `qwen3_tts_gguf/gui/app.py`（头部 import、`LogTab` 类、`CloneTab.__init__`、`main()`）

- [ ] **Step 1: 头部补 import**

```python
import logging
import queue
import threading
from datetime import datetime
from pathlib import Path

from qwen3_tts_gguf import logger
```

（推理相关 import 在 Task 5/6 接线时再加，本任务只搭 UI 与事件管道。）

- [ ] **Step 2: 模块级 QueueLogHandler**（放在 `init_style` 之前）

```python
class QueueLogHandler(logging.Handler):
    """把 inference 的 logger 记录转发到 GUI 事件队列 (线程安全)"""
    def __init__(self, q: "queue.Queue"):
        super().__init__(logging.INFO)
        self.q = q

    def emit(self, record):
        try:
            self.q.put(("log", record.getMessage()))
        except Exception:
            pass
```

- [ ] **Step 3: LogTab 类**（放在 `PlaceholderTab` 之前）

```python
class LogTab(ttkb.Frame):
    """只读日志页：显示历史日志，自动滚到底"""

    def __init__(self, master):
        super().__init__(master, padding=5)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.text = tk.Text(self, wrap="word", state="disabled", relief="flat",
                            padx=6, pady=4, highlightthickness=1,
                            highlightbackground="#c9c9c9", highlightcolor="#1a7f8e")
        ybar = ttkb.Scrollbar(self, orient=VERTICAL, command=self.text.yview)
        self.text.configure(yscrollcommand=ybar.set)
        self.text.grid(row=0, column=0, sticky=NSEW)
        ybar.grid(row=0, column=1, sticky=NS)

    def append(self, line: str):
        self.text.configure(state="normal")
        self.text.insert("end", line + "\n")
        self.text.see("end")
        self.text.configure(state="disabled")
```

- [ ] **Step 4: CloneTab 加状态字段与事件管道**

`CloneTab.__init__` 末尾追加：

```python
        # 引擎与生成状态 (worker 线程写，UI 线程经 ui_queue 读)
        self.engine = None
        self.generating = False
        self.cancel_event = None
        self.ui_queue = queue.Queue()
        # 载入区需随载入状态锁定的控件 (on_load_toggle 后由 _set_loaded 刷新)
        self._load_fields = []
```

`_build_load_group` 里把控件存入 `self._load_fields`（Entry / Combobox 均适用，`Browse` 按钮同样锁定）。改动点：

```python
        self.model_entry = ttkb.Entry(group, textvariable=self.model_dir)
        self.model_entry.grid(row=0, column=1, sticky=EW, pady=4)
        ...
        self.n_ctx_entry = ttkb.Entry(opt_row, textvariable=self.n_ctx, width=8)
        ...
        self.llm_combo = ttkb.Combobox(opt_row, ...)
        ...
        self.onnx_combo = ttkb.Combobox(opt_row, ...)
        ...
        browse_btn = ttkb.Button(group, text="浏览", command=self.on_browse_model, width=14)
        ...
        self._load_fields = [self.model_entry, self.n_ctx_entry, self.llm_combo, self.onnx_combo, browse_btn]
```

新增反馈绑定与轮询方法（`CloneTab` 内）：

```python
    def bind_feedback(self, status_var, progress, log_tab):
        """main() 创建完状态栏与日志页后回接"""
        self.status_var = status_var
        self.progress = progress
        self.log_tab = log_tab
        logger.addHandler(QueueLogHandler(self.ui_queue))
        self.after(100, self._poll)

    def _poll(self):
        """UI 线程: 拉干事件队列 (日志/状态/进度/按钮态)"""
        try:
            while True:
                ev = self.ui_queue.get_nowait()
                kind = ev[0]
                if kind == "log":
                    self.log_tab.append(ev[1])
                elif kind == "status":
                    self.status_var.set(ev[1])
                elif kind == "progress":
                    self.progress.configure(maximum=ev[2], value=ev[1])
                elif kind == "loaded":
                    self._set_loaded(ev[1])
                elif kind == "gen_done":
                    self._set_generating(False)
                elif kind == "error":
                    self.log_tab.append(f"❌ {ev[1]}")
        except queue.Empty:
            pass
        self.after(100, self._poll)
```

- [ ] **Step 5: main() 加状态栏与日志 tab**

`main()` 中 Notebook 创建段改为：

```python
    nb = ttkb.Notebook(app)
    nb.grid(row=0, column=0, sticky=NSEW, padx=6, pady=6)

    clone_tab = CloneTab(nb)
    log_tab = LogTab(nb)
    tabs = [
        (clone_tab, "克隆"),
        (PlaceholderTab(nb, "自定义音色"), "自定义音色"),
        (PlaceholderTab(nb, "音色设计"), "音色设计"),
        (log_tab, "日志"),
    ]
    for tab, title in tabs:
        nb.add(tab, text=pad_title(title))
    highlight_active_tab()

    # 底部状态栏: 状态文字 + 进度条
    bar = ttkb.Frame(app, padding=(12, 4))
    bar.grid(row=1, column=0, sticky=EW)
    status_var = tk.StringVar(value="就绪")
    ttkb.Label(bar, textvariable=status_var).pack(side=LEFT)
    progress = ttkb.Progressbar(bar, mode="determinate", length=240)
    progress.pack(side=RIGHT)

    clone_tab.bind_feedback(status_var, progress, log_tab)
```

窗口高度计算段（`app.update_idletasks()` 附近）移到状态栏创建之后，保持不变。

- [ ] **Step 6: 启动自检**

Run: `D:/anaconda3/envs/fun/python.exe -c "from qwen3_tts_gguf.gui.app import main; print('import ok')"`
Expected: `import ok`。

再人工启动 `D:/anaconda3/envs/fun/python.exe 52-GUI.py` 确认：四个 tab（克隆/自定义音色/音色设计/日志）等宽、底部状态栏显示"就绪"、日志 tab 可见且只读。

- [ ] **Step 7: Commit**

```bash
git add qwen3_tts_gguf/gui/app.py
git commit -m "GUI 骨架：底部状态栏、只读日志 tab、logger->GUI 事件队列"
```

---

### Task 5: GUI 载入/卸载接线

**Files:**
- Modify: `qwen3_tts_gguf/gui/app.py`（import 区、`on_load_toggle`、新 `_load_worker` / `_set_loaded`）

- [ ] **Step 1: 补推理 import**（Task 4 的 import 块追加）

```python
from qwen3_tts_gguf.inference import TTSEngine
```

- [ ] **Step 2: 实现 on_load_toggle / _load_worker / _set_loaded**（替换原 `on_load_toggle` 的 `pass`）

```python
    def on_load_toggle(self):
        if self.generating:
            return
        if self.engine is not None:
            self.engine.shutdown()
            self.engine = None
            self._set_loaded(False)
            self.ui_queue.put(("status", "已卸载"))
            return
        params = dict(model_dir=self.model_dir.get(),
                      n_ctx=int(self.n_ctx.get()),
                      llm=self.llm_device.get(),
                      onnx=self.onnx_provider.get())
        self.load_btn.configure(state="disabled")
        self.ui_queue.put(("status", "正在载入模型…"))
        threading.Thread(target=self._load_worker, kwargs=params, daemon=True).start()

    def _load_worker(self, model_dir, n_ctx, llm, onnx):
        """后台线程: 建引擎 (进程内解码器)。参数在 UI 线程读好传入。"""
        try:
            eng = TTSEngine(model_dir=model_dir, onnx_provider=onnx,
                            llm_use_gpu=(llm == "GPU"), subprocess_decoder=False)
            self.engine = eng if eng else None
            self.ui_queue.put(("loaded", bool(eng)))
            self.ui_queue.put(("status", "引擎就绪" if eng else "载入失败，详见日志"))
        except Exception as e:
            self.engine = None
            logger.exception("载入失败")
            self.ui_queue.put(("loaded", False))
            self.ui_queue.put(("error", f"载入异常: {e}"))

    def _set_loaded(self, ok: bool):
        """载入成功: 按钮变卸载、载入区锁定；失败/卸载: 还原"""
        self.load_btn.configure(text="卸载" if ok else "载入", state="normal")
        for w in self._load_fields:
            w.configure(state="disabled" if ok else "normal")
```

- [ ] **Step 3: 启动自检**

Run: `D:/anaconda3/envs/fun/python.exe -c "from qwen3_tts_gguf.gui.app import main; print('import ok')"`
Expected: `import ok`。

人工启动 GUI：载入 `model-base`（默认参数）→ 按钮变"卸载"、载入区变灰、状态栏"引擎就绪"、日志 tab 出现引擎初始化日志；点"卸载"→ 还原。

- [ ] **Step 4: Commit**

```bash
git add qwen3_tts_gguf/gui/app.py
git commit -m "GUI 克隆页接入载入/卸载（进程内解码器引擎）"
```

---

### Task 6: GUI 生成/停止接线

**Files:**
- Modify: `qwen3_tts_gguf/gui/app.py`（import 区、`on_start_stop`、新 `_gen_worker` / `_set_generating`、`on_open_output`）

- [ ] **Step 1: 补 import**

```python
from qwen3_tts_gguf.inference import TTSConfig
from qwen3_tts_gguf.inference.batch import BatchRunner
from qwen3_tts_gguf.inference.voice import prepare_voice
```

- [ ] **Step 1b: 预填默认克隆源与任务文本**

`_build_infer_group` 中克隆源默认值：

```python
        self.clone_source = tk.StringVar(value="output/elaborate/Vivian.json")
```

任务文本框 `self.task_text` 创建后插入默认示例（`ybar.grid` 之后）：

```python
        self.task_text.insert("1.0",
                              "今天天气真不错，我们一起去公园走走吧。\n"
                              "人工智能的发展速度令人惊叹，短短几年就改变了很多行业。\n"
                              "清晨的菜市场总是热闹非凡，吆喝声此起彼伏。\n"
                              "学习新语言最重要的是动手实践，写出第一个程序就有成就感。")
```

- [ ] **Step 2: 实现 on_start_stop / _set_generating**（替换原 `pass`）

```python
    def on_start_stop(self):
        if self.generating:
            # 停止: 置位取消信号，按钮转"停止中…"直到 worker 收尾
            self.cancel_event.set()
            self.start_btn.configure(state="disabled", text="停止中…")
            self.ui_queue.put(("status", "正在停止…"))
            return

        if self.engine is None:
            self.ui_queue.put(("error", "引擎未载入")); return
        source = self.clone_source.get().strip()
        if not source:
            self.ui_queue.put(("error", "未选择克隆源")); return
        lines = [l.strip() for l in self.task_text.get("1.0", "end").splitlines() if l.strip()]
        if not lines:
            self.ui_queue.put(("error", "任务文本为空")); return

        params = dict(
            source=source, lines=lines,
            language=LANGUAGES[self.param_vars["language"].get()],
            cfg=TTSConfig(
                max_steps=int(self.param_vars["max_steps"].get()),
                temperature=float(self.param_vars["temperature"].get()),
                seed=int(self.param_vars["seed"].get()),
                sub_temperature=float(self.param_vars["sub_temperature"].get()),
                sub_seed=int(self.param_vars["sub_seed"].get()),
                streaming=False,
            ),
            n_ctx=int(self.n_ctx.get()),
            n_paths=max(1, int(self.n_paths.get())),
            out_root=self.output_dir.get(),
        )
        self.cancel_event = threading.Event()
        self._set_generating(True)
        self.ui_queue.put(("progress", 0, len(lines)))
        threading.Thread(target=self._gen_worker, kwargs=params, daemon=True).start()

    def _set_generating(self, on: bool):
        self.generating = on
        self.start_btn.configure(text="停止" if on else "开始生成",
                                 bootstyle=DANGER if on else PRIMARY, state="normal")
        self.load_btn.configure(state="disabled" if on else "normal")
```

- [ ] **Step 3: 实现 _gen_worker**

```python
    def _gen_worker(self, source, lines, language, cfg, n_ctx, n_paths, out_root):
        """后台线程: 准备锚点 -> 分批 clone_batch -> 按序号落盘 wav+json"""
        try:
            voice = prepare_voice(self.engine, self.engine.tokenizer, source)
            if voice is None:
                self.ui_queue.put(("error", "克隆源准备失败，详见日志"))
                return

            runner = BatchRunner(self.engine, n_ctx_per_seq=n_ctx, cancel_event=self.cancel_event)
            out_dir = Path(out_root) / datetime.now().strftime("%Y%m%d-%H%M%S")
            out_dir.mkdir(parents=True, exist_ok=True)

            done = ok = idx = 0
            batches = [lines[i:i + n_paths] for i in range(0, len(lines), n_paths)]
            for bi, batch in enumerate(batches):
                if self.cancel_event.is_set():
                    break
                self.ui_queue.put(("status", f"批次 {bi + 1}/{len(batches)}（{len(batch)} 路）生成中…"))
                tasks = [(text, voice, language, False, cfg) for text in batch]
                for r in runner.clone_batch(tasks):
                    idx += 1
                    if r is not None:
                        r.save(str(out_dir / f"{idx:03d}.wav"))
                        r.save(str(out_dir / f"{idx:03d}.json"))
                        ok += 1
                    else:
                        logger.warning(f"[GUI] 第 {idx} 路已取消，未落盘")
                    done += 1
                    self.ui_queue.put(("progress", done, len(lines)))

            state = "已停止" if self.cancel_event.is_set() else "完成"
            self.ui_queue.put(("status", f"{state} {ok}/{len(lines)} 路 → {out_dir}"))
            self.ui_queue.put(("gen_done",))
        except Exception as e:
            logger.exception("生成失败")
            self.ui_queue.put(("error", f"生成异常: {e}"))
```

- [ ] **Step 4: 实现 on_open_output**（替换原 `pass`；Windows 下直接资源管理器打开）

```python
    def on_open_output(self):
        path = self.output_dir.get()
        if os.path.isdir(path):
            os.startfile(path)
```

（头部补 `import os`。）

- [ ] **Step 5: 启动自检**

Run: `D:/anaconda3/envs/fun/python.exe -c "from qwen3_tts_gguf.gui.app import main; print('import ok')"`
Expected: `import ok`。

人工启动 GUI 全流程（用户本人验收）：
1. 载入 model-base → 引擎就绪。
2. 克隆源选 `output/elaborate/Vivian.json`，任务文本粘 4 行短句，并发路数 4，开始生成。
3. 观察：按钮变"停止"、进度条推进、状态栏批次信息、日志 tab 帧进度。
4. 完成后检查 `output/clone/<时间戳>/001.wav…004.wav` + 同名 `.json`（json 可回喂为克隆源）。
5. 再点开始（8 行任务、路数 2），中途点"停止"→ 按钮过"停止中…"回到"开始生成"，状态栏显示"已停止 n/8 路"，已落盘文件完好。

- [ ] **Step 6: Commit**

```bash
git add qwen3_tts_gguf/gui/app.py
git commit -m "GUI 克隆页接入批量生成/停止，输出按时间戳目录+序号落盘 wav+json"
```

---

### Task 7: 回归验证

**Files:**
- Test: 既有脚本运行，无代码改动

- [ ] **Step 1: 子进程路径回归（47-Batch-Speed）**

Run: `D:/anaconda3/envs/fun/python.exe 47-Batch-Speed.py`
Expected: 与改动前同量级——B=1/16/32 各打印 RTF，无异常；确认 DecoderProxy 路径未被破坏。

- [ ] **Step 2: 交互式终端冒烟（51-Interactive-Clone）**

Run: `D:/anaconda3/envs/fun/python.exe 51-Interactive-Clone.py`，输入一句短文本回车。
Expected: 正常合成并流式播放（set_voice 委托改造未破坏该路径），`/q` 退出。

- [ ] **Step 3: 清理冒烟脚本**

```bash
git rm -f --cached tmp/smoke_*.py 2>/dev/null; rm -f tmp/smoke_inline_decoder.py tmp/smoke_prepare_voice.py tmp/smoke_batch_cancel.py
```

（tmp/ 目录不追踪，直接删文件即可。）

- [ ] **Step 4: Commit（如有遗漏改动）**

```bash
git status --short
# 若有未提交改动: git add -A && git commit -m "回归验证收尾"
```
