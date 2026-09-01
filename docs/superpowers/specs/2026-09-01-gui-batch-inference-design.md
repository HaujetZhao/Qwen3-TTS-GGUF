# GUI 接入批量推理 + inference 端重构 设计文档

日期：2026-09-01
分支：batch-inference

## 目标

把克隆页骨架（`qwen3_tts_gguf/gui/app.py`）接入真实推理链路：GUI 内 decoder 不走子进程；支持批量生成、随时停止、进度与日志展示。为此对 inference 端做三处重构：decoder 双模式、voice 准备链抽出、BatchRunner 取消机制。

非目标：自定义音色 / 音色设计两个 tab（保持占位）；流式播放（GUI 只做批量离线生成）。

## ① Decoder 双模式

**问题**：`TTSEngine` 硬编码拉起 `DecoderProxy`（decoder + speaker 两个子进程），服务 51-Interactive-Clone 的流式播放。GUI 批量场景不需要子进程，但要复用 batch/stream 现有的 `engine.decoder.decode()` 调用链。

**方案**：

- `inference/decoder.py` 新增 `LocalDecoder`：
  - 持有 `StatefulDecoder`，实现与 `DecoderProxy.decode()` 完全相同的签名：`decode(input, task_id=, is_final=, stream=, state=) -> DecodeResult`。
  - 包含 TTSResult 输入分支：ref_codes 预解码对齐 state、audio/final_state 回写（逻辑与 `DecoderProxy.decode` 一致）。
  - 同步直调，无队列、无监听线程；用 `sessions` dict 按 task_id 维护 DecoderState（与 worker 侧 `handle_decode_task` 对齐）。
  - speaker/播放类接口（pause/resume/raw_play/stop）不实现——inline 只服务离线场景。
- `TTSEngine.__init__` 加参数 `subprocess_decoder: bool = True`：
  - True：现有行为不变（DecoderProxy，并行初始化 + wait_until_ready）。
  - False：`self.decoder = LocalDecoder(...)`，同步初始化，跳过 wait_until_ready。
- `batch.py` / `stream.py` 零改动。

## ② Voice 准备链抽出

**问题**：json/wav → TTSResult voice 的构造与规范化链（缺 spk_emb → 解码补齐；缺 final_state → 预解码对齐）长在 `TTSStream._set_voice_from_*` 上，GUI 不走 Stream 也需要。

**方案**：抽到新模块 `inference/voice.py`：

```python
def prepare_voice(engine, tokenizer, source_path, text=None) -> Optional[TTSResult]
```

- `.json` → `TTSResult.from_json`；`.wav` 等音频 → `codec_encoder.encode` + `speaker_encoder.encode`（编码器缺失时报错返回 None）。
- 规范化逻辑（补 spk_emb / 补 final_state）原样迁移。
- `TTSStream.set_voice` 系列改为委托该模块，行为不变。

## ③ BatchRunner 取消与容错

- `BatchRunner.__init__` 加 `cancel_event: Optional[threading.Event] = None`；lockstep 主循环每帧开头检查，置位则 break。
- 取消语义：**当前批整批丢弃**（不落半成品盘）；此前已完成批次不受影响；后续批次不再开跑。
- 现有清理（samplers free、del contexts）挪进 `try/finally`，取消/异常路径同样释放显存。
- 单路 context 溢出由 `raise IndexError`（整批炸）改为该路记日志提前退出，其余路继续。

## ④ GUI 接入

**线程模型**：单个后台 worker 线程 + `queue.Queue` 事件队列；UI 侧 `root.after(100, poll)` 拉取事件更新控件。推理/载入永不阻塞 mainloop。载入与生成互斥（生成中禁用载入按钮，载入中禁用生成按钮）。

**载入/卸载**（`on_load_toggle`）：
- 载入：后台线程建 `TTSEngine(model_dir, onnx_provider, llm_use_gpu=(llm_device=="GPU"), subprocess_decoder=False)`；成功后按钮"载入"→"卸载"；失败记日志并还原。
- 卸载：`engine.shutdown()`，按钮还原。
- 载入参数（模型文件夹/上下文/设备）在载入后锁定不可改。

**生成**（`on_start_stop`）：
- 输入校验：引擎就绪、克隆源存在、文本非空、输出目录有效。
- 文本按行拆任务（跳过空行）；`prepare_voice` 建音色锚点；构建 `BatchRunner(engine, n_ctx_per_seq=上下文大小, cancel_event=...)`。
- 按"并发路数"分批调用 `clone_batch`，每批完成后逐路落盘。
- 输出：`输出目录/年月日时分秒/序号.wav` + `序号.json`（`TTSResult.save` 存档，可直接回喂为克隆源）。序号全局递增（跨批次连续）。
- 参数映射：语言/最大步数/Talker 温度/种子/Predictor 温度/种子 → `TTSConfig` 对应字段。
- 停止：开始后按钮变"停止"，点击置位 `cancel_event`；worker 线程感知后收尾、按钮还原。

**状态栏 + 日志**：
- 窗口底部状态栏：状态文字 + 进度条（已完成路数/总路数）。
- Notebook 加第四个只读"日志"tab（Text + 滚动条，自动滚到底）。
- 日志来源：给 inference 的 `logger` 挂 QueueHandler，事件队列统一由 poll 消费——inference 内部 `logger.info`（帧进度、耗时）自动进 GUI 日志，调用点无需手工插桩。

## 涉及文件

| 文件 | 改动 |
|---|---|
| `inference/decoder.py` | +`LocalDecoder` |
| `inference/engine.py` | +`subprocess_decoder` 参数分流 |
| `inference/voice.py` | 新建，voice 准备链 |
| `inference/stream.py` | set_voice 委托 voice.py |
| `inference/batch.py` | cancel_event、try/finally、单路溢出降级 |
| `gui/app.py` | 事件接线、worker 线程、状态栏、日志 tab |

## 验证

- 回归：`47-Batch-Speed.py`（子进程路径）与 51-Interactive-Clone 行为不变。
- inline 路径：小脚本用 `subprocess_decoder=False` 跑 clone_batch，比对音频与子进程路径一致（同种子）。
- GUI 手测（用户本人）：载入/卸载、生成、停止、日志与进度。
