"""克隆页：载入/推理两区 + 后台线程接线（载入 worker、批量生成 worker）。"""
import os
import queue
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog

import ttkbootstrap as ttkb
import windnd
from ttkbootstrap.constants import *

from qwen3_tts_gguf import logger
from qwen3_tts_gguf.inference import TTSEngine, TTSConfig
from qwen3_tts_gguf.inference.batch import BatchRunner
from qwen3_tts_gguf.inference.voice import prepare_voice

from .log_tab import PrintRedirect, QueueLogHandler

# 克隆源支持的文件类型
CLONE_SOURCE_TYPES = [("克隆源", "*.wav *.json"), ("所有文件", "*.*")]

# 语言下拉框：显示中文，内部值与引擎一致
LANGUAGES = {"中文": "Chinese", "英语": "English", "越南语": "Vietnamese", "泰语": "Thai",
             "印尼语": "Indonesian", "日语": "Japanese", "韩语": "Korean", "阿拉伯语": "Arabic"}

# 设备选项：LLM = Talker + Predictor (GGUF/llama.cpp)；ONNX 组件 = 解码器/编解码器/说话人编码器
LLM_DEVICES = ["GPU", "CPU"]
ONNX_PROVIDERS = ["CUDA", "CPU"]


def _hook_drop_folder(entry, var):
    """文件夹输入框拖拽: 文件夹直填，文件取父目录"""
    def on_drop(file_list):
        # windnd 回调在 UI 线程 (消息循环)，可直接更新变量
        p = file_list[0].decode("gbk", errors="replace")
        if os.path.isfile(p):
            p = os.path.dirname(p)
        var.set(p)
    windnd.hook_dropfiles(entry, func=on_drop)


def _hook_drop_file(entry, var):
    """文件输入框拖拽: 只收文件"""
    def on_drop(file_list):
        p = file_list[0].decode("gbk", errors="replace")
        if os.path.isfile(p):
            var.set(p)
    windnd.hook_dropfiles(entry, func=on_drop)


class CloneTab(ttkb.Frame):
    """语音克隆页：载入区 + 推理区"""

    def __init__(self, master):
        super().__init__(master, padding=10)
        self._build_load_group()
        self._build_infer_group()
        self.columnconfigure(0, weight=1)

        # 引擎与生成状态 (worker 线程写，UI 线程经 ui_queue 读)
        self.engine = None
        self.generating = False
        self.cancel_event = None
        self._gen_thread = None
        self.ui_queue = queue.Queue()
        # 载入区需随载入状态锁定的控件：由 _build_load_group 收集 (由 _set_loaded 刷新)

    # ---------- 载入区 ----------

    def _build_load_group(self):
        group = ttkb.Labelframe(self, text="载入", padding=8)
        group.grid(row=0, column=0, sticky=EW)
        group.columnconfigure(1, weight=1)

        ttkb.Label(group, text="模型文件夹").grid(row=0, column=0, sticky=W, padx=(0, 10), pady=4)
        self.model_dir = tk.StringVar(value="model-base")
        self.model_entry = ttkb.Entry(group, textvariable=self.model_dir)
        self.model_entry.grid(row=0, column=1, sticky=EW, pady=4)
        _hook_drop_folder(self.model_entry, self.model_dir)
        browse_btn = ttkb.Button(group, text="浏览", command=self.on_browse_model, width=14)
        browse_btn.grid(row=0, column=2, padx=(8, 0), pady=4)

        # 上下文 + 设备选项一行，全部从左排
        opt_row = ttkb.Frame(group)
        opt_row.grid(row=1, column=0, columnspan=2, sticky=EW, pady=4)
        ttkb.Label(opt_row, text="上下文大小").pack(side=LEFT, padx=(0, 10))
        self.n_ctx = tk.StringVar(value="2048")
        self.n_ctx_entry = ttkb.Entry(opt_row, textvariable=self.n_ctx, width=8)
        self.n_ctx_entry.pack(side=LEFT, padx=(0, 24))
        ttkb.Label(opt_row, text="LLM 设备").pack(side=LEFT, padx=(0, 10))
        self.llm_device = tk.StringVar(value=LLM_DEVICES[0])
        self.llm_combo = ttkb.Combobox(opt_row, textvariable=self.llm_device, values=LLM_DEVICES,
                                       state="readonly", width=8)
        self.llm_combo.pack(side=LEFT, padx=(0, 24))
        ttkb.Label(opt_row, text="ONNX 组件").pack(side=LEFT, padx=(0, 10))
        self.onnx_provider = tk.StringVar(value=ONNX_PROVIDERS[0])
        self.onnx_combo = ttkb.Combobox(opt_row, textvariable=self.onnx_provider, values=ONNX_PROVIDERS,
                                        state="readonly", width=9)
        self.onnx_combo.pack(side=LEFT)
        self._load_fields = [self.model_entry, self.n_ctx_entry, self.llm_combo, self.onnx_combo, browse_btn]

        # 文本随状态切换：未载入叫"载入"，载入后叫"卸载"（on_load_toggle 里改）
        self.load_btn = ttkb.Button(group, text="载入", command=self.on_load_toggle, width=14, bootstyle=SUCCESS)
        self.load_btn.grid(row=1, column=2, sticky=E, pady=4)

    # ---------- 推理区 ----------

    def _build_infer_group(self):
        group = ttkb.Labelframe(self, text="推理", padding=8)
        group.grid(row=1, column=0, sticky=NSEW, pady=(10, 0))
        group.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)
        group.rowconfigure(2, weight=1)  # 多余空间全部给任务文本框

        # 克隆源
        ttkb.Label(group, text="克隆源").grid(row=0, column=0, sticky=W, padx=(0, 10), pady=4)
        self.clone_source = tk.StringVar(value="output/elaborate/Vivian.json")
        source_entry = ttkb.Entry(group, textvariable=self.clone_source)
        source_entry.grid(row=0, column=1, sticky=EW, pady=4)
        _hook_drop_file(source_entry, self.clone_source)
        ttkb.Button(group, text="选择克隆源", command=self.on_pick_source, width=14).grid(row=0, column=2, padx=(8, 0), pady=4)

        # 参考文本: 音频克隆源的转写；留空则零样本克隆 (json 源自带转写，忽略此框)
        ttkb.Label(group, text="参考文本", width=12).grid(row=1, column=0, sticky=W, padx=(0, 10), pady=4)
        self.ref_text = tk.StringVar()
        ttkb.Entry(group, textvariable=self.ref_text).grid(row=1, column=1, columnspan=2, sticky=EW, pady=4)

        # 任务文本
        text_group = ttkb.Labelframe(group, text="输入文本（每行一个任务，多路批量生成，# 开头为注释）", padding=5)
        text_group.grid(row=2, column=0, columnspan=3, sticky=NSEW, pady=(8, 3))
        text_group.columnconfigure(0, weight=1)
        text_group.rowconfigure(0, weight=1)
        self.task_text = tk.Text(text_group, height=14, wrap="none", relief="flat",
                                 padx=6, pady=4, highlightthickness=1,
                                 highlightbackground="#c9c9c9", highlightcolor="#1a7f8e")
        ybar = ttkb.Scrollbar(text_group, orient=VERTICAL, command=self.task_text.yview)
        self.task_text.configure(yscrollcommand=ybar.set)
        self.task_text.grid(row=0, column=0, sticky=NSEW)
        ybar.grid(row=0, column=1, sticky=NS)
        self.task_text.insert("1.0",
                              "今天天气真不错，我们一起去公园走走吧。\n"
                              "人工智能的发展速度令人惊叹，短短几年就改变了很多行业。\n"
                              "清晨的菜市场总是热闹非凡，吆喝声此起彼伏。\n"
                              "学习新语言最重要的是动手实践，写出第一个程序就有成就感。")

        # 输出文件夹
        ttkb.Label(group, text="输出文件夹").grid(row=3, column=0, sticky=W, padx=(0, 10), pady=4)
        self.output_dir = tk.StringVar(value="./output/clone")
        output_entry = ttkb.Entry(group, textvariable=self.output_dir)
        output_entry.grid(row=3, column=1, sticky=EW, pady=4)
        _hook_drop_folder(output_entry, self.output_dir)
        ttkb.Button(group, text="浏览", command=self.on_browse_output, width=14).grid(row=3, column=2, padx=(8, 0), pady=4)

        self._build_param_grid(group)
        self._build_control_row(group)

    def _build_param_grid(self, group):
        pg = ttkb.Labelframe(group, text="参数", padding=8)
        pg.grid(row=4, column=0, columnspan=3, sticky=EW, pady=8)
        # 三个输入列平分窗口变宽多出的空间
        for col in (1, 3, 5):
            pg.columnconfigure(col, weight=1)

        self.param_vars = {}
        # 三列布局：列1 语言/最大步数，列2 Talker，列3 Predictor
        # (行, 标签列, 键, 标签, 默认值, 控件类型) —— 键对应 TTSConfig 字段
        entries = [
            (0, 0, "language", "语言", next(iter(LANGUAGES)), "combo"),
            (1, 0, "max_steps", "最大步数", "300", "entry"),
            (0, 2, "temperature", "Talker 温度", "0.9", "entry"),
            (1, 2, "seed", "Talker 种子", "42", "entry"),
            (0, 4, "sub_temperature", "Predictor 温度", "0.9", "entry"),
            (1, 4, "sub_seed", "Predictor 种子", "45", "entry"),
        ]
        for row, col, key, label, default, kind in entries:
            # sticky=EW：输入框填满所在列，随窗口变宽而拉宽；width 只是最小尺寸
            if kind == "combo":
                var = tk.StringVar(value=default)
                # combobox 自带箭头和内边距，width=8 才与 width=10 的 entry 等宽（实测 163/164px）
                ttkb.Combobox(pg, textvariable=var, values=list(LANGUAGES), state="readonly", width=8).grid(row=row + 1, column=col + 1, sticky=EW, pady=3)
            else:
                var = tk.StringVar(value=default)
                ttkb.Entry(pg, textvariable=var, width=10).grid(row=row + 1, column=col + 1, sticky=EW, padx=(0, 18), pady=3)
            ttkb.Label(pg, text=label, width=12).grid(row=row + 1, column=col, sticky=W, padx=(0, 6), pady=3)
            self.param_vars[key] = var

    def _build_control_row(self, group):
        bar = ttkb.Frame(group)
        bar.grid(row=5, column=0, columnspan=3, sticky=EW)
        self.start_btn = ttkb.Button(bar, text="开始生成", command=self.on_start_stop, width=14,
                                     bootstyle=PRIMARY, state="disabled")  # 引擎载入前不可生成
        self.start_btn.pack(side=RIGHT)
        ttkb.Button(bar, text="打开输出文件夹", command=self.on_open_output, width=14).pack(side=RIGHT, padx=(0, 8))
        # 并发路数：每轮同时生成的路数，任务文本按行拆分后按此分批
        self.n_paths = tk.IntVar(value=32)
        ttkb.Label(bar, text="并发路数").pack(side=LEFT, padx=(0, 6))
        ttkb.Spinbox(bar, from_=1, to=32, textvariable=self.n_paths, width=5).pack(side=LEFT)

    # ---------- 事件管道 ----------

    def bind_feedback(self, status_var, progress, log_tab):
        """main() 创建完状态栏与日志页后回接"""
        self.status_var = status_var
        self.progress = progress
        self.log_tab = log_tab
        logger.addHandler(QueueLogHandler(self.ui_queue))
        # inference 里的 print（引擎初始化耗时等）也导入日志页；stderr 不动，异常栈走控制台/logger 文件
        import sys
        sys.stdout = PrintRedirect(self.ui_queue, sys.stdout)
        self.after(100, self._poll)

    def _poll(self):
        """UI 线程: 拉取事件队列 (日志/状态/进度/按钮态)

        核心约束：worker 线程只往 ui_queue 塞事件，所有 UI 控件更新都
        在这里（UI 线程）执行，禁止在其他线程直接操作 tkinter 控件。
        """
        if not self.winfo_exists():
            return  # 窗口已销毁，断开 after 轮询链（销毁时序处理）
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

    def _set_loaded(self, ok: bool):
        """载入成功: 按钮变卸载、载入区锁定；失败/卸载: 还原"""
        self.load_btn.configure(text="卸载" if ok else "载入", state="normal")
        for w in self._load_fields:
            w.configure(state="disabled" if ok else "normal")
        # 生成按钮只在引擎就绪时可用；失败/卸载后禁用
        self.start_btn.configure(state="normal" if ok else "disabled")

    def _set_generating(self, on: bool):
        """生成中: 开始变停止、卸载按钮禁用（须先停止再卸载）"""
        self.generating = on
        self.start_btn.configure(text="停止" if on else "开始生成",
                                 bootstyle=DANGER if on else PRIMARY, state="normal")
        self.load_btn.configure(state="disabled" if on else "normal")
        if not on:
            self.progress.configure(value=0)

    # ---------- 载入/卸载 ----------

    def on_browse_model(self):
        path = filedialog.askdirectory(title="选择模型文件夹")
        if path:
            self.model_dir.set(path)

    def on_browse_output(self):
        path = filedialog.askdirectory(title="选择输出文件夹")
        if path:
            self.output_dir.set(path)

    def on_pick_source(self):
        path = filedialog.askopenfilename(title="选择克隆源 (.wav / .json)", filetypes=CLONE_SOURCE_TYPES)
        if path:
            self.clone_source.set(path)

    def on_load_toggle(self):
        if self.generating:
            return
        if self.engine is not None:
            self.engine.shutdown()
            self.engine = None
            self._set_loaded(False)
            self.ui_queue.put(("status", "已卸载"))
            return
        # tkinter 变量必须在 UI 线程读——参数在此读好再传给后台线程
        params = dict(model_dir=self.model_dir.get(),
                      llm=self.llm_device.get(),
                      onnx=self.onnx_provider.get())
        self.load_btn.configure(state="disabled")
        self.start_btn.configure(state="disabled")  # 载入中禁止生成
        self.ui_queue.put(("status", "正在载入模型…"))
        threading.Thread(target=self._load_worker, kwargs=params, daemon=True).start()

    def _load_worker(self, model_dir, llm, onnx):
        """后台线程: 建引擎 (进程内解码器)。参数在 UI 线程读好传入。"""
        try:
            t0 = time.time()
            eng = TTSEngine(model_dir=model_dir, onnx_provider=onnx,
                            llm_use_gpu=(llm == "GPU"), subprocess_decoder=False)
            self.engine = eng if eng else None
            self.ui_queue.put(("loaded", bool(eng)))
            if eng:
                self.ui_queue.put(("status", f"引擎就绪 (耗时 {time.time() - t0:.1f}s)"))
            else:
                self.ui_queue.put(("status", "载入失败，详见日志"))
        except Exception as e:
            self.engine = None
            logger.exception("载入失败")
            self.ui_queue.put(("loaded", False))
            self.ui_queue.put(("error", f"载入异常: {e}"))

    # ---------- 生成/停止 ----------

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
        lines = [l.strip() for l in self.task_text.get("1.0", "end").splitlines()]
        lines = [l for l in lines if l and not l.startswith("#")]  # 空行与 # 注释行剔除
        if not lines:
            self.ui_queue.put(("error", "任务文本为空")); return

        # tkinter 变量必须在 UI 线程读——参数在此读好再传给后台线程
        try:
            cfg = TTSConfig(
                max_steps=int(self.param_vars["max_steps"].get()),
                temperature=float(self.param_vars["temperature"].get()),
                seed=int(self.param_vars["seed"].get()),
                sub_temperature=float(self.param_vars["sub_temperature"].get()),
                sub_seed=int(self.param_vars["sub_seed"].get()),
                streaming=False,
            )
            n_ctx = int(self.n_ctx.get())
            n_paths = max(1, int(self.n_paths.get()))
        except ValueError:
            self.ui_queue.put(("error", "参数格式有误：步数/种子须为整数，温度须为数字"))
            return
        params = dict(
            source=source, lines=lines,
            ref_text=self.ref_text.get().strip(),
            language=LANGUAGES[self.param_vars["language"].get()],
            cfg=cfg,
            n_ctx=n_ctx,
            n_paths=n_paths,
            out_root=self.output_dir.get(),
            engine=self.engine,
        )
        self.cancel_event = threading.Event()
        self._set_generating(True)
        self.ui_queue.put(("progress", 0, len(lines)))
        self._gen_thread = threading.Thread(target=self._gen_worker, kwargs=params, daemon=True)
        self._gen_thread.start()

    def _gen_worker(self, engine, source, lines, ref_text, language, cfg, n_ctx, n_paths, out_root):
        """后台线程: 准备锚点 -> 分批 clone_batch -> 按序号落盘 wav+json"""
        try:
            # 克隆源分流: json 自带转写；音频源按参考文本定模式
            zero_shot = False
            src_path = Path(source)
            if src_path.suffix.lower() == ".json":
                if ref_text:
                    logger.info("[GUI] json 克隆源自带转写，忽略参考文本框")
            elif src_path.suffix.lower() in (".wav", ".mp3", ".flac", ".m4a", ".opus"):
                if not ref_text:
                    json_pair = src_path.with_suffix(".json")
                    if json_pair.exists():
                        # 有同名锚点存档: 用它做普通克隆 (带转写，效果优于零样本)
                        logger.info(f"[GUI] 参考文本为空，改用同名锚点存档 {json_pair.name}")
                        source = str(json_pair)
                    else:
                        logger.info("[GUI] 参考文本为空，走零样本克隆 (仅用音色向量)")
                        zero_shot = True
            else:
                self.ui_queue.put(("error", f"不支持的克隆源类型: {src_path.suffix}"))
                return
            voice = prepare_voice(engine, engine.tokenizer, source,
                                  text=ref_text if (not zero_shot and src_path.suffix.lower() != ".json") else None)
            if voice is None:
                self.ui_queue.put(("error", "克隆源准备失败，详见日志"))
                return
            runner = BatchRunner(engine, n_ctx_per_seq=n_ctx, cancel_event=self.cancel_event)
            out_dir = Path(out_root) / datetime.now().strftime("%Y%m%d-%H%M%S")
            out_dir.mkdir(parents=True, exist_ok=True)

            done = ok = idx = 0
            total_frames = 0
            t_start = time.time()
            batches = [lines[i:i + n_paths] for i in range(0, len(lines), n_paths)]
            for bi, batch in enumerate(batches):
                if self.cancel_event.is_set():
                    break
                self.ui_queue.put(("status", f"批次 {bi + 1}/{len(batches)}（{len(batch)} 路）生成中…"))
                t0 = time.time()
                tasks = [(text, voice, language, zero_shot, cfg) for text in batch]
                results = runner.clone_batch(tasks)
                dt = time.time() - t0
                frames = sum(r.codes.shape[0] for r in results if r is not None)
                total_frames += frames
                audio_s = frames / 12.5  # 每秒 12.5 帧
                if frames > 0:
                    logger.info(f"[GUI] 批次 {bi + 1}/{len(batches)}: {len(batch)} 路, "
                                f"音频 {audio_s:.1f}s, 壁钟 {dt:.2f}s, RTF {dt / audio_s:.3f}")
                # 撞最大步数 = 未自然收束 (EOS)，该路大概率是噪声
                n_hit = 0
                warned = False
                for r in results:
                    if r is not None and r.stats.total_steps >= cfg.max_steps:
                        n_hit += 1
                        if not warned:  # 每批最多提示一次，避免刷屏
                            logger.warning(f"[GUI] 该批次有路达到最大步数未自然结束 (total_steps={r.stats.total_steps})，输出可能异常")
                            warned = True
                if results and n_hit == len(results):
                    # 整批全部撞顶: 克隆源大概率有问题，主动停止后续批次
                    logger.error("[GUI] 本批全部任务达到最大步数未收束，判定克隆源异常，停止后续批次")
                    self.ui_queue.put(("error", "克隆源异常：全部任务未自然收束，已停止后续批次"))
                    self.cancel_event.set()
                    break
                for r in results:
                    idx += 1
                    if r is not None:
                        r.save(str(out_dir / f"{idx:03d}.wav"))
                        r.save(str(out_dir / f"{idx:03d}.json"))
                        ok += 1
                    else:
                        logger.warning(f"[GUI] 第 {idx} 路已取消，未落盘")
                    done += 1
                    self.ui_queue.put(("progress", done, len(lines)))

            dt_total = time.time() - t_start
            total_audio_s = total_frames / 12.5  # 每秒 12.5 帧
            if self.cancel_event.is_set():
                # 停止时不算 RTF，数据不完整
                self.ui_queue.put(("status", f"已停止 {ok}/{len(lines)} 路 → {out_dir}"))
            else:
                rtf = f", RTF {dt_total / total_audio_s:.3f}" if total_audio_s > 0 else ""
                self.ui_queue.put(("status", f"完成 {ok}/{len(lines)} 路{rtf} → {out_dir}"))
        except Exception as e:
            logger.exception("生成失败")
            self.ui_queue.put(("error", f"生成异常: {e}"))
        finally:
            self.ui_queue.put(("gen_done",))  # 单点: 任何结束路径都恢复按钮态

    def on_open_output(self):
        path = os.path.abspath(self.output_dir.get())
        os.makedirs(path, exist_ok=True)  # 用户的意图就是要这个目录，不存在就建出来
        os.startfile(path)

    def shutdown(self):
        """关窗前收尾: 停止生成 -> 卸载引擎。join 有超时，保证快速退出。"""
        if self.cancel_event is not None:
            self.cancel_event.set()
        if self._gen_thread is not None and self._gen_thread.is_alive():
            self._gen_thread.join(timeout=10)
            if self._gen_thread.is_alive():
                # 超时说明 worker 卡在 GPU 调用上；进程即将退出，daemon 线程随之消亡，
                # 此时再 shutdown 会与推理并发释放显存（use-after-free），放弃手动清理
                logger.warning("[GUI] 生成线程停止超时，跳过引擎清理直接退出")
                return
        if self.engine is not None:
            self.engine.shutdown()
            self.engine = None
