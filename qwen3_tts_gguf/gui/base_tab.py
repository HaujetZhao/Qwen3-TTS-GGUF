"""TTS 页面基类：载入区 / 事件管道 / 参数网格 / 控制行 / 任务文本框。

克隆、自定义音色、音色设计三页共有的骨架都在这里。子类提供四件事：
- _build_source_rows: 推理区顶部的源控件（克隆源 / 说话人 / 音色描述），返回占用的行数
- _read_source:       UI 线程读取并校验源参数（开始生成时调用），失败自行报错并返回 None
- _make_tasks:        worker 线程构造批量任务，返回错误字符串表示源准备失败
- _run_batch:         调用对应的 BatchRunner 批量接口

核心约束：worker 线程只往 ui_queue 塞事件，所有 UI 控件更新都在
_poll（UI 线程）里执行，禁止在其他线程直接操作 tkinter 控件。
"""
import os
import queue
import re
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

from .log_tab import PrintRedirect, QueueLogHandler

# 设备选项：LLM = Talker + Predictor (GGUF/llama.cpp)；ONNX 组件 = 解码器/编解码器/说话人编码器
LLM_DEVICES = ["GPU", "CPU"]
ONNX_PROVIDERS = ["CUDA", "CPU"]

# 全量语言（引擎 LANGUAGE_MAP 的 12 种）：自定义音色页用，克隆页用自己的 8 种子集
LANGS_ALL = {
    "中文": "Chinese", "英语": "English", "德语": "German", "西班牙语": "Spanish",
    "日语": "Japanese", "法语": "French", "韩语": "Korean", "俄语": "Russian",
    "意大利语": "Italian", "葡萄牙语": "Portuguese", "四川话": "sichuan_dialect", "北京话": "beijing_dialect",
}

# 默认任务文本存在 texts/<页名>.txt，改文案直接编辑文件即可
TEXTS_DIR = Path(__file__).parent / "texts"


def load_task_text(name):
    """读各页默认任务文本（去首尾空白）"""
    return (TEXTS_DIR / f"{name}.txt").read_text(encoding="utf-8").strip()


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


GROUP_SEP = re.compile(r"^\s*={3,}\s*$")   # 三个以上等号的行: 分组
BLOCK_SEP = re.compile(r"^\s*-{3,}\s*$")   # 三个以上减号的行: 组内分块


def text_slug(text, n_words=10):
    """取 target text 前 n 个词作文件名: CJK 每字一词，英文每单词一词。

    只收集 CJK 字符与英数单词，文件名非法字符（<>:"/\\|?* 等）天然不会进入。
    拼接时拉丁词之间补空格，CJK 字之间直接相连。
    """
    words = []
    i = 0
    while i < len(text) and len(words) < n_words:
        ch = text[i]
        if ch.isascii() and (ch.isalpha() or ch.isdigit()):
            j = i
            while j < len(text) and text[j].isascii() and (text[j].isalpha() or text[j].isdigit()):
                j += 1
            words.append(text[i:j])
            i = j
        elif ord(ch) >= 0x2E80:  # CJK 及日文/全角等区: 每字一词，标点也占位
            words.append(ch)
            i += 1
        else:
            i += 1
    out = ""
    for w in words:
        if out and (out[-1].isascii() or w[0].isascii()):
            out += " "
        out += w
    return out


def parse_task_groups(text):
    """按 === 行分组，组内按 --- 行分块（自定义音色/音色设计的任务格式）。

    组内分出 2 块以上: 前面的块是 instruct，最后一块是 target；
    只有 1 块: 该块就是 target。空行与 # 注释行剔除。
    返回 (instruct, target) 列表，instruct 为 None 表示无描述。
    """
    tasks = []
    group = [[]]  # 当前组: 块列表，每块为行列表

    def flush():
        blocks = [b for b in group if b]
        if not blocks:
            return
        if len(blocks) >= 2:
            instruct = "\n".join("\n".join(b) for b in blocks[:-1])
            target = "\n".join(blocks[-1])
        else:
            instruct, target = None, "\n".join(blocks[0])
        tasks.append((instruct, target))

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if GROUP_SEP.match(line):
            flush()
            group = [[]]
        elif BLOCK_SEP.match(line):
            group.append([])
        else:
            group[-1].append(line)
    flush()
    return tasks


class TTSPageBase(ttkb.Frame):
    """单模式 TTS 页骨架：载入区 + 推理区"""

    MODEL_DEFAULT = ""      # 模型文件夹默认值，子类必填
    OUT_DEFAULT = ""        # 输出文件夹默认值，子类必填
    LANGS = LANGS_ALL       # 语言下拉（显示中文 -> 引擎值），子类可覆写
    TEXT_HINT = "输入文本（每行一个任务，多路批量生成）"
    TASK_TEXT_DEFAULT = load_task_text("clone")  # 取自个人笔记的 16 句金句，每行一个任务

    _log_hooked = False     # 日志 handler / stdout 重定向全局只挂一次

    def __init__(self, master):
        super().__init__(master, padding=10)
        self._build_load_group()
        self._build_infer_group()
        self.columnconfigure(0, weight=1)

        # 引擎与生成状态 (worker 线程写，UI 线程经 ui_queue 读)
        self.engine = None
        self.generating = False
        self.loading = False    # 载入进行中（单模型模式据此跳过载入/卸载中的页）
        self.cancel_event = None
        self._gen_thread = None
        self.ui_queue = queue.Queue()

    # ---------- 载入区 ----------

    def _build_load_group(self):
        group = ttkb.Labelframe(self, text="载入", padding=8)
        group.grid(row=0, column=0, sticky=EW)
        group.columnconfigure(1, weight=1)

        ttkb.Label(group, text="模型文件夹").grid(row=0, column=0, sticky=W, padx=(0, 10), pady=4)
        self.model_dir = tk.StringVar(value=self.MODEL_DEFAULT)
        self.model_entry = ttkb.Entry(group, textvariable=self.model_dir)
        self.model_entry.grid(row=0, column=1, sticky=EW, pady=4)
        _hook_drop_folder(self.model_entry, self.model_dir)
        browse_btn = ttkb.Button(group, text="浏览", command=self.on_browse_model, width=14)
        browse_btn.grid(row=0, column=2, padx=(8, 0), pady=4)

        # 上下文 + 设备选项一行，全部从左排
        opt_row = ttkb.Frame(group)
        opt_row.grid(row=1, column=0, columnspan=2, sticky=EW, pady=4)
        ttkb.Label(opt_row, text="上下文大小").pack(side=LEFT, padx=(0, 10))
        self.n_ctx = tk.StringVar(value="512")
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

        row = self._build_source_rows(group)

        # 任务文本，多余空间全部给它
        text_group = ttkb.Labelframe(group, text=self.TEXT_HINT, padding=5)
        text_group.grid(row=row, column=0, columnspan=3, sticky=NSEW, pady=(8, 3))
        group.rowconfigure(row, weight=1)
        text_group.columnconfigure(0, weight=1)
        text_group.rowconfigure(0, weight=1)
        self.task_text = tk.Text(text_group, height=14, wrap="none", relief="flat",
                                 padx=6, pady=4, highlightthickness=1,
                                 highlightbackground="#c9c9c9", highlightcolor="#1a7f8e")
        ybar = ttkb.Scrollbar(text_group, orient=VERTICAL, command=self.task_text.yview)
        self.task_text.configure(yscrollcommand=ybar.set)
        self.task_text.grid(row=0, column=0, sticky=NSEW)
        ybar.grid(row=0, column=1, sticky=NS)
        self.task_text.insert("1.0", self.TASK_TEXT_DEFAULT)
        row += 1

        # 输出文件夹
        ttkb.Label(group, text="输出文件夹").grid(row=row, column=0, sticky=W, padx=(0, 10), pady=4)
        self.output_dir = tk.StringVar(value=self.OUT_DEFAULT)
        output_entry = ttkb.Entry(group, textvariable=self.output_dir)
        output_entry.grid(row=row, column=1, sticky=EW, pady=4)
        _hook_drop_folder(output_entry, self.output_dir)
        ttkb.Button(group, text="浏览", command=self.on_browse_output, width=14).grid(row=row, column=2, padx=(8, 0), pady=4)
        row += 1

        self._build_param_grid(group, row)
        self._build_control_row(group, row + 1)

    def _build_source_rows(self, group):
        """推理区顶部源控件（克隆源/说话人等），子类按需覆写，返回占用的行数"""
        return 0

    # ---------- 参数与控制行 ----------

    def _param_entries(self):
        """参数网格布局: (行, 标签列, 键, 标签, 默认值, 控件类型) —— 键对应 TTSConfig 字段"""
        return [
            (0, 0, "language", "语言", next(iter(self.LANGS)), "combo"),
            (1, 0, "max_steps", "最大步数", "300", "entry"),
            (0, 2, "temperature", "Talker 温度", "0.8", "entry"),
            (1, 2, "seed", "Talker 种子", "42", "entry"),
            (0, 4, "sub_temperature", "Predictor 温度", "0.2", "entry"),
            (1, 4, "sub_seed", "Predictor 种子", "45", "entry"),
        ]

    def _build_param_grid(self, group, row):
        pg = ttkb.Labelframe(group, text="参数", padding=8)
        pg.grid(row=row, column=0, columnspan=3, sticky=EW, pady=8)
        # 三个输入列平分窗口变宽多出的空间
        for col in (1, 3, 5):
            pg.columnconfigure(col, weight=1)

        self.param_vars = {}
        for r, col, key, label, default, kind in self._param_entries():
            var = tk.StringVar(value=default)
            # sticky=EW：输入框填满所在列，随窗口变宽而拉宽；width 只是最小尺寸
            if kind == "combo":
                # combobox 自带箭头和内边距，width=8 才与 width=10 的 entry 等宽（实测 163/164px）
                ttkb.Combobox(pg, textvariable=var, values=list(self.LANGS), state="readonly", width=8) \
                    .grid(row=r + 1, column=col + 1, sticky=EW, pady=3)
            else:
                ttkb.Entry(pg, textvariable=var, width=10).grid(row=r + 1, column=col + 1, sticky=EW, padx=(0, 18), pady=3)
            ttkb.Label(pg, text=label, width=12).grid(row=r + 1, column=col, sticky=W, padx=(0, 6), pady=3)
            self.param_vars[key] = var

    def _build_control_row(self, group, row):
        bar = ttkb.Frame(group)
        bar.grid(row=row, column=0, columnspan=3, sticky=EW)
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
        """main() 创建完状态栏与日志页后回接。三页共用同一状态栏/进度条/日志页"""
        self.status_var = status_var
        self.progress = progress
        self.log_tab = log_tab
        if not TTSPageBase._log_hooked:
            logger.addHandler(QueueLogHandler(self.ui_queue))
            # inference 里的 print（引擎初始化耗时等）也导入日志页；stderr 不动，异常栈走控制台/logger 文件
            import sys
            sys.stdout = PrintRedirect(self.ui_queue, sys.stdout)
            TTSPageBase._log_hooked = True
        self.after(100, self._poll)

    def _poll(self):
        """UI 线程: 拉取事件队列 (日志/状态/进度/按钮态)"""
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
                elif kind == "play_done":
                    self._refresh_play_btn()
                elif kind == "error":
                    self.log_tab.append(f"❌ {ev[1]}")
                    self.status_var.set(f"❌ {ev[1]}")  # 错误同时上状态栏，避免"没反应"
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
        self._refresh_play_btn()

    def _refresh_play_btn(self):
        """播放按钮状态钩子（仅克隆页有播放，其余页为空）"""
        pass

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

    def unload_others(self, current):
        """载入前钩子：单模型模式下由 app 注入，卸载其他页已载入的引擎"""
        pass

    def on_load_toggle(self):
        if self.generating:
            return
        if self.engine is not None:
            self.engine.shutdown()
            self.engine = None
            self._set_loaded(False)
            self.ui_queue.put(("status", "已卸载"))
            return
        self.unload_others(self)  # 单模型模式: 载入前先卸掉其他页的引擎
        # tkinter 变量必须在 UI 线程读——参数在此读好再传给后台线程
        params = self._load_params()
        self.load_btn.configure(state="disabled")
        self.loading = True
        self.ui_queue.put(("status", "正在载入模型…"))
        threading.Thread(target=self._load_worker, kwargs=params, daemon=True).start()

    def _load_params(self):
        """UI 线程读取载入参数（键与 _load_worker 形参对应），子类可覆写增减"""
        return dict(model_dir=self.model_dir.get(),
                    llm=self.llm_device.get(),
                    onnx=self.onnx_provider.get())

    def _create_engine(self, model_dir, llm, onnx):
        """构建引擎实例，子类可覆写调整载入范围"""
        return TTSEngine(model_dir=model_dir, onnx_provider=onnx,
                         llm_use_gpu=(llm == "GPU"), subprocess_decoder=False,
                         chunk_size=64)

    def _load_worker(self, model_dir, onnx, llm=None):
        """后台线程: 建引擎 (进程内解码器)。参数在 UI 线程读好传入。"""
        try:
            t0 = time.time()
            eng = self._create_engine(model_dir, llm, onnx)
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
        finally:
            self.loading = False

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
        text = self.task_text.get("1.0", "end")
        src = self._read_source()
        if src is None:
            return

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
        self.cancel_event = threading.Event()
        self._set_generating(True)
        self._gen_thread = threading.Thread(
            target=self._gen_worker,
            kwargs=dict(engine=self.engine, text=text, cfg=cfg, n_ctx=n_ctx,
                        n_paths=n_paths, out_root=self.output_dir.get(), src=src),
            daemon=True)
        self._gen_thread.start()

    def _read_source(self):
        """UI 线程: 读源参数并校验，失败自行报错并返回 None。无源参数的页不用覆写"""
        return {}

    def _make_tasks(self, text, cfg, src):
        """worker 线程: 解析任务文本构造批量任务，返回错误字符串表示失败。子类实现"""
        raise NotImplementedError

    def _run_batch(self, runner, tasks):
        """调对应的 BatchRunner 批量接口。子类实现"""
        raise NotImplementedError

    def _gen_worker(self, engine, text, cfg, n_ctx, n_paths, out_root, src):
        """后台线程: 解析任务 -> 分批生成 -> 按序号落盘 wav+json"""
        try:
            tasks = self._make_tasks(text, cfg, src)
            if isinstance(tasks, str):  # 源准备失败，字符串即错误信息
                self.ui_queue.put(("error", tasks))
                return
            if not tasks:
                self.ui_queue.put(("error", "任务文本为空"))
                return
            self.ui_queue.put(("progress", 0, len(tasks)))
            runner = BatchRunner(engine, n_ctx_per_seq=n_ctx, cancel_event=self.cancel_event)
            out_dir = Path(out_root) / datetime.now().strftime("%Y%m%d-%H%M%S")
            out_dir.mkdir(parents=True, exist_ok=True)

            done = ok = idx = 0
            total_frames = 0
            t_start = time.time()
            batches = [tasks[i:i + n_paths] for i in range(0, len(tasks), n_paths)]
            for bi, batch in enumerate(batches):
                if self.cancel_event.is_set():
                    break
                self.ui_queue.put(("status", f"批次 {bi + 1}/{len(batches)}（{len(batch)} 路）生成中…"))
                t0 = time.time()
                results = self._run_batch(runner, batch)
                dt = time.time() - t0
                frames = sum(r.codes.shape[0] for r in results if r is not None)
                total_frames += frames
                audio_s = frames / 12.5  # 每秒 12.5 帧
                if frames > 0:
                    logger.info(f"[GUI] 批次 {bi + 1}/{len(batches)}: {len(batch)} 路, "
                                f"音频 {audio_s:.1f}s, 壁钟 {dt:.2f}s, RTF {dt / audio_s:.3f}")
                # 撞最大步数 = 未自然收束 (EOS)，可能是文本过长；照常落盘，只提示
                for r in results:
                    if r is not None and r.stats.total_steps >= cfg.max_steps:
                        logger.warning(f"[GUI] 有任务达到最大步数未自然结束 (total_steps={r.stats.total_steps})，音频可能不完整")
                        break  # 每批最多提示一次，避免刷屏
                for r in results:
                    idx += 1
                    if r is not None:
                        # 文件名: 序号 - target text 前 10 词（tasks 各模式的 [0] 都是 target text）
                        slug = text_slug(tasks[idx - 1][0])
                        stem = f"{idx:03d}-{slug}" if slug else f"{idx:03d}"
                        r.save(str(out_dir / f"{stem}.wav"))
                        r.save(str(out_dir / f"{stem}.json"))
                        ok += 1
                    else:
                        logger.warning(f"[GUI] 第 {idx} 路已取消，未落盘")
                    done += 1
                    self.ui_queue.put(("progress", done, len(tasks)))

            dt_total = time.time() - t_start
            total_audio_s = total_frames / 12.5  # 每秒 12.5 帧
            if self.cancel_event.is_set():
                # 停止时不算 RTF，数据不完整
                self.ui_queue.put(("status", f"已停止 {ok}/{len(tasks)} 路 → {out_dir}"))
            else:
                rtf = f", RTF {dt_total / total_audio_s:.3f}" if total_audio_s > 0 else ""
                self.ui_queue.put(("status", f"完成 {ok}/{len(tasks)} 路{rtf} → {out_dir}"))
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
