"""
Qwen3-TTS GGUF 图形界面（ttkbootstrap）。

当前只搭建页面骨架，功能后续再填：
- 克隆 Tab：载入区（模型/后端/上下文）+ 推理区（克隆源/任务文本/输出/参数）
- 自定义音色 / 音色设计 Tab：占位

克隆走批量生成，不做流式。
"""
import tkinter as tk
from tkinter import filedialog, ttk

import ttkbootstrap as ttkb
from ttkbootstrap.constants import *

UI_SCALE = 1.25        # 整体缩放系数
DEBUG_TOPMOST = True   # 调试阶段保持窗口占据前台

# 克隆源支持的文件类型
CLONE_SOURCE_TYPES = [("克隆源", "*.wav *.json"), ("所有文件", "*.*")]

# 语言下拉框：显示中文，内部值与引擎一致
LANGUAGES = {"中文": "Chinese", "英语": "English", "越南语": "Vietnamese", "泰语": "Thai",
             "印尼语": "Indonesian", "日语": "Japanese", "韩语": "Korean", "阿拉伯语": "Arabic"}

# 设备选项：LLM = Talker + Predictor (GGUF/llama.cpp)；ONNX 组件 = 解码器/编解码器/说话人编码器
LLM_DEVICES = ["GPU", "CPU"]
ONNX_PROVIDERS = ["CUDA", "CPU"]


def pad_title(title, width=6):
    """用全角空格把 tab 标题补到同样宽度，使三个 tab 等宽"""
    pad = width - len(title)
    return "　" * (pad // 2) + title + "　" * (pad - pad // 2)


def init_style(root):
    """分组框描边 + 标题着色，否则 flatly 下 Labelframe 只剩一行悬空文字"""
    style = ttkb.Style.get_instance()
    style.configure("TLabelframe", borderwidth=1, relief="solid", bordercolor="#c9c9c9")
    style.configure("TLabelframe.Label", foreground="#1a7f8e", font=("微软雅黑", 10, "bold"))


def highlight_active_tab():
    """选中 tab 用普通按钮同款深色。必须等 Notebook 创建完再调——ttkbootstrap 建控件时会重刷样式"""
    style = ttkb.Style.get_instance()
    style.map("TNotebook.Tab", background=[("selected", "#2c3e50")], foreground=[("selected", "#ffffff")])


class CloneTab(ttkb.Frame):
    """语音克隆页：载入区 + 推理区"""

    def __init__(self, master):
        super().__init__(master, padding=10)
        self._build_load_group()
        self._build_infer_group()
        self.columnconfigure(0, weight=1)

    # ---------- 载入区 ----------

    def _build_load_group(self):
        group = ttkb.Labelframe(self, text="载入", padding=8)
        group.grid(row=0, column=0, sticky=EW)
        group.columnconfigure(1, weight=1)

        ttkb.Label(group, text="模型文件夹").grid(row=0, column=0, sticky=W, padx=(0, 10), pady=4)
        self.model_dir = tk.StringVar(value="model-base")
        ttkb.Entry(group, textvariable=self.model_dir).grid(row=0, column=1, sticky=EW, pady=4)
        ttkb.Button(group, text="浏览", command=self.on_browse_model, width=14).grid(row=0, column=2, padx=(8, 0), pady=4)

        # 上下文 + 设备选项一行，全部从左排
        opt_row = ttkb.Frame(group)
        opt_row.grid(row=1, column=0, columnspan=2, sticky=EW, pady=4)
        ttkb.Label(opt_row, text="上下文大小").pack(side=LEFT, padx=(0, 10))
        self.n_ctx = tk.StringVar(value="2048")
        ttkb.Entry(opt_row, textvariable=self.n_ctx, width=8).pack(side=LEFT, padx=(0, 24))
        ttkb.Label(opt_row, text="LLM 设备").pack(side=LEFT, padx=(0, 10))
        self.llm_device = tk.StringVar(value=LLM_DEVICES[0])
        ttkb.Combobox(opt_row, textvariable=self.llm_device, values=LLM_DEVICES,
                      state="readonly", width=8).pack(side=LEFT, padx=(0, 24))
        ttkb.Label(opt_row, text="ONNX 组件").pack(side=LEFT, padx=(0, 10))
        self.onnx_provider = tk.StringVar(value=ONNX_PROVIDERS[0])
        ttkb.Combobox(opt_row, textvariable=self.onnx_provider, values=ONNX_PROVIDERS,
                      state="readonly", width=9).pack(side=LEFT)

        # 文本随状态切换：未载入叫"载入"，载入后叫"卸载"（on_load_toggle 里改）
        self.load_btn = ttkb.Button(group, text="载入", command=self.on_load_toggle, width=14, bootstyle=SUCCESS)
        self.load_btn.grid(row=1, column=2, sticky=E, pady=4)

    # ---------- 推理区 ----------

    def _build_infer_group(self):
        group = ttkb.Labelframe(self, text="推理", padding=8)
        group.grid(row=1, column=0, sticky=NSEW, pady=(10, 0))
        group.columnconfigure(1, weight=1)
        group.rowconfigure(1, weight=1)  # 多余空间全部给任务文本框
        self.rowconfigure(1, weight=1)

        # 克隆源
        ttkb.Label(group, text="克隆源").grid(row=0, column=0, sticky=W, padx=(0, 10), pady=4)
        self.clone_source = tk.StringVar()
        ttkb.Entry(group, textvariable=self.clone_source).grid(row=0, column=1, sticky=EW, pady=4)
        ttkb.Button(group, text="选择克隆源", command=self.on_pick_source, width=14).grid(row=0, column=2, padx=(8, 0), pady=4)

        # 任务文本
        text_group = ttkb.Labelframe(group, text="输入文本（每行一个任务，多路批量生成）", padding=5)
        text_group.grid(row=1, column=0, columnspan=3, sticky=NSEW, pady=(8, 3))
        text_group.columnconfigure(0, weight=1)
        text_group.rowconfigure(0, weight=1)
        self.task_text = tk.Text(text_group, height=14, wrap="none", relief="flat",
                                 padx=6, pady=4, highlightthickness=1,
                                 highlightbackground="#c9c9c9", highlightcolor="#1a7f8e")
        ybar = ttkb.Scrollbar(text_group, orient=VERTICAL, command=self.task_text.yview)
        self.task_text.configure(yscrollcommand=ybar.set)
        self.task_text.grid(row=0, column=0, sticky=NSEW)
        ybar.grid(row=0, column=1, sticky=NS)

        # 输出文件夹
        ttkb.Label(group, text="输出文件夹").grid(row=2, column=0, sticky=W, padx=(0, 10), pady=4)
        self.output_dir = tk.StringVar(value="./output/clone")
        ttkb.Entry(group, textvariable=self.output_dir).grid(row=2, column=1, sticky=EW, pady=4)
        ttkb.Button(group, text="浏览", command=self.on_browse_output, width=14).grid(row=2, column=2, padx=(8, 0), pady=4)

        self._build_param_grid(group)
        self._build_control_row(group)

    def _build_param_grid(self, group):
        pg = ttkb.Labelframe(group, text="参数", padding=8)
        pg.grid(row=3, column=0, columnspan=3, sticky=EW, pady=8)
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
        bar.grid(row=4, column=0, columnspan=3, sticky=EW)
        self.start_btn = ttkb.Button(bar, text="开始生成", command=self.on_start_stop, width=14, bootstyle=PRIMARY)
        self.start_btn.pack(side=RIGHT)
        ttkb.Button(bar, text="打开输出文件夹", command=self.on_open_output, width=14).pack(side=RIGHT, padx=(0, 8))
        # 并发路数：每轮同时生成的路数，任务文本按行拆分后按此分批
        self.n_paths = tk.IntVar(value=32)
        ttkb.Label(bar, text="并发路数").pack(side=LEFT, padx=(0, 6))
        ttkb.Spinbox(bar, from_=1, to=32, textvariable=self.n_paths, width=5).pack(side=LEFT)

    # ---------- 事件（功能后续再填） ----------

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
        pass

    def on_start_stop(self):
        pass

    def on_open_output(self):
        pass


class PlaceholderTab(ttkb.Frame):
    """占位页：自定义音色 / 音色设计"""

    def __init__(self, master, text):
        super().__init__(master, padding=20)
        ttkb.Label(self, text=f"{text}：待实现，先做克隆", font=("微软雅黑", 12)).pack(expand=True)


def main():
    app = ttkb.Window(title="Qwen3-TTS GGUF", themename="flatly")
    # 在系统 DPI 给定的 scaling 基础上按系数放大（字体与以字符计宽的控件同步变大）
    base = float(app.tk.call("tk", "scaling"))
    app.tk.call("tk", "scaling", base * UI_SCALE)
    init_style(app)
    if DEBUG_TOPMOST:
        app.attributes("-topmost", True)

    app.columnconfigure(0, weight=1)
    app.rowconfigure(0, weight=1)

    nb = ttkb.Notebook(app)
    nb.grid(row=0, column=0, sticky=NSEW, padx=6, pady=6)

    tabs = [
        (CloneTab(nb), "克隆"),
        (PlaceholderTab(nb, "自定义音色"), "自定义音色"),
        (PlaceholderTab(nb, "音色设计"), "音色设计"),
    ]
    for tab, title in tabs:
        nb.add(tab, text=pad_title(title))
    highlight_active_tab()

    # 窗口取内容的自然尺寸（所有控件铺开到舒适状态后由 Tk 计算），不超出屏幕
    app.update_idletasks()
    w, h = app.winfo_reqwidth(), app.winfo_reqheight()
    h = min(h, app.winfo_screenheight() - 80)
    app.geometry(f"{w}x{h}")

    app.mainloop()
