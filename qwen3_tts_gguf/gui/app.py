"""
Qwen3-TTS GGUF 图形界面（ttkbootstrap）——组装层。

- 克隆 Tab：载入区（模型/后端/上下文）+ 推理区（克隆源/任务文本/输出/参数）
- 自定义音色 / 音色设计 Tab：占位
- 日志 Tab：只读日志 + 清除

克隆走批量生成，不做流式。页面实现拆在 clone_tab / log_tab，此处只做样式与组装。
"""
import logging
import queue
import tkinter as tk

import ttkbootstrap as ttkb
from ttkbootstrap.constants import *

from .clone_tab import CloneTab
from .log_tab import LogTab

UI_SCALE = 1.25        # 整体缩放系数
DEBUG_TOPMOST = True   # 调试阶段保持窗口占据前台


def pad_title(title, width=6):
    """用全角空格把 tab 标题补到同样宽度，使四个 tab 等宽"""
    pad = width - len(title)
    return "　" * (pad // 2) + title + "　" * (pad - pad // 2)


class QueueLogHandler(logging.Handler):
    """把 inference 的 logger 记录转发到 GUI 事件队列 (线程安全)"""
    def __init__(self, q: "queue.Queue"):
        super().__init__(logging.INFO)  # GUI 只展示 INFO 及以上，debug 进文件日志
        self.q = q

    def emit(self, record):
        try:
            self.q.put(("log", record.getMessage()))
        except Exception:
            self.handleError(record)


def init_style(root):
    """分组框描边 + 标题着色，否则 flatly 下 Labelframe 只剩一行悬空文字"""
    style = ttkb.Style.get_instance()
    style.configure("TLabelframe", borderwidth=1, relief="solid", bordercolor="#c9c9c9")
    style.configure("TLabelframe.Label", foreground="#1a7f8e", font=("微软雅黑", 10, "bold"))


def highlight_active_tab():
    """选中 tab 用普通按钮同款深色。必须等 Notebook 创建完再调——ttkbootstrap 建控件时会重刷样式"""
    style = ttkb.Style.get_instance()
    style.map("TNotebook.Tab", background=[("selected", "#2c3e50")], foreground=[("selected", "#ffffff")])


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

    def on_close():
        clone_tab.shutdown()   # 停止生成 -> 卸载引擎
        app.destroy()

    # 显式接管关窗：先停生成/卸载引擎，destroy 后 _poll 由 winfo_exists 断开轮询链
    app.protocol("WM_DELETE_WINDOW", on_close)

    # 窗口取内容的自然尺寸（所有控件铺开到舒适状态后由 Tk 计算），不超出屏幕
    app.update_idletasks()
    w, h = app.winfo_reqwidth(), app.winfo_reqheight()
    h = min(h, app.winfo_screenheight() - 80)
    app.geometry(f"{w}x{h}")

    app.mainloop()
