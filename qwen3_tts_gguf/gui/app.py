"""
Qwen3-TTS GGUF 图形界面（ttkbootstrap）——组装层。

- 克隆 / 自定义音色 / 音色设计 Tab：各自载入区（模型/后端/上下文）+ 推理区
- 设置 Tab：全局开关（如单模型模式）
- 日志 Tab：只读日志 + 清除

各模式走批量生成，不做流式。页面实现拆在 clone_tab / custom_tab / design_tab /
settings_tab / log_tab，此处只做样式与组装。
"""
import tkinter as tk

import ttkbootstrap as ttkb
from ttkbootstrap.constants import *

from .clone_tab import CloneTab
from .custom_tab import CustomTab
from .design_tab import DesignTab
from .tools_tab import ToolsTab
from .settings_tab import SettingsTab
from .log_tab import LogTab

UI_SCALE = 1.25        # 整体缩放系数
DEBUG_TOPMOST = False  # 调试阶段可置 True 让窗口保持前台


def pad_title(title, width=6):
    """用全角空格把 tab 标题补到同样宽度，使各 tab 等宽"""
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
    custom_tab = CustomTab(nb)
    design_tab = DesignTab(nb)
    tools_tab = ToolsTab(nb)
    settings_tab = SettingsTab(nb)
    log_tab = LogTab(nb)
    tabs = [
        (clone_tab, "克隆"),
        (custom_tab, "自定义音色"),
        (design_tab, "音色设计"),
        (tools_tab, "工具"),
        (settings_tab, "设置"),
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

    # 带引擎的页（含工具页）：共用状态栏/进度条/日志，参与单模型互斥与关窗收尾
    eng_tabs = (clone_tab, custom_tab, design_tab, tools_tab)

    for tab in eng_tabs:
        tab.bind_feedback(status_var, progress, log_tab)

    def unload_others(current):
        """单模型模式: 在 current 页启动载入前，卸载其他页已载入的引擎。

        生成中/载入中的页跳过（生成中卸载是 use-after-free，载入中无法中断）。
        """
        if not settings_tab.single_model.get():
            return
        for t in eng_tabs:
            if t is not current and t.engine is not None and not t.generating and not t.loading:
                t.on_load_toggle(evict=True)

    for t in eng_tabs:
        t.unload_others = unload_others

    def on_close():
        for tab in eng_tabs:
            tab.shutdown()   # 停止生成 -> 卸载引擎
        app.destroy()

    # 显式接管关窗：先停生成/卸载引擎，destroy 后 _poll 由 winfo_exists 断开轮询链
    app.protocol("WM_DELETE_WINDOW", on_close)

    # 窗口取内容的自然尺寸（所有控件铺开到舒适状态后由 Tk 计算），不超出屏幕
    app.update_idletasks()
    w, h = app.winfo_reqwidth(), app.winfo_reqheight()
    h = min(h, app.winfo_screenheight() - 80)
    app.geometry(f"{w}x{h}")

    app.mainloop()
