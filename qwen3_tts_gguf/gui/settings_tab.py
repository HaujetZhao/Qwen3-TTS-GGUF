"""设置页：全局开关。"""
import tkinter as tk

import ttkbootstrap as ttkb
from ttkbootstrap.constants import *


class SettingsTab(ttkb.Frame):
    """设置页。选项的生效接线在 app / 各页，这里只持有变量"""

    def __init__(self, master):
        super().__init__(master, padding=10)
        group = ttkb.Labelframe(self, text="模型", padding=8)
        group.grid(row=0, column=0, sticky=EW)
        # 单模型模式：切换 Tab 时载入当前页模型，卸载其他页已载入的模型
        self.single_model = tk.BooleanVar(value=True)
        ttkb.Checkbutton(group, variable=self.single_model, text="只载入一个模型（切换 Tab 时自动载入当前页模型，卸载其他页）") \
            .pack(anchor=W)
