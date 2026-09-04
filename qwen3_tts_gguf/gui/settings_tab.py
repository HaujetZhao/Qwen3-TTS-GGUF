"""设置页：全局开关 + 界面语言。"""
import tkinter as tk

import ttkbootstrap as ttkb
from ttkbootstrap.constants import *

from .i18n import t, available_langs, load_config, save_lang


class SettingsTab(ttkb.Frame):
    """设置页。选项的生效接线在 app / 各页，这里只持有变量"""

    def __init__(self, master):
        super().__init__(master, padding=10)
        group = ttkb.Labelframe(self, text=t("settings.model_group"), padding=8)
        group.grid(row=0, column=0, sticky=EW)
        # 单模型模式：切换 Tab 时载入当前页模型，卸载其他页已载入的模型
        self.single_model = tk.BooleanVar(value=True)
        ttkb.Checkbutton(group, variable=self.single_model, text=t("settings.single_model")) \
            .pack(anchor=W)

        # 界面语言：选中即写配置，重启后生效
        lang_group = ttkb.Labelframe(self, text=t("settings.lang_group"), padding=8)
        lang_group.grid(row=1, column=0, sticky=EW, pady=(10, 0))
        self.lang_var = tk.StringVar(value=load_config())
        ttkb.Combobox(lang_group, textvariable=self.lang_var, values=available_langs(),
                      state="readonly", width=8) \
            .pack(side=LEFT)
        self.lang_var.trace_add("write", lambda *_: save_lang(self.lang_var.get()))
        ttkb.Label(lang_group, text=t("settings.lang_hint"), foreground="#888").pack(side=LEFT, padx=(10, 0))
