"""日志页：只读日志 Text，自动滚到底，带清除按钮。"""
import tkinter as tk

import ttkbootstrap as ttkb
from ttkbootstrap.constants import *


class LogTab(ttkb.Frame):
    """只读日志页：显示历史日志，自动滚到底"""

    def __init__(self, master):
        super().__init__(master, padding=5)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # 顶部一行：清除按钮右对齐
        top = ttkb.Frame(self)
        top.grid(row=0, column=0, columnspan=2, sticky=EW)
        ttkb.Button(top, text="清除日志", command=self.clear, width=10).pack(side=RIGHT)

        self.text = tk.Text(self, wrap="word", state="disabled", relief="flat",
                            padx=6, pady=4, highlightthickness=1,
                            highlightbackground="#c9c9c9", highlightcolor="#1a7f8e")
        ybar = ttkb.Scrollbar(self, orient=VERTICAL, command=self.text.yview)
        self.text.configure(yscrollcommand=ybar.set)
        self.text.grid(row=1, column=0, sticky=NSEW)
        ybar.grid(row=1, column=1, sticky=NS)

    def append(self, line: str):
        self.text.configure(state="normal")
        self.text.insert("end", line + "\n")
        self.text.see("end")
        self.text.configure(state="disabled")

    def clear(self):
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")
