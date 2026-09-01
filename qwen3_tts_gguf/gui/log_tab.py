"""日志页：只读日志 Text，自动滚到底，带清除按钮。日志管道 QueueLogHandler 也归此模块。"""
import logging
import queue
import tkinter as tk

import ttkbootstrap as ttkb
from ttkbootstrap.constants import *


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
