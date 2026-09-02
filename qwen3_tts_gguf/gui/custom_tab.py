"""自定义音色页：内置说话人 -> custom_batch。风格描述写在任务文本组内（--- 上块）。"""
import tkinter as tk

import ttkbootstrap as ttkb
from ttkbootstrap.constants import *

from qwen3_tts_gguf.inference.schema.constants import SPEAKER_MAP

from .base_tab import TTSPageBase, LANGS_ALL, load_task_text, parse_task_groups


class CustomTab(TTSPageBase):
    """自定义音色页：内置音色可加风格描述控制语气；自定义 ID 2800-3071"""

    MODEL_DEFAULT = "model-custom"
    OUT_DEFAULT = "./output/custom"
    LANGS = {"自动": None, **LANGS_ALL}  # "自动"传 None，模型按文本自适应（官方支持）
    TEXT_HINT = "输入文本（=== 分隔多组；组内 --- 分隔：上方为风格描述，下方为朗读文本）"
    TASK_TEXT_DEFAULT = load_task_text("custom")

    def _build_source_rows(self, group):
        ttkb.Label(group, text="说话人").grid(row=0, column=0, sticky=W, padx=(0, 10), pady=4)
        # 可编辑下拉：可选内置音色，也可直接输 2800-3071 的说话人 ID
        self.speaker = tk.StringVar(value="vivian")
        ttkb.Combobox(group, textvariable=self.speaker, values=list(SPEAKER_MAP), width=14) \
            .grid(row=0, column=1, sticky=W, pady=4)
        ttkb.Label(group, text="自定义 ID 2800-3071").grid(row=0, column=2, sticky=W, padx=(8, 0), pady=4)
        return 1

    # ---------- 生成 ----------

    def _read_source(self):
        speaker = self.speaker.get().strip()
        if not speaker:
            self.ui_queue.put(("error", "未填写说话人"))
            return None
        return dict(speaker=int(speaker) if speaker.isdigit() else speaker,
                    language=self.LANGS[self.param_vars["language"].get()])

    def _make_tasks(self, text, cfg, src):
        # 组内 --- 上块为风格描述（可缺省），下块为朗读文本
        return [(target, src["speaker"], src["language"], instruct or "", cfg)
                for instruct, target in parse_task_groups(text)]

    def _run_batch(self, runner, tasks):
        return runner.custom_batch(tasks)
