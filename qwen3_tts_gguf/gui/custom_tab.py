"""自定义音色页：内置说话人 -> custom_batch。风格描述写在任务文本组内（--- 上块）。"""
import tkinter as tk

import ttkbootstrap as ttkb
from ttkbootstrap.constants import *

from qwen3_tts_gguf.inference.schema.constants import SPEAKER_MAP

from .base_tab import TTSPageBase, LANGS_ALL, load_task_text, parse_task_groups
from .i18n import t


class CustomTab(TTSPageBase):
    """自定义音色页：内置音色可加风格描述控制语气；自定义 ID 2800-3071"""

    MODEL_DEFAULT = "model-custom"
    OUT_DEFAULT = "./output/custom"
    LANGS = ("Auto", *LANGS_ALL)  # Auto 传 None，模型按文本自适应（官方支持）
    TEXT_HINT_KEY = "custom.text_hint"
    TASK_TEXT_DEFAULT = load_task_text("custom")

    def _build_source_rows(self, group):
        ttkb.Label(group, text=t("custom.speaker")).grid(row=0, column=0, sticky=W, padx=(0, 10), pady=4)
        # 可编辑下拉：可选内置音色，也可直接输 2800-3071 的说话人 ID
        self.speaker = tk.StringVar(value="vivian")
        ttkb.Combobox(group, textvariable=self.speaker, values=list(SPEAKER_MAP), width=14) \
            .grid(row=0, column=1, sticky=W, pady=4)
        ttkb.Label(group, text=t("custom.custom_id_hint")).grid(row=0, column=2, sticky=W, padx=(8, 0), pady=4)
        return 1

    # ---------- 生成 ----------

    def _read_source(self):
        speaker = self.speaker.get().strip()
        if not speaker:
            self.ui_queue.put(("error", t("custom.no_speaker")))
            return None
        return dict(speaker=int(speaker) if speaker.isdigit() else speaker,
                    language=self._selected_language())

    def _make_tasks(self, text, cfg, src):
        # 组内 --- 上块为风格描述（可缺省），下块为朗读文本
        return [(target, src["speaker"], src["language"], instruct or "", cfg)
                for instruct, target in parse_task_groups(text)]

    def _run_batch(self, runner, tasks):
        return runner.custom_batch(tasks)
