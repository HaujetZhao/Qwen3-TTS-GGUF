"""音色设计页：design_batch。音色描述写在任务文本组内（--- 上块），官方无 language。"""
from .base_tab import TTSPageBase, load_task_text, parse_task_groups


class DesignTab(TTSPageBase):
    """音色设计页：纯文字描述设计音色"""

    MODEL_DEFAULT = "model-design"
    OUT_DEFAULT = "./output/design"
    TEXT_HINT = "输入文本（=== 分隔多组；组内 --- 分隔：上方为音色描述，下方为朗读文本）"
    TASK_TEXT_DEFAULT = load_task_text("design")

    def _param_entries(self):
        # 设计页无语言选项，语种由描述与文本决定
        return [e for e in super()._param_entries() if e[2] != "language"]

    def _make_tasks(self, text, cfg, src):
        # 组内 --- 上块为音色描述，下块为朗读文本；官方 design 无 language，传 None
        return [(target, instruct, None, cfg)
                for instruct, target in parse_task_groups(text)]

    def _run_batch(self, runner, tasks):
        return runner.design_batch(tasks)
