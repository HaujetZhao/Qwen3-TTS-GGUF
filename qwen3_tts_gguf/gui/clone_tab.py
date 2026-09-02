"""克隆页：克隆源 + 参考文本 -> clone_batch。载入区/管道/参数等骨架在 base_tab。"""
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import sounddevice as sd
import ttkbootstrap as ttkb
from ttkbootstrap.constants import *

from qwen3_tts_gguf import logger
from qwen3_tts_gguf.inference.schema.constants import SAMPLE_RATE
from qwen3_tts_gguf.inference.schema.result import TTSResult
from qwen3_tts_gguf.inference.utils.audio import load_audio
from qwen3_tts_gguf.inference.voice import prepare_voice

from .base_tab import TTSPageBase, LANGS_ALL, _hook_drop_file

# 克隆源支持的文件类型
CLONE_SOURCE_TYPES = [("克隆源", "*.wav *.json"), ("所有文件", "*.*")]


class CloneTab(TTSPageBase):
    """语音克隆页"""

    MODEL_DEFAULT = "model-base"
    OUT_DEFAULT = "./output/clone"
    # 语言下拉：官方支持 10 语言 + 2 方言；"自动"传 None 走模型自适应
    LANGS = {"自动": None, **LANGS_ALL}

    def _build_source_rows(self, group):
        ttkb.Label(group, text="克隆源").grid(row=0, column=0, sticky=W, padx=(0, 10), pady=4)
        self.clone_source = tk.StringVar(value="output/elaborate/Vivian.json")
        source_entry = ttkb.Entry(group, textvariable=self.clone_source)
        source_entry.grid(row=0, column=1, sticky=EW, pady=4)
        _hook_drop_file(source_entry, self.clone_source)
        ttkb.Button(group, text="选择克隆源", command=self.on_pick_source, width=14).grid(row=0, column=2, padx=(8, 0), pady=4)
        self.clone_source.trace_add("write", lambda *_: self._refresh_play_btn())

        # 参考文本: 音频克隆源的转写；留空则零样本克隆 (json 源自带转写，忽略此框)
        ttkb.Label(group, text="参考文本", width=12).grid(row=1, column=0, sticky=W, padx=(0, 10), pady=4)
        self.ref_text = tk.StringVar()
        ttkb.Entry(group, textvariable=self.ref_text).grid(row=1, column=1, sticky=EW, pady=4)
        # json 源须先经引擎解码，未载入时不可播放；音频源直接播
        self.play_btn = ttkb.Button(group, text="播放克隆源", command=self.on_play_source, width=14, state="disabled")
        self.play_btn.grid(row=1, column=2, padx=(8, 0), pady=4)
        return 2

    def _refresh_play_btn(self):
        """播放按钮：有源即可播音频；json 源须引擎已载入 (要 decode)"""
        src = self.clone_source.get().strip()
        is_json = src.lower().endswith(".json")
        ok = bool(src) and (not is_json or self.engine is not None)
        self.play_btn.configure(state="normal" if ok else "disabled")

    def on_pick_source(self):
        path = filedialog.askopenfilename(title="选择克隆源 (.wav / .json)", filetypes=CLONE_SOURCE_TYPES)
        if path:
            self.clone_source.set(path)

    def on_play_source(self):
        source = self.clone_source.get().strip()
        if Path(source).suffix.lower() == ".json":
            # 解码要跑 ONNX，放后台线程，播完/失败后经 play_done 恢复按钮态
            self.play_btn.configure(state="disabled")
            self.ui_queue.put(("status", "正在解码克隆源音频…"))
            threading.Thread(target=self._play_worker, args=(source,), daemon=True).start()
        else:
            try:
                audio = load_audio(source, SAMPLE_RATE)
            except Exception as e:
                self.ui_queue.put(("error", f"读取克隆源音频失败: {e}"))
                return
            sd.play(audio, samplerate=SAMPLE_RATE)
            self.ui_queue.put(("status", f"正在播放 {Path(source).name}"))

    def _play_worker(self, source):
        try:
            res = TTSResult.from_json(source)
            self.engine.decode(res)  # 回写 res.audio
            if res.audio is None or len(res.audio) == 0:
                self.ui_queue.put(("error", "克隆源 JSON 无可播放的音频数据"))
                return
            res.play(blocking=False)
            self.ui_queue.put(("status", "正在播放克隆源音频"))
        except Exception as e:
            logger.exception("解码克隆源失败")
            self.ui_queue.put(("error", f"解码克隆源失败: {e}"))
        finally:
            self.ui_queue.put(("play_done",))

    # ---------- 生成 ----------

    def _read_source(self):
        source = self.clone_source.get().strip()
        if not source:
            self.ui_queue.put(("error", "未选择克隆源"))
            return None
        return dict(source=source,
                    ref_text=self.ref_text.get().strip(),
                    language=self.LANGS[self.param_vars["language"].get()])

    def _make_tasks(self, text, cfg, src):
        """克隆按行拆任务；克隆源分流: json 自带转写；音频源按参考文本定模式"""
        lines = [l.strip() for l in text.splitlines()]
        lines = [l for l in lines if l and not l.startswith("#")]  # 空行与 # 注释行剔除
        source, ref_text, language = src["source"], src["ref_text"], src["language"]
        zero_shot = False
        src_path = Path(source)
        if src_path.suffix.lower() == ".json":
            if ref_text:
                logger.info("[GUI] json 克隆源自带转写，忽略参考文本框")
        elif src_path.suffix.lower() in (".wav", ".mp3", ".flac", ".m4a", ".opus"):
            if not ref_text:
                json_pair = src_path.with_suffix(".json")
                if json_pair.exists():
                    # 有同名锚点存档: 用它做普通克隆 (带转写，效果优于零样本)
                    logger.info(f"[GUI] 参考文本为空，改用同名锚点存档 {json_pair.name}")
                    source = str(json_pair)
                else:
                    logger.info("[GUI] 参考文本为空，走零样本克隆 (仅用音色向量)")
                    zero_shot = True
        else:
            return f"不支持的克隆源类型: {src_path.suffix}"
        voice = prepare_voice(self.engine, self.engine.tokenizer, source,
                              text=ref_text if (not zero_shot and src_path.suffix.lower() != ".json") else None)
        if voice is None:
            return "克隆源准备失败，详见日志"
        return [(text, voice, language, zero_shot, cfg) for text in lines]

    def _run_batch(self, runner, tasks):
        return runner.clone_batch(tasks)
