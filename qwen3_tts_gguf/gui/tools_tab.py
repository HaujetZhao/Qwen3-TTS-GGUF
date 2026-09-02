"""工具页：只载编码器 + 解码器（不载 LLM），放音色/音频维护工具。

- 更新 JSON 音色向量：从 codes 重提 spk_emb 写回原 JSON
- WAV → JSON：wav + 参考文本 -> 旁边生成同名 .json 音色锚点
- JSON → WAV：文件夹内逐个 .json 解码出同名 .wav

复用 TTSPageBase 的载入管道（ui_queue / _poll / 载入卸载），推理骨架不适用，
推理区整体换成工具面板。
"""
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import ttkbootstrap as ttkb
from ttkbootstrap.constants import *

from qwen3_tts_gguf import logger
from qwen3_tts_gguf.inference import TTSEngine
from qwen3_tts_gguf.inference.schema.result import TTSResult
from qwen3_tts_gguf.inference.utils.audio import load_audio

from .base_tab import TTSPageBase, ONNX_PROVIDERS, _hook_drop_file, _hook_drop_folder

JSON_TYPES = [("JSON", "*.json"), ("所有文件", "*.*")]
AUDIO_TYPES = [("音频", "*.wav *.mp3 *.flac *.m4a *.opus"), ("所有文件", "*.*")]


class ToolsTab(TTSPageBase):
    """工具页"""

    MODEL_DEFAULT = "model-base"

    # ---------- 载入区（比生成页瘦：无上下文/LLM 设备选项） ----------

    def _build_load_group(self):
        group = ttkb.Labelframe(self, text="载入", padding=8)
        group.grid(row=0, column=0, sticky=EW)
        group.columnconfigure(1, weight=1)

        ttkb.Label(group, text="模型文件夹").grid(row=0, column=0, sticky=W, padx=(0, 10), pady=4)
        self.model_dir = tk.StringVar(value=self.MODEL_DEFAULT)
        self.model_entry = ttkb.Entry(group, textvariable=self.model_dir)
        self.model_entry.grid(row=0, column=1, sticky=EW, pady=4)
        _hook_drop_folder(self.model_entry, self.model_dir)
        browse_btn = ttkb.Button(group, text="浏览", command=self.on_browse_model, width=14)
        browse_btn.grid(row=0, column=2, padx=(8, 0), pady=4)

        opt_row = ttkb.Frame(group)
        opt_row.grid(row=1, column=0, columnspan=2, sticky=EW, pady=4)
        ttkb.Label(opt_row, text="ONNX 组件").pack(side=LEFT, padx=(0, 10))
        self.onnx_provider = tk.StringVar(value=ONNX_PROVIDERS[0])
        self.onnx_combo = ttkb.Combobox(opt_row, textvariable=self.onnx_provider, values=ONNX_PROVIDERS,
                                        state="readonly", width=9)
        self.onnx_combo.pack(side=LEFT)
        ttkb.Label(opt_row, text="只载入 编码器 + 解码器（不载 Talker / Predictor）", foreground="#888") \
            .pack(side=LEFT, padx=(24, 0))

        self.load_btn = ttkb.Button(group, text="载入", command=self.on_load_toggle, width=14, bootstyle=SUCCESS)
        self.load_btn.grid(row=1, column=2, sticky=E, pady=4)

        self._load_fields = [self.model_entry, self.onnx_combo, browse_btn]

    def _load_params(self):
        # 工具页不载 LLM，无设备选项
        return dict(model_dir=self.model_dir.get(), onnx=self.onnx_provider.get())

    def _create_engine(self, model_dir, llm, onnx):
        return TTSEngine(model_dir=model_dir, onnx_provider=onnx, load_llm=False,
                         subprocess_decoder=False, chunk_size=64)

    # ---------- 工具区 ----------

    def _build_infer_group(self):
        wrap = ttkb.Frame(self)
        wrap.grid(row=1, column=0, sticky=EW, pady=(10, 0))
        wrap.columnconfigure(0, weight=1)

        self.tool_buttons = []  # 引擎载入后才可用

        g = self._tool_frame(wrap, 0, "更新 JSON 音色向量")
        self.upd_json_path = tk.StringVar()
        self._path_row(g, 0, "JSON 文件/文件夹", self.upd_json_path, filetypes=JSON_TYPES)
        self._tool_tail(g, 1, "从 codes 重新提取 spk_emb 写回原文件；用于修复缺失/维度不符的旧锚点",
                        self.on_update_json)

        g = self._tool_frame(wrap, 1, "WAV → JSON（生成音色锚点）")
        self.wav_path = tk.StringVar()
        self._path_row(g, 0, "WAV 文件", self.wav_path, filetypes=AUDIO_TYPES)
        self.ref_text = tk.StringVar()
        self._text_row(g, 1, "参考文本", self.ref_text)
        self._tool_tail(g, 2, "提取音色特征，在 WAV 旁生成同名 .json 音色锚点",
                        self.on_wav_to_json)

        g = self._tool_frame(wrap, 2, "JSON → WAV（批量解码）")
        self.json_dir = tk.StringVar()
        self._path_row(g, 0, "JSON 文件夹", self.json_dir, is_dir=True)
        self._tool_tail(g, 1, "逐个解码文件夹内的 .json，在旁边生成同名 .wav",
                        self.on_json_to_wav)

    def _tool_frame(self, parent, row, title):
        g = ttkb.Labelframe(parent, text=title, padding=8)
        g.grid(row=row, column=0, sticky=EW, pady=(0 if row == 0 else 10, 0))
        g.columnconfigure(1, weight=1)
        return g

    def _path_row(self, g, row, label, var, is_dir=False, filetypes=None):
        ttkb.Label(g, text=label).grid(row=row, column=0, sticky=W, padx=(0, 10), pady=4)
        entry = ttkb.Entry(g, textvariable=var)
        entry.grid(row=row, column=1, sticky=EW, pady=4)
        (_hook_drop_folder if is_dir else _hook_drop_file)(entry, var)

        def browse():
            if is_dir:
                path = filedialog.askdirectory(title=f"选择{label}")
            else:
                path = filedialog.askopenfilename(title=f"选择{label}", filetypes=filetypes)
            if path:
                var.set(path)
        ttkb.Button(g, text="浏览", command=browse, width=14).grid(row=row, column=2, padx=(8, 0), pady=4)

    def _text_row(self, g, row, label, var):
        ttkb.Label(g, text=label).grid(row=row, column=0, sticky=W, padx=(0, 10), pady=4)
        ttkb.Entry(g, textvariable=var).grid(row=row, column=1, sticky=EW, pady=4)

    def _tool_tail(self, g, row, hint, command):
        """每个工具的收尾行：左侧灰字说明，右侧执行按钮"""
        ttkb.Label(g, text=hint, foreground="#888").grid(row=row, column=0, columnspan=2, sticky=W, pady=(6, 0))
        btn = ttkb.Button(g, text="执行", command=command, width=14, bootstyle=PRIMARY, state="disabled")
        btn.grid(row=row, column=2, sticky=E, pady=(6, 0))
        self.tool_buttons.append(btn)

    # ---------- 状态切换 ----------

    def _set_loaded(self, ok):
        """载入成功: 按钮变卸载、载入区锁定、工具按钮解锁；失败/卸载(含被他页挤掉): 全部还原禁用"""
        self.load_btn.configure(text="卸载" if ok else "载入", state="normal")
        for w in self._load_fields:
            w.configure(state="disabled" if ok else "normal")
        for b in self.tool_buttons:
            b.configure(state="normal" if ok else "disabled")

    def _set_generating(self, on):
        """工具执行中: 全部按钮锁死（含卸载，卸载是 use-after-free）；结束恢复（此时引擎必在）"""
        self.generating = on  # 复用基类 busy 标记: 单模型互斥据此跳过本页
        self.load_btn.configure(state="disabled" if on else "normal")
        for b in self.tool_buttons:
            b.configure(state="disabled" if on else "normal")
        if not on:
            self.progress.configure(value=0)

    # ---------- 执行管道 ----------

    def _start_tool(self, fn):
        """UI 线程: 锁按钮 -> 后台线程执行；结束经 gen_done 事件恢复"""
        if self.generating:
            return
        self._set_generating(True)
        self.ui_queue.put(("status", "工具执行中…"))
        threading.Thread(target=self._tool_worker, args=(fn,), daemon=True).start()

    def _tool_worker(self, fn):
        try:
            fn()
        except Exception as e:
            logger.exception("工具执行失败")
            self.ui_queue.put(("error", f"工具执行异常: {e}"))
        finally:
            self.ui_queue.put(("gen_done",))

    def _collect_jsons(self, path):
        """输入是文件取自身，是文件夹取其中全部 .json"""
        p = Path(path)
        return [p] if p.is_file() else sorted(p.glob("*.json"))

    # ---------- 三个工具 ----------

    def on_update_json(self):
        files = self._collect_jsons(self.upd_json_path.get().strip())
        if not files:
            self.ui_queue.put(("error", "未找到 JSON 文件"))
            return
        self._start_tool(lambda: self._update_json(files))

    def _update_json(self, files):
        """逐个锚点：解码出音频后重提 spk_emb，写回原文件"""
        ok = 0
        for i, f in enumerate(files):
            self.ui_queue.put(("status", f"更新音色向量 {i + 1}/{len(files)}: {f.name}"))
            res = TTSResult.from_json(str(f))
            if not res.is_valid_anchor:
                logger.warning(f"[GUI] 跳过无效锚点: {f.name}")
                continue
            had_audio = res.audio is not None
            if res.audio is None:
                self.engine.decode(res)  # spk_emb 从解码音频重提
            self.engine.encode(res)
            res.save_json(str(f), include_audio=had_audio)  # 原来带音频的存档保持带音频
            ok += 1
            self.ui_queue.put(("progress", i + 1, len(files)))
        self.ui_queue.put(("status", f"音色向量已更新 {ok}/{len(files)}"))

    def on_wav_to_json(self):
        wav = self.wav_path.get().strip()
        text = self.ref_text.get().strip()
        if not wav:
            self.ui_queue.put(("error", "未选择 WAV 文件"))
            return
        if not text:
            self.ui_queue.put(("error", "参考文本为空，无法生成锚点"))
            return
        self._start_tool(lambda: self._wav_to_json(wav, text))

    def _wav_to_json(self, wav, text):
        """音频 -> codes + spk_emb，存为同名 .json 锚点"""
        self.ui_queue.put(("status", f"正在提取音色特征: {Path(wav).name}"))
        samples = load_audio(wav)
        res = TTSResult(text=text, text_ids=self.engine.tokenizer.encode(text).ids,
                        spk_emb=self.engine.speaker_encoder.encode(samples),
                        codes=self.engine.codec_encoder.encode(samples))
        out = Path(wav).with_suffix(".json")
        res.save(str(out))
        self.ui_queue.put(("progress", 1, 1))
        self.ui_queue.put(("status", f"已生成锚点: {out.name}"))

    def on_json_to_wav(self):
        files = self._collect_jsons(self.json_dir.get().strip())
        if not files:
            self.ui_queue.put(("error", "未找到 JSON 文件"))
            return
        self._start_tool(lambda: self._json_to_wav(files))

    def _json_to_wav(self, files):
        """逐个锚点解码出音频，存为旁边同名 .wav"""
        ok = 0
        for i, f in enumerate(files):
            self.ui_queue.put(("status", f"解码 {i + 1}/{len(files)}: {f.name}"))
            res = TTSResult.from_json(str(f))
            if not res.is_valid_anchor:
                logger.warning(f"[GUI] 跳过无效锚点: {f.name}")
                continue
            self.engine.decode(res)
            res.save(str(f.with_suffix(".wav")))
            ok += 1
            self.ui_queue.put(("progress", i + 1, len(files)))
        self.ui_queue.put(("status", f"解码完成 {ok}/{len(files)}"))
