# CLAUDE.md

用 llama.cpp 跑 Qwen3-TTS 的推理引擎：Talker/Predictor 走 GGUF（CUDA/Vulkan），Encoder/Decoder 走 ONNX Runtime。

## 结构

- `qwen3_tts_gguf/inference/` — 核心引擎
  - `engine.py` 入口；`batch.py` 多路 lockstep 批量；`stream.py` 流式
  - `llama.py` llama.cpp 的 ctypes 绑定（与 llama.cpp 版本强耦合，升级须对照 Experience/09）
  - `decoder.py` / `encoder.py` ONNX 组件；`talker.py` / `predictor.py` LLM 侧
  - `workers/` 子进程（解码、说话人、录音）；`bin/` 存放 llama.cpp DLL（不入 git）
- `qwen3_tts_gguf/gui/` — ttkbootstrap GUI，入口 `python 52-GUI.py`（即 `__main__.py`）
  - `base_tab.py` 页面基类，承载共用的载入/批量生成流程；各 tab 继承它
- `qwen3_tts_gguf/export/` — HF 权重转 GGUF（含 llama.cpp 官方 convert 脚本的改版）
- 根目录编号脚本 = 流水线阶段：`1x` 小组件导出、`2x` Talker、`3x` Predictor、`4x` 推理示例、`5x` 交互/GUI/性能剖析
- `Qwen3-TTS-main/` — 官方 PyTorch 实现，仅导出时参考；`Experience/` — 踩坑记录；`model-*` — 导出产物（不入 git）

## 依赖与环境

- 真相源：`pyproject.toml` + `uv.lock`；`requirements.txt` 仅供 pip 用户，改依赖需手动同步
- extras：`dml` / `cuda`（ONNX 运行时二选一）、`export`（torch cu132 + 导出工具链，仅 1x~3x 脚本需要）
- 日常：`uv sync --extra dml`；Python >= 3.14，运行用 `.venv/Scripts/python`（原 conda fun 环境已废弃）

## 文档

- 面向用户的完整说明（导出流程、性能数据、原理）在 `readme.md`
- llama.cpp 版本兼容坑与排查方法在 `Experience/09`（b10258 起 penalties 采样器签名变更）
