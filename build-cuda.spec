# -*- mode: python ; coding: utf-8 -*-
#
# 打包 GUI 为可分发 exe: llama.cpp cuda13 + onnxruntime-cuda (原 onnxruntime-gpu 包) 版。
#
# 用法:
#   uv sync --extra cuda
#   uv run pyinstaller build-cuda.spec
#
# vulkan 版见 build-vulkan.spec；两个 spec 分开写, 避免 build/ 缓存来回失效拖慢构建。
# CUDA runtime (cudart/cublas/cudnn) 不打进包, 假设用户已自行安装, 构建时过滤掉。
#
# 项目代码 (qwen3_tts_gguf) 与模型不打包, 构建后 junction 进产物目录
# (不占双份磁盘, 7z 打发行 zip 时解引用为真实文件)。
# 项目包 junction 在 internal/ 里: frozen 程序的 sys.path 本来就含 internal,
# 直接导入, 无需 runtime hook; 改 junction 的源码即刻生效, 不用重新打包。
# 注意: bin/ 会停留为最后构建的变体, 从源码跑时让 extras 与 bin 匹配
# (切完 extras 重跑一次对应 spec 即可)。

import os
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files

ROOT = Path(SPECPATH)
BIN_DIR = ROOT / "qwen3_tts_gguf" / "inference" / "bin"

VARIANT = "cuda"
LLAMA_ZIP = "llama-b10621-bin-win-cuda-13.3-x64.zip"
DIST_DIR = ROOT / "dist" / f"Qwen3-TTS-GUI-{VARIANT}"

# junction 进产物目录的目录: 项目代码 + 模型目录 + 音色锚点
JUNCTIONS = [
    ("internal/qwen3_tts_gguf", ROOT / "qwen3_tts_gguf"),
    ("model-base", ROOT / "model-base"),
    ("model-custom", ROOT / "model-custom"),
    ("model-design", ROOT / "model-design"),
    ("output/elaborate", ROOT / "output" / "elaborate"),
]

hiddenimports = [
    "tkinter",
    "numpy",
    "gguf",
    "tokenizers",
    "onnxruntime",
    "scipy.fft",
    "scipy.signal.windows",
    "sounddevice",
    "soundfile",
    "ttkbootstrap",
    "windnd",
]

# bin 不入 git, 清空后从 ref/ 下的 zip 解压出对应变体的 llama.cpp DLL
if BIN_DIR.exists():
    shutil.rmtree(BIN_DIR)
print(f"[{VARIANT}] 解压 {LLAMA_ZIP} -> {BIN_DIR}")
with zipfile.ZipFile(ROOT / "ref" / LLAMA_ZIP) as zf:
    zf.extractall(BIN_DIR)

a = Analysis(
    ["52-GUI.py"],
    pathex=[],
    binaries=[],
    datas=collect_data_files("ttkbootstrap"),  # 主题资源 (assets/)
    hiddenimports=hiddenimports,
    excludes=["qwen3_tts_gguf"],
    noarchive=False,
)

# 过滤 PyInstaller 从系统/环境收集来的 CUDA runtime DLL (用户自装, 不捆绑);
# onnxruntime_providers_cuda.dll 是 ORT 自身组件, 不在过滤范围
CUDA_DLL = re.compile(r"cudart|cublas|cudnn|nvidia gpu computing toolkit|\\cuda\\v|\\nvidia\\", re.I)
kept = []
for name, src, kind in a.binaries:
    if CUDA_DLL.search(name) or CUDA_DLL.search(str(src)):
        print(f"[{VARIANT}] 排除 CUDA runtime DLL: {name}")
        continue
    kept.append((name, src, kind))
a.binaries = kept

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Qwen3-TTS-GUI",
    console=False,
    contents_directory="internal",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name=f"Qwen3-TTS-GUI-{VARIANT}",
)

for dest, target in JUNCTIONS:
    link = DIST_DIR / dest
    if os.path.lexists(link):
        if not os.path.isjunction(link):
            raise SystemExit(f"{link} 已存在且不是 junction, 请手动处理")
        os.rmdir(link)  # 只移除 junction 本身, 不动目标
    link.parent.mkdir(parents=True, exist_ok=True)  # 嵌套目标如 output/elaborate
    print(f"junction {link} -> {target}")
    subprocess.check_call(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                          stdout=subprocess.DEVNULL)

print(f"\n完成: {DIST_DIR}")
