"""把 dist/ 下的产物压成发行 zip，输出到 dist/。

用法:
    uv run python zip_release.py                    # 默认 vulkan 版
    uv run python zip_release.py Qwen3-TTS-GUI-cuda

排除规则:
    - model-*、log/ 不进包 (模型另有独立分发)
    - internal/ 整体进包, 其中的 qwen3_tts_gguf junction 被 7z 解引用为真实
      文件——发行包含项目源码与 llama DLL, 用户机器上没有 link 目标
    - output/elaborate 音色锚点解引用为真实文件打进包
"""
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).parent
DIST = ROOT / "dist"
SEVEN_ZIP_CANDIDATES = [
    r"C:\Program Files\7-Zip\7z.exe",
    r"C:\Program Files (x86)\7-Zip\7z.exe",
]


def find_7z() -> str:
    found = shutil.which("7z")
    if found:
        return found
    for p in SEVEN_ZIP_CANDIDATES:
        if Path(p).exists():
            return p
    raise SystemExit("找不到 7z.exe, 请安装 7-Zip")


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "Qwen3-TTS-GUI-vulkan"
    src = DIST / name
    if not (src / "Qwen3-TTS-GUI.exe").exists():
        raise SystemExit(f"{src} 不是构建产物目录")
    out = DIST / f"{src.name}-{date.today():%Y%m%d}.zip"
    out.unlink(missing_ok=True)  # 7z a 对已存在的包是追加, 先删

    subprocess.run(
        [find_7z(), "a", "-tzip", "-mx5", "-xr!log", str(out),
         "Qwen3-TTS-GUI.exe", "internal", str(Path("output") / "elaborate")],
        cwd=src,
        check=True,
    )
    print(f"完成: {out} ({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
