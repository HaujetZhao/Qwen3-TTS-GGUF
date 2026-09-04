"""GUI 多语言：TOML 字典 + 语义 key。

启动时 set_lang 一次（重启切语言），t() 查不到回退返回 key 本身。
字典按页面分节，见 locales/<lang>.toml。
"""
import tomllib
from pathlib import Path

LOCALES_DIR = Path(__file__).parent / "locales"
CONFIG_PATH = Path.home() / ".qwen3_tts_gui.toml"

_catalog = {}


def set_lang(code):
    """读 locales/<code>.toml 进全局字典"""
    global _catalog
    with open(LOCALES_DIR / f"{code}.toml", "rb") as f:
        _catalog = tomllib.load(f)


def t(key):
    """按 . 逐层取嵌套表，任何一层缺失返回 key 本身"""
    node = _catalog
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return key
        node = node[part]
    return node


def available_langs():
    """扫 locales/*.toml，供设置页下拉"""
    return sorted(p.stem for p in LOCALES_DIR.glob("*.toml"))


# ponytail: 单键配置，设置多了再迁完整配置层
def load_config():
    """读界面语言配置，文件不存在/缺键默认 zh"""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "rb") as f:
            return tomllib.load(f).get("language", "zh")
    return "zh"


def save_lang(code):
    """写界面语言配置（content 固定格式，手拼即可）"""
    CONFIG_PATH.write_text(f'language = "{code}"\n', encoding="utf-8")
