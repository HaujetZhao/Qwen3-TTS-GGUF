"""
启动 Qwen3-TTS GGUF 图形界面
等价于: python -m qwen3_tts_gguf.gui
"""
import multiprocessing

from qwen3_tts_gguf.gui.app import main

if __name__ == "__main__":
    multiprocessing.freeze_support()  # frozen exe 下 multiprocessing 子进程必要
    main()
