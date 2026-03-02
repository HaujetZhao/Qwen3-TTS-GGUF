import os
import sys
import time
import torch
import random
import numpy as np
import soundfile as sf
import subprocess
import functools
import types
import traceback
from pathlib import Path
from export_config import Models; MODEL_DIR = Models.clone.source
from qwen3_tts_gguf.inference.schema.result import TTSResult, Timing
from qwen3_tts_gguf.inference.capturer import OfficialCapturer

ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR / "Qwen3-TTS-main"))
from qwen_tts import Qwen3TTSModel

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def play_audio(file_path):
    print(f"正在播放 {file_path}...")
    # 使用 PowerShell 播放音频
    try:
        subprocess.run(["powershell", "-c", f"(New-Object Media.SoundPlayer '{file_path}').PlaySync();"], check=True)
    except Exception as e:
        print(f"播放音频失败: {e}")

def main():
    # 使用 GPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 检查模型文件
    if not os.path.exists(MODEL_DIR):
        print(f"错误：找不到模型路径：{MODEL_DIR}")
        return

    print(f"正在从 {MODEL_DIR} 加载模型")
    
    try:
        print("开始加载模型...")
        # 定义数据类型
        dtype = torch.bfloat16
        
        # 载入模型
        t_load_start = time.time()
        set_seed(47)
        tts = Qwen3TTSModel.from_pretrained(
            MODEL_DIR,
            device_map=device,
            dtype=dtype,
        )
        t_load_end = time.time()
        load_time = t_load_end - t_load_start
        print(f"模型加载完成，耗时 {load_time:.4f} 秒。")

        # --- 初始化自动捕获器 ---
        capturer = OfficialCapturer(tts)
        
        # 设定生成参数
        # 这里使用 01 生成的第一个音频作为参考音频
        ref_audio = "output/sample.wav" 
        ref_text = "你好，我是千问，你今天过得好吗？"
        target_text = "我是一只可以克隆任何声音的小猫咪，喵喵喵。"
        
        if not os.path.exists(ref_audio):
            print(f"警告：参考音频 {ref_audio} 不存在，将尝试使用官方示例 URL")
            ref_audio = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"
            ref_text = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."

        # 连续生成多个音频
        for i in range(3):
            print(f"\n--- [Voice Clone] 正在生成第 {i+1} 个音频 ---")
            t_infer_start = time.time()
            # 现在直接返回 TTSResult
            res = tts.generate_voice_clone(
                text=target_text,
                language="Chinese",
                ref_audio=ref_audio,
                ref_text=ref_text,
                temperature=0.8, 
                subtalker_temperature=0.8, 
            )
            t_infer_end = time.time()
            infer_time = t_infer_end - t_infer_start
            print(f"推理完成，耗时 {infer_time:.4f} 秒。")

            # --- 保存并播放音频 ---
            output_base = f"output/clone_{i+1}"
            res.save_json(f"{output_base}.json")
            res.save_wav(f"{output_base}.wav")
            print(f"音频与元数据已保存至: {output_base}.*")
            res.play()

    except Exception as e:
        print(f"推理失败: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
