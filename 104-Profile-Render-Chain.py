"""
104-Profile-Render-Chain.py - 深入分析渲染链条启动耗时
"""
import os
import sys
import time
import multiprocessing as mp

# 确保能找到 qwen3_tts_gguf 包
sys.path.append(os.getcwd())

from qwen3_tts_gguf.mouth_decoder import StatefulMouthDecoder

def profile_mouth_load(onnx_path):
    print(f"📦 [Profile] 开始计时 MouthWorker 模型加载 (DML 端)...")
    t0 = time.time()
    # 模拟 Worker 内部加载
    os.environ["OMP_NUM_THREADS"] = "4"
    decoder = StatefulMouthDecoder(onnx_path, use_dml=True)
    dt = time.time() - t0
    print(f"✅ [Profile] 模型加载完成: {dt:.2f}s (Provider: {decoder.active_provider})")
    return dt

def profile_speaker_init():
    print(f"🎧 [Profile] 开始计时 SpeakerWorker 声卡驱动初始化...")
    t0 = time.time()
    import sounddevice as sd
    import numpy as np
    
    # 模拟由于回调开启导致的驱动握手
    def dummy_callback(outdata, frames, time_info, status):
        outdata.fill(0)
        
    try:
        with sd.OutputStream(samplerate=24000, channels=1, callback=dummy_callback, blocksize=512):
            dt = time.time() - t0
            print(f"✅ [Profile] 声卡驱动初始化完成: {dt:.2f}s")
            return dt
    except Exception as e:
        print(f"❌ [Profile] 声卡初始化失败: {e}")
        return 0

def run_analysis():
    print("\n" + "="*60)
    print("🔍 渲染链条各阶段耗时深度分析")
    print("="*60)
    
    MODEL_PATH = "model/qwen3_tts_decoder_stateful.onnx"
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 找不到模型: {MODEL_PATH}")
        return

    # 1. 测试主进程直接加载的理论耗时
    # 这排除了进程间通讯和 Python 解释器启动的干扰
    print("\n[阶段 1: 算法核心加载测试]")
    mouth_dt = profile_mouth_load(MODEL_PATH)
    speaker_dt = profile_speaker_init()
    
    # 2. 测试进程创建开销
    print("\n[阶段 2: 多进程环境切换测试]")
    t_p_s = time.time()
    p = mp.Process(target=time.sleep, args=(0.1,))
    p.start()
    p.join()
    t_p_e = time.time()
    print(f"🚀 [Profile] 纯子进程拉起时间: {t_p_e - t_p_s - 0.1:.4f}s")

    print("\n" + "="*60)
    print("📊 诊断结果摘要")
    print(f"  - 模型加载占起动耗时约: {mouth_dt/(mouth_dt+speaker_dt+0.1)*100:.1f}%")
    print(f"  - 驱动探测占起动耗时约: {speaker_dt/(mouth_dt+speaker_dt+0.1)*100:.1f}%")
    print("\n💡 建议:")
    if mouth_dt > 3:
        print("   -> 加载耗时主要在 DML 显存分配和算符编译。这是 Windows GPU 初始化的固有开销。")
    if speaker_dt > 1:
        print("   -> 声卡驱动响应慢。可能是由于系统中存在多个音频设备或驱动负载较高。")
    print("="*60)

if __name__ == "__main__":
    run_analysis()
