import time
import queue
import numpy as np
import sounddevice as sd
from ..schema.protocol import SpeakerRequest, SpeakerResponse

def handle_command(cmd: SpeakerRequest, state: dict):
    """处理来自主进程的播放指令 (仅修改物理状态)"""
    if cmd is None or cmd.msg_type == "EXIT":
        state["stop"] = True
        return
    
    if cmd.msg_type == "STOP":
        state["current_data"] = np.zeros((0, 1), dtype=np.float32)
        state["started"] = False
        return
    
    if cmd.msg_type == "PAUSE":
        state["paused"] = True
        return

    if cmd.msg_type == "CONTINUE":
        state["paused"] = False
        return
    
    if cmd.msg_type == "AUDIO":
        if cmd.audio is not None and len(cmd.audio) > 0:
             state["current_data"] = np.concatenate(
                 [state["current_data"], cmd.audio.reshape(-1, 1).astype(np.float32)], 
                 axis=0
             )


def sync_playback_status(state: dict, result_queue):
    """统一状态反馈：对比逻辑状态与物理状态，仅在变化时发送信号"""
    if result_queue is None: return
    
    # 1. 判定当前的真实物理目标状态
    if state.get("paused", False):
        target = "PAUSED"
    elif state.get("started", False):
        target = "PLAYING"
    else:
        target = "IDLE"
        
    # 2. 对比上一次反馈给 Proxy 的状态
    if target == state["playback_state"]:
        return
        
    # 3. 状态转换发送信号
    msg_map = {
        "PAUSED": "PAUSED",
        "PLAYING": "STARTED",
        "IDLE": "FINISHED"
    }
    
    result_queue.put(SpeakerResponse(msg_type=msg_map[target]))
    state["playback_state"] = target


def fill_audio(outdata, frames, state: dict):
    """从缓冲区提取数据填充输出设备 (物理逻辑)"""
    # 0. 如果暂停，填充 0 并返回
    if state.get("paused", False):
        outdata.fill(0)
        return

    # 1. 检查物理缓冲是否足够 (Threshold)
    if not state["started"]:
        if len(state["current_data"]) >= state["threshold"]: 
            state["started"] = True
        else: 
            outdata.fill(0)
            return
            
    # 3. 填充输出设备
    avail = len(state["current_data"])
    to_copy = min(avail, frames)
    if to_copy > 0:
        outdata[:to_copy] = state["current_data"][:to_copy]
        state["current_data"] = state["current_data"][to_copy:]
    
    # 4. 物理状态重置 (播放完毕)
    if to_copy < frames:
        outdata[to_copy:].fill(0)
        state["started"] = False


def speaker_worker_proc(play_queue, result_queue=None, sample_rate=24000):
    """
    音频播放子进程工人 (SpeakerWorker)。
    采用“状态驱动 + 回调机制”架构，确保音频播放的低延迟与高平滑度。
    """
    state = {
        "current_data": np.zeros((0, 1), dtype=np.float32), 
        "started": False,         
        "threshold": 1200,          
        "stop": False,
        "paused": False,          # 暂停标记
        "playback_state": "IDLE"  # IDLE, PLAYING 用于控制反馈信号
    } 
    
    def audio_callback(outdata, frames, time_info, status):
        # 1. 异步抓取并处理所有可用指令包
        while True:
            try:
                command = play_queue.get_nowait()
                handle_command(command, state)
            except queue.Empty: 
                break
            
        # 2. 填充音频数据 (物理逻辑)
        fill_audio(outdata, frames, state)

        # 3. 统一反馈状态 (由于 above 逻辑可能修改了 physical flags)
        sync_playback_status(state, result_queue)

    try:
        with sd.OutputStream(samplerate=sample_rate, channels=1, callback=audio_callback, blocksize=2048):
            # 握手
            if result_queue:
                result_queue.put(SpeakerResponse(msg_type="READY"))
            
            while True:
                time.sleep(0.2)
                if state.get("stop"): break
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"❌ [SpeakerWorker] 异常: {e}")
