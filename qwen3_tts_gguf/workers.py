import os
import time
import numpy as np
import queue
import soundfile as sf

from .protocol import DecodeRequest, DecodeResult, SpeakerRequest

def wav_writer_proc(record_queue, filename, sample_rate=24000):
    abs_filename = os.path.abspath(filename)
    os.makedirs(os.path.dirname(abs_filename), exist_ok=True)
    try:
        f = sf.SoundFile(abs_filename, mode='w', samplerate=sample_rate, channels=1)
    except:
        abs_filename = abs_filename.replace(".wav", f"_{int(time.time())}.wav")
        f = sf.SoundFile(abs_filename, mode='w', samplerate=sample_rate, channels=1)
    try:
        while True:
            chunk = record_queue.get()
            if chunk is None: break
            if isinstance(chunk, str) and chunk == "CLEAR": continue
            f.write(chunk.flatten().astype(np.float32))
            f.flush()
    except: pass
    finally: f.close()

def decoder_worker_proc(codes_queue, pcm_queue, decoder_onnx_path, record_queue=None):
    """
    解码子进程工人 (DecoderWorker)。
    使用 protocol.py 定义的强类型协议通信。
    """
    from qwen3_tts_gguf.decoder import StatefulDecoder
    
    # 强制关闭多线程竞争，防止干扰主进程
    os.environ["OMP_NUM_THREADS"] = "4"
    
def decoder_worker_proc(codes_queue, pcm_queue, decoder_onnx_path, record_queue=None):
    """
    解码子进程工人 (DecoderWorker)。
    使用 protocol.py 定义的强类型协议通信。
    支持多会话状态管理 (Session-based State Management)。
    """
    from qwen3_tts_gguf.decoder import StatefulDecoder
    
    # 强制关闭多线程竞争，防止干扰主进程
    os.environ["OMP_NUM_THREADS"] = "4"
    
    decoder = StatefulDecoder(decoder_onnx_path, use_dml=True)
    # 向 Proxy 发送就绪信号
    pcm_queue.put(DecodeResult(msg_type="READY", task_id="decoder"))
    print(f"🔊 [DecoderWorker] 已就绪 (Provider: {decoder.active_provider})")
    
    # 会话状态管理 {task_id: DecoderState}
    sessions = {}
    
    try:
        while True:
            req: DecodeRequest = codes_queue.get()
            
            # 毒丸：退出信号
            if req is None:
                pcm_queue.put(None)
                if record_queue: record_queue.put(None)
                break
                
            # 处理 RESET 指令 (相当于强制结束会话)
            if req.msg_type == "RESET":
                if req.task_id in sessions:
                    del sessions[req.task_id] # 彻底销毁记忆
                if record_queue: record_queue.put("CLEAR")
                continue
            
            # 处理解码请求
            if req.msg_type in ["DECODE", "DECODE_CHUNK"]:
                codes_all = np.array(req.codes, dtype=np.int64)
                if codes_all.ndim == 1:
                    codes_all = codes_all.reshape(-1, 16)
                
                # 获取当前会话状态 (如果不存在则传 None，decoder 会自动新建)
                current_state = sessions.get(req.task_id)
                
                # 自动切分超长序列
                chunk_step = 50
                n_total = codes_all.shape[0]
                
                try:
                    for start_idx in range(0, n_total, chunk_step):
                        end_idx = min(start_idx + chunk_step, n_total)
                        is_last_chunk = (end_idx == n_total)
                        # 只有在整段代码的最后一部分，且 req.is_final 为 True 时，才传递 is_final=True
                        current_is_final = req.is_final and is_last_chunk
                        
                        codes = codes_all[start_idx:end_idx]
                        
                        t0 = time.time()
                        # 纯函数式调用：输入 state，输出 new_state
                        audio, new_state = decoder.decode(codes, state=current_state, is_final=current_is_final)
                        dt = time.time() - t0
                        
                        # 更新会话状态
                        sessions[req.task_id] = new_state
                        current_state = new_state # 为下一个 chunk 准备
                        
                        # 回传结果
                        if len(audio) > 0:
                            res = DecodeResult(msg_type="AUDIO", task_id=req.task_id, audio=audio.copy(), compute_time=dt)
                            pcm_queue.put(res)
                            if record_queue: record_queue.put(audio.copy())
                        else:
                            # 发送空包以维持时序心跳
                            pcm_queue.put(DecodeResult(msg_type="AUDIO", task_id=req.task_id, audio=np.array([], dtype=np.float32), compute_time=dt))
                    
                    # 只有当这是整个请求的结束，且是 DECODE 类型(默认是一次性) 或 is_final 为 True 时，才清理状态
                    if req.is_final:
                        if req.task_id in sessions:
                            del sessions[req.task_id] # 任务结束，释放显存/内存
                            
                    if req.msg_type == "DECODE":
                        # 非流式模式最后发送一个 None audio 表示结束
                        pcm_queue.put(DecodeResult(msg_type="AUDIO", task_id=req.task_id, audio=None, compute_time=0))
                        
                except Exception as e:
                    print(f"⚠️ [DecoderWorker] 解码异常: {e}")
                    # 发生异常，清理该会话
                    if req.task_id in sessions:
                        del sessions[req.task_id]
                        
                    pcm_queue.put(DecodeResult(msg_type="AUDIO", task_id=req.task_id, audio=np.array([], dtype=np.float32), compute_time=0))
                    if req.msg_type == "DECODE":
                         pcm_queue.put(DecodeResult(msg_type="AUDIO", task_id=req.task_id, audio=None, compute_time=0))

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"❌ [DecoderWorker] 崩溃: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"❌ [DecoderWorker] 崩溃: {e}")
        import traceback
        traceback.print_exc()


def speaker_worker_proc(play_queue, result_queue=None, sample_rate=24000):
    import sounddevice as sd
    
    # 状态控制
    state = {
        "current_data": np.zeros((0, 1), dtype=np.float32), 
        "started": False, 
        "prefill": 1200, 
        "playing_task_id": -1, # 当前正在播放的任务 ID
        "stop": False
    } 
    
    def audio_callback(outdata, frames, time_info, status):
        # 抓取当前所有可用数据
        while True:
            try:
                cmd: SpeakerRequest = play_queue.get_nowait()
                if cmd is None:
                    state["stop"] = True
                    break
                
                # 处理 STOP 指令
                if cmd.msg_type == "STOP":
                    if cmd.task_id == state["playing_task_id"]:
                        state["current_data"] = np.zeros((0, 1), dtype=np.float32)
                        state["started"] = False
                    continue
                
                # 处理 PLAY 指令
                if cmd.msg_type == "PLAY":
                    # 抢占式切歌：如果 ID 变了，清空缓冲区
                    if cmd.task_id != state["playing_task_id"]:
                        state["current_data"] = np.zeros((0, 1), dtype=np.float32)
                        state["started"] = False
                        state["playing_task_id"] = cmd.task_id
                    
                    if cmd.audio is not None and len(cmd.audio) > 0:
                         state["current_data"] = np.concatenate(
                             [state["current_data"], cmd.audio.reshape(-1, 1).astype(np.float32)], 
                             axis=0
                        )

            except queue.Empty: break
            
        if not state["started"]:
            if len(state["current_data"]) >= state["prefill"]: 
                state["started"] = True
            else: 
                outdata.fill(0); return
                
        avail = len(state["current_data"])
        to_copy = min(avail, frames)
        if to_copy > 0:
            outdata[:to_copy] = state["current_data"][:to_copy]
            state["current_data"] = state["current_data"][to_copy:]
        if to_copy < frames:
            outdata[to_copy:].fill(0)
            state["started"] = False

    try:
        # blocksize 调小以降低系统缓冲
        with sd.OutputStream(samplerate=sample_rate, channels=1, callback=audio_callback, blocksize=512):
            if result_queue:
                result_queue.put(DecodeResult(msg_type="READY", task_id="speaker"))
            
            # 主循环：监听退出信号
            while True:
                time.sleep(0.1)
                # 虽然回调已经在读，但我们通过一个标记来判断是否该退出
                if state.get("stop"):
                    break
    except KeyboardInterrupt:
        pass # 静默退出
    except Exception as e:
        print(f"  ❌ [SpeakerWorker] 异常: {e}")
