import os
import time
import numpy as np
import queue
import soundfile as sf

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

def decoder_worker_proc(codes_queue, pcm_queue, mouth_onnx_path, record_queue=None):
    """
    解码子进程工人。
    
    请求协议: (type, task_id, payload, is_final)
    响应协议: (type, task_id, pcm_data, compute_time)
    """
    from qwen3_tts_gguf.mouth_decoder import StatefulMouthDecoder
    
    # 强制关闭多线程竞争，防止干扰主进程
    os.environ["OMP_NUM_THREADS"] = "4"
    
    decoder = StatefulMouthDecoder(mouth_onnx_path, use_dml=True)
    # 向 Proxy 发送就绪信号
    pcm_queue.put(("READY", "mouth", None, 0))
    print(f"  🔊 [MouthWorker] 已就绪 (Provider: {decoder.active_provider})")
    
    try:
        while True:
            msg = codes_queue.get()
            
            # 毒丸：退出信号
            if msg is None:
                pcm_queue.put(None)
                if record_queue: record_queue.put(None)
                break
                
            msg_type, task_id, payload, is_final = msg
            
            if msg_type == "RESET":
                decoder.reset()
                if record_queue: record_queue.put("CLEAR")
                continue
            
            if msg_type in ["DECODE", "DECODE_CHUNK"]:
                codes_all = np.array(payload, dtype=np.int64)
                if codes_all.ndim == 1:
                    codes_all = codes_all.reshape(-1, 16)
                
                # 自动切分超长序列 (防止单次推理显存溢出或超时)
                chunk_step = 50
                n_total = codes_all.shape[0]
                
                try:
                    for start_idx in range(0, n_total, chunk_step):
                        end_idx = min(start_idx + chunk_step, n_total)
                        is_last_chunk = (end_idx == n_total)
                        # 仅在最后一个 chunk 且任务本身是 is_final 时，传递 is_final=True
                        current_is_final = is_final and is_last_chunk
                        
                        codes = codes_all[start_idx:end_idx]
                        
                        t0 = time.time()
                        audio = decoder.decode(codes, is_final=current_is_final)
                        dt = time.time() - t0
                        
                        # 回传结果
                        if len(audio) > 0:
                            pcm_queue.put(("AUDIO", task_id, audio.copy(), dt))
                            if record_queue: record_queue.put(audio.copy())
                        else:
                            # 即使没产出音频也推个空的，维持心跳
                            pcm_queue.put(("AUDIO", task_id, np.array([], dtype=np.float32), dt))
                    
                    # 如果是同步任务，且已处理完，发送一个特殊的结束标识 (pcm=None)
                    if msg_type == "DECODE":
                        pcm_queue.put(("AUDIO", task_id, None, 0))
                        
                except Exception as e:
                    print(f"  ⚠️ [MouthWorker] 解码异常: {e} (已触发状态重置)")
                    decoder.reset()
                    # 回传空音频及结束标识以防主进程死等
                    pcm_queue.put(("AUDIO", task_id, np.array([], dtype=np.float32), 0))
                    if msg_type == "DECODE":
                        pcm_queue.put(("AUDIO", task_id, None, 0))

    except KeyboardInterrupt:
        pass # 静默退出
    except Exception as e:
        print(f"  ❌ [MouthWorker] 崩溃: {e}")
        import traceback
        traceback.print_exc()


def speaker_worker_proc(pcm_queue, result_queue=None, sample_rate=24000):
    import sounddevice as sd
    
    state = {"current_data": np.zeros((0, 1), dtype=np.float32), "started": False, "prefill": 1200} # 50ms prefill
    def audio_callback(outdata, frames, time_info, status):
        # 抓取当前所有可用数据
        while True:
            try:
                new_item = pcm_queue.get_nowait()
                if new_item is None:
                    state["stop"] = True
                    break
                if not isinstance(new_item, np.ndarray): continue
                # 打印调试信息 (仅在第一次接收到数据时)
                if len(state["current_data"]) == 0:
                    pass 
                state["current_data"] = np.concatenate([state["current_data"], new_item.reshape(-1, 1).astype(np.float32)], axis=0)
            except queue.Empty: break
            
        if not state["started"]:
            if len(state["current_data"]) >= state["prefill"]: 
                state["started"] = True
                # print("  🔔 [Speaker] 开始物理输出...") # 回调内不建议 print，但调试可用
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
                result_queue.put(("READY", "speaker", None, 0))
            
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
