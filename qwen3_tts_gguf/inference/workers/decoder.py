import os
import time
import numpy as np
from ..schema.protocol import DecodeRequest, DecoderResponse

def handle_decode_task(req: DecodeRequest, decoder, sessions, pcm_queue, record_queue=None):
    """处理单次解码任务 (顶层解耦函数)"""
    codes_all = np.array(req.codes, dtype=np.int64)
    if codes_all.ndim == 1:
        codes_all = codes_all.reshape(-1, 16)
    
    current_state = sessions.get(req.task_id)
    is_task_final = req.is_final or (req.msg_type == "DECODE")
    chunk_step = 50
    n_total = codes_all.shape[0]
    
    try:
        for start_idx in range(0, n_total, chunk_step):
            end_idx = min(start_idx + chunk_step, n_total)
            is_last_chunk = (end_idx == n_total)
            current_is_final = is_task_final and is_last_chunk
            
            codes = codes_all[start_idx:end_idx]
            t0 = time.time()
            audio, new_state = decoder.decode(codes, state=current_state, is_final=current_is_final)
            dt = time.time() - t0
            
            sessions[req.task_id] = new_state
            current_state = new_state
            
            # 回传音频
            pcm_data = audio.copy() if len(audio) > 0 else np.array([], dtype=np.float32)
            pcm_queue.put(DecoderResponse(msg_type="AUDIO", task_id=req.task_id, audio=pcm_data, compute_time=dt))
            if len(audio) > 0 and record_queue:
                record_queue.put(audio.copy())
        
        if is_task_final:
            if req.task_id in sessions: del sessions[req.task_id]
            # 信号: FINISH 表示结束
            pcm_queue.put(DecoderResponse(msg_type="FINISH", task_id=req.task_id))
            
    except Exception as e:
        print(f"⚠️ [DecoderWorker] 解码异常: {e}")
        traceback.print_exc()
        if req.task_id in sessions: del sessions[req.task_id]
        pcm_queue.put(DecoderResponse(msg_type="FINISH", task_id=req.task_id))


def decoder_worker_proc(codes_queue, pcm_queue, decoder_onnx_path, record_queue=None):
    """
    解码子进程工人 (DecoderWorker)。
    支持多会话状态管理 (Session-based State Management)。
    """
    from ..decoder import StatefulDecoder
    os.environ["OMP_NUM_THREADS"] = "4"
    
    decoder = StatefulDecoder(decoder_onnx_path, use_dml=True)
    pcm_queue.put(DecoderResponse(msg_type="READY", task_id="decoder"))
    print(f"🔊 [DecoderWorker] 已就绪 (Provider: {decoder.active_provider})")
    
    sessions = {} # {task_id: DecoderState}

    try:
        while True:
            req: DecodeRequest = codes_queue.get()
            if req is None:
                pcm_queue.put(None)
                if record_queue: record_queue.put(None)
                break
                
            if req.msg_type in ["STOP", "RESET"]:
                if req.task_id in sessions: 
                    del sessions[req.task_id]
                if record_queue: record_queue.put("CLEAR")
                continue
            
            if req.msg_type in ["DECODE", "DECODE_CHUNK"]:
                handle_decode_task(req, decoder, sessions, pcm_queue, record_queue)
                
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"❌ [DecoderWorker] 崩溃: {e}")
        import traceback
        traceback.print_exc()
