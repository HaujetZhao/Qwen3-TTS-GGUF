import os
import time
import numpy as np
from ..schema.protocol import DecodeRequest, DecoderResponse, DecoderSession

def handle_decode_task(req: DecodeRequest, decoder, sessions, response_queue, record_queue=None):
    """处理单次解码任务 (顶层解耦函数)"""
    codes_all = np.array(req.codes, dtype=np.int64)
    if codes_all.ndim == 1:
        codes_all = codes_all.reshape(-1, 16)
    
    # 获取或初始化会话信息
    session = sessions.get(req.task_id, DecoderSession())
    current_state = session.state
    curr_index = session.index
    if current_state is None and req.state is not None:
        current_state = req.state
        
    is_task_final = req.is_final or (req.msg_type == "DECODE")
    
    try:
        t0 = time.time()
        # StatefulDecoder.decode 内部会自动处理分片和历史采样点抵消
        audio, new_state = decoder.decode(codes_all, state=current_state, is_final=is_task_final)
        dt = time.time() - t0
        
        # 更新会话状态
        session.state = new_state
        session.index += 1
        sessions[req.task_id] = session
        
        # 回传音频
        pcm_data = audio.copy() if len(audio) > 0 else np.array([], dtype=np.float32)
        response_queue.put(DecoderResponse(
            msg_type="AUDIO", 
            task_id=req.task_id, 
            index=curr_index, 
            audio=pcm_data, 
            compute_time=dt
        ))
        
        if len(audio) > 0 and record_queue:
            record_queue.put(audio.copy())
        
        if is_task_final:
            # 信号: FINISH 表示结束，携带最后的状态。使用当前的 index (已自增)
            response_queue.put(DecoderResponse(
                msg_type="FINISH", 
                task_id=req.task_id, 
                index=session.index, 
                state=new_state
            ))
            if req.task_id in sessions: del sessions[req.task_id]
            
    except Exception as e:
        print(f"⚠️ [DecoderWorker] 解码异常: {e}")
        traceback.print_exc()
        if req.task_id in sessions: del sessions[req.task_id]
        response_queue.put(DecoderResponse(msg_type="FINISH", task_id=req.task_id))


def decoder_worker_proc(codes_queue, pcm_queue, decoder_onnx_path, chunk_size=12, record_queue=None):
    """
    解码子进程工人 (DecoderWorker)。
    支持多会话状态管理 (Session-based State Management)。
    """
    from ..decoder import StatefulDecoder
    os.environ["OMP_NUM_THREADS"] = "4"
    
    decoder = StatefulDecoder(decoder_onnx_path, use_dml=True, chunk_size=chunk_size)
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
