"""
proxy.py - 解码器多进程代理
分离自 decoder.py，负责主进程与 Worker 之间的通过协议通信。
"""
import multiprocessing as mp
import atexit
import threading
import queue
import time
import numpy as np
from typing import Optional

from .protocol import DecodeRequest, DecodeResult, SpeakerRequest

class DecoderProxy:
    """
    解码器多进程代理 (DecoderProxy)。
    
    它负责在独立进程中拉起 DecoderWorker 和 SpeakerWorker，
    并提供线程安全的任务队列接口。
    """
    def __init__(self, onnx_path: str, use_dml: bool = True):
        self.onnx_path = onnx_path
        self.use_dml = use_dml
        
        # 任务控制
        self.task_counter = 0
        self.active_task_id = 0
        
        # 通讯队列 (传输 Protocol 对象)
        self.codes_q = mp.Queue()     # 主 -> Decoder (DecodeRequest)
        self.result_q = mp.Queue()    # Decoder -> Proxy (DecodeResult)
        self.play_q = mp.Queue()      # Proxy -> Playback (SpeakerRequest)
        
        # 进程对象
        self.decoder_proc = None
        self.play_proc = None
        
        # 结果监听线程 (负责从 result_q 收集数据)
        self.results = {}             # task_id -> list of (pcm, time)
        self.streaming_results = {}   # task_id -> bool
        self.ready_states = {"decoder": False, "speaker": False}
        self.stop_listener = False
        self.listener_thread = None
        
        self.start()
        
        # 注册自动退出逻辑
        atexit.register(self.shutdown)

    def start(self):
        """启动工作进程"""
        from qwen3_tts_gguf.workers import decoder_worker_proc, speaker_worker_proc
        
        # 1. 解调子进程 (Decoder)
        self.decoder_proc = mp.Process(
            target=decoder_worker_proc,
            args=(self.codes_q, self.result_q, self.onnx_path),
            daemon=True
        )
        self.decoder_proc.start()
        
        # 2. 播放子进程 (Speaker)
        # 监听独立的 play_q，并向 result_q 反馈就绪状态
        self.play_proc = mp.Process(
            target=speaker_worker_proc,
            args=(self.play_q, self.result_q),
            daemon=True
        )
        self.play_proc.start()
        
        # 3. 握手：子进程启动后会回传一条消息确认已就绪
        self._active_provider = "Pending..."
        
        # 4. 启动本地监听器
        self.listener_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listener_thread.start()

    @property
    def active_provider(self) -> str:
        """兼容性属性：返回后端名称"""
        return "Multiprocessing (Worker)"

    def _listen_loop(self):
        """从结果队列中抓取数据并分类转发"""
        while not self.stop_listener:
            try:
                msg: DecodeResult = self.result_q.get(timeout=0.1)
                if msg is None: break
                
                # 处理就绪信号
                if msg.msg_type == "READY":
                    # task_id 复用为 worker name: "decoder" or "speaker"
                    self.ready_states[str(msg.task_id)] = True 
                    continue
                
                # 正常音频数据
                task_id = msg.task_id
                pcm = msg.audio
                dt = msg.compute_time
                
                # 1. 存入结果字典供同步获取
                if task_id not in self.results:
                    self.results[task_id] = []
                self.results[task_id].append((pcm, dt))
                
                # 2. 如果该任务标记为流式播放，转发给播放进程
                if task_id in self.streaming_results:
                    if pcm is not None and len(pcm) > 0:
                        # 封装为 SpeakerRequest 发送
                        self.play_q.put(SpeakerRequest(task_id=task_id, msg_type="PLAY", audio=pcm))
            except queue.Empty:
                continue
            except Exception as e:
                break

    def reset(self):
        """重置子进程中的解码器状态"""
        self.active_task_id = self.task_counter
        self.task_counter += 1
        # 发送 RESET 指令
        req = DecodeRequest(task_id=self.active_task_id, msg_type="RESET")
        self.codes_q.put(req)

    def cancel_current(self):
        """取消当前任务 (新增)"""
        if self.active_task_id == 0: return
        
        # 1. 停止 Speaker
        self.play_q.put(SpeakerRequest(task_id=self.active_task_id, msg_type="STOP"))
        # 2. 停止 Decoder (尚未实现 CANCEL 类型，暂时还是用 RESET 近似替代或忽略)
        # TODO: 在 Protocol 中增加 CANCEL 类型

    def wait_until_ready(self, timeout=10):
        """阻塞直到所有工作进程就绪"""
        t0 = time.time()
        while time.time() - t0 < timeout:
            if all(self.ready_states.values()):
                return True
            time.sleep(0.1)
        return False

    def decode(self, codes: np.ndarray, is_final: bool = False, stream: bool = False) -> np.ndarray:
        """
        跨进程解码。
        
        如果 stream=True，则仅将任务推入队列，不等待结果。
        如果 stream=False，则阻塞直到获取本次任务的所有结果（离线模式）。
        """
        task_id = self.task_counter
        self.task_counter += 1
        
        msg_type = "DECODE_CHUNK" if stream else "DECODE"
        if stream:
            self.streaming_results[task_id] = True
            
        # 构造请求对象
        req = DecodeRequest(task_id=task_id, msg_type=msg_type, codes=codes, is_final=is_final)
        self.codes_q.put(req)
        
        if stream:
            return np.array([], dtype=np.float32)
        
        # 同步等待 (离线模式)
        start_wait = time.time()
        collected_pcm = []
        is_done = False
        while not is_done and (time.time() - start_wait < 30.0): # 30秒超时
            if task_id in self.results:
                msg_list = self.results[task_id]
                while msg_list:
                    pcm, dt = msg_list.pop(0)
                    if pcm is None:
                        is_done = True
                        break
                    collected_pcm.append(pcm)
            if not is_done:
                time.sleep(0.01)
        
        if task_id in self.results:
            del self.results[task_id]
            
        if not collected_pcm:
            return np.array([], dtype=np.float32)
        return np.concatenate(collected_pcm)

    def raw_play(self, pcm: np.ndarray):
        """直接向播放进程推送原始 PCM 数据 (24kHz, float32)"""
        if pcm is not None and len(pcm) > 0:
            # 这里的 task_id 暂时用 -1 表示系统级播放
            self.play_q.put(SpeakerRequest(task_id=-1, msg_type="PLAY", audio=pcm))

    def shutdown(self):
        """彻底关闭所有子进程，防止僵尸进程及主进程挂起"""
        self.stop_listener = True
        
        # 1. 向子进程发送毒丸
        try:
            if self.decoder_proc and self.decoder_proc.is_alive():
                self.codes_q.put(None)
            if self.play_proc and self.play_proc.is_alive():
                self.play_q.put(None) 
        except: pass
            
        # 2. 依次清理子进程 (硬限时 join + terminate)
        for p in [self.decoder_proc, self.play_proc]:
            if p and p.is_alive():
                p.join(timeout=0.3) 
                if p.is_alive():
                    try: p.terminate()
                    except: pass
        
        # 3. 停止监听线程
        if self.listener_thread:
            self.listener_thread.join(timeout=0.3)
            
        # 4. 清理并销毁队列 (取消 join_thread 以防主线程在此挂起)
        for q in [self.codes_q, self.result_q, self.play_q]:
            try:
                q.cancel_join_thread() # 关键：不强制等待缓冲区数据刷完，主进程可立即退出
                while not q.empty():
                    q.get_nowait()
                q.close()
            except: pass
