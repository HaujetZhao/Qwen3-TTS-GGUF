"""
proxy.py - 解码器多进程代理
分离自 decoder.py，负责主进程与 Worker 之间的协议通信。
"""
import multiprocessing as mp
import atexit
import threading
import queue
import time
import numpy as np
from typing import Optional, Union
from .schema.result import TTSResult, DecodeResult

from .schema.protocol import DecodeRequest, DecoderResponse, SpeakerRequest, SpeakerResponse

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
        
        # 通讯队列
        self.codes_q = mp.Queue()     # 主 -> Decoder (DecodeRequest)
        self.result_q = mp.Queue()    # Decoder/Speaker -> Proxy (DecoderResponse / SpeakerResponse)
        self.play_q = mp.Queue()      # Proxy -> Playback (SpeakerRequest)
        
        # 进程对象
        self.decoder_proc = None
        self.play_proc = None
        
        # 状态存储
        self.results = {}             # task_id -> list of np.ndarray
        self.events = {}              # task_id -> threading.Event
        self.streaming_results = {}   # task_id -> bool
        self.ready_states = {"decoder": False, "speaker": False}
        self.speaker_status = "IDLE"  # IDLE, PLAYING, PAUSED
        self.speaker_idle = threading.Event()
        self.speaker_idle.set() # 初始为就绪/闲置状态
        
        # 解码进度追踪
        self.active_decoder_tasks = set()
        self.decoder_idle = threading.Event()
        self.decoder_idle.set()
        
        self.stop_listener = False
        self.listener_thread = None
        
        self.start()
        
        # 注册自动退出逻辑
        atexit.register(self.shutdown)

    def start(self):
        """启动工作进程"""
        from .workers.decoder import decoder_worker_proc
        from .workers.speaker import speaker_worker_proc
        
        # 1. 解调子进程 (Decoder)
        self.decoder_proc = mp.Process(
            target=decoder_worker_proc,
            args=(self.codes_q, self.result_q, self.onnx_path),
            daemon=True
        )
        self.decoder_proc.start()
        
        # 2. 播放子进程 (Speaker)
        self.play_proc = mp.Process(
            target=speaker_worker_proc,
            args=(self.play_q, self.result_q),
            daemon=True
        )
        self.play_proc.start()
        
        # 3. 启动本地监听器
        self.listener_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listener_thread.start()

    def _listen_loop(self):
        """从结果队列中抓取数据并分类转发 (后台储蓄罐)"""
        while not self.stop_listener:
            try:
                msg = self.result_q.get(timeout=0.1)
                if msg is None: break
                
                if isinstance(msg, DecoderResponse):
                    self._handle_decoder_msg(msg)
                elif isinstance(msg, SpeakerResponse):
                    self._handle_speaker_msg(msg)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️ [Proxy] 监听异常: {e}")
                import traceback
                traceback.print_exc()
                break

    def _handle_decoder_msg(self, msg: DecoderResponse):
        """处理来自解码器的消息"""
        if msg.msg_type == "READY":
            self.ready_states["decoder"] = True
            return
            
        task_id = msg.task_id
        
        # A1. 消息累积 (储蓄罐)
        if msg.msg_type == "AUDIO":
            msg.recv_time = time.time() # 记录 Proxy 收到音频的具体时刻
            if task_id not in self.results:
                self.results[task_id] = []
            
            # 存储完整消息对象，以保留 compute_time
            self.results[task_id].append(msg)
            
            # 如果该任务标记为实时流式播放，同步转发给播放器线路
            if self.streaming_results.get(task_id):
                self.play_q.put(SpeakerRequest(msg_type="AUDIO", audio=msg.audio))

        # A2. 收到结束信号 (闹钟)
        elif msg.msg_type == "FINISH":
            if task_id in self.events:
                self.events[task_id].set()
            
            # 从活跃解码集合中移除
            if task_id in self.active_decoder_tasks:
                self.active_decoder_tasks.remove(task_id)
                if not self.active_decoder_tasks:
                    self.decoder_idle.set()

    def _handle_speaker_msg(self, msg: SpeakerResponse):
        """处理来自播放器的消息"""
        if msg.msg_type == "READY":
            self.ready_states["speaker"] = True
        elif msg.msg_type == "STARTED":
            self.speaker_status = "PLAYING"
            self.speaker_idle.clear()
        elif msg.msg_type == "FINISHED":
            self.speaker_status = "IDLE"
            self.speaker_idle.set()
        elif msg.msg_type == "PAUSED":
            self.speaker_status = "PAUSED"
            self.speaker_idle.set()


    def wait_until_ready(self, timeout=10):
        """阻塞直到所有工作进程就绪"""
        t0 = time.time()
        while time.time() - t0 < timeout:
            if all(self.ready_states.values()):
                return True
            time.sleep(0.1)
        return False

    def join_speaker(self, timeout: Optional[float] = None):
        """阻塞等待播放器列表任务清空"""
        self.speaker_idle.wait(timeout=timeout)

    def join_decoder(self, timeout: Optional[float] = None):
        """阻塞等待解码器完成所有当前任务"""
        self.decoder_idle.wait(timeout=timeout)

    def decode(self, input: Union[np.ndarray, TTSResult], task_id="default", is_final: bool = False, stream: bool = False) -> np.ndarray:
        """
        累积式跨进程解码。
        
        1. 流式推送包 (stream=True): 立即返回空数组，后台自动累积数据。
        2. 离线/终结包 (stream=False 或 is_final=True): 阻塞等待所有片段到齐，拼接并返回完整音频。
        3. 支持传入 TTSResult: 自动提取 codes，并在完成后写回 wav 属性及耗时统计。
        """
        # 参数预处理
        if isinstance(input, TTSResult):
            codes = input.codes
            is_final = True # 对象解码默认为离线完成模式
        else:
            codes = input

        t_start = time.time()
        
        # 初始化状态位与储蓄罐
        if task_id not in self.results:
            self.results[task_id] = []
        if task_id not in self.events:
            self.events[task_id] = threading.Event()
        else:
            self.events[task_id].clear()
            
        if stream:
            self.streaming_results[task_id] = True
            
        # 追踪活跃任务
        if is_final or not stream:
            self.active_decoder_tasks.add(task_id)
            self.decoder_idle.clear()

        # 构造并发送请求
        # 对于 stream=False (离线模式)，Worker 内部会根据 msg_type="DECODE" 自动判定 is_final
        msg_type = "DECODE_CHUNK" if stream else "DECODE"
        req = DecodeRequest(task_id=task_id, msg_type=msg_type, codes=codes, is_final=is_final)
        self.codes_q.put(req)
        
        # 逻辑分支 A: 中间的流式包，无需等待
        if stream and not is_final:
            return np.array([], dtype=np.float32)
        
        # 逻辑分支 B: 离线模式或流式终包，开始同步等待
        self.events[task_id].wait(timeout=30.0)
        
        # 从消息列表中提取音频碎片和统计信息
        responses = self.results.get(task_id, [])
        
        # 结果打包
        result = DecodeResult(responses=responses)
            
        # 清理该任务的残留资源
        if task_id in self.results: del self.results[task_id]
        if task_id in self.events: del self.events[task_id]
        if task_id in self.streaming_results: del self.streaming_results[task_id]

        # 结果回写 (如果输入是对象)
        if isinstance(input, TTSResult):
            input.audio = result.audio
            if input.stats:
                input.stats.decoder_compute_times = result.chunk_compute_times
        
        return result

    def get_decode_result(self, task_id) -> DecodeResult:
        """从储蓄罐中提取已有的 DecodeResult（用于流式过程中的快照统计）"""
        responses = self.results.get(task_id, [])
        return DecodeResult(responses=responses)

    def pause(self):
        """发送暂停指令"""
        self.play_q.put(SpeakerRequest(msg_type="PAUSE"))
        self.speaker_status = "PAUSED"

    def resume(self):
        """发送继续指令"""
        self.play_q.put(SpeakerRequest(msg_type="CONTINUE"))

    def stop(self, task_id="default") -> np.ndarray:
        """
        显式停止并清理特定任务，返回已累积的音频。
        1. 通知 Decoder 丢弃该 Session 缓存。
        2. 通知 Speaker 截断播放并清空硬件缓冲。
        3. 提取并返回 Proxy 本地的累积片段，随后清理缓存。
        """
        # A. Decoder 线路停止
        self.codes_q.put(DecodeRequest(task_id=task_id, msg_type="STOP"))
        
        # B. Speaker 线路停止 (Speaker 本身不感 ID)
        self.play_q.put(SpeakerRequest(msg_type="STOP"))
        self.speaker_status = "IDLE"
        self.speaker_idle.set()
        
        # C. 提取已累积的音频
        collected_pcm = self.results.get(task_id, [])
        final_pcm = np.concatenate(collected_pcm) if collected_pcm else np.array([], dtype=np.float32)

        # D. Proxy 本地清理
        if task_id in self.active_decoder_tasks:
            self.active_decoder_tasks.remove(task_id)
            if not self.active_decoder_tasks:
                self.decoder_idle.set()

        if task_id in self.results: del self.results[task_id]
        if task_id in self.events: 
            self.events[task_id].set() # 唤醒可能阻塞在 decode 的线程
            del self.events[task_id]
        if task_id in self.streaming_results: del self.streaming_results[task_id]

        return final_pcm

    def raw_play(self, pcm: np.ndarray):
        """直接推送 PCM"""
        if pcm is not None and len(pcm) > 0:
            self.play_q.put(SpeakerRequest(msg_type="AUDIO", audio=pcm))

    def shutdown(self):
        """关闭所有子进程"""
        self.stop_listener = True
        
        # 1. 向子进程发送毒丸
        try:
            if self.decoder_proc and self.decoder_proc.is_alive():
                self.codes_q.put(None)
            if self.play_proc and self.play_proc.is_alive():
                self.play_q.put(SpeakerRequest(msg_type="EXIT")) 
        except: pass
            
        # 2. 依次清理子进程
        for p in [self.decoder_proc, self.play_proc]:
            if p and p.is_alive():
                p.join(timeout=0.3) 
                if p.is_alive():
                    try: p.terminate()
                    except: pass
        
        # 3. 停止监听线程
        if self.listener_thread:
            self.listener_thread.join(timeout=0.3)
            
        # 4. 清理并销毁队列
        for q in [self.codes_q, self.result_q, self.play_q]:
            try:
                q.cancel_join_thread()
                while not q.empty():
                    q.get_nowait()
                q.close()
            except: pass
