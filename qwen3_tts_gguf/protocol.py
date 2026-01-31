"""
protocol.py - 进程间通信协议定义
"""
from dataclasses import dataclass, field
from typing import Optional, Union, List
import numpy as np

@dataclass
class IPCPayload:
    """所有 IPC 消息的基类"""
    task_id: Union[str, int]

@dataclass
class DecoderState:
    """解码器的核心记忆 (不跨进程传输，仅在 Worker 内部流转)"""
    pre_conv_history: Optional[np.ndarray] = None
    latent_buffer: Optional[np.ndarray] = None
    conv_history: Optional[np.ndarray] = None
    kv_cache: List[np.ndarray] = field(default_factory=list)

@dataclass
class DecodeRequest(IPCPayload):
    """主进程 -> DecoderWorker"""
    msg_type: str = "DECODE"  # DECODE, DECODE_CHUNK, RESET
    codes: Optional[np.ndarray] = None
    is_final: bool = False

@dataclass
class DecodeResult(IPCPayload):
    """DecoderWorker -> Proxy"""
    msg_type: str = "AUDIO"   # AUDIO, READY, ERROR
    audio: Optional[np.ndarray] = None
    compute_time: float = 0.0

@dataclass
class SpeakerRequest(IPCPayload):
    """Proxy -> SpeakerWorker"""
    msg_type: str = "PLAY"    # PLAY, STOP
    audio: Optional[np.ndarray] = None
