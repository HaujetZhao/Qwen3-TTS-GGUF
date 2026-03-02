"""
protocol.py - 进程间通信协议定义
"""
from dataclasses import dataclass, field
from typing import Optional, Union, List
import numpy as np

@dataclass
class DecoderState:
    """解码器的核心记忆 (不跨进程传输，仅在 Worker 内部流转)"""
    pre_conv_history: Optional[np.ndarray] = None
    latent_buffer: Optional[np.ndarray] = None
    conv_history: Optional[np.ndarray] = None
    kv_cache: List[np.ndarray] = field(default_factory=list)

@dataclass
class DecodeRequest:
    """主进程 -> DecoderWorker"""
    task_id: Union[str, int]
    msg_type: str = "DECODE"  # DECODE, DECODE_CHUNK, STOP, RESET
    codes: Optional[np.ndarray] = None
    is_final: bool = False

@dataclass
class DecoderResponse:
    """DecoderWorker -> Proxy"""
    task_id: Union[str, int]
    msg_type: str = "AUDIO"   # AUDIO, FINISH, READY, ERROR
    audio: Optional[np.ndarray] = None
    compute_time: float = 0.0
    recv_time: float = 0.0    # Proxy 接收到消息的时间 (由 Proxy 填充)

@dataclass
class SpeakerResponse:
    """SpeakerWorker -> Proxy"""
    msg_type: str = "READY"   # READY, STARTED, FINISHED, PAUSED

@dataclass
class SpeakerRequest:
    """Proxy -> SpeakerWorker"""
    msg_type: str = "AUDIO"    # AUDIO, STOP, EXIT, PAUSE, CONTINUE
    audio: Optional[np.ndarray] = None
