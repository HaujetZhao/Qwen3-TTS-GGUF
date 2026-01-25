import torch
import torch.nn as nn
from . import logger
import torch.nn as nn
# 假设调用者已将 Qwen3-TTS 父目录或本身加入 sys.path
try:
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model
except ImportError:
    # Fallback/Warning if path not set
    Qwen3TTSTokenizerV2Model = None


class CodecEncoderExportWrapper(nn.Module):
    """
    Qwen3-TTS 音频编码器导出包装类。
    
    职责：
    1. 接收原始波形 (input_values)。
    2. 执行编码逻辑，输出离散码。
    """
    def __init__(self, model: Qwen3TTSTokenizerV2Model):
        super().__init__()
        self.encoder = model.encoder
        self.config = model.config
        self.encoder.eval()

    def forward(self, input_values):
        """
        Args:
            input_values (torch.FloatTensor): Shape [Batch, Time] (raw waveform)
            
        Returns:
            audio_codes (torch.LongTensor): Shape [Batch, Time, Q]
        """
        # 1. 增加 Channel 维度: [B, T] -> [B, 1, T]
        x = input_values.unsqueeze(1)
        
        # 2. 编码
        # outputs 是 Qwen3TTSTokenizerV2EncoderOutput
        # audio_codes 是 list of tensors? 还是 tensor?
        # 根据源代码: recommended to use return_dict=True
        # 但是我们直接调用 self.encoder.encode
        
        # 查看 modeling_qwen3_tts_tokenizer_v2.py: 
        # encoder.encode(input_values=..., return_dict=True) -> keys: audio_codes [List] ?
        # Wait, encoder.encode returns MimiEncoderOutput-like object?
        # 源代码: 
        # encoded_frames = self.encoder.encode(input_values=input_values.unsqueeze(1), return_dict=True)
        # audio_codes = encoded_frames.audio_codes[:, :self.encoder_valid_num_quantizers]
        
        # 我们不能直接调用 encoder.encode 因为它返回的是中间特征。
        # 我们需要模拟 Qwen3TTSTokenizerV2Model.encode 的逻辑：
        # encoded_frames = self.encoder.encode(...)
        # audio_codes = encoded_frames.audio_codes ...
        
        # 这里的 self.encoder 是 Qwen3TTSTokenizerV2Encoder (MimiModel subclass)
        # MimiModel.encode returns [audio_codes, ...]
        
        # 最好是直接调用 wrapper 里的逻辑来匹配 model.encode 的行为。
        
        encoded_frames = self.encoder(input_values=x, return_dict=True)
        # encoded_frames from MimiModel forward/encode usually has 'audio_codes' if it's quantization enabled?
        # Qwen3 implementation uses `self.encoder.encode` which seems to come from MimiModel.
        
        # 修正：直接用 Qwen3TTSTokenizerV2Model.encode 的逻辑复刻比较安全，或者直接 trace model.encode
        # 但 model.encode 有很多预处理。
        # 让我们只需 model.encoder.encode (MimiModel 的方法)。
        
        # 在 modeling...py 中:
        # self.encoder = Qwen3TTSTokenizerV2Encoder...
        # encode 方法: self.encoder.encode(input_values=input_values.unsqueeze(1), return_dict=True)
        
        outputs = self.encoder.encode(x, return_dict=True)
        
        # outputs.audio_codes 是一个 Tensor [B, Q, T] (MimiModel output)
        codes = outputs.audio_codes
        
        # Qwen3 逻辑：
        # audio_codes = encoded_frames.audio_codes[:, :self.encoder_valid_num_quantizers]
        # (截取有效量化器数量)
        valid_q = self.config.encoder_valid_num_quantizers
        codes = codes[:, :valid_q, :]
        
        # Qwen3 逻辑: transpose to [B, T, Q] ?
        # 11-Export 脚本里我们没看到 encode 的详细逻辑，inference.py 里调用了 model.encode
        # inference.py: res.audio_codes shape: [B, T, Q] (注释里写的)
        
        # 检查 modeling...py: `audio_codes = [code...transpose(0, 1)...]` it returns list of codes if padding mask used?
        # 如果没有 padding mask，它应该是一个 Tensor。
        # 为了 ONNX，我们希望是一个 Tensor [B, T, Q]。
        
        codes = codes.transpose(1, 2) # [B, Q, T] -> [B, T, Q]
        return codes

class CodecExportWrapper(nn.Module):
    """
    Qwen3-TTS 音频解码器导出包装类。
    
    职责：
    1. 接收与原始模型一致的输入格式 (audio_codes)。
    2. 执行解码逻辑，绕过 Python 层面的 chunked_decode，直接导出核心推理图。
    3. 确保输出格式符合预期 (waveform)。
    
    遵循原则：
    - 高内聚：封装所有模型特定的调整逻辑。
    - 语义清晰：forward 方法清晰描述了“从 Code 到 Audio”的过程。
    """
    def __init__(self, model: Qwen3TTSTokenizerV2Model):
        super().__init__()
        # 只持有需要导出的子模块，减少耦合
        self.decoder = model.decoder
        self.config = model.config
        
        # 确保处于评估模式
        self.decoder.eval()

    def forward(self, audio_codes):
        """
        Args:
            audio_codes (torch.LongTensor):Shape [Batch, Time, NumQuantizers]
            
        Returns:
            audio_values (torch.FloatTensor): Shape [Batch, Time]
        """
        # 1. 转置 Input: [B, T, Q] -> [B, Q, T]
        # 这是为了匹配 Qwen3TTSTokenizerV2Decoder.forward 的输入要求
        codes = audio_codes.transpose(1, 2)
        
        # 2. 调用核心解码器
        # 注意：这里我们不使用 chunked_decode，因为 ONNX 导出通常希望导出一个完整的计算图。
        # 推理时的 Chunking 应该由调用者控制，或者输入的长度本身就是 Chunk。
        wav = self.decoder(codes)
        
        # 3. 后处理
        # decoder 输出为 [B, 1, T_out] (unsqueeze 过的) 或者 [B, T_out]
        # 查看源代码: return wav.clamp(min=-1, max=1)
        # 形状溯源: last conv is Conv1d(..., 1, ...), so [B, 1, T]
        
        return wav.squeeze(1)

def load_kv_cache_model(model_path):
    """
    辅助函数：加载模型并准备导出。
    """
    # 动态导入，避免硬编码依赖路径，假设 Qwen3-TTS 在 sys.path 或相对路径可达
    # 在 11-Export... 脚本中需要处理 sys.path
    pass
