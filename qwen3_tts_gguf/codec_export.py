import torch
import torch.nn as nn
from . import logger

# 引入本地模型定义
try:
    from .tokenizer_12hz.modeling_tokenizer import Qwen3TTSTokenizerV2Model
except ImportError:
    Qwen3TTSTokenizerV2Model = None
    logger.warning("Could not import Qwen3TTSTokenizerV2Model from .modeling_tokenizer")

class CodecEncoderExportWrapper(nn.Module):
    """
    Qwen3-TTS 音频编码器导出包装类。
    
    【核心修复 V3】
    针对 JIT Trace 报 `unordered_map` 的问题，这通常是因为 Transformers 的子模块
    （MimiEncoder 或 Quantizer）内部返回了 ModelOutput 对象（类似字典）。
    
    本类采取以下策略：
    1. 强制覆盖 config.return_dict = False。
    2. 对于 Quantizer 这种可能不遵守 config 的组件，尝试通过属性访问手动剥离对象。
    """
    def __init__(self, model: Qwen3TTSTokenizerV2Model):
        super().__init__()
        self.config = model.config
        
        # 1. 强力覆写配置
        self.config.return_dict = False
        self.config.use_cache = False
        
        # 2. 提取核心模块 (MimiModel 内部组件)
        self.mimi_model = model.encoder
        self.cnn_encoder = self.mimi_model.encoder 
        self.transformer = self.mimi_model.encoder_transformer
        self.downsample = self.mimi_model.downsample
        self.quantizer = self.mimi_model.quantizer
        
        # 3. 锁定为评估模式，并递归禁用 return_dict/use_cache
        self.eval()
        for m in self.modules():
            if hasattr(m, 'config'):
                m.config.return_dict = False
                m.config.use_cache = False

    def forward(self, input_values):
        """
        Args:
            input_values (torch.FloatTensor): Shape [Batch, Time]
        Returns:
            audio_codes (torch.LongTensor): Shape [Batch, Time, Q]
        """
        # 1. Input: [B, T] -> [B, 1, T]
        x = input_values.unsqueeze(1)
        
        # 2. Step 1: CNN Encoder (SEANet)
        # Returns [B, Hidden, T_inner]
        cnn_out = self.cnn_encoder(x)
        if isinstance(cnn_out, (tuple, list)):
            hidden_states = cnn_out[0]
        else:
            hidden_states = cnn_out

        # 3. Step 2: Transformer Encoder
        # Transformer expects [B, T_inner, Hidden]
        hidden_states = hidden_states.transpose(1, 2)
        
        # return_dict=False is critical here
        trans_out = self.transformer(hidden_states, return_dict=False)
        
        if isinstance(trans_out, (tuple, list)):
            hidden_states = trans_out[0]
        else:
            hidden_states = trans_out
            
        # 4. Step 3: Downsample (if configured)
        # Downsample expects [B, Hidden, T_inner]
        hidden_states = hidden_states.transpose(1, 2)
        if self.downsample is not None:
            hidden_states = self.downsample(hidden_states)

        # 5. Step 4: Quantizer (RVQ)
        # MimiResidualVectorQuantizer.encode returns [Q, B, T]
        codes = self.quantizer.encode(hidden_states)
        
        # Defense against potential tuple/ModelOutput (though RVQ usually returns Tensor)
        if isinstance(codes, (tuple, list)):
            codes = codes[0]
        elif hasattr(codes, 'audio_codes'):
            codes = codes.audio_codes

        # 6. Qwen3 Slice Logic: Just take the valid number of quantizers
        valid_q = self.config.encoder_valid_num_quantizers
        codes = codes[:valid_q, :, :] # Slice first dim [Q]
        
        # 7. Transpose to final format: [Q, B, T] -> [B, T, Q]
        codes = codes.transpose(0, 1).transpose(1, 2)
        
        return codes

class CodecExportWrapper(nn.Module):
    """
    Qwen3-TTS 音频解码器导出包装类 (保持不变)。
    """
    def __init__(self, model: Qwen3TTSTokenizerV2Model):
        super().__init__()
        self.decoder = model.decoder
        self.config = model.config
        self.decoder.eval()

    def forward(self, audio_codes):
        # Input: [B, T, Q] -> [B, Q, T]
        codes = audio_codes.transpose(1, 2)
        wav = self.decoder(codes)
        # Output: [B, 1, T] -> [B, T]
        return wav.squeeze(1)

class StreamingCodecExportWrapper(nn.Module):
    """
    Qwen3-TTS 音频解码器流式验证包装类。
    内部模拟逐帧/逐片推理，手动管理状态，但对外接口保持为 (audio_codes)。
    用于验证有状态拆解逻辑是否与原版等价。
    """
    def __init__(self, model: Qwen3TTSTokenizerV2Model):
        super().__init__()
        self.decoder = model.decoder
        self.config = model.config
        self.decoder.eval()

    def forward(self, audio_codes):
        """
        验证逻辑：
        1. 内部模拟逐帧调用 Transformer 并传递 KV Cache。
        2. 卷积层暂时保持全序列处理（因为卷积状态的管理逻辑在下一阶段）。
        """
        # 1. 映射为向量 [B, Q, T]
        codes = audio_codes.transpose(1, 2)
        hidden = self.decoder.quantizer.decode(codes)
        
        # 2. 初步卷积处理 [B, Hidden, T] -> [B, T, Hidden]
        hidden = self.decoder.pre_conv(hidden).transpose(1, 2)
        
        # 3. 核心：模拟流式 Transformer 缓存机制
        from transformers.cache_utils import DynamicCache
        B, T, H = hidden.shape
        past_key_values = DynamicCache()
        all_hidden = []
        
        for i in range(T):
            # 逐帧取特征
            frame_hidden = hidden[:, i:i+1, :]
            # position_ids 会在 pre_transformer 内部根据 past_key_values 长度自动计算
            outputs = self.decoder.pre_transformer(
                inputs_embeds=frame_hidden,
                past_key_values=past_key_values,
                use_cache=True
            )
            all_hidden.append(outputs.last_hidden_state)
            past_key_values = outputs.past_key_values
            
        # 合并 Transformer 的输出 [B, T, Hidden]
        hidden = torch.cat(all_hidden, dim=1)
        
        # 4. 后续卷积与上采样流水线
        hidden = hidden.permute(0, 2, 1) # [B, Hidden, T]
        
        # 依次通过 DecoderDecoderBlock
        for blocks in self.decoder.upsample:
            for block in blocks:
                hidden = block(hidden)
        
        wav = hidden
        for block in self.decoder.decoder:
            wav = block(wav)
            
        # 返回波形 [B, T_audio]
        return wav.squeeze(1).clamp(min=-1, max=1)

class StatefulCodecExportWrapper(nn.Module):
    """
    Qwen3-TTS 终极有状态流式包装器。
    """
    
    # ========== 物理常数 (针对 12Hz 模型) ==========
    # 前瞻帧数：为保证卷积链条对齐所需的未来最小上下文。
    # 物理层级 4 层上采样决定了理论最小值为 4 帧。
    LOOKAHEAD_FRAMES = 4
    
    # 预处理层 pre_conv 使用 kernel_size=3，因此固定需要 2 帧历史。
    PRE_CONV_HISTORY_WINDOW = 2
    # 后端卷积链历史，测试发现 4 帧即可满足状态恢复
    CONV_HISTORY_WINDOW = 4
    # KV Cache 滑动窗口 (根据模型 config.sliding_window 设置)
    KV_CACHE_WINDOW_SIZE = 72
    # ===============================================

    def __init__(self, model: Qwen3TTSTokenizerV2Model):
        super().__init__()
        self.decoder = model.decoder
        self.config = model.config
        self.decoder.eval()
        # 24000 / 12.5 = 1920
        self.samples_per_frame = self.config.decode_upsample_rate

    def forward(self, audio_codes, past_key_values=None, latent_buffer=None, 
                conv_history=None, pre_conv_history=None, is_last_chunk=False):
        """
        Qwen3-TTS 流式推理核心逻辑 (已修复 pre_conv 断层与对齐问题)。
        """
        B, N, Q = audio_codes.shape
        device = audio_codes.device
        
        # 1. 码本解码与预处理历史
        codes = audio_codes.transpose(1, 2)
        quantized = self.decoder.quantizer.decode(codes) # [B, Dim, N]
        
        if pre_conv_history is not None:
            # 拼接历史以供 pre_conv 使用
            quant_full = torch.cat([pre_conv_history, quantized], dim=-1)
            h_quant_len = pre_conv_history.shape[-1]
        else:
            quant_full = quantized
            h_quant_len = 0
            
        # 执行 pre_conv [B, Hidden, h+N]
        hidden_all = self.decoder.pre_conv(quant_full)
        # 裁切掉历史部分
        hidden = hidden_all[:, :, h_quant_len:].transpose(1, 2) # [B, N, Hidden]
        
        # 保存新的 pre_conv 历史
        next_pre_conv_hist = quantized[:, :, -self.PRE_CONV_HISTORY_WINDOW:].clone() if not is_last_chunk else None
        
        # 2. Transformer 推理 (有状态)
        outputs = self.decoder.pre_transformer(
            inputs_embeds=hidden,
            past_key_values=past_key_values,
            use_cache=True
        )
        new_hidden = outputs.last_hidden_state.transpose(1, 2) # [B, Hidden, N]
        next_pkv = outputs.past_key_values
        
        # KV Cache 滑动窗口裁剪 (极简实现)
        if next_pkv is not None:
            next_pkv.crop(self.KV_CACHE_WINDOW_SIZE)
        
        # 3. 维护 Latent Buffer (物理延迟对齐)
        if latent_buffer is None:
            accumulated = new_hidden
        else:
            accumulated = torch.cat([latent_buffer, new_hidden], dim=-1)
        
        total_acc_frames = accumulated.shape[-1]
        
        # 4. 确定本次“可下发”给卷积链的帧数
        if is_last_chunk:
            num_frames_to_finalize = total_acc_frames
            next_latent = None
        elif total_acc_frames <= self.LOOKAHEAD_FRAMES:
            return torch.zeros(B, 0, device=device), next_pkv, accumulated, conv_history, next_pre_conv_hist
        else:
            num_frames_to_finalize = total_acc_frames - self.LOOKAHEAD_FRAMES
            next_latent = accumulated[:, :, -self.LOOKAHEAD_FRAMES:].clone()

        finalize_hidden = accumulated[:, :, :num_frames_to_finalize]
        
        # 5. 卷积历史拼接
        if conv_history is not None:
            conv_chain_input = torch.cat([conv_history, finalize_hidden], dim=-1)
            h_len = conv_history.shape[-1]
        else:
            conv_chain_input = finalize_hidden
            h_len = 0
            
        if not is_last_chunk and next_latent is not None:
             conv_chain_input = torch.cat([conv_chain_input, next_latent], dim=-1)
        
        # 6. 执行卷积链推理
        curr = conv_chain_input
        for blocks in self.decoder.upsample:
            for block in blocks: curr = block(curr)
        for block in self.decoder.decoder:
            curr = block(curr)
            
        wav = curr.squeeze(1).clamp(min=-1, max=1)
        
        # 7. 精确对齐裁剪
        start_samples = h_len * self.samples_per_frame
        end_samples = start_samples + num_frames_to_finalize * self.samples_per_frame
        
        actual_end = min(end_samples, wav.shape[-1])
        final_wav = wav[:, start_samples:actual_end]
        
        # 8. 更新卷积历史
        if is_last_chunk:
            next_conv_hist = None
        else:
            next_conv_hist = finalize_hidden[:, :, -self.CONV_HISTORY_WINDOW:].clone()
            
        return final_wav, next_pkv, next_latent, next_conv_hist, next_pre_conv_hist

class SpeakerEncoderExportWrapper(nn.Module):
    """
    Qwen3-TTS 说话人编码器导出包装类。
    """
    def __init__(self, speaker_encoder):
        super().__init__()
        self.speaker_encoder = speaker_encoder
        self.speaker_encoder.eval()

    def forward(self, mels):
        """
        Args:
            mels (torch.FloatTensor): [Batch, Seq, MelDim(128)]
        Returns:
            spk_emb (torch.FloatTensor): [Batch, 512]
        """
        return self.speaker_encoder(mels)

class MTPStepExportWrapper(nn.Module):
    """
    Qwen3-TTS MTP (Code Predictor) 单步导出包装类。
    因为 MTP 有 15 个不同的 Head 和 Embedding，我们将它们全部包含进来。
    """
    def __init__(self, code_predictor):
        super().__init__()
        self.code_predictor_model = code_predictor.model
        self.lm_heads = code_predictor.lm_head # ModuleList of 15 Linears
        self.small_to_mtp_projection = code_predictor.small_to_mtp_projection
        self.eval()

    def forward(self, inputs_embeds, step_idx, past_key_values_list=None):
        """
        Args:
            inputs_embeds: [Batch, Seq, Hidden]
            step_idx: int (0 to 14)
            past_key_values_list: Optional list/tuple of past KV
        """
        # 1. Project if needed
        hidden_states = self.small_to_mtp_projection(inputs_embeds)
        
        # 2. Model Forward
        # 注意：这里我们简化处理，假设一次跑一步或者一段。
        # 为了 ONNX 导出方便，我们可能需要更具体的 KV Cache 处理。
        # 但 MTP 总共只有 15 步，且序列很短（总共 16 2?），不带 KV Cache 跑 full self-attn 也可以。
        # 考虑到复杂度，这里先实现一个全序列版本的，推理时我们可以拼接序列。
        
        outputs = self.code_predictor_model(
            inputs_embeds=hidden_states,
            past_key_values=None, # 简化：不使用 KV Cache 避免 ONNX 复杂化
            use_cache=False
        )
        
        last_hidden = outputs.last_hidden_state[:, -1:, :]
        
        # 3. Apply the specific head for this step
        # 由于 ONNX 不支持动态索引 ModuleList，我们需要用一种 trick 或者导出多个模型。
        # 或者使用一个逻辑合并所有的 Head。
        # 这里为了导出一个模型，我们写一个 loop (ONNX 会展开它或者我们可以用 index_select)
        
        # 假设 step_idx 是 tensor
        all_logits = []
        for head in self.lm_heads:
            all_logits.append(head(last_hidden))
        
        # [15, B, 1, Vocab] -> [B, 1, Vocab]
        stacked_logits = torch.stack(all_logits, dim=0)
        # 使用 step_idx 选取对应的 logits
        logits = stacked_logits[step_idx] 
        
        return logits
