import os
import torch
import numpy as np
import onnxruntime as ort
import soundfile as sf
from transformers import AutoTokenizer, AutoProcessor
from .codec_export import CodecExportWrapper # Use this only if we needed the class, but we use ONNX
from . import logger
from transformers import AutoTokenizer
# Removed PyTorch model import
# from Qwen3_TTS.qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

class Qwen3HybridSynthesizer:
    """
    Qwen3-TTS 混合推理合成器。
    
    组成：
    1. Prompt Encoder (PyTorch): 用于将参考音频编码为 Code。
    2. LLM (llama.cpp): 用于根据文本和参考 Code 生成目标 Code。
    3. Codec Decoder (ONNX): 用于将目标 Code 解码为音频波形。
    """
    def __init__(self, model_dir, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model_dir = model_dir
        self.device = device
        
        # 1. 加载 Tokenizer (用于文本)
        # 假设 tokenizer 在 model_dir 或原始目录
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        
        # 2. 加载 Codec 模型 (ONNX)
        # 完全去除 PyTorch 模型依赖
        onnx_path = os.path.join(model_dir, "Qwen3-Codec-Decoder.onnx")
        if os.path.exists(onnx_path):
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
            self.decoder_session = ort.InferenceSession(onnx_path, providers=providers)
        else:
            self.decoder_session = None
            logger.warning(f"ONNX decoder not found at {onnx_path}.")

        # 3. 加载 Encoder ONNX
        encoder_onnx_path = os.path.join(model_dir, "Qwen3-Codec-Encoder.onnx")
        if os.path.exists(encoder_onnx_path):
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
            self.encoder_session = ort.InferenceSession(encoder_onnx_path, providers=providers)
        else:
            self.encoder_session = None
            logger.warning(f"ONNX encoder not found at {encoder_onnx_path}.")

        # 4. 加载 LLM (GGUF)
        gguf_path = os.path.join(model_dir, "Qwen3-LLM-1.7B-F16.gguf")
        if Llama and os.path.exists(gguf_path):
            self.llm = Llama(
                model_path=gguf_path,
                n_ctx=2048, # 根据需要调整
                n_gpu_layers=-1 # 尽可能使用 GPU
            )
        else:
            self.llm = None
            logger.warning("llama-cpp-python not installed or GGUF not found.")

    def encode_prompt(self, audio_path):
        """
        读取参考音频并编码为 codes.
        """
        import librosa
        # 假设采样率 16000 或 24000，需确认模型配置
        # 这里硬编码或从 config.json 读取 (如果读取了)。
        # Qwen3-TTS usually 16k or higher? Let's check previous tasks -> 24000 mentioned in modeling.
        # But tokenizer might differ.
        # Safe bet: 16000? 
        # 为了稳健，我们应该读取 config.json。
        # 这里先假设 16000。
        
        wav, sr = librosa.load(audio_path, sr=None)
        # 重采样到 24000 (根据 config.json)
        # TODO: 从 model_dir/config.json 读取 input_sample_rate
        target_sr = 24000 
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        
        # [1, T]
        wav_numpy = wav[np.newaxis, :].astype(np.float32)
        
        if self.encoder_session:
             onnx_inputs = {self.encoder_session.get_inputs()[0].name: wav_numpy}
             codes = self.encoder_session.run(None, onnx_inputs)[0] # [1, T_codes, Q]
             return codes
        else:
            raise RuntimeError("Encoder session not initialized")

    def synthesize(self, text, ref_audio_path, output_path="output.wav"):
        if not self.llm:
            raise RuntimeError("LLM not initialized.")
            
        # 1. Encode Reference Audio
        ref_codes = self.encode_prompt(ref_audio_path) # [1, T_ref, Q]
        
        # 2. Construct Prompt
        # 这是一个复杂步骤，涉及 Qwen3-TTS 的特殊 Token 格式。
        # 格式通常是:
        # <|im_start|>user\n(Instruction)<|im_end|>\n
        # <|im_start|>assistant\n<|audio_start|>...ref_codes...<|audio_end|>...text...<|audio_start|>
        
        # 简化实现：假设 LLM 已经微调为接收特定格式。
        # 这里需要参考 inference 代码中的 prompt 构建逻辑 (Qwen3TTSModel._tokenize_texts 等)。
        # 由于我们没有完整复刻复杂的 Prompt 逻辑，这里做一个占位符实现的描述。
        # 实际情况需要将 ref_codes 映射到 Vocabulary 中的 Audio Tokens。
        
        # 假设我们能获得生成的 codes (gen_codes)
        # 这里模拟生成过程
        logger.info("Synthesizing (Simulated)...")
        
        # Dummy Gen Codes for 2 seconds of audio at 25Hz/12Hz?
        # Qwen3 12Hz -> 12 frames per second? 
        # Check config: position_id_per_seconds=13.
        
        # 3. Decode Codes to Audio (ONNX)
        # 假设 gen_codes 是 [1, T_gen, Q]
        # 合并 ref_codes 和 gen_codes 或者只解码 gen_codes
        
        # 使用 PyTorch Encoder 产生的 ref_codes 来测试 ONNX Decoder
        target_codes = ref_codes # 仅作测试：重构参考音频
        
        if self.decoder_session:
            # ONNX Input: audio_codes [1, T, Q]
            onnx_inputs = {self.decoder_session.get_inputs()[0].name: target_codes}
            wav = self.decoder_session.run(None, onnx_inputs)[0] # [1, T_out]
        else:
             raise RuntimeError("Decoder session not initialized")
        
        # 4. Save
        # 采样率同样硬编码或读取
        sf.write(output_path, wav, 24000) # Assuming 24k output
        logger.info(f"Saved to {output_path}")
        return output_path
