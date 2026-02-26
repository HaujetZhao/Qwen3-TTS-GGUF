"""
13-Export-Decoder.py
Qwen3-TTS Stateful Codec Decoder ONNX 导出脚本 (V4 Dynamo 版)。

核心经验参考：
- 使用 dynamo=True 路径
- 强制 eager 模式并锁定 head_dim 以优化 DML 稳定性
- 使用联合模型架构避免算子断层
- 外部裁剪波形 (valid_samples)
"""
import os
import torch
import numpy as np
from qwen3_tts_gguf.export.codec_export import StatefulDecoderDynamoCombined
from qwen3_tts_gguf.export.tokenizer_12hz.modeling_tokenizer import Qwen3TTSTokenizerV2Model

from export_config import MODEL_DIR, EXPORT_DIR

def main():
    # 1. 配置
    ONNX_FILENAME = "qwen3_tts_decoder.onnx"
    os.makedirs(EXPORT_DIR, exist_ok=True)
    
    device = "cpu"
    
    # 2. 加载原始模型
    print("🚀 正在加载模型...")
    tokenizer_load_path = os.path.join(MODEL_DIR, "speech_tokenizer") if os.path.exists(os.path.join(MODEL_DIR, "speech_tokenizer")) else MODEL_DIR
    model = Qwen3TTSTokenizerV2Model.from_pretrained(tokenizer_load_path, torch_dtype=torch.float32).to(device)
    
    # 【CRITICAL】应用 Transformer 导出优化补丁
    # 这两行确保了计算图在 Dynamo 分析阶段是确定且 DML 友好的
    model.config.decoder_config._attn_implementation = "eager"
    model.config.decoder_config.head_dim = 64
    
    # 包装全量模型
    wrapper = StatefulDecoderDynamoCombined(model.decoder).to(device).eval()
    
    # 3. 获取关键配置
    num_layers = wrapper.num_layers
    cfg = model.decoder.config
    num_heads = cfg.num_key_value_heads if hasattr(cfg, 'num_key_value_heads') else cfg.num_attention_heads
    head_dim = cfg.head_dim
    
    print(f"   num_layers={num_layers}, num_heads={num_heads}, head_dim={head_dim}")

    # 4. 创建 Dummy Inputs (匹配新 wrapper 签名)
    B = 1
    N = 3
    Q = 16
    
    dummy_audio_codes = torch.zeros(B, N, Q, dtype=torch.long, device=device)
    dummy_pre_conv_h = torch.zeros(B, 512, 0, device=device) 
    dummy_latent_buf = torch.zeros(B, 1024, 0, device=device)
    dummy_conv_h = torch.zeros(B, 1024, 0, device=device)
    dummy_is_last = torch.tensor([0.0], device=device)
    
    # KV Cache
    dummy_kv = []
    for _ in range(num_layers * 2):
        dummy_kv.append(torch.zeros(B, num_heads, 0, head_dim, device=device))

    dummy_inputs = (
        dummy_audio_codes,
        dummy_pre_conv_h,
        dummy_latent_buf,
        dummy_conv_h,
        dummy_is_last,
        *dummy_kv
    )
    
    # 5. 定义动态维度 (Modern Dynamo Style)
    batch = torch.export.Dim("batch", min=1, max=8)
    num_frames = torch.export.Dim("num_frames", min=1, max=1024)
    past_seq = torch.export.Dim("past_seq", min=0, max=72)
    pre_conv_seq = torch.export.Dim("pre_conv_seq", min=0, max=2)
    latent_seq = torch.export.Dim("latent_seq", min=0, max=4)
    conv_seq = torch.export.Dim("conv_seq", min=0, max=4)
    
    dynamic_shapes = (
        {0: batch, 1: num_frames},      # audio_codes
        {0: batch, 2: pre_conv_seq},    # pre_conv_history
        {0: batch, 2: latent_seq},      # latent_buffer
        {0: batch, 2: conv_seq},        # conv_history
        None,                           # is_last
        tuple([{0: batch, 2: past_seq}] * (num_layers * 2)) # *past_kv_flat
    )
    
    # 6. 设置名称
    input_names = ["audio_codes", "pre_conv_history", "latent_buffer", "conv_history", "is_last"]
    output_names = ["final_wav", "valid_samples", "next_pre_conv_history", "next_latent_buffer", "next_conv_history"]
    
    for i in range(num_layers): input_names.append(f"past_key_{i}")
    for i in range(num_layers): input_names.append(f"past_value_{i}")
    for i in range(num_layers): output_names.append(f"next_key_{i}")
    for i in range(num_layers): output_names.append(f"next_value_{i}")
    
    # 7. 执行联合导出
    onnx_path = os.path.join(EXPORT_DIR, ONNX_FILENAME)
    print(f"📦 正在使用 dynamo=True 执行 Decoder 全量联合导出: {onnx_path}")
    
    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                dummy_inputs,
                onnx_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_shapes=dynamic_shapes,
                opset_version=18,
                dynamo=True,
            )
        
        file_size = os.path.getsize(onnx_path) / 1024 / 1024
        print(f"✅ ONNX 导出成功！文件大小: {file_size:.2f} MB")
        
    except Exception as e:
        print(f"❌ 导出失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
