import os
import sys
import torch
# import logging (Removed)

# 添加项目根目录到 sys.path 以便导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# 添加 Qwen3-TTS 目录以便导入 qwen_tts
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Qwen3-TTS"))

from qwen3_tts_gguf import logger
from qwen3_tts_gguf.codec_export import CodecExportWrapper, CodecEncoderExportWrapper

# 移除本地 logging 配置，直接使用导入的 logger

def main():
    # 1. 配置路径
    MODEL_DIR = r'./Qwen3-TTS-12Hz-1.7B-CustomVoice'
    OUTPUT_DIR = r'./model'
    ONNX_PATH = os.path.join(OUTPUT_DIR, 'Qwen3-Codec-Decoder.onnx')
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    logger.info(f"正在加载模型: {MODEL_DIR}")
    
    # 检测 speech_tokenizer 子目录
    tokenizer_model_dir = os.path.join(MODEL_DIR, "speech_tokenizer")
    if os.path.exists(tokenizer_model_dir):
        logger.info(f"检测到 speech_tokenizer 子目录，使用: {tokenizer_model_dir}")
        load_path = tokenizer_model_dir
    else:
        load_path = MODEL_DIR
    
    # 2. 加载 PyTorch 模型
    try:
        # 使用 Hugging Face 风格加载
        model = Qwen3TTSTokenizerV2Model.from_pretrained(load_path)
        model.eval()
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        return

    # 3. 准备导出包装器 (Decoder)
    logger.info("准备 ONNX 导出包装器 (Decoder)...")
    decoder_wrapper = CodecExportWrapper(model)
    
    # Decoder Dummy Input
    # Shape: [Batch=1, Time=100, NumQuantizers=ModelConfig.num_quantizers]
    num_quantizers = model.config.decoder_config.num_quantizers
    decoder_dummy_input = torch.randint(0, 1024, (1, 100, num_quantizers), dtype=torch.long)
    
    # Decoder Export
    logger.info(f"开始导出 Decoder ONNX 到: {ONNX_PATH}")
    try:
        torch.onnx.export(
            decoder_wrapper,
            (decoder_dummy_input,),
            ONNX_PATH,
            input_names=['audio_codes'],
            output_names=['audio_values'],
            dynamic_axes={
                'audio_codes': {0: 'batch_size', 1: 'sequence_length'},
                'audio_values': {0: 'batch_size', 1: 'sequence_length'}
            },
            opset_version=18,
            do_constant_folding=True
        )
        logger.info("Decoder ONNX 导出成功！")
    except Exception as e:
        logger.error(f"Decoder ONNX 导出失败: {e}")

    ONNX_ENCODER_PATH = os.path.join(OUTPUT_DIR, 'Qwen3-Codec-Encoder.onnx')
    logger.info("准备 ONNX 导出包装器 (Encoder)...")
    encoder_wrapper = CodecEncoderExportWrapper(model)
    
    # Encoder Dummy Input
    # Shape: [Batch=1, Time=24000*3] (3 seconds audio at 24000Hz)
    encoder_dummy_input = torch.randn(1, 24000 * 3)
    
    # Encoder Export
    logger.info(f"开始导出 Encoder ONNX 到: {ONNX_ENCODER_PATH}")
    try:
        torch.onnx.export(
            encoder_wrapper,
            (encoder_dummy_input,),
            ONNX_ENCODER_PATH,
            input_names=['input_values'], # raw audio
            output_names=['audio_codes'],
            dynamic_axes={
                'input_values': {0: 'batch_size', 1: 'sequence_length'},
                'audio_codes': {0: 'batch_size', 1: 'sequence_length'}
            },
            opset_version=18,
            do_constant_folding=True
        )
        logger.info("Encoder ONNX 导出成功！")
    except Exception as e:
        logger.error(f"Encoder ONNX 导出失败: {e}")

    # 6. 验证 (可选: 简单的形状检查 - Decoder)
    # 比较 PyTorch 及其 ONNX 的输出形状
    with torch.no_grad():
        torch_out = decoder_wrapper(decoder_dummy_input)
        logger.info(f"PyTorch Output Shape: {torch_out.shape}")
        
    # TODO: 使用 onnxruntime 进行数值验证 (如果安装了)
    try:
        import onnxruntime
        sess = onnxruntime.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
        onnx_out = sess.run(None, {'audio_codes': decoder_dummy_input.numpy()})[0]
        logger.info(f"ONNX Runtime Output Shape: {onnx_out.shape}")
        
        # 简单数值对比
        import numpy as np
        if np.allclose(torch_out.numpy(), onnx_out, atol=1e-4):
            logger.info("数值验证通过！(PyTorch vs ONNX)")
        else:
            logger.warning("数值验证存在差异，请检查！")
            
    except ImportError:
        logger.warning("未安装 onnxruntime，跳过验证。")

if __name__ == "__main__":
    main()
