import os
import sys
import torch
from qwen3_tts_gguf import logger
from qwen3_tts_gguf.export.codec_export import CodecEncoderExportWrapper
# 确保导入的是本地的 model definition (它现在使用 Internal Mimi)
from qwen3_tts_gguf.export.tokenizer_12hz.modeling_tokenizer import Qwen3TTSTokenizerV2Model

# 添加项目根目录到 sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Qwen3-TTS"))

from export_config import MODEL_DIR, EXPORT_DIR

def main():
    # 1. 配置路径
    ONNX_PATH = os.path.join(EXPORT_DIR, 'qwen3_tts_codec_encoder.onnx')
    
    os.makedirs(EXPORT_DIR, exist_ok=True)
    
    # 路径处理
    logger.info(f"正在加载模型: {MODEL_DIR}")
    tokenizer_model_dir = os.path.join(MODEL_DIR, "speech_tokenizer")
    load_path = tokenizer_model_dir if os.path.exists(tokenizer_model_dir) else MODEL_DIR
    
    # 2. 加载 PyTorch 模型
    try:
        # 这将加载使用 Internal Mimi 的模型，已经包含了 Dynamo Fix
        model = Qwen3TTSTokenizerV2Model.from_pretrained(load_path, torch_dtype=torch.float32)
        model.eval()
        
        # 全局禁用 return_dict (双重保险)
        model.config.return_dict = False
        if hasattr(model, 'encoder'):
            model.encoder.config.return_dict = False
        
        # 移除 Weight Norm (Mimi 内部提供了便捷方法)
        logger.info("Removing weight norm...")
        model.encoder.apply(lambda m: m.remove_weight_norm() if hasattr(m, 'remove_weight_norm') else None)
            
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. 准备导出包装器
    logger.info("准备 ONNX 导出包装器 (Full Mimi Encoder)...")
    encoder_wrapper = CodecEncoderExportWrapper(model).eval()
    
    # 4. Dummy Input (1s audio at 24kHz)
    dummy_input = torch.randn(1, 24000) 
    
    # 5. Export
    logger.info(f"开始导出 Encoder ONNX 到: {ONNX_PATH}")
    
    try:
        torch.onnx.export(
            encoder_wrapper,
            (dummy_input,),
            ONNX_PATH,
            input_names=['input_values'],
            output_names=['audio_codes'],
            dynamic_axes={
                'input_values': {0: 'batch_size', 1: 'sequence_length'},
                'audio_codes': {0: 'batch_size', 1: 'sequence_length'}
            },
            opset_version=18,  # 使用 opset 18，避免版本转换失败
            do_constant_folding=True
        )
        logger.info("✅ Encoder ONNX 导出成功！")
        
    except Exception as e:
        logger.error(f"❌ Encoder ONNX 导出失败: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()
