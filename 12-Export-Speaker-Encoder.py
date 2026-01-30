import os
import sys
import torch
import torch.nn as nn
from qwen_tts import Qwen3TTSModel
from qwen3_tts_gguf.codec_export import SpeakerEncoderExportWrapper

# 添加项目根目录到 sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

def main():
    # 1. 配置路径
    MODEL_PATH = os.path.abspath("Qwen3-TTS-12Hz-1.7B-Base")
    OUTPUT_DIR = os.path.abspath("model")
    ONNX_PATH = os.path.join(OUTPUT_DIR, 'qwen3_tts_speaker_encoder.onnx')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"载入 Base 模型以导出 Speaker Encoder: {MODEL_PATH}")
    
    try:
        # 加载完整模型
        tts = Qwen3TTSModel.from_pretrained(
            MODEL_PATH, 
            device_map="cpu", 
            torch_dtype=torch.float32
        )
        
        # 2. 提取 Speaker Encoder 模块
        # 在 Qwen3TTSForConditionalGeneration 中，它是 self.speaker_encoder
        speaker_encoder_module = tts.model.speaker_encoder
        
        if speaker_encoder_module is None:
            print("❌ 该模型不包含 Speaker Encoder。")
            return

        # 3. 使用包装器准备导出
        # 注意：SpeakerEncoder 的输入是 Mel 谱图 [Batch, Seq, 128]
        # 为了方便 GGUF 推理，我们通常导出这个核心部分
        wrapper = SpeakerEncoderExportWrapper(speaker_encoder_module).eval()

        # 4. 准备 Dummy Input
        # Speaker Encoder 的输入维度是 (Batch, Seq, MelDim=128)
        # 我们假设输入一个 1 秒音频对应的 Mel 长度 (约 94)
        dummy_input = torch.randn(1, 100, 128)

        # 5. 执行导出
        print(f"开始导出 Speaker Encoder ONNX 到: {ONNX_PATH}")
        torch.onnx.export(
            wrapper,
            (dummy_input,),
            ONNX_PATH,
            input_names=['mels'],
            output_names=['spk_emb'],
            dynamic_axes={
                'mels': {0: 'batch_size', 1: 'sequence_length'},
                'spk_emb': {0: 'batch_size'}
            },
            opset_version=18,
            do_constant_folding=True
        )
        
        print(f"✅ Speaker Encoder 导出成功！")
        print(f"输出文件: {ONNX_PATH}")

    except Exception as e:
        print(f"❌ 导出失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
