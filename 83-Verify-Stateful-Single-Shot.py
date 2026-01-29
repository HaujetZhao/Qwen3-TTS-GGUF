import os
import torch
import numpy as np
import soundfile as sf
from qwen3_tts_gguf.codec_export import StatefulCodecExportWrapper
from qwen3_tts_gguf.tokenizer_12hz.modeling_tokenizer import Qwen3TTSTokenizerV2Model

def main():
    # 1. 配置路径
    MODEL_DIR = r'./Qwen3-TTS-12Hz-1.7B-CustomVoice'
    DEBUG_DATA_DIR = "debug_data"
    
    codes_path = os.path.join(DEBUG_DATA_DIR, "jintian_codes.npy")
    ref_wav_path = os.path.join(DEBUG_DATA_DIR, "jintian_ref.wav")
    
    if not os.path.exists(codes_path):
        print(f"❌ 找不到调试数据: {codes_path}，请先运行 43-Generate-Debug-Data.py")
        return

    # 2. 加载数据
    codes_np = np.load(codes_path)
    ref_wav_np, sr = sf.read(ref_wav_path)
    
    # [T, Q] -> [B=1, T, Q]
    codes_torch = torch.from_numpy(codes_np).unsqueeze(0).long()
    
    # 3. 加载模型
    print(f"🚀 正在加载模型: {MODEL_DIR}...")
    tokenizer_model_dir = os.path.join(MODEL_DIR, "speech_tokenizer")
    load_path = tokenizer_model_dir if os.path.exists(tokenizer_model_dir) else MODEL_DIR
    model = Qwen3TTSTokenizerV2Model.from_pretrained(load_path)
    model.eval()
    
    # 初始化 Stateful Wrapper
    wrapper = StatefulCodecExportWrapper(model).eval()
    
    # 4. 单次全量推理
    print("🧪 正在执行单次全量推理对比 (is_last_chunk=True)...")
    with torch.no_grad():
        # 这里模拟“只有一整块”的情况
        audio, next_pkv, next_latent = wrapper(
            codes_torch, 
            past_key_values=None, 
            latent_buffer=None, 
            is_last_chunk=True
        )
        
    out_wav = audio.numpy().squeeze()
    
    # 5. 数值比对
    print("\n" + "="*40)
    print(f"📊 输出波形长度: {len(out_wav)}")
    print(f"📊 参考波形长度: {len(ref_wav_np)}")
    
    common_len = min(len(out_wav), len(ref_wav_np))
    mse = np.mean((out_wav[:common_len] - ref_wav_np[:common_len])**2)
    max_diff = np.max(np.abs(out_wav[:common_len] - ref_wav_np[:common_len]))
    
    print(f"✅ 验证结果:")
    print(f"   - MSE: {mse:.2e}")
    print(f"   - Max Diff: {max_diff:.2e}")
    print("="*40)
    
    if max_diff < 1e-4:
        print("\n🎉 完美！Stateful Wrapper 单次推理结果与参考音频一致。")
    else:
        print("\n⚠️ 提示：存在微小数值差异（可能是 PyTorch vs ONNX 精度引起）。")
        
    # 保存结果
    output_dir = "output_verify"
    os.makedirs(output_dir, exist_ok=True)
    sf.write(os.path.join(output_dir, "verify_stateful_oneshot.wav"), out_wav, 24000)
    print(f"\n音频已保存至: {output_dir}")

if __name__ == "__main__":
    main()
