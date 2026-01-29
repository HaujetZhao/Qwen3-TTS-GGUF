"""
87-Verify-ONNX-Runtime.py
验证导出的 ONNX 模型的正确性和推理速度。
"""
import os
import time
import numpy as np
import torch
import onnxruntime as ort
import soundfile as sf

from qwen3_tts_gguf.tokenizer_12hz.modeling_tokenizer import Qwen3TTSTokenizerV2Model
from qwen3_tts_gguf.codec_export import StatefulCodecONNXWrapper

def main():
    # 配置
    MODEL_PATH = "./Qwen3-TTS-12Hz-1.7B-CustomVoice"
    ONNX_PATH = "onnx_export/qwen3_tts_decoder_stateful.onnx"
    
    device = "cpu"
    chunk_size = 3
    
    # 1. 加载 PyTorch 模型
    print("🚀 正在加载 PyTorch 模型...")
    tokenizer_path = os.path.join(MODEL_PATH, "speech_tokenizer") if os.path.exists(os.path.join(MODEL_PATH, "speech_tokenizer")) else MODEL_PATH
    model = Qwen3TTSTokenizerV2Model.from_pretrained(tokenizer_path).to(device)
    wrapper = StatefulCodecONNXWrapper(model).to(device)
    wrapper.eval()
    
    # 2. 加载 ONNX 模型
    print("📦 正在加载 ONNX 模型...")
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_session = ort.InferenceSession(ONNX_PATH, sess_options, providers=['CPUExecutionProvider'])
    
    # 打印输入输出信息
    print("\n📋 ONNX 模型输入:")
    for inp in ort_session.get_inputs():
        print(f"   {inp.name}: {inp.shape}")
    print("\n📋 ONNX 模型输出:")
    for out in ort_session.get_outputs():
        print(f"   {out.name}: {out.shape}")
    
    # 3. 准备测试数据
    print("\n📊 正在准备测试数据...")
    codes_path = os.path.join("output_verify", "codes_ref.npy")
    ref_wav_path = os.path.join("output_verify", "reference.wav")
    
    if os.path.exists(codes_path):
        codes_ref = np.load(codes_path)
        if codes_ref.ndim == 2:
            codes_ref = codes_ref[np.newaxis, ...]
        ref_wav, sr = sf.read(ref_wav_path) if os.path.exists(ref_wav_path) else (None, 24000)
    else:
        # 自动生成测试数据
        print("   ⚠️ 未找到预存数据，生成随机测试码...")
        total_test_frames = 16
        Q = 16  # 量化器数量
        codes_ref = np.random.randint(0, 1024, (1, total_test_frames, Q), dtype=np.int64)
        ref_wav = None
    
    total_frames = codes_ref.shape[1]
    print(f"   总帧数: {total_frames}")
    
    # 4. 获取配置
    num_layers = wrapper.num_layers
    cfg = wrapper.decoder.config
    num_heads = cfg.num_key_value_heads if hasattr(cfg, 'num_key_value_heads') else cfg.num_attention_heads
    head_dim = cfg.head_dim
    
    # 5. 初始化状态
    def init_states_np():
        pre_conv_h = np.zeros((1, 512, 0), dtype=np.float32)
        latent_buf = np.zeros((1, 1024, 0), dtype=np.float32)
        conv_h = np.zeros((1, 1024, 0), dtype=np.float32)
        pkv = []
        for _ in range(num_layers):
            pkv.append(np.zeros((1, num_heads, 0, head_dim), dtype=np.float32))  # K
        for _ in range(num_layers):
            pkv.append(np.zeros((1, num_heads, 0, head_dim), dtype=np.float32))  # V
        return pre_conv_h, latent_buf, conv_h, pkv
    
    def init_states_torch():
        pre_conv_h = torch.zeros(1, 512, 0, device=device)
        latent_buf = torch.zeros(1, 1024, 0, device=device)
        conv_h = torch.zeros(1, 1024, 0, device=device)
        pkv = []
        for _ in range(num_layers):
            pkv.append(torch.zeros(1, num_heads, 0, head_dim, device=device))
        for _ in range(num_layers):
            pkv.append(torch.zeros(1, num_heads, 0, head_dim, device=device))
        return pre_conv_h, latent_buf, conv_h, pkv
    
    # 6. PyTorch 推理
    print("\n🔥 PyTorch 流式推理...")
    pre_conv_h, latent_buf, conv_h, pkv_list = init_states_torch()
    pytorch_wavs = []
    pytorch_times = []
    
    with torch.no_grad():
        for i in range(0, total_frames, chunk_size):
            chunk_codes = torch.from_numpy(codes_ref[:, i:i+chunk_size, :]).to(device)
            is_last = torch.tensor([1.0 if i + chunk_size >= total_frames else 0.0], device=device)
            
            t0 = time.perf_counter()
            outputs = wrapper(chunk_codes, is_last, pre_conv_h, latent_buf, conv_h, *pkv_list)
            t1 = time.perf_counter()
            pytorch_times.append(t1 - t0)
            
            chunk_wav_full = outputs[0]
            valid_len = int(outputs[1].item())
            pre_conv_h = outputs[2]
            latent_buf = outputs[3]
            conv_h = outputs[4]
            pkv_list = outputs[5:]
            
            chunk_wav = chunk_wav_full[:, :valid_len].cpu().numpy()
            if valid_len > 0:
                pytorch_wavs.append(chunk_wav.flatten())
    
    pytorch_wav = np.concatenate(pytorch_wavs)
    pytorch_total_time = sum(pytorch_times)
    print(f"   总耗时: {pytorch_total_time*1000:.2f} ms")
    print(f"   输出长度: {len(pytorch_wav)}")
    
    # 7. ONNX Runtime 推理
    print("\n⚡ ONNX Runtime 流式推理...")
    pre_conv_h_np, latent_buf_np, conv_h_np, pkv_np = init_states_np()
    onnx_wavs = []
    onnx_times = []
    
    input_names = [inp.name for inp in ort_session.get_inputs()]
    output_names = [out.name for out in ort_session.get_outputs()]
    
    for i in range(0, total_frames, chunk_size):
        chunk_codes = codes_ref[:, i:i+chunk_size, :].astype(np.int64)
        is_last = np.array([1.0 if i + chunk_size >= total_frames else 0.0], dtype=np.float32)
        
        # 构建输入字典
        feed_dict = {
            "audio_codes": chunk_codes,
            "is_last": is_last,
            "pre_conv_history": pre_conv_h_np,
            "latent_buffer": latent_buf_np,
            "conv_history": conv_h_np,
        }
        for j in range(num_layers):
            feed_dict[f"past_key_{j}"] = pkv_np[j]
            feed_dict[f"past_value_{j}"] = pkv_np[num_layers + j]
        
        t0 = time.perf_counter()
        outputs = ort_session.run(output_names, feed_dict)
        t1 = time.perf_counter()
        onnx_times.append(t1 - t0)
        
        # 解包输出
        chunk_wav_full = outputs[0]
        valid_len = int(outputs[1][0])
        pre_conv_h_np = outputs[2]
        latent_buf_np = outputs[3]
        conv_h_np = outputs[4]
        for j in range(num_layers):
            pkv_np[j] = outputs[5 + j]
            pkv_np[num_layers + j] = outputs[5 + num_layers + j]
        
        chunk_wav = chunk_wav_full[:, :valid_len]
        if valid_len > 0:
            onnx_wavs.append(chunk_wav.flatten())
    
    onnx_wav = np.concatenate(onnx_wavs)
    onnx_total_time = sum(onnx_times)
    print(f"   总耗时: {onnx_total_time*1000:.2f} ms")
    print(f"   输出长度: {len(onnx_wav)}")
    
    # 8. 对比结果
    print("\n" + "="*50)
    print("📊 验证结果:")
    
    min_len = min(len(pytorch_wav), len(onnx_wav))
    mse = np.mean((pytorch_wav[:min_len] - onnx_wav[:min_len])**2)
    max_diff = np.max(np.abs(pytorch_wav[:min_len] - onnx_wav[:min_len]))
    
    print(f"   PyTorch 输出长度: {len(pytorch_wav)}")
    print(f"   ONNX    输出长度: {len(onnx_wav)}")
    print(f"   MSE: {mse:.2e}")
    print(f"   Max Diff: {max_diff:.2e}")
    
    print("\n⏱️ 性能对比:")
    print(f"   PyTorch 总耗时: {pytorch_total_time*1000:.2f} ms")
    print(f"   ONNX RT 总耗时: {onnx_total_time*1000:.2f} ms")
    print(f"   加速比: {pytorch_total_time/onnx_total_time:.2f}x")
    
    # 计算实时率
    audio_duration = len(onnx_wav) / 24000  # 假设 24kHz
    rtf_pytorch = pytorch_total_time / audio_duration
    rtf_onnx = onnx_total_time / audio_duration
    print(f"\n📈 实时率 (RTF):")
    print(f"   PyTorch: {rtf_pytorch:.4f} (1/{1/rtf_pytorch:.1f}x 实时)")
    print(f"   ONNX RT: {rtf_onnx:.4f} (1/{1/rtf_onnx:.1f}x 实时)")
    
    print("="*50)
    
    if mse < 1e-5:
        print("🎉 验证通过！ONNX 模型与 PyTorch 输出等价。")
    else:
        print("⚠️ 警告：数值差异较大，请检查模型。")

if __name__ == "__main__":
    main()
