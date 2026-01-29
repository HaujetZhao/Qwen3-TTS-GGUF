"""
91-Quantize-ONNX.py
对 Stateful Decoder 进行 INT8 量化并对比性能。
注意：CPU 上 ConvInteger 未实现，因此仅量化 MatMul 权重。
"""
import os
import time
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType, QuantFormat

def main():
    ONNX_PATH = "onnx_export/qwen3_tts_decoder_stateful.onnx"
    QUANT_PATH = "onnx_export/qwen3_tts_decoder_stateful_int8.onnx"
    
    # 1. 动态量化 (仅 MatMul，跳过 Conv)
    print("🔧 正在进行 INT8 动态量化 (仅 MatMul)...")
    print(f"   输入: {ONNX_PATH}")
    print(f"   输出: {QUANT_PATH}")
    
    t0 = time.time()
    quantize_dynamic(
        model_input=ONNX_PATH,
        model_output=QUANT_PATH,
        weight_type=QuantType.QInt8,
        # 关键：使用 QOperator 格式，排除 Conv 类算子
        op_types_to_quantize=['MatMul', 'Gemm', 'Conv', 'ConvTranspose'],  # 仅量化这些算子
    )
    print(f"✅ 量化完成！耗时: {time.time()-t0:.1f}s")

    
    # 检查文件大小
    orig_size = os.path.getsize(ONNX_PATH) / 1e6
    quant_size = os.path.getsize(QUANT_PATH) / 1e6
    print(f"   原始大小: {orig_size:.1f} MB")
    print(f"   量化后:   {quant_size:.1f} MB (压缩率: {quant_size/orig_size*100:.0f}%)")
    
    # 2. 性能对比
    TOTAL_FRAMES = 200
    CHUNK_SIZE = 25
    Q = 16
    NUM_LAYERS = 8
    NUM_HEADS = 16
    HEAD_DIM = 64
    
    codes_ref = np.random.randint(0, 1024, (1, TOTAL_FRAMES, Q), dtype=np.int64)
    
    def init_states():
        pre_conv = np.zeros((1, 512, 0), dtype=np.float32)
        latent = np.zeros((1, 1024, 0), dtype=np.float32)
        conv = np.zeros((1, 1024, 0), dtype=np.float32)
        pkv = []
        for _ in range(NUM_LAYERS):
            pkv.append(np.zeros((1, NUM_HEADS, 0, HEAD_DIM), dtype=np.float32))
        for _ in range(NUM_LAYERS):
            pkv.append(np.zeros((1, NUM_HEADS, 0, HEAD_DIM), dtype=np.float32))
        return pre_conv, latent, conv, pkv
    
    def run_inference(sess):
        pre_conv, latent, conv, pkv = init_states()
        output_names = [out.name for out in sess.get_outputs()]
        
        t_total = 0
        for i in range(0, TOTAL_FRAMES, CHUNK_SIZE):
            chunk = codes_ref[:, i:i+CHUNK_SIZE, :]
            is_last = np.array([1.0 if i + CHUNK_SIZE >= TOTAL_FRAMES else 0.0], dtype=np.float32)
            
            feed = {
                "audio_codes": chunk,
                "is_last": is_last,
                "pre_conv_history": pre_conv,
                "latent_buffer": latent,
                "conv_history": conv,
            }
            for j in range(NUM_LAYERS):
                feed[f"past_key_{j}"] = pkv[j]
                feed[f"past_value_{j}"] = pkv[NUM_LAYERS + j]
            
            ts = time.perf_counter()
            outputs = sess.run(output_names, feed)
            te = time.perf_counter()
            t_total += (te - ts)
            
            pre_conv = outputs[2]
            latent = outputs[3]
            conv = outputs[4]
            for j in range(NUM_LAYERS):
                pkv[j] = outputs[5 + j]
                pkv[NUM_LAYERS + j] = outputs[5 + NUM_LAYERS + j]
        
        return t_total
    
    # 加载两个模型
    print("\n📦 加载原始 FP32 模型...")
    sess_fp32 = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
    
    print("📦 加载量化 INT8 模型...")
    sess_int8 = ort.InferenceSession(QUANT_PATH, providers=['DmlExecutionProvider'])
    
    # 预热
    run_inference(sess_fp32)
    run_inference(sess_int8)
    
    # 正式测试
    print(f"\n🔥 性能对比测试 ({TOTAL_FRAMES} 帧, Chunk={CHUNK_SIZE})...")
    
    ITERS = 3
    
    t_fp32 = 0
    for _ in range(ITERS):
        t_fp32 += run_inference(sess_fp32)
    t_fp32 /= ITERS
    
    t_int8 = 0
    for _ in range(ITERS):
        t_int8 += run_inference(sess_int8)
    t_int8 /= ITERS
    
    # 报告
    print("\n" + "="*50)
    print("📊 性能对比结果")
    print("="*50)
    print(f"  FP32 模型: {t_fp32*1000:.1f} ms")
    print(f"  INT8 模型: {t_int8*1000:.1f} ms")
    print(f"  加速比:    {t_fp32/t_int8:.2f}x")
    print(f"  模型压缩:  {orig_size:.1f} MB -> {quant_size:.1f} MB")
    print("="*50)
    
    # 精度验证
    print("\n🔬 精度验证 (单次推理)...")
    pre_conv, latent, conv, pkv = init_states()
    chunk = codes_ref[:, :CHUNK_SIZE, :]
    is_last = np.array([0.0], dtype=np.float32)
    
    feed = {
        "audio_codes": chunk,
        "is_last": is_last,
        "pre_conv_history": pre_conv,
        "latent_buffer": latent,
        "conv_history": conv,
    }
    for j in range(NUM_LAYERS):
        feed[f"past_key_{j}"] = pkv[j]
        feed[f"past_value_{j}"] = pkv[NUM_LAYERS + j]
    
    out_fp32 = sess_fp32.run(None, feed)
    out_int8 = sess_int8.run(None, feed)
    
    # 对比音频输出
    wav_fp32 = out_fp32[0]
    wav_int8 = out_int8[0]
    
    mse = np.mean((wav_fp32 - wav_int8) ** 2)
    max_diff = np.max(np.abs(wav_fp32 - wav_int8))
    
    print(f"  MSE:      {mse:.2e}")
    print(f"  Max Diff: {max_diff:.2e}")
    
    if mse < 1e-4:
        print("  ✅ 精度损失可接受")
    else:
        print("  ⚠️ 精度损失较大，建议检查")

if __name__ == "__main__":
    main()
