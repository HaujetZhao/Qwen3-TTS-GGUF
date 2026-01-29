"""
93-Optimize-DML-IOBinding.py
使用 IOBinding 优化 DirectML 推理，让状态留在 GPU 上避免反复拷贝。
"""
import os
import time
import numpy as np
import onnxruntime as ort

# 配置
ONNX_PATH = "onnx_export/qwen3_tts_decoder_stateful.onnx"
TOTAL_FRAMES = 200
CHUNK_SIZE = 25
Q = 16
NUM_LAYERS = 8
NUM_HEADS = 16
HEAD_DIM = 64

def run_naive_dml():
    """朴素方式：每次推理都拷贝状态（慢）"""
    sess = ort.InferenceSession(ONNX_PATH, providers=['DmlExecutionProvider', 'CPUExecutionProvider'])
    
    codes = np.random.randint(0, 1024, (1, TOTAL_FRAMES, Q), dtype=np.int64)
    pre_conv = np.zeros((1, 512, 0), dtype=np.float32)
    latent = np.zeros((1, 1024, 0), dtype=np.float32)
    conv = np.zeros((1, 1024, 0), dtype=np.float32)
    pkv = [np.zeros((1, NUM_HEADS, 0, HEAD_DIM), dtype=np.float32) for _ in range(NUM_LAYERS * 2)]
    output_names = [out.name for out in sess.get_outputs()]
    
    t_total = 0
    for i in range(0, TOTAL_FRAMES, CHUNK_SIZE):
        chunk = codes[:, i:i+CHUNK_SIZE, :]
        is_last = np.array([1.0 if i + CHUNK_SIZE >= TOTAL_FRAMES else 0.0], dtype=np.float32)
        
        feed = {"audio_codes": chunk, "is_last": is_last,
                "pre_conv_history": pre_conv, "latent_buffer": latent, "conv_history": conv}
        for j in range(NUM_LAYERS):
            feed[f"past_key_{j}"] = pkv[j]
            feed[f"past_value_{j}"] = pkv[NUM_LAYERS + j]
        
        ts = time.perf_counter()
        outputs = sess.run(output_names, feed)
        t_total += time.perf_counter() - ts
        
        pre_conv, latent, conv = outputs[2], outputs[3], outputs[4]
        for j in range(NUM_LAYERS):
            pkv[j] = outputs[5 + j]
            pkv[NUM_LAYERS + j] = outputs[5 + NUM_LAYERS + j]
    
    return t_total, sess

def run_iobinding_dml():
    """IOBinding 方式：让输出留在 GPU 上"""
    print("   [DEBUG] 创建 session...")
    sess = ort.InferenceSession(ONNX_PATH, providers=['DmlExecutionProvider', 'CPUExecutionProvider'])
    
    input_names = [inp.name for inp in sess.get_inputs()]
    output_names = [out.name for out in sess.get_outputs()]
    print(f"   [DEBUG] 输入数: {len(input_names)}, 输出数: {len(output_names)}")
    
    codes = np.random.randint(0, 1024, (1, TOTAL_FRAMES, Q), dtype=np.int64)
    
    # 初始状态（CPU numpy 数组）
    pre_conv = np.zeros((1, 512, 0), dtype=np.float32)
    latent = np.zeros((1, 1024, 0), dtype=np.float32)
    conv = np.zeros((1, 1024, 0), dtype=np.float32)
    pkv = [np.zeros((1, NUM_HEADS, 0, HEAD_DIM), dtype=np.float32) for _ in range(NUM_LAYERS * 2)]
    
    # 保存上一次的 GPU 输出 OrtValue（用于下次输入）
    prev_outputs = None
    
    t_total = 0
    chunk_idx = 0
    
    for i in range(0, TOTAL_FRAMES, CHUNK_SIZE):
        chunk_idx += 1
        print(f"   [DEBUG] Chunk {chunk_idx}: frames {i}-{i+CHUNK_SIZE}")
        
        chunk = codes[:, i:i+CHUNK_SIZE, :]
        is_last = np.array([1.0 if i + CHUNK_SIZE >= TOTAL_FRAMES else 0.0], dtype=np.float32)
        
        # 创建 IOBinding
        io_binding = sess.io_binding()
        
        # 绑定 CPU 输入（音频码每帧不同，必须从 CPU 传）
        io_binding.bind_cpu_input("audio_codes", chunk)
        io_binding.bind_cpu_input("is_last", is_last)
        
        if prev_outputs is None:
            # 第一帧：从 CPU 传入初始空状态
            io_binding.bind_cpu_input("pre_conv_history", pre_conv)
            io_binding.bind_cpu_input("latent_buffer", latent)
            io_binding.bind_cpu_input("conv_history", conv)
            for j in range(NUM_LAYERS):
                io_binding.bind_cpu_input(f"past_key_{j}", pkv[j])
                io_binding.bind_cpu_input(f"past_value_{j}", pkv[NUM_LAYERS + j])
        else:
            # 后续帧：使用上一帧的 GPU 输出作为输入
            io_binding.bind_ortvalue_input("pre_conv_history", prev_outputs[2])
            io_binding.bind_ortvalue_input("latent_buffer", prev_outputs[3])
            io_binding.bind_ortvalue_input("conv_history", prev_outputs[4])
            for j in range(NUM_LAYERS):
                io_binding.bind_ortvalue_input(f"past_key_{j}", prev_outputs[5 + j])
                io_binding.bind_ortvalue_input(f"past_value_{j}", prev_outputs[5 + NUM_LAYERS + j])
        
        # 绑定输出到 DML 设备（让 RT 自动分配）
        for name in output_names:
            io_binding.bind_output(name, 'dml')
        
        ts = time.perf_counter()
        sess.run_with_iobinding(io_binding)
        t_total += time.perf_counter() - ts
        
        # 保存输出引用（仍在 GPU 上）
        prev_outputs = io_binding.get_outputs()
        prev_io_binding = io_binding  # 保留引用以便最后拷贝
        print(f"   [DEBUG] Chunk {chunk_idx} 完成, 输出数: {len(prev_outputs)}")
    
    # 最后拷贝音频到 CPU - 使用 copy_outputs_to_cpu
    print("   [DEBUG] 拷贝最终输出到 CPU...")
    prev_io_binding.synchronize_outputs()
    cpu_outputs = prev_io_binding.copy_outputs_to_cpu()
    final_wav = cpu_outputs[0]
    print(f"   [DEBUG] 最终输出 shape: {final_wav.shape}")

    
    return t_total, final_wav

def main():
    print("="*60)
    print("📊 DirectML IOBinding 优化测试")
    print("="*60)
    
    providers = ort.get_available_providers()
    if 'DmlExecutionProvider' not in providers:
        print("❌ DmlExecutionProvider 不可用！")
        print(f"   可用 providers: {providers}")
        return
    
    print(f"✅ DirectML 可用")
    print(f"📊 测试配置: {TOTAL_FRAMES} 帧, Chunk={CHUNK_SIZE}")
    
    # 1. 朴素方式
    print("\n🔥 测试朴素方式 (每次拷贝)...")
    run_naive_dml()  # 预热
    t_naive, _ = run_naive_dml()
    print(f"   耗时: {t_naive*1000:.1f} ms")
    
    # 2. IOBinding 方式
    print("\n🔥 测试 IOBinding 方式 (状态留在 GPU)...")
    try:
        t_iobind, wav = run_iobinding_dml()
        print(f"   耗时: {t_iobind*1000:.1f} ms")
        print(f"   输出 Shape: {wav.shape}")
        
        print("\n" + "="*50)
        print("📊 性能对比")
        print("="*50)
        print(f"  朴素方式:    {t_naive*1000:.1f} ms")
        print(f"  IOBinding:   {t_iobind*1000:.1f} ms")
        print(f"  加速比:      {t_naive/t_iobind:.2f}x")
        print("="*50)
        
    except Exception as e:
        print(f"❌ IOBinding 失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

