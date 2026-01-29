"""
90-Profile-ONNX.py
使用 ONNX Runtime 内置 Profiling 功能分析算子级别耗时。
"""
import os
import json
import time
import numpy as np
import onnxruntime as ort
from collections import defaultdict

def main():
    ONNX_PATH = "onnx_export/qwen3_tts_decoder_stateful.onnx"
    TOTAL_FRAMES = 200
    CHUNK_SIZE = 25
    Q = 16
    NUM_LAYERS = 8
    NUM_HEADS = 16
    HEAD_DIM = 64
    
    # 1. 启用 Profiling
    sess_options = ort.SessionOptions()
    sess_options.enable_profiling = True
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    print(f"📦 正在加载 ONNX 模型: {ONNX_PATH}")
    sess = ort.InferenceSession(ONNX_PATH, sess_options, providers=['CPUExecutionProvider'])
    
    # 2. 准备数据
    codes_ref = np.random.randint(0, 1024, (1, TOTAL_FRAMES, Q), dtype=np.int64)
    output_names = [out.name for out in sess.get_outputs()]
    
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
    
    # 3. 运行流式推理
    print(f"\n🔥 运行流式推理 (Chunk={CHUNK_SIZE}, 共 {TOTAL_FRAMES} 帧)...")
    pre_conv, latent, conv, pkv = init_states()
    
    t_start = time.perf_counter()
    chunk_count = 0
    
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
        
        outputs = sess.run(output_names, feed)
        
        # 更新状态
        pre_conv = outputs[2]
        latent = outputs[3]
        conv = outputs[4]
        for j in range(NUM_LAYERS):
            pkv[j] = outputs[5 + j]
            pkv[NUM_LAYERS + j] = outputs[5 + NUM_LAYERS + j]
        
        chunk_count += 1
    
    total_time = time.perf_counter() - t_start
    print(f"   总耗时: {total_time*1000:.1f} ms | Chunks: {chunk_count}")
    
    # 4. 结束 Profiling 并分析
    profile_file = sess.end_profiling()
    print(f"\n📊 Profiling 文件: {profile_file}")
    
    # 5. 解析 Profile JSON
    with open(profile_file, 'r') as f:
        profile_data = json.load(f)
    
    # 聚合算子耗时
    op_times = defaultdict(float)
    op_counts = defaultdict(int)
    
    for event in profile_data:
        if event.get("cat") == "Node":
            op_type = event.get("args", {}).get("op_name", "Unknown")
            dur_us = event.get("dur", 0)  # 微秒
            op_times[op_type] += dur_us / 1000  # 转毫秒
            op_counts[op_type] += 1
    
    # 6. 输出报告
    print("\n" + "="*70)
    print("📊 ONNX Runtime 算子级别耗时分析")
    print("="*70)
    
    total_op_time = sum(op_times.values())
    sorted_ops = sorted(op_times.items(), key=lambda x: x[1], reverse=True)
    
    print(f"{'算子类型':<25} {'累计耗时(ms)':>12} {'调用次数':>10} {'占比':>8}")
    print("-"*60)
    
    for op, t in sorted_ops[:20]:  # Top 20
        pct = t / total_op_time * 100 if total_op_time > 0 else 0
        cnt = op_counts[op]
        print(f"{op:<25} {t:>12.2f} {cnt:>10} {pct:>7.1f}%")
    
    print("-"*60)
    print(f"{'TOTAL (Top 20)':<25} {sum(t for _, t in sorted_ops[:20]):>12.2f}")
    print(f"{'TOTAL (All)':<25} {total_op_time:>12.2f}")
    
    # 7. 卷积类算子汇总
    conv_ops = ["Conv", "ConvTranspose", "ConvInteger"]
    conv_total = sum(op_times.get(op, 0) for op in conv_ops)
    
    # 注意力类算子
    attn_ops = ["MatMul", "Softmax", "Attention", "FusedAttention"]
    attn_total = sum(op_times.get(op, 0) for op in attn_ops)
    
    print("\n" + "="*70)
    print("📊 按类别汇总")
    print("="*70)
    print(f"  卷积类 (Conv*):     {conv_total:>10.2f} ms ({conv_total/total_op_time*100:.1f}%)")
    print(f"  注意力类 (MatMul等): {attn_total:>10.2f} ms ({attn_total/total_op_time*100:.1f}%)")
    print(f"  其他:               {total_op_time - conv_total - attn_total:>10.2f} ms")
    
    # 清理
    if os.path.exists(profile_file):
        os.remove(profile_file)
        print(f"\n✅ 已清理临时 Profile 文件")

if __name__ == "__main__":
    main()
