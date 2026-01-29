"""
89-Profile-PyTorch.py
使用 PyTorch 对新版 StatefulCodecONNXWrapper 进行细粒度性能分析。
重点分析：记忆填充(memmove/cat)、Transformer、卷积解码各阶段耗时。
"""
import os
import time
import numpy as np
import torch

from qwen3_tts_gguf.tokenizer_12hz.modeling_tokenizer import Qwen3TTSTokenizerV2Model
from qwen3_tts_gguf.codec_export import TraceableKVStack

def main():
    # 配置
    MODEL_PATH = "./Qwen3-TTS-12Hz-1.7B-CustomVoice"
    TOTAL_FRAMES = 200
    Q = 16
    device = "cpu"
    
    # 测试不同的 chunk size
    CHUNK_SIZES = [3, 12, 25, 50]
    
    # 1. 加载模型
    print("🚀 正在加载模型...")
    tokenizer_path = os.path.join(MODEL_PATH, "speech_tokenizer") if os.path.exists(os.path.join(MODEL_PATH, "speech_tokenizer")) else MODEL_PATH
    model = Qwen3TTSTokenizerV2Model.from_pretrained(tokenizer_path).to(device)
    decoder = model.decoder.to(device).eval()
    trans = decoder.pre_transformer
    
    # 获取配置
    cfg = decoder.config
    num_layers = cfg.num_hidden_layers
    num_heads = cfg.num_key_value_heads if hasattr(cfg, 'num_key_value_heads') else cfg.num_attention_heads
    head_dim = cfg.head_dim
    KV_WINDOW = 72
    LOOKAHEAD = 4
    
    print(f"   num_layers={num_layers}, num_heads={num_heads}, head_dim={head_dim}")
    
    # 2. 准备测试数据
    codes_ref = np.random.randint(0, 1024, (1, TOTAL_FRAMES, Q), dtype=np.int64)
    
    # 3. 细粒度 forward (手动拆解 wrapper 逻辑以打桩)
    def run_profiled_forward(chunk_size):
        """运行一次完整流式推理，返回各阶段累计耗时"""
        times = {
            "quantize": 0,
            "pre_conv_cat": 0,  # 拼接历史
            "pre_conv_run": 0,  # pre_conv 卷积
            "kv_init": 0,       # KV Stack 初始化
            "rope_mask": 0,     # RoPE + Mask 生成
            "transformer": 0,   # 8层 Transformer
            "post_proj": 0,     # 后投影
            "latent_cat": 0,    # Latent buffer 拼接
            "conv_decode": 0,   # 上采样 + 卷积解码
            "slice_output": 0,  # 切片输出
            "kv_update": 0,     # KV Cache 更新 (在 layer 内部，这里估算)
        }
        
        # 初始化状态
        pre_conv_h = torch.zeros(1, 512, 0, device=device)
        latent_buf = torch.zeros(1, 1024, 0, device=device)
        conv_h = torch.zeros(1, 1024, 0, device=device)
        pkv_keys = [torch.zeros(1, num_heads, 0, head_dim, device=device) for _ in range(num_layers)]
        pkv_vals = [torch.zeros(1, num_heads, 0, head_dim, device=device) for _ in range(num_layers)]
        
        chunk_count = 0
        
        with torch.no_grad():
            for i in range(0, TOTAL_FRAMES, chunk_size):
                chunk_codes = torch.from_numpy(codes_ref[:, i:i+chunk_size, :]).to(device)
                is_last = (i + chunk_size >= TOTAL_FRAMES)
                B, N, _ = chunk_codes.shape
                
                # ============ 1. Quantize ============
                t0 = time.perf_counter()
                codes_t = chunk_codes.transpose(1, 2)
                quantized = decoder.quantizer.decode(codes_t)
                times["quantize"] += time.perf_counter() - t0
                
                # ============ 2. Pre-conv cat ============
                t0 = time.perf_counter()
                quant_full = torch.cat([pre_conv_h, quantized], dim=-1)
                h_len = pre_conv_h.size(2)
                times["pre_conv_cat"] += time.perf_counter() - t0
                
                # ============ 3. Pre-conv run ============
                t0 = time.perf_counter()
                hidden_all = decoder.pre_conv(quant_full)
                hidden = hidden_all[:, :, h_len:].transpose(1, 2)
                times["pre_conv_run"] += time.perf_counter() - t0
                
                pre_conv_h = quantized[:, :, -2:]  # 更新历史
                
                # ============ 4. KV Init ============
                t0 = time.perf_counter()
                kv_stack = TraceableKVStack(pkv_keys.copy(), pkv_vals.copy(), KV_WINDOW)
                past_len = kv_stack.get_seq_length()
                total_len = past_len + N
                times["kv_init"] += time.perf_counter() - t0
                
                # ============ 5. RoPE + Mask ============
                t0 = time.perf_counter()
                h = trans.input_proj(hidden)
                position_ids = torch.arange(past_len, total_len, device=device).unsqueeze(0)
                pos_embeddings = trans.rotary_emb(h, position_ids)
                
                q_idx = torch.arange(N, device=device).unsqueeze(1)
                k_idx = torch.arange(total_len, device=device).unsqueeze(0)
                mask_cond = (k_idx <= (past_len + q_idx)) & (k_idx > (past_len + q_idx - KV_WINDOW))
                attn_mask = torch.where(mask_cond, 0.0, -float("inf")).unsqueeze(0).unsqueeze(0)
                times["rope_mask"] += time.perf_counter() - t0
                
                # ============ 6. Transformer ============
                t0 = time.perf_counter()
                for layer in trans.layers:
                    h = layer(h, attention_mask=attn_mask, position_ids=position_ids,
                              past_key_values=kv_stack, use_cache=True, position_embeddings=pos_embeddings)
                times["transformer"] += time.perf_counter() - t0
                
                # ============ 7. Post projection ============
                t0 = time.perf_counter()
                h = trans.norm(h)
                new_hidden = trans.output_proj(h).transpose(1, 2)
                times["post_proj"] += time.perf_counter() - t0
                
                # ============ 8. Latent cat ============
                t0 = time.perf_counter()
                accumulated = torch.cat([latent_buf, new_hidden], dim=-1)
                num_finalize = accumulated.size(2) if is_last else max(0, accumulated.size(2) - LOOKAHEAD)
                latent_buf = accumulated[:, :, -LOOKAHEAD:]
                times["latent_cat"] += time.perf_counter() - t0
                
                # ============ 9. Conv decode ============
                t0 = time.perf_counter()
                conv_input = torch.cat([conv_h, accumulated], dim=-1)
                curr = conv_input
                for blocks in decoder.upsample:
                    for block in blocks: curr = block(curr)
                for block in decoder.decoder:
                    curr = block(curr)
                wav = curr.squeeze(1).clamp(-1, 1)
                times["conv_decode"] += time.perf_counter() - t0
                
                # ============ 10. Slice output ============
                t0 = time.perf_counter()
                start_samples = conv_h.size(2) * 1920
                final_wav = wav[:, start_samples:]
                finalize_hidden = accumulated[:, :, :num_finalize]
                conv_h = finalize_hidden[:, :, -4:] if num_finalize >= 4 else finalize_hidden
                times["slice_output"] += time.perf_counter() - t0
                
                # 更新 KV
                pkv_keys = kv_stack.key_cache
                pkv_vals = kv_stack.value_cache
                
                chunk_count += 1
        
        return times, chunk_count
    
    # 4. 运行对比
    print("\n" + "="*70)
    print(f"📊 细粒度性能分析 ({TOTAL_FRAMES} 帧)")
    print("="*70)
    
    results = {}
    for cs in CHUNK_SIZES:
        print(f"\n🔥 测试 Chunk Size = {cs}...")
        times, n_chunks = run_profiled_forward(cs)
        results[cs] = (times, n_chunks)
        
        total = sum(times.values())
        print(f"   总耗时: {total*1000:.1f} ms | Chunks: {n_chunks}")
    
    # 5. 汇总表格
    print("\n" + "="*70)
    print("📊 各阶段耗时对比 (单位: ms)")
    print("="*70)
    
    stages = ["quantize", "pre_conv_cat", "pre_conv_run", "kv_init", "rope_mask", 
              "transformer", "post_proj", "latent_cat", "conv_decode", "slice_output"]
    
    # Header
    header = f"{'Stage':<15}" + "".join([f"{cs:>10}" for cs in CHUNK_SIZES])
    print(header)
    print("-" * len(header))
    
    for stage in stages:
        row = f"{stage:<15}"
        for cs in CHUNK_SIZES:
            t = results[cs][0][stage] * 1000
            row += f"{t:>10.1f}"
        print(row)
    
    # Total
    print("-" * len(header))
    row = f"{'TOTAL':<15}"
    for cs in CHUNK_SIZES:
        t = sum(results[cs][0].values()) * 1000
        row += f"{t:>10.1f}"
    print(row)
    
    # 占比分析
    print("\n" + "="*70)
    print("📊 耗时占比分析 (Chunk=25)")
    print("="*70)
    
    times_25 = results[25][0]
    total_25 = sum(times_25.values())
    sorted_stages = sorted(times_25.items(), key=lambda x: x[1], reverse=True)
    
    for stage, t in sorted_stages:
        pct = t / total_25 * 100
        bar = "█" * int(pct / 2)
        print(f"  {stage:<15} {t*1000:7.1f} ms ({pct:5.1f}%) {bar}")

if __name__ == "__main__":
    main()
