import os
import sys
import numpy as np
import ctypes
import qwen3_tts_gguf.nano_llama as nano_llama

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 模型与路径配置
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CAPTURED_DIR = os.path.join(PROJECT_ROOT, "captured_craftsman")
MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "qwen3_tts_craftsman_advanced.gguf")

def compare_vectors(official, gguf_out, vec_name):
    """对比两个向量的相似度"""
    off_flat = official.flatten()
    gguf_flat = gguf_out.flatten()
    
    # 余弦相似度
    norm_off = np.linalg.norm(off_flat)
    norm_gguf = np.linalg.norm(gguf_flat)
    cos_sim = np.dot(off_flat, gguf_flat) / (norm_off * norm_gguf + 1e-9)
    # MAE
    mae = np.mean(np.abs(off_flat - gguf_flat))
    
    mark = "✅" if cos_sim > 0.9999 else "⚠️"
    print(f"  {mark} [{vec_name}] CosSim: {cos_sim:.6f}, MAE: {mae:.6f}")
    return cos_sim

def run_full_verification():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 找不到 GGUF 模型: {MODEL_PATH}")
        return
    
    print(f"--- 正在开启全量工匠 (15步) 闭环验证 ---")
    print(f"加载模型: {MODEL_PATH}")
    
    model = nano_llama.load_model(MODEL_PATH, n_gpu_layers=0)
    if not model: return
        
    ctx_params = nano_llama.llama_context_default_params()
    ctx_params.n_ctx = 512
    ctx_params.embeddings = True # 需要获取隐藏层
    ctx = nano_llama.llama_init_from_model(model, ctx_params)
    n_embd = nano_llama.llama_model_n_embd(model)
    vocab = nano_llama.llama_model_get_vocab(model)
    n_vocab = nano_llama.llama_vocab_n_tokens(vocab) # 应为 30720
    
    print(f"模型参数: n_embd={n_embd}, n_vocab={n_vocab}")

    # 获取全量 Embedding 权重 (用于构造输入校验)
    print("正在从 GGUF 提取 Token Embedding 权重用于输入对齐检查...")
    all_embd_weights = nano_llama.get_token_embeddings_gguf(MODEL_PATH)
    
    prev_gguf_id = None

    # 循环验证 15 步
    for i in range(15):
        print(f"\n>>> Step {i} 验证 <<<")
        
        # 1. 加载官方捕获的输入 (已投影的 Hidden States)
        input_path = os.path.join(CAPTURED_DIR, f"step_{i}_projected_input.npy")
        if not os.path.exists(input_path):
            print(f"  ⚠️ 缺少输入数据，跳过 step {i}")
            continue
            
        proj_input = np.load(input_path).astype(np.float32)
        # 形状可能是 (1, 2, 1024) 或 (1, 1, 1024)
        flat_input = proj_input.reshape(-1, n_embd)
        n_tokens = flat_input.shape[0]
        
        # --- 新增：输入校验逻辑 & 自回归切换 ---
        if i > 0 and prev_gguf_id is not None:
            # 构造输入：取前一帧 ID 在当前前一步 (i-1) 对应的 Embedding
            gguf_input_idx = int(prev_gguf_id + (i - 1) * 2048)
            constructed_input = all_embd_weights[gguf_input_idx]
            
            # 对比官方输入（仅用于参考）
            official_input_this_step = flat_input[0]
            compare_vectors(official_input_this_step, constructed_input, f"S{i} Input Emb")
            
            # 【关键修改】：将构造的输入作为本步实际推理输入，实现真自回归
            flat_input = constructed_input.reshape(1, n_embd)
            n_tokens = 1

        # 2. 构造 Batch 并推理 (由于 i>0 时 flat_input 已被替换，这里实现了自回归)
        batch = nano_llama.llama_batch_init(n_tokens, n_embd, 1)
        batch.n_tokens = n_tokens
        
        # 注入数据 (Embeddings 模式)
        embd_data = np.ascontiguousarray(flat_input.astype(np.float32))
        ctypes.memmove(batch.embd, embd_data.ctypes.data, embd_data.nbytes)
        
        # 计算当前位置 (KV Cache 累积)
        start_pos = 0 if i == 0 else (i + 1)
        
        for t in range(n_tokens):
            batch.pos[t] = start_pos + t
            batch.n_seq_id[t] = 1
            batch.seq_id[t][0] = 0
            batch.logits[t] = 1 if t == n_tokens - 1 else 0
            
        ret = nano_llama.llama_decode(ctx, batch)
        if ret != 0:
            print(f"  ❌ llama_decode 失败: {ret}")
            break

        # 3. 验证 Hidden State 对齐 (Transformer 输出)
        # 获取 GGUF 输出的 Embeddings (Hidden States)
        out_hidden_ptr = nano_llama.llama_get_embeddings(ctx)
        gguf_hidden_all = np.ctypeslib.as_array(out_hidden_ptr, shape=(n_tokens, n_embd)).copy()
        gguf_hidden_last = gguf_hidden_all[-1]
        
        official_hidden_path = os.path.join(CAPTURED_DIR, f"step_{i}_output_hidden.npy")
        if os.path.exists(official_hidden_path):
            official_hidden = np.load(official_hidden_path).astype(np.float32)
            off_hidden_last = official_hidden.flatten().reshape(-1, n_embd)[-1]
            compare_vectors(off_hidden_last, gguf_hidden_last, f"S{i} Hidden")

        # 4. 验证 Token ID 对齐 (LM Head & Shift 逻辑)
        # 获取 Logits [n_tokens, 30720]
        logits_ptr = nano_llama.llama_get_logits(ctx)
        all_logits = np.ctypeslib.as_array(logits_ptr, shape=(n_tokens, n_vocab)).copy()
        # 取最后一个 token 的预测分布
        gguf_logits_last = all_logits[-1]
        
        # 应用 Shift: 提取当前步对应的 2048 维度区域
        shift_start = i * 2048
        shift_end = (i + 1) * 2048
        step_logits = gguf_logits_last[shift_start : shift_end]
        
        gguf_id = np.argmax(step_logits)
        prev_gguf_id = gguf_id # 记录本次预测 ID 用于下一波输入校验
        
        # 加载官方选中的 ID
        official_id_path = os.path.join(CAPTURED_DIR, f"step_{i}_output_ids.npy")
        official_id = np.load(official_id_path).flatten()[0]
        
        if gguf_id == official_id:
            print(f"  ✅ [S{i} Token ID] Match! ID: {gguf_id}")
        else:
            print(f"  ❌ [S{i} Token ID] Mismatch! GGUF: {gguf_id}, Official: {official_id}")
            # 辅助排查：检查 official_id 在 GGUF Logits 中的位置
            full_gguf_id = np.argmax(gguf_logits_last)
            print(f"     DEBUG: Full GGUF Argmax: {full_gguf_id} (Relative: {full_gguf_id % 2048}, Table: {full_gguf_id // 2048})")

        # 清理 Batch
        nano_llama.llama_batch_free(batch)

    # 最终清理
    nano_llama.llama_free(ctx)
    nano_llama.llama_model_free(model)
    print(f"\n--- 验证结束 ---")

if __name__ == "__main__":
    run_full_verification()
