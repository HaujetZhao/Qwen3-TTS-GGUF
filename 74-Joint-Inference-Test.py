import os
import ctypes
import numpy as np
import torch
import qwen3_tts_gguf.nano_llama as nano_llama

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 路径配置
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
MASTER_GGUF = os.path.join(MODEL_DIR, "qwen3_tts_talker.gguf")
CRAFTSMAN_GGUF = os.path.join(MODEL_DIR, "qwen3_tts_craftsman_advanced.gguf")
MASTER_HEAD_PATH = os.path.join(MODEL_DIR, "codec_head_weight.npy")
PROJ_PT_PATH = os.path.join(MODEL_DIR, "craftsman_hf/master_to_craftsman_proj.pt")

def load_assets():
    print("正在加载外部权重资产...")
    assets = {
        "master_head": np.load(MASTER_HEAD_PATH),
        "emb_tables": [np.load(os.path.join(MODEL_DIR, f"codec_embedding_{i}.npy")) for i in range(16)],
        "proj": torch.load(PROJ_PT_PATH, map_location="cpu")
    }
    print("✅ 资产加载完成。")
    return assets

def apply_projection(hidden_2048, proj_assets):
    """将 2048 维隐藏层投影到 1024 维"""
    w = proj_assets["weight"].float().numpy() # [1024, 2048]
    b = proj_assets["bias"].float().numpy()   # [1024]
    return hidden_2048 @ w.T + b

def compare_vectors(official, gguf_out, vec_name, threshold_cos=0.999, threshold_mae=1e-3):
    """对比两个向量的相似度"""
    off_flat = official.flatten()
    gguf_flat = gguf_out.flatten()
    
    # 余弦相似度
    norm_off = np.linalg.norm(off_flat)
    norm_gguf = np.linalg.norm(gguf_flat)
    if norm_off == 0 or norm_gguf == 0:
        cos_sim = 0.0
    else:
        cos_sim = np.dot(off_flat, gguf_flat) / (norm_off * norm_gguf)
        
    # MAE
    mae = np.mean(np.abs(off_flat - gguf_flat))
    
    mark = "✅" if cos_sim > threshold_cos else "⚠️"
    print(f"  {mark} [{vec_name}] CosSim: {cos_sim:.6f}, MAE: {mae:.6f}")
    return cos_sim

def run_joint_inference():
    print("=== [74] 大师与工匠联合对齐验证 (Joint Alignment) ===\n")
    
    CAPTURED_DIR = os.path.join(PROJECT_ROOT, "captured_assembly")
    if not os.path.exists(CAPTURED_DIR):
        print(f"❌ 找不到捕获数据目录: {CAPTURED_DIR}")
        return

    # 1. 加载模型与资产
    assets = load_assets()
    
    print("加载 GGUF 模型...")
    master_model = nano_llama.load_model(MASTER_GGUF, n_gpu_layers=0)
    craftsman_model = nano_llama.load_model(CRAFTSMAN_GGUF, n_gpu_layers=0)
    
    ctx_params = nano_llama.llama_context_default_params()
    ctx_params.n_ctx = 2048 # 根据实际 Sequence Length 调整
    ctx_params.embeddings = True
    
    m_ctx = nano_llama.llama_init_from_model(master_model, ctx_params)
    c_ctx = nano_llama.llama_init_from_model(craftsman_model, ctx_params)
    
    m_embd_dim = 2048
    c_embd_dim = 1024
    
    # --- 步骤 A: 大师 Prefill ---
    print("\n--- 步骤 A: 大师 Prefill ---")
    
    # 加载真实 Prefill 输入
    prefill_input = np.load(os.path.join(CAPTURED_DIR, "prefill_input_embeds.npy")).astype(np.float32) # [1, T, 2048]
    n_tokens = prefill_input.shape[1]
    
    m_batch = nano_llama.llama_batch_init(n_tokens * 4, m_embd_dim, 1)
    m_batch.n_tokens = n_tokens
    
    # 注入 Prefill 数据
    flat_input = np.ascontiguousarray(prefill_input[0])
    ctypes.memmove(m_batch.embd, flat_input.ctypes.data, flat_input.nbytes)
    
    # M-RoPE 4维位置设置 (T, H, W, Delta)
    for i in range(n_tokens):
        m_batch.pos[i] = i
        m_batch.pos[n_tokens + i] = i
        m_batch.pos[2 * n_tokens + i] = i
        m_batch.pos[3 * n_tokens + i] = 0
        
        m_batch.n_seq_id[i] = 1
        m_batch.seq_id[i][0] = 0
        m_batch.logits[i] = 1 if i == n_tokens - 1 else 0
    
    nano_llama.llama_decode(m_ctx, m_batch)
    
    # 获取 Prefill 最后一帧输出
    m_out_ptr = nano_llama.llama_get_embeddings(m_ctx)
    m_hidden_all = np.ctypeslib.as_array(m_out_ptr, shape=(n_tokens, m_embd_dim))
    m_hidden_last = m_hidden_all[-1].copy()
    
    # 验证 Prefill 输出
    official_prefill_hidden = np.load(os.path.join(CAPTURED_DIR, "prefill_output_hidden.npy")).astype(np.float32)
    # official shape maybe [1, 1, 2048]
    compare_vectors(official_prefill_hidden, m_hidden_last, "Prefill Output Hidden")
    
    # 预测第一个 Code (Table 0)
    # 注意：Official Capture 并没有保证 master_step_0_result_codes[0] 一定是 master head 的输出结果。
    # 官方逻辑是：predict -> code -> lookup. 
    # 我们先验证这个 predict 动作。
    logits_0 = m_hidden_last @ assets["master_head"].T
    code_0 = np.argmax(logits_0)
    
    official_codes = np.load(os.path.join(CAPTURED_DIR, "master_step_0_result_codes.npy")) # [1, 16]
    official_code_0 = official_codes[0, 0]
    
    if code_0 == official_code_0:
        print(f"  ✅ Code 0 Match: {code_0}")
    else:
        print(f"  ❌ Code 0 Mismatch! Mine: {code_0}, Official: {official_code_0}")
    
    # 查原始 2048 词表
    emb_0 = assets["emb_tables"][0][code_0] # [2048]
    
    # --- 步骤 B: 构造工匠输入并投影 ---
    print("\n--- 步骤 B: 工匠推理 (15 步自回归) ---")
    # 工匠输入拼接: [Master Hidden, Emb 0] -> [2, 2048]
    # 注意：Master Hidden 应该是 [1, 2048]，emb_0 是 [2048]。由于 GGUF 工匠需要 [1, 2, 1024]，我们堆叠这两个
    combined_input = np.stack([m_hidden_last, emb_0], axis=0) # [2, 2048]
    
    # 执行维度投影 -> [2, 1024]
    projected_input = apply_projection(combined_input, assets["proj"]) # [2, 1024]
    
    # 验证输入 (参考 Only)
    # official_craftsman_input = np.load(...) 
    # capturing script saves projected input [1, 2, 2048] BEFORE projection??? 
    # No, captured data name is "craftsman_step_0_input_2048.npy". Wait.
    # Code says: inputs_embeds = kwargs.get('inputs_embeds')...
    # 25-Capture: save "craftsman_step_0_input_2048.npy" with shape [1, 2, 2048] usually
    # If using Advanced GGUF (which expects 1024), we manually project it.
    
    # 给工匠起跑
    c_batch = nano_llama.llama_batch_init(16, c_embd_dim, 1)
    # Step 0: Feed 2 tokens (Master Hidden + Code 0 Emb)
    c_batch.n_tokens = 2
    ctypes.memmove(c_batch.embd, projected_input.ctypes.data, projected_input.nbytes)
    for i in range(2):
        c_batch.pos[i] = i
        c_batch.n_seq_id[i] = 1
        c_batch.seq_id[i][0] = 0
        c_batch.logits[i] = 1 if i == 1 else 0
        
    nano_llama.llama_decode(c_ctx, c_batch)
    
    # 获取 Logits 并自回归
    c_logits_ptr = nano_llama.llama_get_logits(c_ctx)
    all_logits = np.ctypeslib.as_array(c_logits_ptr, shape=(2, 30720)) 
    last_logits = all_logits[1]
    
    generated_codes = [code_0]
    
    # 自回归生成剩余 15 个 Code (1...15) wait, total 16 codes (0...15).
    # Master produces Code 0. Craftsman produces Code 1..15.
    # Wait. Craftsman Step 0 (input: M + C0) -> Prediction: Code 1.
    # ...
    # Step 14 (input: C14) -> Prediction: Code 15.
    
    for step in range(1, 16): # Want codes 1 to 15
        # 提取当前 Table (Step) 的分布
        # GGUF Logic: Logits offset is i * 2048. 
        # But wait, Step 1 (predicting Code 1) uses Table 1? 
        # Let's check 72 script: "step_logits = gguf_logits_last[i*2048 : (i+1)*2048]"
        # Loop i from 0 to 14.
        
        # Here step runs 1..15.
        # Logits index for predicting Code `k` is `(k-1)*2048`? 
        # No.
        # Step 0 (Loop i=0): Predicts Code 1 (using Table 1 weights?). 
        # Actually:
        # Code 0 (Master) -> [Craftsman] -> Predicts Code 1.
        # With Virtual Vocab: The head for Step `t` is at offset `t*2048`.
        # Step 0 of Craftsman corresponds to predicting Code 1.
        # So we should look at offset 0? Or offset 2048?
        # Let's align with 72 script.
        # 72 script: Loop i=0..14. 
        # i=0: Predicts ID (official ID is 355). 355 is Code 1 (Code 0 is 1159).
        # i=0 uses offset 0*2048.
        # So: Step `t` (0-indexed) of Craftsman predicts Code `t+1`, using logits offset `t*2048`.
        
        craftsman_step_idx = step - 1 # 0..14
        logits_offset = craftsman_step_idx * 2048
        
        table_logits = last_logits[logits_offset : logits_offset + 2048]
        code = np.argmax(table_logits)
        generated_codes.append(code)
        
        # 如果不是最后一步，准备下一步输入
        if step < 15:
            c_batch.n_tokens = 1
            c_batch.pos[0] = step + 1 # 0,1 used. Next is 2.
            
            # 构造输入: 使用刚刚预测出的 code 在当前 Craftsman Step (t) 对应的表?
            # 72 fix: "prev_gguf_id + (i-1)*2048".
            # Input for Step `t+1` comes from Code `t+1` (just predicted).
            # Which table?
            # Step 0 (predicts C1). Step 1 input is C1 embedding.
            # C1 embedding comes from Table 1?
            # 72 script logic: "gguf_input_idx = int(prev_gguf_id + (i - 1) * 2048)"
            # where i is loop index 0..14. Wait.
            # If i=1 (Step 1), input is from (1-1)*2048 = 0*2048? No.
            # 72 script Input alignment check: i=1 input matches Table 0??
            # Please re-verify 72 script comments.
            # "gguf_input_idx = int(prev_gguf_id + (i - 1) * 2048)" where i starts from 1. 
            # If i=1, offset is 0. So input is from Table 0? 
            # But prev_gguf_id was from Logits offset (i-1)*2048? No.
            
            # Let's simplify:
            # Code 0 (Table 0). 
            # Code 1 (Table 1).
            # ...
            # Code k (Table k).
            # The "Glue" Logic uses Sum(Emb_k[Code_k]).
            # The "Autoregression" Logic uses Code_k[Emb_k] as input for next step?
            # Usually: Input to predict `k+1` is `Emb_k`.
            # So:
            # To predict Code 1 -> Input is Emb_0[Code_0] (This is what we fed in Step 0: [M, Emb0])
            # To predict Code 2 -> Input is Emb_1[Code_1]
            # ...
            # To predict Code k+1 -> Input is Emb_k[Code_k]
            
            # So for next step (step+1, which predicts step+2), input is Emb_step[code].
            # Emb table index is `step`. (Since we just got `code` which is Code `step`).
            
            emb_idx = step # e.g. Just got Code 1. Need Table 1.
            
            # 查原始 2048 表并投影
            next_emb_2048 = assets["emb_tables"][emb_idx][code]
            next_emb_1024 = apply_projection(next_emb_2048, assets["proj"])
            
            ctypes.memmove(c_batch.embd, next_emb_1024.ctypes.data, next_emb_1024.nbytes)
            c_batch.logits[0] = 1
            
            nano_llama.llama_decode(c_ctx, c_batch)
            last_logits = np.ctypeslib.as_array(nano_llama.llama_get_logits(c_ctx), shape=(30720,))

    print(f"生成的完整分码列表 (16个):")
    print(f"  Mine: {generated_codes}")
    print(f"  Auth: {official_codes[0].tolist()}")
    
    if np.array_equal(generated_codes, official_codes[0]):
        print("  ✅ All Codes Match!")
    else:
        print("  ❌ Codes Mismatch.")

    # --- 步骤 C: 求和反馈给大师 ---
    print("\n--- 步骤 C: 反馈求和并对齐 ---")
    summed_vector = np.zeros(m_embd_dim, dtype=np.float32)
    for i, code in enumerate(generated_codes):
        # 查原始 2048 表
        summed_vector += assets["emb_tables"][i][code]
        
    # 叠加 Trailing Text / Pad
    trailing_text = np.load(os.path.join(CAPTURED_DIR, "trailing_text_hidden.npy")).astype(np.float32)
    # 假设 Step 0
    generation_step = 0 
    if generation_step < trailing_text.shape[1]:
        summed_vector += trailing_text[0, generation_step]
    else:
        tts_pad = np.load(os.path.join(CAPTURED_DIR, "tts_pad_embed.npy")).astype(np.float32)
        summed_vector += tts_pad[0]
        
    # 验证最终输入
    official_backbone_input = np.load(os.path.join(CAPTURED_DIR, "master_step_0_backbone_input.npy")).astype(np.float32)
    # official [1, 1, 2048]
    compare_vectors(official_backbone_input, summed_vector, "Master Next Input (Summed)")

    # (可选) 驱动大师二次推理
    m_batch.n_tokens = 1
    m_batch.pos[0] = n_tokens # T
    m_batch.pos[1] = n_tokens # H
    m_batch.pos[2] = n_tokens # W
    m_batch.pos[3] = 0        # Delta
    
    ctypes.memmove(m_batch.embd, summed_vector.ctypes.data, summed_vector.nbytes)
    m_batch.logits[0] = 1
    
    nano_llama.llama_decode(m_ctx, m_batch)
    print("✅ 大师二次推理完成 (No Crash).")

    # 清理
    nano_llama.llama_batch_free(m_batch)
    nano_llama.llama_batch_free(c_batch)
    nano_llama.llama_free(m_ctx)
    nano_llama.llama_free(c_ctx)
    nano_llama.llama_model_free(master_model)
    nano_llama.llama_model_free(craftsman_model)

if __name__ == "__main__":
    run_joint_inference()
