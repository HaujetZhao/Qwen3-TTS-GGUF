import os
import ctypes
import numpy as np
import torch
import qwen3_tts_gguf.nano_llama as nano_llama

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 1. 路径与配置
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
MASTER_GGUF = os.path.join(MODEL_DIR, "qwen3_tts_talker.gguf")
CRAFTSMAN_GGUF = os.path.join(MODEL_DIR, "qwen3_tts_craftsman_advanced.gguf")
MASTER_HEAD_PATH = os.path.join(MODEL_DIR, "codec_head_weight.npy")
PROJ_PT_PATH = os.path.join(MODEL_DIR, "craftsman_hf/master_to_craftsman_proj.pt")
CAPTURED_DIR = os.path.join(PROJECT_ROOT, "captured_assembly")
FULL_GEN_DIR = os.path.join(PROJECT_ROOT, "captured_full_gen")

EOS_TOKEN_ID = 2150

# 2. 辅助函数
def load_assets():
    print("[1/5] 正在加载模型权重与词表资产...")
    assets = {
        "master_head": np.load(MASTER_HEAD_PATH),
        "emb_tables": [np.load(os.path.join(MODEL_DIR, f"codec_embedding_{i}.npy")) for i in range(16)],
        "proj": torch.load(PROJ_PT_PATH, map_location="cpu")
    }
    
    # 辅助数据
    assets["prefill_input"] = np.load(os.path.join(CAPTURED_DIR, "prefill_input_embeds.npy")).astype(np.float32)
    assets["trailing_text"] = np.load(os.path.join(CAPTURED_DIR, "trailing_text_hidden.npy")).astype(np.float32)
    assets["tts_pad"] = np.load(os.path.join(CAPTURED_DIR, "tts_pad_embed.npy")).astype(np.float32)
    
    # 真值
    try:
        assets["official_codes"] = np.load(os.path.join(FULL_GEN_DIR, "full_generated_codes.npy"))
        print(f"  ✅ 发现官方真值 (共 {len(assets['official_codes'])} 步)")
    except Exception:
        print("  ⚠️ 未发现官方真值，将进行盲跑。")
        assets["official_codes"] = None
        
    print("✅ 资产加载完成。")
    return assets

def apply_projection(hidden_2048, proj_assets):
    """维度投影: 2048 -> 1024"""
    w = proj_assets["weight"].float().numpy() # [1024, 2048]
    b = proj_assets["bias"].float().numpy()   # [1024]
    return hidden_2048 @ w.T + b

def run_full_pipeline():
    print("=== [75] 大师 & 工匠全链路 GGUF 联合推理流程 ===\n")
    
    assets = load_assets()
    
    # 初始化模型
    print("[2/5] 初始化 GGUF 推理引擎...")
    m_model = nano_llama.load_model(MASTER_GGUF, n_gpu_layers=0)
    c_model = nano_llama.load_model(CRAFTSMAN_GGUF, n_gpu_layers=0)
    
    m_ctx_params = nano_llama.llama_context_default_params()
    m_ctx_params.n_ctx = 4096
    m_ctx_params.embeddings = True
    m_ctx = nano_llama.llama_init_from_model(m_model, m_ctx_params)
    
    c_ctx_params = nano_llama.llama_context_default_params()
    c_ctx_params.n_ctx = 512
    c_ctx_params.embeddings = True
    c_ctx = nano_llama.llama_init_from_model(c_model, c_ctx_params)
    
    m_embd_dim = 2048
    c_embd_dim = 1024
    
    # ---------------------------------------------------------
    # 步骤 1: 大师 Prefill
    # ---------------------------------------------------------
    print("\n[3/5] 执行大师 Prefill...")
    prefill_input = assets["prefill_input"]
    n_tokens = prefill_input.shape[1]
    
    # 准备 Batch 对象 (复用)
    m_batch = nano_llama.llama_batch_init(4096, m_embd_dim, 1)
    c_batch = nano_llama.llama_batch_init(32, c_embd_dim, 1)
    
    # ---------------------------------------------------------
    # 步骤 1: 大师 Prefill
    # ---------------------------------------------------------
    print("\n[3/5] 执行大师 Prefill...")
    prefill_input = assets["prefill_input"]
    n_tokens = prefill_input.shape[1]
    
    m_batch.n_tokens = n_tokens
    
    flat_prefill = np.ascontiguousarray(prefill_input[0])
    ctypes.memmove(m_batch.embd, flat_prefill.ctypes.data, flat_prefill.nbytes)
    
    # M-RoPE 4D Pos
    for i in range(n_tokens):
        m_batch.pos[i] = i
        m_batch.pos[n_tokens + i] = i
        m_batch.pos[2 * n_tokens + i] = i
        m_batch.pos[3 * n_tokens + i] = 0
        m_batch.n_seq_id[i] = 1
        m_batch.seq_id[i][0] = 0
        m_batch.logits[i] = 1 if i == n_tokens - 1 else 0
        
    nano_llama.llama_decode(m_ctx, m_batch)
    
    # 获取初始隐藏层 [1, 2048]
    m_out_ptr = nano_llama.llama_get_embeddings(m_ctx)
    m_hidden_last = np.ctypeslib.as_array(m_out_ptr, shape=(n_tokens, m_embd_dim))[-1].copy()
    
    current_m_pos = n_tokens
    generated_sequence = []
    
    # ---------------------------------------------------------
    # 步骤 2: 自回归循环 (Max 50 steps)
    # ---------------------------------------------------------
    print(f"\n[4/5] 进入生成循环 (最大 50 步)...")
    
    for step_idx in range(50):
        # A. 大师预测 Code 0
        m_logits = m_hidden_last @ assets["master_head"].T
        code_0 = np.argmax(m_logits)
        
        if code_0 == EOS_TOKEN_ID:
            print(f"  🛑 观测到 EOS ({EOS_TOKEN_ID})，生成结束。")
            break
            
        # B. 工匠生成 15 个分码
        step_codes = [code_0]
        step_embeds_2048 = [assets["emb_tables"][0][code_0].copy()]
        
        # 构造工匠输入并投影
        # [Master Hidden, Code 0 Emb] -> [2, 2048] -> [2, 1024]
        c_in_2048 = np.stack([m_hidden_last, step_embeds_2048[0]], axis=0)
        c_in_1024 = apply_projection(c_in_2048, assets["proj"])
        
        # 复用工匠 Context: 每一轮开始前清理 KV Cache
        mem = nano_llama.llama_get_memory(c_ctx)
        nano_llama.llama_memory_clear(mem, True)
        
        c_batch.n_tokens = 2
        ctypes.memmove(c_batch.embd, c_in_1024.ctypes.data, c_in_1024.nbytes)
        for i in range(2):
            c_batch.pos[i] = i
            c_batch.n_seq_id[i] = 1
            c_batch.seq_id[i][0] = 0
            c_batch.logits[i] = 1 if i == 1 else 0
            
        nano_llama.llama_decode(c_ctx, c_batch)
        
        # 自回归剩余 15 步
        c_logits_ptr = nano_llama.llama_get_logits(c_ctx)
        last_logits_all = np.ctypeslib.as_array(c_logits_ptr, shape=(2, 30720))
        last_logits = last_logits_all[1] # 取最后一帧
        
        for c_step in range(1, 16):
            # 预测
            logits_offset = (c_step - 1) * 2048
            table_logits = last_logits[logits_offset : logits_offset + 2048]
            code = np.argmax(table_logits)
            step_codes.append(code)
            
            # 追加到 Sum 列表
            step_embeds_2048.append(assets["emb_tables"][c_step][code].copy())
            
            if c_step < 15:
                # 投喂下一步输入 (Emb_step -> Predict Code_step+1)
                next_in_2048 = step_embeds_2048[-1]
                next_in_1024 = apply_projection(next_in_2048, assets["proj"])
                
                c_batch.n_tokens = 1
                c_batch.pos[0] = c_step + 1
                ctypes.memmove(c_batch.embd, next_in_1024.ctypes.data, next_in_1024.nbytes)
                c_batch.logits[0] = 1
                
                nano_llama.llama_decode(c_ctx, c_batch)
                last_logits = np.ctypeslib.as_array(nano_llama.llama_get_logits(c_ctx), shape=(30720,))

        generated_sequence.append(step_codes)
        print(f"  Step {step_idx:02}: Codes {step_codes}")
        
        # C. 汇总求和并反馈给大师
        summed = np.sum(step_embeds_2048, axis=0) # [2048]
        
        # 叠加文本偏置
        if step_idx < assets["trailing_text"].shape[1]:
            summed += assets["trailing_text"][0, step_idx]
        else:
            summed += assets["tts_pad"].flatten()
            
        # D. 大师下一步推理
        m_batch.n_tokens = 1
        ctypes.memmove(m_batch.embd, summed.ctypes.data, summed.nbytes)
        
        # M-RoPE 4D
        pos = current_m_pos
        m_batch.pos[0] = pos
        m_batch.pos[1] = pos
        m_batch.pos[2] = pos
        m_batch.pos[3] = 0
        m_batch.logits[0] = 1
        
        nano_llama.llama_decode(m_ctx, m_batch)
        current_m_pos += 1
        
        # 更新下一个循环的隐藏层
        m_out_ptr = nano_llama.llama_get_embeddings(m_ctx)
        m_hidden_last = np.ctypeslib.as_array(m_out_ptr, shape=(1, m_embd_dim))[0].copy()

    # ---------------------------------------------------------
    # 步骤 3: 结果核验
    # ---------------------------------------------------------
    print("\n[5/5] 推理完成，正在执行序列核验...")
    mine = np.array(generated_sequence)
    auth = assets["official_codes"]
    
    if auth is not None:
        min_len = min(len(mine), len(auth))
        match = np.array_equal(mine[:min_len], auth[:min_len])
        if match:
            print("\n✅ 全序列完全对齐！Master GGUF + Craftsman GGUF 联合流水线通过验证。")
        else:
            print("\n❌ 序列存在不一致。")
            diffs = (mine[:min_len] != auth[:min_len]).any(axis=1)
            first_diff = np.where(diffs)[0][0]
            print(f"  首次分歧发生在 Step {first_diff}:")
            print(f"  Mine: {mine[first_diff]}")
            print(f"  Auth: {auth[first_diff]}")
    else:
        print(f"生成的序列长度: {len(mine)}")
        
    # 清理
    nano_llama.llama_batch_free(m_batch)
    nano_llama.llama_batch_free(c_batch)
    nano_llama.llama_free(m_ctx)
    nano_llama.llama_free(c_ctx)
    nano_llama.llama_model_free(m_model)
    nano_llama.llama_model_free(c_model)

if __name__ == "__main__":
    run_full_pipeline()
