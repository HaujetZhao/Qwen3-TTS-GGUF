import os
import sys
import numpy as np
import ctypes
import qwen3_tts_gguf.nano_llama as nano_llama

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # 绕过 OpenMP 冲突

# 模型与路径配置
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CAPTURED_DIR = os.path.join(PROJECT_ROOT, "captured_craftsman")
MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "qwen3_tts_craftsman.gguf")

def compare_vectors(official, gguf_out, vec_name):
    """对比两个向量的相似度"""
    off_flat = official.flatten()
    gguf_flat = gguf_out.flatten()
    
    if len(off_flat) != len(gguf_flat):
        print(f"[{vec_name}] ❌ 维度不匹配: Official {len(off_flat)} vs GGUF {len(gguf_flat)}")
        return 0.0

    # 余弦相似度
    norm_off = np.linalg.norm(off_flat)
    norm_gguf = np.linalg.norm(gguf_flat)
    cos_sim = np.dot(off_flat, gguf_flat) / (norm_off * norm_gguf + 1e-9)
    
    # MAE (Mean Absolute Error)
    mae = np.mean(np.abs(off_flat - gguf_flat))
    
    mark = "✅" if cos_sim > 0.999 else "⚠️"
    print(f"{mark} [{vec_name}] CosSim: {cos_sim:.6f}, MAE: {mae:.6f}")
    return cos_sim

def run_verification():
    # 1. 检查文件存在性
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 找不到 GGUF 模型: {MODEL_PATH}")
        return
    
    input_path = os.path.join(CAPTURED_DIR, "step_0_input_2048.npy")
    output_path = os.path.join(CAPTURED_DIR, "step_0_output_hidden.npy")
    
    if not os.path.exists(input_path) or not os.path.exists(output_path):
        print(f"❌ 找不到捕获数据 (step_0_input/output)")
        return
    
    # 2. 加载 GGUF 模型
    print(f"正在加载 GGUF 模型: {MODEL_PATH}")
    # 强制使用 CPU，避免量化误差干扰验证
    model = nano_llama.load_model(MODEL_PATH, n_gpu_layers=0) 
    if not model:
        print("❌ 模型加载失败")
        return
        
    ctx_params = nano_llama.llama_context_default_params()
    ctx_params.n_ctx = 2048
    ctx_params.embeddings = True  # 开启 Embedding 模式以获取 hidden states
    ctx = nano_llama.llama_init_from_model(model, ctx_params)
    
    n_embd = nano_llama.llama_model_n_embd(model)
    print(f"GGUF n_embd: {n_embd} (应为 1024)")
    
    # 3. 加载并预处理输入数据
    # 原始输入是 [1, 1, 2048] 的 Master Output
    raw_input = np.load(input_path).astype(np.float32)
    print(f"原始输入维度: {raw_input.shape}")
    
    # 我们需要在 Python 端做第一次投影 (small_to_mtp)，模拟 2048 -> 1024
    # 因为我们在提取权重时，只对 Embedding 做了投影，
    # 但工匠的第 0 步输入并非 Embedding，而是外部传入的 Master Hidden State
    # 所以我们需要加载那个投影矩阵手动算一下
    
    print("正在加载投影层权重以预处理输入...")
    from safetensors.torch import load_file
    hf_weights = load_file(os.path.join(PROJECT_ROOT, "model", "craftsman_hf", "model.safetensors"))
    # 从 HF 权重中找回投影矩阵（注意：我们之前提取时并没保存投影层到 GGUF，而是留在外面了）
    # 但等等，我们在 60-Extract-Craftsman-HF.py 里没有保存单独的投影层文件！
    # 不过我们有保存到 model.safetensors 里吗？
    # 检查一下 60 脚本代码... 是的，我们提取出的 model.safetensors 本质上是给 GGUF 转换用的，
    # 并没有包含那个 'small_to_mtp_projection'。
    
    # 哎呀，这里是个小疏忽。我们在 export 时把投影层用来处理 embedding 之后就丢弃了（或者只存在内存里）。
    # 但对于 Step 0，输入是 Hidden State，也需要经过同样的投影。
    # 所以我们需要重新从原始 safetensors 加载投影矩阵。
    
    orig_model_path = os.path.join(PROJECT_ROOT, "Qwen3-TTS-12Hz-1.7B-CustomVoice", "model.safetensors")
    orig_weights = load_file(orig_model_path)
    proj_w = orig_weights["talker.code_predictor.small_to_mtp_projection.weight"].float().numpy() # [1024, 2048]
    proj_b = orig_weights["talker.code_predictor.small_to_mtp_projection.bias"].float().numpy()   # [1024]
    
    # 执行投影: Input [1, 2048] @ W.T [2048, 1024] + b
    # 注意维度：nn.Linear(2048, 1024) 的 weight 形状是 [1024, 2048]
    # Input [1, 2048] @ W.T [2048, 1024] -> [1, 1024]
    flat_input = raw_input.reshape(-1, 2048) # [1, 2048]
    projected_input = flat_input @ proj_w.T + proj_b
    
    print(f"投影后输入维度: {projected_input.shape} (应为 [n_tokens, 1024])")
    n_tokens = projected_input.shape[0] # 获取实际 token 数 (例如 2)
    
    # 4. 构造 Batch 并推理
    # 这里的输入还是 Embeddings 模式（直接注入 hidden state）
    batch = nano_llama.llama_batch_init(n_tokens, n_embd, 1)
    batch.n_tokens = n_tokens
    
    # 注入数据 (projected_input)
    embd_data = np.ascontiguousarray(projected_input.astype(np.float32))
    ctypes.memmove(batch.embd, embd_data.ctypes.data, embd_data.nbytes)
    
    # 设置位置 - Step 0
    # 注意：llama_batch_init 返回的是结构体，其数字段是 C 指针
    # 我们需要通过数组索引来赋值
    for i in range(n_tokens):
        batch.pos[i] = i
        batch.n_seq_id[i] = 1
        batch.seq_id[i][0] = 0
        # 只需要最后一个 token 输出
        batch.logits[i] = 1 if i == n_tokens - 1 else 0
    
    # 关键修正：确保所有 seq_id 初始化（虽然这里只有一个）。
    # 另外，验证 batch.pos 等是否真的是可写数组。
    # 实际上 nano_llama.py 中定义的是 POINTER(c_int)，所以 batch.pos[0] = 0 是合法的。
    # 但崩溃发生在 llama_decode，可能是因为 embeddings 数据传递有问题，
    # 或者 context 初始化时有问题。
    
    # 再检查一下 ctx_params.embeddings = True 是否生效
    print(f"DEBUG: Batch n_tokens={batch.n_tokens}, embd={batch.embd}")
    
    # 5. 执行解码
    print("正在执行 GGUF 推理...")
    ret = nano_llama.llama_decode(ctx, batch)
    if ret != 0:
        print(f"❌ llama_decode 失败: {ret}")
        return

    # 6. 获取输出
    out_ptr = nano_llama.llama_get_embeddings(ctx)
    gguf_output = np.ctypeslib.as_array(out_ptr, shape=(n_tokens, n_embd)).copy()
    
    # 7. 对比官方输出
    official_output = np.load(output_path).astype(np.float32)
    print(f"官方输出维度: {official_output.shape}") 
    official_last = official_output.flatten().reshape(-1, 1024)[-1]
    gguf_last = gguf_output[-1]
    
    compare_vectors(official_last, gguf_last, "Step 0 Hidden State")
    
    # 8. 校验 Logits (Token ID)
    n_vocab = nano_llama.llama_vocab_n_tokens(nano_llama.llama_model_get_vocab(model))
    logits_ptr = nano_llama.llama_get_logits(ctx)
    # 重要修正：如果 n_tokens > 1 且开了 embeddings，llama.cpp 可能会输出所有 token 的 logits
    # 我们需要取最后一个 token 的预测结果
    all_logits = np.ctypeslib.as_array(logits_ptr, shape=(n_tokens, n_vocab)).copy()
    gguf_logits = all_logits[-1]
    
    gguf_id = np.argmax(gguf_logits)
    
    # 加载官方 ID
    official_id_path = os.path.join(CAPTURED_DIR, "step_0_output_ids.npy")
    if os.path.exists(official_id_path):
        official_id = np.load(official_id_path).flatten()[0]
        if gguf_id == official_id:
            print(f"✅ [Step 0 Token ID] Match! ID: {gguf_id}")
        else:
            print(f"❌ [Step 0 Token ID] Mismatch! GGUF: {gguf_id}, Official: {official_id}")
            print(f"   DEBUG: GGUF Token 0 Argmax: {np.argmax(all_logits[0])}")
    else:
        print(f"⚠️ 找不到官方 ID 文件: {official_id_path}")
    
    # 清理
    nano_llama.llama_batch_free(batch)
    nano_llama.llama_free(ctx)
    nano_llama.llama_model_free(model)

if __name__ == "__main__":
    run_verification()
