import os
import sys
import numpy as np
import ctypes
import qwen3_tts_gguf.nano_llama as nano_llama

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 模型与路径配置
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CAPTURED_DIR = os.path.join(PROJECT_ROOT, "captured_craftsman")
# 诊断用单表 GGUF
SIMPLE_MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "qwen3_tts_craftsman.gguf")

def run_diagnostic():
    if not os.path.exists(SIMPLE_MODEL_PATH):
        print(f"❌ 找不到 Simple GGUF 模型: {SIMPLE_MODEL_PATH}")
        return
    
    print(f"--- [DIAGNOSTIC] Step 0 Token Mismatch ---\n")
    
    model = nano_llama.load_model(SIMPLE_MODEL_PATH, n_gpu_layers=0)
    if not model: return
        
    ctx_params = nano_llama.llama_context_default_params()
    ctx_params.n_ctx = 512
    ctx_params.embeddings = True
    ctx = nano_llama.llama_init_from_model(model, ctx_params)
    
    n_embd = nano_llama.llama_model_n_embd(model)
    vocab_ptr = nano_llama.llama_model_get_vocab(model)
    n_vocab = nano_llama.llama_vocab_n_tokens(vocab_ptr)
    
    # --- 1. 准备 Step 0 输入 (2 Tokens) ---
    input_path = os.path.join(CAPTURED_DIR, "step_0_input_2048.npy")
    raw_input = np.load(input_path).astype(np.float32) # [1, 2, 2048]
    
    # 手动投影 (同 64 脚本)
    from safetensors.torch import load_file
    orig_model_path = os.path.join(PROJECT_ROOT, "Qwen3-TTS-12Hz-1.7B-CustomVoice", "model.safetensors")
    orig_weights = load_file(orig_model_path)
    proj_w = orig_weights["talker.code_predictor.small_to_mtp_projection.weight"].float().numpy()
    proj_b = orig_weights["talker.code_predictor.small_to_mtp_projection.bias"].float().numpy()
    
    flat_input = raw_input.reshape(-1, 2048)
    proj_input = flat_input @ proj_w.T + proj_b # [2, 1024]
    n_tokens = proj_input.shape[0]
    
    # --- 2. 构造 Batch: 要求输出所有 Token 的 Logits ---
    batch = nano_llama.llama_batch_init(n_tokens, n_embd, 1)
    batch.n_tokens = n_tokens
    
    embd_data = np.ascontiguousarray(proj_input.astype(np.float32))
    ctypes.memmove(batch.embd, embd_data.ctypes.data, embd_data.nbytes)
    
    for i in range(n_tokens):
        batch.pos[i] = i
        batch.n_seq_id[i] = 1
        batch.seq_id[i][0] = 0
        batch.logits[i] = 1 # 全部开启！
        
    print(f"正在执行诊断推理 (n_tokens={n_tokens})...")
    nano_llama.llama_decode(ctx, batch)
    
    # --- 3. 获取所有 Logits 并分析 ---
    # llama_get_logits 通常返回最后一个 token 的 logits，
    # 但如果开启了多个，则连续排列 [n_token_with_logits, n_vocab]
    logits_ptr = nano_llama.llama_get_logits(ctx)
    all_logits = np.ctypeslib.as_array(logits_ptr, shape=(n_tokens, n_vocab)).copy()
    
    print("\n--- 结果对比 ---")
    official_id_path = os.path.join(CAPTURED_DIR, "step_0_output_ids.npy")
    official_id = np.load(official_id_path).flatten()[0]
    print(f"官方捕获 ID: {official_id}")
    
    for t in range(n_tokens):
        t_logits = all_logits[t]
        pred_id = np.argmax(t_logits)
        conf = t_logits[pred_id]
        
        # 看看官方 ID 在这里的排名和数值
        off_val = t_logits[official_id]
        rank = (t_logits > off_val).sum() + 1
        
        print(f"[Token {t}] Argmax: {pred_id} (conf: {conf:.4f}), Official({official_id}) val: {off_val:.4f}, Rank: {rank}")

    # 清理
    nano_llama.llama_batch_free(batch)
    nano_llama.llama_free(ctx)
    nano_llama.llama_model_free(model)

if __name__ == "__main__":
    run_diagnostic()
