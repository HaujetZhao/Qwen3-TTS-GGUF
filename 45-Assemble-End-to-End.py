import os
import sys
import ctypes
import numpy as np
import onnxruntime as ort
import qwen3_tts_gguf.nano_llama as nano_llama

# 路径配置
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CAPTURED_DIR = os.path.join(PROJECT_ROOT, "captured_assembly")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
GGUF_PATH = os.path.join(MODEL_DIR, "qwen3_tts_talker.gguf")
ONNX_PATH = os.path.join(MODEL_DIR, "qwen3_tts_predictor.onnx")

def run_assembly():
    print(f"--- 组装联合测试 (Assembly Test) ---")
    
    # 1. 加载捕获的数据
    print("加载捕获数据...")
    try:
        prefill_input_embeds = np.load(os.path.join(CAPTURED_DIR, "prefill_input_embeds.npy")).astype(np.float32)
        official_prefill_hidden = np.load(os.path.join(CAPTURED_DIR, "prefill_output_hidden.npy")).astype(np.float32)
        official_craftsman_input = np.load(os.path.join(CAPTURED_DIR, "craftsman_step_0_input_2048.npy")).astype(np.float32)
        official_final_codes = np.load(os.path.join(CAPTURED_DIR, "master_step_0_result_codes.npy"))
    except FileNotFoundError as e:
        print(f"❌ 缺少捕获文件: {e}")
        return

    # 提取 official last_id_hidden (raw embedding)
    # craftsman_input = [past_hidden, last_id_hidden] (Dim: [B, 2, 2048])
    # 注意：Official Capture 并没有保证 dim=2 是 cat 顺序，需检查 modeling 代码
    # modeling: torch.cat((past_hidden, last_id_hidden), dim=1)
    # past_hidden 通常是 [B, 1, 2048]. so index 0 is past_hidden, index 1 is last_id_hidden.
    
    official_last_id_hidden = official_craftsman_input[:, 1:2, :] # [B, 1, 2048]
    
    # 2. 运行 Master GGUF
    print(f"\n[Master GGUF] Loading {GGUF_PATH}...")
    model = nano_llama.load_model(GGUF_PATH, n_gpu_layers=0)
    if not model: return
    
    ctx_params = nano_llama.llama_context_default_params()
    ctx_params.n_ctx = 2048
    ctx_params.embeddings = True
    ctx = nano_llama.llama_init_from_model(model, ctx_params)
    n_embd = nano_llama.llama_model_n_embd(model)
    
    # 准备 GGUF Batch (Prefill)
    n_tokens = prefill_input_embeds.shape[1]
    batch = nano_llama.llama_batch_init(n_tokens * 4, n_embd, 1)
    batch.n_tokens = n_tokens
    
    full_embd = np.ascontiguousarray(prefill_input_embeds[0])
    ctypes.memmove(batch.embd, full_embd.ctypes.data, full_embd.nbytes)
    
    current_pos = 0 # 初始位置
    for k in range(n_tokens):
        pos = current_pos + k
        # M-RoPE 填充 (Mocking position layout, assuming pure text prompt or handled similarly)
        # Prefill 阶段通常是 Prompt，包含 Vision 吗？ "今天天气不错" 是纯文本。
        # 纯文本时 pos 是一样的。
        batch.pos[k] = pos
        batch.pos[n_tokens + k] = pos
        batch.pos[2 * n_tokens + k] = pos
        batch.pos[3 * n_tokens + k] = 0
        batch.n_seq_id[k] = 1
        batch.seq_id[k][0] = 0
        batch.logits[k] = 1 if k == n_tokens - 1 else 0 # 只计算最后一个 token 的 output
    
    print(f"[Master GGUF] Running Prefill ({n_tokens} tokens)...")
    ret = nano_llama.llama_decode(ctx, batch)
    if ret != 0:
        print(f"❌ GGUF 推理失败: {ret}")
        return
        
    out_ptr = nano_llama.llama_get_embeddings(ctx)
    gguf_full_hidden = np.ctypeslib.as_array(out_ptr, shape=(n_tokens, n_embd))
    gguf_last_hidden = gguf_full_hidden[-1].copy() # [2048]
    gguf_past_hidden = gguf_last_hidden.reshape(1, 1, n_embd) # [1, 1, 2048]
    
    # 验证 Master Output
    # official_prefill_hidden [1, 1, 2048]
    diff_master = np.abs(official_prefill_hidden - gguf_past_hidden).mean()
    cos_master = np.dot(official_prefill_hidden.flatten(), gguf_past_hidden.flatten()) / (
        np.linalg.norm(official_prefill_hidden) * np.linalg.norm(gguf_past_hidden)
    )
    
    print(f"  [Check 1] GGUF Output vs Official Output")
    pass_mark_1 = "✅" if diff_master < 1e-3 and cos_master > 0.999 else "⚠️"
    print(f"  {pass_mark_1} MAE: {diff_master:.6f}, CosSim: {cos_master:.6f}")
    
    # 3. 组装输入 (Glue)
    # Hybrid Input = GGUF Output + Official Last ID Embedding
    hybrid_craftsman_input = np.concatenate([gguf_past_hidden, official_last_id_hidden], axis=1) # [1, 2, 2048]
    
    # 4. 运行 Craftsman ONNX
    # 4. 运行 Craftsman ONNX (Step 0)
    print(f"\n[Craftsman ONNX] Loading {ONNX_PATH}...")
    
    # Load Heads
    heads_path = os.path.join(MODEL_DIR, "qwen3_tts_predictor_heads.npy")
    print(f"[Craftsman ONNX] Loading Heads from {heads_path}...")
    if not os.path.exists(heads_path):
        print(f"❌ Heads file missing")
        return
    predictor_heads = np.load(heads_path) # [15, 3072, 2048]

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(ONNX_PATH, sess_options, providers=['CPUExecutionProvider'])
    
    # Prepare Empty Pasts (10 tensors) for Step 0
    # Shape: (1, 8, 0, 128)
    current_pasts = {f"past_{i}": np.zeros((1, 8, 0, 128), dtype=np.float32) for i in range(10)}
    
    inputs = {'inputs_embeds': hybrid_craftsman_input}
    inputs.update(current_pasts)
    
    print(f"[Craftsman ONNX] Running Inference (Step 0)...")
    outputs = sess.run(None, inputs)
    onnx_hidden = outputs[0] # [1, 1, 2048]
    
    # 5. 计算 Code
    # Step 0 对应 Head 0 (预测第 1 个 code group, input 是 group 0)
    head_weight = predictor_heads[0] # (3072, 2048)
    onnx_logits = onnx_hidden[0, -1] @ head_weight.T # [3072]
    onnx_code = np.argmax(onnx_logits)
    
    # Official Codes: [1, 16] -> [input, pred_0, pred_1, ... pred_14]
    # We compare onnx_code vs official_final_codes[0, 1]
    official_code_val = official_final_codes[0, 1]
    
    print(f"  [Check 2] ONNX Code vs Official Code (Step 0)")
    print(f"  Official: {official_code_val}")
    print(f"  ONNX    : {onnx_code}")
    
    match = (official_code_val == onnx_code)
    
    if match and diff_master < 0.1: # Relaxed GGUF check due to BF16 diff
        print("\n结论: GGUF 大师 + ONNX 工匠 组装测试通过！")
    else:
        print("\n结论: 组装测试存在差异。")
    
    # Cleanup
    nano_llama.llama_batch_free(batch)
    nano_llama.llama_free(ctx)
    nano_llama.llama_model_free(model)

if __name__ == "__main__":
    run_assembly()
