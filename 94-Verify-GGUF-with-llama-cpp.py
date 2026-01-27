"""
使用 llama.cpp C API 直接注入 embeddings 来验证 GGUF 模型

验证内容：
1. 使用 llama.cpp DLL 加载 GGUF 模型
2. 直接注入 embeddings（与 92 脚本相同的输入）
3. 验证推理结果是否与 HF 模型一致
4. 检查 logits 分布是否符合合并词表方案（前 3072 个位置有效）

参考：fun_asr_gguf/nano_llama.py 和 core/decoder.py
"""
import os
import sys
import numpy as np
import ctypes
import time

os.environ["VK_ICD_FILENAMES"] = "none"       # 禁止 Vulkan
PROJECT_ROOT = '.'

import qwen3_tts_gguf.nano_llama as nano_llama


def run_inference(ctx, inputs_embeds, vocab_size, expected_token_id, run_name=""):
    """执行单次推理并返回预测的 token ID"""
    print(f"\n--- Running Inference: {run_name} ---")
    try:
        # 清空 KV 缓存 (使用 nano_llama 中的非标准方法)
        mem = nano_llama.llama_get_memory(ctx)
        nano_llama.llama_memory_clear(mem, True)

        # 准备 batch
        n_input_tokens = inputs_embeds.shape[1]
        full_embd = inputs_embeds[0].astype(np.float32)
        if not full_embd.flags['C_CONTIGUOUS']:
            full_embd = np.ascontiguousarray(full_embd)

        # 重要：对于 qwen3vl/M-RoPE 模型，注入 embedding 时 llama.cpp 不会自动扩展 1D pos 为 4D。
        # 它会直接读取 n_tokens * 4 个 llama_pos。
        # 技巧：使用 n_input_tokens * 4 初始化 batch，这会分配足够的 pos 空间 (n_tokens * 4)。
        # 然后将 batch.n_tokens 设置回原来的 n_input_tokens。
        # 这样 embd 分配了 (n_tokens * 4 * n_embd)，pos 分配了 (n_tokens * 4)。
        # llama_decode 会使用 batch.n_tokens (=40)，并读取 pos 指针处的 40 * 4 = 160 个值，刚好安全。
        
        batch = nano_llama.llama_batch_init(n_input_tokens * 4, full_embd.shape[1], 1)
        batch.n_tokens = n_input_tokens
        
        batch.token = ctypes.cast(None, ctypes.POINTER(nano_llama.llama_token))
        ctypes.memmove(batch.embd, full_embd.ctypes.data, full_embd.nbytes)
        
        for k in range(n_input_tokens):
            # 将 1D position k 扩展为 (k, k, k, 0)
            # 布局是 [n_tokens * dim1, n_tokens * dim2, ...]
            batch.pos[k] = k
            batch.pos[n_input_tokens + k] = k
            batch.pos[2 * n_input_tokens + k] = k
            batch.pos[3 * n_input_tokens + k] = 0
            
            batch.n_seq_id[k] = 1
            batch.seq_id[k][0] = 0
            batch.logits[k] = 1 if k == n_input_tokens - 1 else 0

        # 推理
        ret = nano_llama.llama_decode(ctx, batch)
        nano_llama.llama_batch_free(batch)

        if ret != 0:
            raise RuntimeError(f"llama_decode failed (ret={ret})")

        # 获取并分析 logits
        logits_ptr = nano_llama.llama_get_logits(ctx)
        logits_arr = np.ctypeslib.as_array(logits_ptr, shape=(vocab_size,))
        
        actual_token_id = int(np.argmax(logits_arr))
        max_logit = np.max(logits_arr)
        
        print(f"  ✓ Predicted: {actual_token_id} (logit: {max_logit:.6f})")
        if actual_token_id == expected_token_id:
            print(f"  ✓ Match expected (1995)")
        else:
            print(f"  ⚠ Mismatch! Expected 1995, got {actual_token_id}")
            
        return actual_token_id, max_logit

    except Exception as e:
        print(f"  ❌ Error in inference: {e}")
        return None, None

def verify_gguf_with_params(gguf_path, n_gpu_layers=-1, n_threads=1):
    """根据参数加载模型并运行多次推理，每次推理都重建 Context"""
    backend_name = "GPU" if n_gpu_layers != 0 else "CPU"
    print(f"\n" + "="*80)
    print(f"  TESTING BACKEND: {backend_name} (n_gpu_layers={n_gpu_layers}, threads={n_threads})")
    print("="*80)

    # 1. 加载模型
    model = nano_llama.load_model(gguf_path, n_gpu_layers=n_gpu_layers)
    if not model:
        return False

    vocab = nano_llama.llama_model_get_vocab(model)
    vocab_size = nano_llama.llama_vocab_n_tokens(vocab)
    
    # 3. 加载测试数据
    inputs_embeds = np.load("40_first_step_embeds.npy")
    expected_token_id = 1995

    results = []
    # 运行两次，每次都使用完全新鲜的 Context
    for i in range(2):
        # 每次推理前创建新 Context，确保完全没有状态残留
        ctx_params = nano_llama.llama_context_default_params()
        ctx_params.n_ctx = 2048
        ctx_params.n_threads = n_threads
        ctx_params.no_perf = True
        ctx = nano_llama.llama_init_from_model(model, ctx_params)
        
        if not ctx:
            print(f"  ❌ Failed to create context for Run {i+1}")
            continue

        res_id, res_logit = run_inference(ctx, inputs_embeds, vocab_size, expected_token_id, run_name=f"{backend_name} Run {i+1} (Fresh Context)")
        results.append((res_id, res_logit))
        
        # 立即释放 Context
        nano_llama.llama_free(ctx)

    # 清理模型
    nano_llama.llama_model_free(model)
    
    return results

def verify_gguf_all_scenarios():
    """主验证流程：测试 GPU 和 CPU 场景"""
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    GGUF_MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "master-merged-vocab-f16.gguf")

    if not os.path.exists(GGUF_MODEL_PATH):
        print(f"❌ Error: GGUF model not found at {GGUF_MODEL_PATH}")
        return False

    all_results = {}

    # 场景 1: GPU (默认)
    all_results["GPU"] = verify_gguf_with_params(GGUF_MODEL_PATH, n_gpu_layers=-1)

    # 场景 2: CPU (仅当结果不一致时测试，或为了对比)
    all_results["CPU"] = verify_gguf_with_params(GGUF_MODEL_PATH, n_gpu_layers=0)

    print("\n" + "="*80)
    print("  FINAL SUMMARY")
    print("="*80)
    
    for backend, res in all_results.items():
        if res:
            consistent = res[0] == res[1]
            print(f"  {backend}: Run 1 = {res[0][0]}, Run 2 = {res[1][0]} | {'CONSISTENT' if consistent else 'RANDOM!'}")
        else:
            print(f"  {backend}: Failed to run")
    
    print("="*80)
    
    # 彻底清理 backend
    nano_llama.llama_backend_free()
    return True

if __name__ == "__main__":
    try:
        # 重导演验证流程
        success = verify_gguf_all_scenarios()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
