"""
[84] Verify GGUF Loading & Numerical Correctness (80 Series)
仿照 94-Verify-GGUF-with-llama-cpp.py，验证 3072 词表的 GGUF 模型。

验证内容：
1. 加载 master-codec-only-3072-f16.gguf
2. 注入 40_first_step_embeds.npy (模拟第一步推理)
3. 检查 Logits 输出形状是否为 3072
4. 验证预测结果 (Expected: 1995)
"""
import os
import sys
import numpy as np
import ctypes
import qwen3_tts_gguf.nano_llama as nano_llama

# os.environ["VK_ICD_FILENAMES"] = "none" # 禁用 Vulkan 以保证数值一致性测试稳健
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))



def run_inference(ctx, inputs_embeds, vocab_size, expected_token_id):
    """执行单次推理"""
    print(f"\n--- 正在运行推理 ---")
    
    # 1. 准备 Batch Setup (仿照 94 的 IMRoPE 兼容逻辑)
    # 输入 shape: [1, n_tokens, n_embd]
    n_input_tokens = inputs_embeds.shape[1]
    dim = inputs_embeds.shape[2]
    
    full_embd = inputs_embeds[0].astype(np.float32)
    if not full_embd.flags['C_CONTIGUOUS']:
        full_embd = np.ascontiguousarray(full_embd)

    # 分配 4倍 token 容量以容纳 M-RoPE 的多重 pos
    batch = nano_llama.llama_batch_init(n_input_tokens * 4, dim, 1)
    batch.n_tokens = n_input_tokens # 逻辑上包含 n_input_tokens 个 token
    
    # 注入 embedding 数据
    batch.token = ctypes.cast(None, ctypes.POINTER(nano_llama.llama_token)) # 指明这是纯 embedding 输入
    ctypes.memmove(batch.embd, full_embd.ctypes.data, full_embd.nbytes)
    
    # 设置 Pos (IMRoPE 模式: 3个 pos + 1个 padding)
    for k in range(n_input_tokens):
        batch.pos[k] = k
        batch.pos[n_input_tokens + k] = k
        batch.pos[2 * n_input_tokens + k] = k
        batch.pos[3 * n_input_tokens + k] = 0
        
        batch.n_seq_id[k] = 1
        batch.seq_id[k][0] = 0
        # 只计算最后一个 token 的 logits
        batch.logits[k] = 1 if k == n_input_tokens - 1 else 0

    # 2. 推理
    ret = nano_llama.llama_decode(ctx, batch)
    nano_llama.llama_batch_free(batch)

    if ret != 0:
        print(f"❌ llama_decode 失败，错误代码 {ret}")
        return False

    # 3. 获取 Logits
    logits_ptr = nano_llama.llama_get_logits(ctx)
    # 转换为 numpy 数组
    logits_arr = np.ctypeslib.as_array(logits_ptr, shape=(vocab_size,))
    
    max_logit_idx = int(np.argmax(logits_arr))
    max_logit_val = np.max(logits_arr)
    
    print(f"  词表大小: {vocab_size}")
    print(f"  生成的 Token ID: {max_logit_idx}")
    print(f"  期望的 Token ID: {expected_token_id}")
    print(f"  Max Logit 值:    {max_logit_val:.4f}")
    
    if max_logit_idx == expected_token_id:
        print("  ✅ 结果匹配成功")
    else:
        print(f"  ⚠️ 结果不匹配，期望 {expected_token_id}，实际得到 {max_logit_idx}")
    
    # 额外的数值检查
    if vocab_size != 3072:
        print(f"  ⚠️ 实际词表大小为 {vocab_size}，期望大小为 3072")
    
    return True

def main():
    # 定义文件名常量
    FILE_GGUF   = os.path.join("model", "qwen3_tts_talker.gguf")
    FILE_EMBDS  = "20-saved-input-embds.npy"
    FILE_LOGITS = "20-saved-input-logits.npy"

    path_gguf   = os.path.join(PROJECT_ROOT, FILE_GGUF)
    path_embds  = os.path.join(PROJECT_ROOT, FILE_EMBDS)
    path_logits = os.path.join(PROJECT_ROOT, FILE_LOGITS)

    if not os.path.exists(path_gguf):
        print(f"❌ 未找到模型文件: {path_gguf}")
        return
    
    if not os.path.exists(path_embds):
        print(f"❌ 未找到输入嵌入文件: {path_embds}")
        return

    print(f"正在加载模型: {path_gguf}")
    # 加载模型 (CPU mode for verification usually suffices and is deterministic)
    model = nano_llama.load_model(path_gguf, n_gpu_layers=0)
    if not model:
        print("❌ 模型加载失败")
        return

    vocab = nano_llama.llama_model_get_vocab(model)
    vocab_size = nano_llama.llama_vocab_n_tokens(vocab)
    print(f"模型已成功加载，词表大小: {vocab_size}")

    # 创建 Context
    ctx_params = nano_llama.llama_context_default_params()
    ctx_params.n_ctx = 2048
    ctx_params.n_threads = 4
    ctx = nano_llama.llama_init_from_model(model, ctx_params)
    
    if not ctx:
        print("❌ 创建推理上下文失败")
        nano_llama.llama_model_free(model)
        return

    # 加载输入
    embeds = np.load(path_embds)
    print(f"成功加载输入嵌入: {path_embds}, 形状: {embeds.shape}")

    # 动态加载官方 Logits 以获取期望结果
    if os.path.exists(path_logits):
        official_logits = np.load(path_logits)
        # 假设 logits shape 是 (1, 3072) 或者 (1, 14, 3072)
        # 前面的脚本输出显示 shape 是 (1, 3072)
        if len(official_logits.shape) == 2:
             # (1, 3072)
             expected_logit_arr = official_logits[0]
        elif len(official_logits.shape) == 3:
             # (1, 14, 3072) -> 取最后一个位置
             expected_logit_arr = official_logits[0, -1, :]
        else:
             expected_logit_arr = official_logits.flatten()

        expected_token_id = int(np.argmax(expected_logit_arr))
        expected_val = float(np.max(expected_logit_arr))
        print(f"基于官方数据计算的期望结果: Token ID {expected_token_id} (官方 Logit 值: {expected_val:.4f})")
    else:
        print(f"⚠️ 未找到官方 Logits 文件 {path_logits}，使用默认期望值 1995")
        expected_token_id = 1995

    # 运行验证
    run_inference(ctx, embeds, vocab_size, expected_token_id=expected_token_id)

    # 清理
    nano_llama.llama_free(ctx)
    nano_llama.llama_model_free(model)
    print("\n验证完成。")

if __name__ == "__main__":
    main()
