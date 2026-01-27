import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import ctypes
import numpy as np
import time
import qwen3_tts_gguf.nano_llama as nano_llama

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
GGUF_PATH = os.path.join(MODEL_DIR, "qwen3_tts_talker.gguf")
MASTER_HEAD_PATH = os.path.join(MODEL_DIR, "codec_head_weight.npy")

def benchmark_master_optimized():
    print("\n=== Optimized Master Benchmark (GGUF + GPU + C-Logits) ===")
    
    # 1. Load Model with GPU Offloading (-1 means use GPU for all layers)
    print("Loading model with n_gpu_layers=-1 (Vulkan)...")
    gguf_model = nano_llama.load_model(GGUF_PATH, n_gpu_layers=-1)
    if not gguf_model:
        print("❌ Load Model Failed")
        return

    ctx_params = nano_llama.llama_context_default_params()
    ctx_params.n_ctx = 1024
    ctx_params.embeddings = True
    gguf_ctx = nano_llama.llama_init_from_model(gguf_model, ctx_params)
    n_embd = nano_llama.llama_model_n_embd(gguf_model)
    v_size = 2152 # Custom Voice Vocab Size
    
    # 2. Setup Benchmark
    n_steps = 100
    batch = nano_llama.llama_batch_init(512, n_embd, 1)
    
    # Dummy Prefill
    dummy_embed = np.random.randn(1, n_embd).astype(np.float32)
    batch.n_tokens = 1
    batch.pos[0] = 0
    batch.n_seq_id[0] = 1
    batch.seq_id[0][0] = 0
    batch.logits[0] = 1 # Request logits
    ctypes.memmove(batch.embd, np.ascontiguousarray(dummy_embed).ctypes.data, dummy_embed.nbytes)
    
    nano_llama.llama_decode(gguf_ctx, batch)
    
    # Check if we can get logits from C instead of manual NumPy
    logits_ptr = nano_llama.llama_get_logits(gguf_ctx)
    if logits_ptr:
        print("✅ C-level Logits available in GGUF!")
    else:
        print("⚠️ C-level Logits NOT available. Will use manual projection.")

    codec_table = np.load(os.path.join(MODEL_DIR, "codec_embedding_0.npy"))
    master_head_weight = np.load(MASTER_HEAD_PATH)

    # 3. Test A: Manual CPU Projection (Baseline with GPU Backbone)
    print(f"\n[Test A] Manual CPU Projection (GPU Backbone, NumPy Head)")
    start_time = time.time()
    current_pos = 1
    for i in range(n_steps):
        # Decode (GPU)
        # Note: llama_get_embeddings gets output of transformer, not logits head.
        out_ptr = nano_llama.llama_get_embeddings(gguf_ctx)
        current_hidden = np.ctypeslib.as_array(out_ptr, shape=(1, n_embd))
        
        # Logits (CPU NumPy)
        logits = current_hidden @ master_head_weight.T
        token_id = np.argmax(logits)
        
        # Next
        next_embed = codec_table[token_id % 2048].reshape(1, n_embd)
        batch.pos[0] = current_pos
        current_pos += 1
        ctypes.memmove(batch.embd, np.ascontiguousarray(next_embed).ctypes.data, next_embed.nbytes)
        nano_llama.llama_decode(gguf_ctx, batch)

    duration = time.time() - start_time
    print(f"  Speed: {n_steps/duration:.2f} t/s")

    # 4. Test B: C-level Logits Projection (If available)
    if logits_ptr:
        print(f"\n[Test B] C-level Logits Projection (GPU Backbone + GPU/C Head)")
        start_time = time.time()
        current_pos = 1 + n_steps # Offset from Test A
        for i in range(n_steps):
            # Logits (Direct from C)
            logits_ptr = nano_llama.llama_get_logits(gguf_ctx)
            # Vocab size for Qwen3-TTS Master is 2152
            logits_arr = np.ctypeslib.as_array(logits_ptr, shape=(v_size,))
            token_id = np.argmax(logits_arr)
            
            # Next
            next_embed = codec_table[token_id % 2048].reshape(1, n_embd)
            batch.pos[0] = current_pos
            current_pos += 1
            ctypes.memmove(batch.embd, np.ascontiguousarray(next_embed).ctypes.data, next_embed.nbytes)
            nano_llama.llama_decode(gguf_ctx, batch)
            
        duration = time.time() - start_time
        print(f"  Speed: {n_steps/duration:.2f} t/s")

    nano_llama.llama_batch_free(batch)
    nano_llama.llama_free(gguf_ctx)
    nano_llama.llama_model_free(gguf_model)

if __name__ == "__main__":
    benchmark_master_optimized()
