import os
import sys
import numpy as np

# 确保能导入 qwen3_tts_gguf
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from qwen3_tts_gguf import GGUFEngine

def main():
    GGUF_PATH = os.path.abspath("model/Qwen3-Talker-F16.gguf")
    
    if not os.path.exists(GGUF_PATH):
        print(f"❌ Error: GGUF model not found at {GGUF_PATH}")
        return

    print("--- Starting Injection Test (GGUF DLL) ---")
    
    # 1. 初始化引擎
    # 我们先用 CPU 运行 (n_gpu_layers=0)
    engine = GGUFEngine(GGUF_PATH, n_gpu_layers=0)
    
    print(f"Initializing GGUF engine...")
    if not engine.initialize():
        print("❌ Error: Engine initialization failed.")
        return

    try:
        # 2. 准备随机数据
        # 模拟 14 个 tokens, 2048 维度
        test_seq_len = 14
        hidden_size = 2048
        print(f"Generating random embeddings: ({test_seq_len}, {hidden_size})")
        random_embeds = np.random.randn(test_seq_len, hidden_size).astype(np.float32)
        
        # 3. 运行推理 (注入)
        print("Feeding embeddings to GGUF...")
        logits = engine.get_logits_from_embeddings(random_embeds)
        
        # 4. 验证输出
        print(f"Success! Logits received.")
        print(f"Logits shape: {logits.shape}")
        
        # 检查 argmax 是否落在我们的 codec 范围内 (0-3072)
        # 虽然是随机数据，不太可能落到 3072 之后（因为那是 Padding 的 0）
        final_token = np.argmax(logits)
        print(f"Argmax token ID: {final_token}")
        
        if logits.shape[0] == 151936:
            print("✅ GGUF output shape is correct (padded vocab size).")
        else:
            print(f"⚠️ Warning: Unexpected logits shape: {logits.shape[0]}")

    except Exception as e:
        print(f"❌ Execution failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 5. 清理
        print("Cleaning up engine...")
        engine.cleanup()
    
    print("\n--- Test Finished ---")

if __name__ == "__main__":
    main()
