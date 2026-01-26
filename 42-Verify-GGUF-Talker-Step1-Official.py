import os
import sys
import numpy as np

# 确保能导入 qwen3_tts_gguf
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from qwen3_tts_gguf import GGUFEngine

def main():
    GGUF_PATH = os.path.abspath("model/Qwen3-Talker-F16.gguf")
    OFFICIAL_EMBEDS_PATH = os.path.abspath("40_first_step_embeds.npy")
    OFFICIAL_LOGITS_PATH = os.path.abspath("40_first_step_logits.npy")
    
    if not os.path.exists(GGUF_PATH):
        print(f"❌ Error: GGUF model not found at {GGUF_PATH}")
        return
    if not os.path.exists(OFFICIAL_EMBEDS_PATH):
        print(f"❌ Error: Official embeddings not found at {OFFICIAL_EMBEDS_PATH}")
        return

    print("--- Brain Collision Test (Step 1: Official Data) ---")
    
    # 1. 加载官方数据
    print(f"Loading official embeddings from {OFFICIAL_EMBEDS_PATH}...")
    official_embeds = np.load(OFFICIAL_EMBEDS_PATH)
    # Shape is (1, 14, 2048), engine expects (seq_len, hidden_size)
    if official_embeds.ndim == 3:
        official_embeds = official_embeds[0]
    
    # 2. 初始化引擎
    engine = GGUFEngine(GGUF_PATH, n_gpu_layers=0)
    print("Initializing GGUF engine...")
    if not engine.initialize():
        print("❌ Error: Engine initialization failed.")
        return

    try:
        # 3. 运行推理 (注入官方 Embeddings)
        print(f"Injecting {official_embeds.shape[0]} official tokens into GGUF...")
        gguf_logits_full = engine.get_logits_from_embeddings(official_embeds)
        
        # 4. 获取结果
        # 只取前 3072 位 Codec 区域
        gguf_logits_codec = gguf_logits_full[:3072]
        gguf_code0 = np.argmax(gguf_logits_codec)
        
        print("\n--- GGUF Inference Result ---")
        print(f"Argmax Code 0: {gguf_code0}")
        
        # 5. 对比官方 Logits
        if os.path.exists(OFFICIAL_LOGITS_PATH):
            off_logits = np.load(OFFICIAL_LOGITS_PATH).flatten()
            print(f"Official Code 0: {np.argmax(off_logits)}")
            
            diff = np.abs(off_logits - gguf_logits_codec)
            print(f"Max Absolute Difference (First 3072): {np.max(diff):.8f}")
            print(f"Mean Absolute Difference: {np.mean(diff):.8f}")
            
            if gguf_code0 == np.argmax(off_logits):
                print("\n✅ MATCH: GGUF reproduced the exact same first code!")
            else:
                print("\n❌ MISMATCH: GGUF produced a different code.")
        else:
            print("⚠️ Warning: Official logits not found for comparison.")

    except Exception as e:
        print(f"❌ Execution failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Cleaning up engine...")
        engine.cleanup()
    
    print("\n--- Test Finished ---")

if __name__ == "__main__":
    main()
