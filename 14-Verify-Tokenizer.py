
import os
import sys
import numpy as np
import ctypes
from pathlib import Path

# Add fun_asr_gguf to path to use its nano_llama wrapper
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT / "fun_asr_gguf"))

try:
    import nano_llama
except ImportError:
    print("Error: Could not import nano_llama. Make sure fun_asr_gguf is in the directory.")
    sys.exit(1)

# Configuration
MODEL_DIR = PROJECT_ROOT / "model"
GGUF_PATH = MODEL_DIR / "Qwen3-LLM-1.7B-F16.gguf" # We assume the GGUF is also here or needs to be copied/pointed to.
# Wait, the GGUF is currently in the root or model dir? 
# The user said "后续我们导出的模型也是要保留在这个位置", but currently only the embeddings are there.
# The `13` script loaded the PyTorch model. We don't have the GGUF of Qwen3-TTS yet.
# BUT, we can use the original Qwen2.5-1.5B GGUF or similar if we have it, OR we rely on the tokenizer from the pytorch model path?
# 
# Actually, `fun_asr_gguf` uses `llama_model_load_from_file` which REQUIRES a GGUF file.
# If we don't have a GGUF file yet, we can't use `nano_llama` to tokenize using the GGUF vocab.
#
# HOWEVER, we can stick to using `transformers` tokenizer to verify the ID mapping against our exported numpy file.
# This confirms that "Input Text -> Tokenizer -> ID -> Numpy Lookup -> Vector" works.

from transformers import AutoTokenizer

def verify_tokenizer_and_lookup():
    print(f"[1/3] Loading Tokenizer from {PROJECT_ROOT / 'Qwen3-TTS-12Hz-1.7B-CustomVoice'}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(PROJECT_ROOT / "Qwen3-TTS-12Hz-1.7B-CustomVoice", trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        return

    print(f"[2/3] Loading Exported Text Embeddings from {MODEL_DIR}...")
    try:
        text_embed_table = np.load(MODEL_DIR / "text_embedding_projected.npy")
        print(f"    Loaded table shape: {text_embed_table.shape}")
    except Exception as e:
        print(f"Failed to load embeddings: {e}")
        return

    print("[3/3] Testing 'Hello World'...")
    test_text = "Hello World"
    # Qwen tokenizer usually adds special tokens?
    # Let's see what IDs we get
    encoded = tokenizer(test_text, return_tensors="np")
    input_ids = encoded["input_ids"][0]
    print(f"    Text: '{test_text}'")
    print(f"    Token IDs: {input_ids}")
    
    print("    Looking up vectors...")
    vectors = text_embed_table[input_ids]
    print(f"    Retrieved Vectors Shape: {vectors.shape}")
    
    if vectors.shape[0] == len(input_ids) and vectors.shape[1] == 2048:
        print("SUCCESS: Tokenizer IDs map correctly to Embedding Table dimensions.")
    else:
        print("FAIL: Dimension mismatch.")

if __name__ == "__main__":
    verify_tokenizer_and_lookup()
