
import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add the source directory to sys.path
# Assuming we are running from C:\Users\Haujet\Desktop\qwen3-tts
# and the source code is in C:\Users\Haujet\Desktop\qwen3-tts\Qwen3-TTS
PROJECT_ROOT = Path(__file__).parent
SOURCE_DIR = PROJECT_ROOT / "Qwen3-TTS"
sys.path.append(str(SOURCE_DIR))

try:
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
except ImportError:
    print(f"Error: Could not import qwen_tts from {SOURCE_DIR}")
    print("Please make sure the Qwen3-TTS repository is cloned correctly.")
    sys.exit(1)

# Configuration
MODEL_PATH = PROJECT_ROOT / "Qwen3-TTS-12Hz-1.7B-CustomVoice"
OUTPUT_DIR = PROJECT_ROOT / "model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def export_embeddings():
    print(f"[1/5] Loading model from {MODEL_PATH}...")
    try:
        model = Qwen3TTSForConditionalGeneration.from_pretrained(
            MODEL_PATH, 
            torch_dtype=torch.float32, # Use float32 for clean export
            device_map="cpu"             # Export on CPU to avoid OOM
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("[2/5] Exporting Text Embeddings (Projected)...")
    # Text embeddings need to be projected to hidden_size
    # Location: model.talker.get_text_embeddings() -> [151936, text_hidden_size]
    # Projection: model.talker.text_projection -> MLP -> [151936, hidden_size]
    with torch.no_grad():
        raw_text_embed = model.talker.get_text_embeddings().weight
        print(f"    Raw Text Embed Shape: {raw_text_embed.shape}")
        
        # We project them in chunks to save memory if needed, but 150k is okay on CPU
        print("    Projecting text embeddings (this may take a moment)...")
        projected_text_embed = model.talker.text_projection(raw_text_embed)
        print(f"    Projected Text Embed Shape: {projected_text_embed.shape}")
        
        np.save(OUTPUT_DIR / "text_embedding_projected.npy", projected_text_embed.numpy())
        print(f"    Saved to {OUTPUT_DIR / 'text_embedding_projected.npy'}")

    print("[3/5] Exporting Codec 0 Embedding (Talker Table 0)...")
    # Location: model.talker.get_input_embeddings()
    # This table contains Code 0 tokens, Special Tokens, and Speaker IDs
    with torch.no_grad():
        codec_0_embed = model.talker.get_input_embeddings().weight
        print(f"    Codec 0 Embed Shape: {codec_0_embed.shape}")
        np.save(OUTPUT_DIR / "codec_embedding_0.npy", codec_0_embed.numpy())
        print(f"    Saved to {OUTPUT_DIR / 'codec_embedding_0.npy'}")

    print("[4/5] Exporting Codec 1-15 Embeddings (Code Predictor Tables)...")
    # Location: model.talker.code_predictor.get_input_embeddings() (ModuleList)
    with torch.no_grad():
        # Inspect code_predictor.codec_embedding
        codec_layers = model.talker.code_predictor.get_input_embeddings()
        print(f"    Found {len(codec_layers)} layers in Code Predictor.")
        
        for i, layer in enumerate(codec_layers):
            layer_idx = i + 1
            embed_weight = layer.weight
            print(f"    Exporting Codec {layer_idx} Table (Shape: {embed_weight.shape})...")
            np.save(OUTPUT_DIR / f"codec_embedding_{layer_idx}.npy", embed_weight.numpy())
            
        print(f"    Saved all {len(codec_layers)} tables to {OUTPUT_DIR}")

    print("[5/5] Verifying Exports...")
    verify_exports(model)

def verify_exports(model):
    print("    Verifying Text Embedding Lookup...")
    saved_text = np.load(OUTPUT_DIR / "text_embedding_projected.npy")
    # Check ID 100
    id_to_check = 100
    with torch.no_grad():
        model_out = model.talker.text_projection(model.talker.get_text_embeddings()(torch.tensor([id_to_check])))
    
    npy_out = saved_text[id_to_check]
    
    if np.allclose(model_out.numpy()[0], npy_out, atol=1e-5):
        print("    [PASS] Text Embedding matches.")
    else:
        print("    [FAIL] Text Embedding mismatch!")
        diff = np.abs(model_out.numpy()[0] - npy_out).max()
        print(f"    Max Diff: {diff}")

    print("    Verifying Codec 0 Lookup...")
    saved_c0 = np.load(OUTPUT_DIR / "codec_embedding_0.npy")
    id_to_check = 500
    with torch.no_grad():
        model_out = model.talker.get_input_embeddings()(torch.tensor([id_to_check]))
        
    npy_out = saved_c0[id_to_check]
    if np.allclose(model_out.numpy()[0], npy_out, atol=1e-5):
        print("    [PASS] Codec 0 Embedding matches.")
    else:
        print("    [FAIL] Codec 0 Embedding mismatch!")

    print("SUCCESS: All tables exported and verified.")

if __name__ == "__main__":
    export_embeddings()
