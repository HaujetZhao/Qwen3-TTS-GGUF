import torch
import numpy as np
import os

def find_nearest(target, table, name):
    # Flatten table if needed
    dist = torch.norm(table - target, dim=1)
    min_dist, idx = torch.min(dist, dim=0)
    return min_dist.item(), idx.item()

def main():
    # Load intercepted data
    data = torch.load("31_intercepted_embeds.pt")
    intercepted = data['inputs_embeds'][0] # [14, 2048]
    
    # Load exported tables
    # Table A: Text Projected
    text_table = torch.from_numpy(np.load("model/text_embedding_projected.npy"))
    # Table B: Codec 0 (includes tags and speaker IDs)
    codec0_table = torch.from_numpy(np.load("model/codec_embedding_0.npy"))
    
    # Special Vectors for fusion
    # From config: codec_pad_id=2148, tts_pad_token_id=151671
    # Note: text_table is already projected
    text_pad_vec = text_table[151671]
    codec_pad_vec = codec0_table[2148]
    
    print(f"Analyzing 14 tokens in sequence...\n")
    
    for i in range(14):
        vec = intercepted[i]
        
        # 1. Check if it's purely Text
        d_text, idx_text = find_nearest(vec, text_table, "Text")
        
        # 2. Check if it's purely Codec
        d_codec, idx_codec = find_nearest(vec, codec0_table, "Codec")
        
        # 3. Check for typical Fusion (Text + Codec)
        # Case A: Text_Pad + Codec_X
        # Codec_X = vec - Text_Pad
        codec_x = vec - text_pad_vec
        d_f1, idx_f1 = find_nearest(codec_x, codec0_table, "Codec-Fused")
        
        # Case B: Codec_Pad + Text_X
        # Text_X = vec - Codec_Pad
        text_x = vec - codec_pad_vec
        d_f2, idx_f2 = find_nearest(text_x, text_table, "Text-Fused")

        results = [
            (d_text, f"Text ID: {idx_text}"),
            (d_codec, f"Codec ID: {idx_codec}"),
            (d_f1, f"Fused (Text_Pad + Codec ID: {idx_f1})"),
            (d_f2, f"Fused (Codec_Pad + Text ID: {idx_f2})")
        ]
        
        # Find best match
        best_d, best_msg = min(results, key=lambda x: x[0])
        
        print(f"Token {i:2}: {best_msg:40} (Dist: {best_d:.6f})")

if __name__ == "__main__":
    main()
