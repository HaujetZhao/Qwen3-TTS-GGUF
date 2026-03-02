
import os
import torch
import numpy as np
from pathlib import Path
from safetensors.torch import load_file

from export_config import MODEL_DIR, EXPORT_DIR

# Configuration
MODEL_PATH = Path(MODEL_DIR)
OUTPUT_DIR = Path(EXPORT_DIR) / 'embeddings'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def export_embeddings():
    model_file = MODEL_PATH / "model.safetensors"
    print(f"[1/5] 正在从 {model_file} 加载权重 (safetensors 模式)...")
    
    if not model_file.exists():
        print(f"❌ 未找到权重文件: {model_file}")
        return

    weights = load_file(str(model_file))
    
    print("[2/5] 正在导出文本嵌入层 (Text Embeddings, 已投影)...")
    with torch.no_grad():
        # Text embedding
        raw_text_embed = weights["talker.model.text_embedding.weight"]
        print(f"    原始文本嵌入形状: {raw_text_embed.shape}")
        
        # Text Projection (MLP: fc1 -> silu -> fc2)
        print("    正在计算文本嵌入层投影 (可能需要一点时间)...")
        w1 = weights["talker.text_projection.linear_fc1.weight"]
        b1 = weights["talker.text_projection.linear_fc1.bias"]
        w2 = weights["talker.text_projection.linear_fc2.weight"]
        b2 = weights["talker.text_projection.linear_fc2.bias"]
        
        # 计算逻辑：y = fc2(silu(fc1(x)))
        # 为了省内村，分块处理或者是直接计算
        h = torch.nn.functional.linear(raw_text_embed, w1, b1)
        h = torch.nn.functional.silu(h)
        projected_text_embed = torch.nn.functional.linear(h, w2, b2)
        
        print(f"    投影后的文本嵌入形状: {projected_text_embed.shape}")
        
        # 保存为 fp16 节省空间
        save_path = OUTPUT_DIR / "text_embedding_projected.npy"
        np.save(save_path, projected_text_embed.float().numpy().astype(np.float16))
        print(f"    已保存至: {save_path}")

    print("[3/5] 正在导出 Codec 0 嵌入层 (Talker 表 0)...")
    with torch.no_grad():
        codec_0_embed = weights["talker.model.codec_embedding.weight"]
        print(f"    Codec 0 嵌入层形状: {codec_0_embed.shape}")
        np.save(OUTPUT_DIR / "codec_embedding_0.npy", codec_0_embed.float().numpy())
        print(f"    已保存至: {OUTPUT_DIR / 'codec_embedding_0.npy'}")

    print("[4/5] 正在导出 Codec 1-15 嵌入层 (Code Predictor 表)...")
    with torch.no_grad():
        codec_layers_found = 0
        for i in range(32): # 探测上限
            key = f"talker.code_predictor.model.codec_embedding.{i}.weight"
            if key in weights:
                layer_idx = i + 1
                embed_weight = weights[key]
                print(f"    正在导出 Codec {layer_idx} 表 (形状: {embed_weight.shape})...")
                np.save(OUTPUT_DIR / f"codec_embedding_{layer_idx}.npy", embed_weight.float().numpy())
                codec_layers_found += 1
            else:
                if codec_layers_found > 0: # 找到一些了，说明后面没连续的了
                    break
        
        print(f"    已将所有 {codec_layers_found} 个表保存至 {OUTPUT_DIR}")

    print("[5/5] 验证导出完成。")
    print("成功：所有表均已导出。如有偏差请对比原始脚本。")

if __name__ == "__main__":
    export_embeddings()
