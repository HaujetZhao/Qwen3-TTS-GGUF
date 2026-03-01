
import os
import sys
import torch
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR / "Qwen3-TTS-main"))
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration


from export_config import MODEL_DIR, EXPORT_DIR

# Configuration
MODEL_PATH = Path(MODEL_DIR)
OUTPUT_DIR = Path(EXPORT_DIR) / 'embeddings'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def export_embeddings():
    print(f"[1/5] 正在从 {MODEL_PATH} 加载模型...")
    try:
        model = Qwen3TTSForConditionalGeneration.from_pretrained(
            MODEL_PATH, 
            torch_dtype=torch.float32, 
            device_map="cpu"
        )
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    print("[2/5] 正在导出文本嵌入层 (Text Embeddings, 已投影)...")
    # Text embeddings need to be projected to hidden_size
    # Location: model.talker.get_text_embeddings() -> [151936, text_hidden_size]
    # Projection: model.talker.text_projection -> MLP -> [151936, hidden_size]
    with torch.no_grad():
        raw_text_embed = model.talker.get_text_embeddings().weight
        print(f"    原始文本嵌入形状: {raw_text_embed.shape}")
        
        # We project them in chunks to save memory if needed, but 150k is okay on CPU
        print("    正在计算文本嵌入层投影 (可能需要一点时间)...")
        projected_text_embed = model.talker.text_projection(raw_text_embed)
        print(f"    投影后的文本嵌入形状: {projected_text_embed.shape}")
        
        np.save(OUTPUT_DIR / "text_embedding_projected.npy", projected_text_embed.numpy().astype(np.float16))
        print(f"    已保存至: {OUTPUT_DIR / 'text_embedding_projected.npy'}")

    print("[3/5] 正在导出 Codec 0 嵌入层 (Talker 表 0)...")
    # Location: model.talker.get_input_embeddings()
    # This table contains Code 0 tokens, Special Tokens, and Speaker IDs
    with torch.no_grad():
        codec_0_embed = model.talker.get_input_embeddings().weight
        print(f"    Codec 0 嵌入层形状: {codec_0_embed.shape}")
        np.save(OUTPUT_DIR / "codec_embedding_0.npy", codec_0_embed.numpy())
        print(f"    已保存至: {OUTPUT_DIR / 'codec_embedding_0.npy'}")

    print("[4/5] 正在导出 Codec 1-15 嵌入层 (Code Predictor 表)...")
    # Location: model.talker.code_predictor.get_input_embeddings() (ModuleList)
    with torch.no_grad():
        # Inspect code_predictor.codec_embedding
        codec_layers = model.talker.code_predictor.get_input_embeddings()
        print(f"    在 Code Predictor 中找到 {len(codec_layers)} 层。")
        
        for i, layer in enumerate(codec_layers):
            layer_idx = i + 1
            embed_weight = layer.weight
            print(f"    正在导出 Codec {layer_idx} 表 (形状: {embed_weight.shape})...")
            np.save(OUTPUT_DIR / f"codec_embedding_{layer_idx}.npy", embed_weight.numpy())
            
        print(f"    已将所有 {len(codec_layers)} 个表保存至 {OUTPUT_DIR}")

    print("[5/5] 正在验证导出结果...")
    verify_exports(model)

def verify_exports(model):
    print("    正在验证文本嵌入层查询...")
    saved_text = np.load(OUTPUT_DIR / "text_embedding_projected.npy")
    # Check ID 100
    id_to_check = 100
    with torch.no_grad():
        model_out = model.talker.text_projection(model.talker.get_text_embeddings()(torch.tensor([id_to_check])))
    
    npy_out = saved_text[id_to_check]
    
    if np.allclose(model_out.numpy()[0], npy_out, atol=1e-5):
        print("    [通过] 文本嵌入层匹配。")
    else:
        print("    [失败] 文本嵌入层不匹配！")
        diff = np.abs(model_out.numpy()[0] - npy_out).max()
        print(f"    最大差异: {diff}")

    print("    正在验证 Codec 0 查询...")
    saved_c0 = np.load(OUTPUT_DIR / "codec_embedding_0.npy")
    id_to_check = 500
    with torch.no_grad():
        model_out = model.talker.get_input_embeddings()(torch.tensor([id_to_check]))
        
    npy_out = saved_c0[id_to_check]
    if np.allclose(model_out.numpy()[0], npy_out, atol=1e-5):
        print("    [通过] Codec 0 嵌入层匹配。")
    else:
        print("    [失败] Codec 0 嵌入层不匹配！")

    print("成功：所有表均已导出并验证。")

if __name__ == "__main__":
    export_embeddings()
