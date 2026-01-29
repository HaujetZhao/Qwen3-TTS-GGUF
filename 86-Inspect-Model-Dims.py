"""
86-Inspect-Model-Dims.py
检查模型的实际维度配置，用于确定正确的 head_dim。
"""
import os
from qwen3_tts_gguf.tokenizer_12hz.modeling_tokenizer import Qwen3TTSTokenizerV2Model

MODEL_PATH = "./Qwen3-TTS-12Hz-1.7B-CustomVoice"
tokenizer_path = os.path.join(MODEL_PATH, "speech_tokenizer") if os.path.exists(os.path.join(MODEL_PATH, "speech_tokenizer")) else MODEL_PATH

print("🔍 正在加载模型并检查维度...")
model = Qwen3TTSTokenizerV2Model.from_pretrained(tokenizer_path)
cfg = model.decoder.config

print(f"\n📊 配置信息:")
print(f"   hidden_size       = {cfg.hidden_size}")
print(f"   num_attention_heads = {cfg.num_attention_heads}")
print(f"   num_key_value_heads = {getattr(cfg, 'num_key_value_heads', 'N/A')}")
print(f"   head_dim (config) = {getattr(cfg, 'head_dim', 'N/A')}")

print(f"\n📐 权重维度 (第0层 Attention):")
attn = model.decoder.pre_transformer.layers[0].self_attn
print(f"   q_proj.weight.shape = {attn.q_proj.weight.shape}")
print(f"   k_proj.weight.shape = {attn.k_proj.weight.shape}")
print(f"   v_proj.weight.shape = {attn.v_proj.weight.shape}")
print(f"   o_proj.weight.shape = {attn.o_proj.weight.shape}")

# 逆推 head_dim
# k_proj 的输出维度 = num_kv_heads * head_dim
k_out_dim = attn.k_proj.weight.shape[0]
num_kv_heads = getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads)
inferred_head_dim = k_out_dim // num_kv_heads

print(f"\n🧮 逆推结果:")
print(f"   k_proj 输出维度 = {k_out_dim}")
print(f"   num_kv_heads   = {num_kv_heads}")
print(f"   推断 head_dim  = {k_out_dim} / {num_kv_heads} = {inferred_head_dim}")
