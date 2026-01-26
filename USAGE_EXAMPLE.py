"""
独立大师模型的使用示例

展示如何使用提取出来的独立大师模型
"""
import torch
from bare_master.configuration import Qwen3TTSTalkerConfig
from bare_master.modeling import Qwen3TTSTalkerModel

# ============================================================
# 方法 1: 使用 from_pretrained (推荐)
# ============================================================
print("Loading master model using from_pretrained()...")

model = Qwen3TTSTalkerModel.from_pretrained(
    "Standalone-Bare-Master",  # 模型路径
    dtype=torch.bfloat16,       # 数据类型
    device_map="cuda"           # 设备映射
)

model.eval()
print("Model loaded successfully!")


# ============================================================
# 方法 2: 手动加载配置和模型
# ============================================================
# config = Qwen3TTSTalkerConfig.from_pretrained("Standalone-Bare-Master")
# model = Qwen3TTSTalkerModel(config).to("cuda").to(torch.bfloat16)
#
# from safetensors import safe_open
# with safe_open("Standalone-Bare-Master/model.safetensors", framework="pt", device="cuda") as f:
#     state_dict = {key: f.get_tensor(key) for key in f.keys()}
# model.load_state_dict(state_dict)


# ============================================================
# 推理示例
# ============================================================
import torch.nn.functional as F

# 准备输入 (这里使用 inputs_embeds 作为示例)
# 在实际使用中，你需要先准备 token embeddings
inputs_embeds = torch.randn(1, 10, 2048).to("cuda").to(torch.bfloat16)
attention_mask = torch.ones(1, 10, device="cuda")

with torch.no_grad():
    # 前向传播
    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask
    )

    last_hidden_state = outputs.last_hidden_state  # [1, 10, 2048]
    print(f"Output shape: {last_hidden_state.shape}")

    # 如果需要预测下一个 codec token
    next_hidden = last_hidden_state[:, -1, :]  # 取最后一个 token

    # 加载 codec head 权重
    from safetensors import safe_open
    with safe_open("Standalone-Bare-Master/codec_head.safetensors", framework="pt", device="cuda") as f:
        codec_head_weight = f.get_tensor("weight")

    # 应用 codec head
    logits = torch.matmul(next_hidden.float(), codec_head_weight.float().T)
    predicted_token_id = torch.argmax(logits, dim=-1).item()

    print(f"Predicted token ID: {predicted_token_id}")


# ============================================================
# 保存修改后的模型
# ============================================================
# model.save_pretrained("Path/To/Save")
# 这会保存 config.json 和 model.safetensors


# ============================================================
# 模型信息
# ============================================================
print("\n" + "="*50)
print("Master Model Information")
print("="*50)
print(f"Architecture: Qwen3TTSTalkerModel (LLM Backbone)")
print(f"Hidden Size: {model.config.hidden_size}")
print(f"Num Layers: {model.config.num_hidden_layers}")
print(f"Num Attention Heads: {model.config.num_attention_heads}")
print(f"Vocab Size (codec): {model.config.vocab_size}")
print(f"Text Vocab Size: {model.config.text_vocab_size}")
print("="*50)
