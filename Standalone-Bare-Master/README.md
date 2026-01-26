# 独立大师模型 (Standalone Master Model)

这是从 Qwen3-TSS 原始模型中提取出来的**大师**（LLM Backbone）独立版本。

## 模型结构

### 原始模型组成
Qwen3-TTS 原始模型包含三个主要部分：

1. **大师 (Master/LLM Backbone)**: 28层 Transformer, 2048 隐藏层
2. **Codec Head**: 简单的线性层 (2048 -> 3072)
3. **工匠 (Craftsman/Code Predictor)**: 5层小型 Transformer, 1024 隐藏层

### 本模型包含

- ✅ **大师 LLM Backbone** (完整)
- ✅ **Codec Head** (单独文件)
- ❌ **工匠 Code Predictor** (已移除)

## 文件说明

```
Standalone-Bare-Master/
├── model.safetensors          (3.3GB) - 大师模型权重
├── codec_head.safetensors     (13MB)  - Codec Head 权重
├── config.json                        - 大师模型配置
├── metadata.json                      - 模型元数据
└── README.md                          - 本文件
```

## 快速开始

### 1. 使用标准 HuggingFace 方法加载 (推荐)

```python
from bare_master.modeling import Qwen3TTSTalkerModel

model = Qwen3TTSTalkerModel.from_pretrained(
    "Standalone-Bare-Master",
    dtype=torch.bfloat16,
    device_map="cuda"
)
```

### 2. 手动加载

```python
from bare_master.configuration import Qwen3TTSTalkerConfig
from bare_master.modeling import Qwen3TTSTalkerModel
from safetensors import safe_open

# 加载配置
config = Qwen3TTSTalkerConfig.from_pretrained("Standalone-Bare-Master")

# 创建模型
model = Qwen3TTSTalkerModel(config).to("cuda").to(torch.bfloat16)

# 加载权重
with safe_open("Standalone-Bare-Master/model.safetensors", framework="pt", device="cuda") as f:
    state_dict = {key: f.get_tensor(key) for key in f.keys()}
model.load_state_dict(state_dict)
```

## 推理示例

```python
import torch
from safetensors import safe_open

# 准备输入
inputs_embeds = torch.randn(1, 10, 2048).to("cuda").to(torch.bfloat16)
attention_mask = torch.ones(1, 10, device="cuda")

# 前向传播
with torch.no_grad():
    outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    last_hidden_state = outputs.last_hidden_state

    # 预测下一个 codec token
    next_hidden = last_hidden_state[:, -1, :]

    # 加载并应用 codec head
    with safe_open("Standalone-Bare-Master/codec_head.safetensors", framework="pt", device="cuda") as f:
        codec_head_weight = f.get_tensor("weight")

    logits = torch.matmul(next_hidden.float(), codec_head_weight.float().T)
    predicted_token_id = torch.argmax(logits, dim=-1).item()

print(f"Predicted token ID: {predicted_token_id}")
```

## 模型配置

```json
{
  "hidden_size": 2048,
  "num_hidden_layers": 28,
  "num_attention_heads": 16,
  "num_key_value_heads": 8,
  "vocab_size": 3072,
  "text_vocab_size": 151936,
  "max_position_embeddings": 32768,
  "rope_theta": 1000000,
  "use_cache": true
}
```

## 提取过程

这个模型是通过以下脚本从原始模型中提取的：

1. `85-Extract-Master-Weights.py` - 提取大师权重
2. `86-Verify-Extracted-Master.py` - 验证提取的模型
3. `87-Verify-Standlone-Loading.py` - 验证标准加载方法

验证结果：
- ✅ Token ID 预测正确 (1995/1995)
- ✅ 权重加载成功
- ✅ 标准 `from_pretrained` 方法支持

## 优势

相比原始完整模型：

1. **显存占用更低**: 不加载工匠部分 (节省约 100MB)
2. **模块化**: 可以独立使用、优化和部署
3. **灵活性**: 方便进行量化、剪枝等优化
4. **清晰度**: 专注于 LLM Backbone 的功能

## 使用场景

- 独立的音频 codec 语言模型
- 需要自定义优化和部署
- 模型研究和实验
- 嵌入式或边缘设备部署

## 依赖

```
torch
safetensors
transformers
```

## 注意事项

1. 本模型只包含大师部分，如果需要完整的 TTS 功能，还需要工匠部分
2. Codec Head 权重在单独的文件中
3. 推理时需要注意数据类型 (建议使用 bfloat16)

## 版本历史

- **2026-01-26**: 初始版本，从 Qwen3-TTS-12Hz-1.7B-CustomVoice 提取
