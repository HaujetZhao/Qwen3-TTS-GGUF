# Qwen3-TTS Stateful Codec Decoder ONNX 导出经验总结

在将 Qwen3-TTS 的 Stateful Codec Decoder（流式音频解码器）导出为 ONNX 的过程中，我们面临了比 Code Predictor 更复杂的挑战。以下是核心经验总结：

## 1. 彻底消除 Python 分支 (if-free Forward)
**挑战**：原始模型的 `forward` 方法中大量使用 `if is_last_chunk:` 等条件分支。ONNX/JIT Trace 无法记录 Python 层面的条件跳转，会导致导出的图只包含单一分支的逻辑。

**对策**：
- **算子替代**：用 `torch.where(is_last > 0.5, A, B)` 替换所有 `if is_last:` 分支。
- **静态输出尺寸**：将 `final_wav` 的切片从 `wav[:, start:actual_end]`（数据依赖）改为 `wav[:, start:]`（输入依赖），同时返回 `valid_samples` 让客户端自行截断。这避免了 TensorRT 等引擎在编译期无法预判输出 Buffer 大小的问题。

## 2. 抛弃 DynamicCache，手写 TraceableKVStack
**挑战**：`transformers` 库的 `DynamicCache` 是一个复杂的 Python 对象，其内部状态更新无法被 JIT Trace 捕获。更糟的是，库内部的 `create_causal_mask` 等辅助函数会尝试调用 Cache 对象上不存在的方法（如 `get_mask_sizes`），导致 `AttributeError`。

**对策**：
- **自定义 TraceableKVStack**：模拟 `Cache.update()` 接口，内部使用纯 `torch.cat` + 负索引切片实现 KV 拼接与滑动窗口裁剪。
- **接管 Transformer 循环**：不再调用 `pre_transformer.forward()`，而是在 Wrapper 内部手动遍历 `for layer in self.trans.layers`，并手动计算因果掩码（Causal Mask）。这彻底切断了与 `transformers` 库复杂内部逻辑的耦合。

## 3. 隐藏的维度陷阱 (head_dim: 64 ≠ 512/16)
**挑战**：按常规公式 `hidden_size / num_heads = 512 / 16 = 32`，但模型配置中硬编码了 `head_dim = 64`。这导致 Dummy KV Cache 的维度与实际权重不匹配，Trace 阶段报 `RuntimeError: Sizes of tensors must match`。

**对策**：
- **直接读取配置**：使用 `cfg.head_dim` 而非公式推导。
- **编写诊断脚本**：`86-Inspect-Model-Dims.py` 可快速打印所有关键维度，避免盲猜。

## 4. 显式参数签名 (21+ 个输入/输出)
**挑战**：Stateful Decoder 需要传递大量状态：`audio_codes, is_last, pre_conv_h, latent_buf, conv_h, past_key_0..7, past_value_0..7`。使用 `*args` 会导致 `dynamic_axes` 无法正确映射。

**对策**：
- **静态展开**：在导出脚本中显式列出所有 21 个输入名和 23 个输出名。
- **精准映射 dynamic_axes**：为每个 KV Tensor 单独指定 `{2: "past_seq_i"}`，确保不同序列长度下的通用性。

## 5. 攻克 Trace 常量固化陷阱 (Symbolic Dimensions)
**挑战**：在 `torch.onnx.export` (Trace 模式) 中，如果你直接将 `tensor.shape[i]` 转为 Python `int` 或 `float` 使用，Tracer 会记录下导出那一刻的具体数字（常量），而不是“取形状”的操作。这会导致导出的模型失去动态性，在处理不同长度的输入或历史记录时发生偏移或错位。

**对策**：
- **禁止原生转换**：严禁在 `forward` 中对 `Tensor.size()` 结果使用 `int()`、`float()` 或 `item()`。
- **符号化获取 (Zeros + Size 技巧)**：使用 `torch.zeros(1) + tensor.size(dim)` 等运算强制生成符号化的 Tensor 节点。这会迫使 ONNX 记录一个 `GatherShape` 算子，确保推理时实时获取维度。
- **纯 Tensor 逻辑链**：确保 `valid_samples`、`start_samples` 等关键标志位的计算全部由 Tensor 算子完成，不含有任何 Python 标量常量的硬编码。

## 6. 导出流程最佳实践
1. **先验证 PyTorch 等价性**：使用 `82-Verify-ONNX-Wrapper-PyTorch.py` 确认基础逻辑。
2. **维度诊断**：用 `86-Inspect-Model-Dims.py` 检查所有关键 Shape。
3. **强制 JIT 路径**：设置 `dynamo=False`。
4. **端到端验证 (关键)**：编写 `87-Verify-ONNX-Runtime.py` 对比 PyTorch 和 ONNX RT 的输出。如果发现长度对不齐或 MSE 飙升，优先检查是否有维度被常量固化。

## 7. 关键文件清单
| 文件 | 用途 |
|------|------|
| `codec_export.py` | 包含 `StatefulCodecONNXWrapper` 和 `TraceableKVStack` |
| `82-Verify-ONNX-Wrapper-PyTorch.py` | PyTorch 侧流式推理验证 |
| `85-Export-Stateful-ONNX.py` | ONNX 导出脚本 |
| `86-Inspect-Model-Dims.py` | 维度诊断工具 |
| `onnx_export/qwen3_tts_decoder_stateful.onnx` | 导出的 ONNX 模型 (435 MB) |
