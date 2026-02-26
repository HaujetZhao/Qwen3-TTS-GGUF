# Torch Dynamo 导出经验：避开 math.ceil 与零长度张量陷阱

在将 Qwen3-TTS 模型通过 Torch Dynamo (`torch.export`) 导出为 ONNX，并使用 DirectML (DML) 后端运行时，我们遇到了两个非常隐蔽但致命的问题。本文记录了这些问题的根源及其解决方法。

---

## 1. 陷阱一：在中途使用 `math.ceil` 导致导出失败

### 现象
在导出代码中，如果对带有符号属性（Symbolic Shapes）的张量维度进行 `math.ceil()` 计算，会触发如下错误：
`torch._guards.SourceContext.GuardOnDataDependentSymNode`

### 原因
Torch Dynamo 在追踪（Tracing）模型时，会将张量的维度表示为符号整数（SymInt）。`math.ceil()` 或类似的浮点操作会将这些 SymInt 强制转换为普通的浮点数。
1. **丢失符号属性**：一旦转换为浮点数，Dynamo 就会失去对该维度动态范围的追踪。
2. **触发保护异常**：Dynamo 为了保证导出的计算图在不同输入长度下都成立，禁止在追踪过程中出现依赖于具体数值的布尔判断（Data-dependent guards），而 `float(SymInt)` 往往会触发这种保护。

### 解决方法：使用纯整数算术
**永远不要在建模代码中使用浮点除法和 `math.ceil` 算 Stride 或 Padding。**

*   **错误做法**:
    ```python
    pad = math.ceil((length - kernel) / stride)
    ```
*   **正确做法 (向上取整等价公式)**:
    利用公式 `ceil(x / y) = (x + y - 1) // y`：
    ```python
    # 纯整数运算，保持符号属性
    pad = (length - kernel + stride - 1) // stride
    ```

---

## 2. 陷阱二：DML 的 80070057 报错与零长度张量

### 现象
音频推理流式解码时，间歇性抛出 DirectML 错误：
`[E:onnxruntime:...] Status Message: ... Exception(2) 80070057`

通过 Profiling 发现，在报错的前一步，某个算子的输出 Shape 变成了 `[1, 768, 0]`（最后一维长度为 0）。

### 原因：切片逻辑溢出
在处理上采样或卷积 Padding 时，常见的“切头去尾”逻辑（Slicing）在流式推理初期极易出错：
```python
# 典型逻辑：剪掉左右 Padding
hidden_state = hidden_state[..., left_pad : total_len - right_pad]
```
如果 `total_len` 刚好等于或小于 `left_pad + right_pad`，切片结果就会变成一个 **零长度张量（Empty Tensor）**。
*   **CPU/CUDA**: 通常能处理这种空张量（或者静默失败）。
*   **DirectML**: 其卷积（Conv）和转置卷积（ConvTranspose）算子的后端驱动通常**不支持宽度为 0 的资源描述符**，会直接抛出 `80070057`（Access Denied / Invalid Argument）导致推理进程崩溃。

### 解决方法：加固切片逻辑
确保切片后的长度永远大于 0，并消除对 `math` 库的依赖。

```python
# 在 __init__ 中
pad = kernel_size - stride
self.left_pad = (pad + 1) // 2
self.right_pad = pad - self.left_pad

# 在 forward 中
total_len = hidden_state.shape[-1]
start = self.left_pad
# 这种写法在 Dynamo 导出时会生成稳定的符号判断
end = total_len - self.right_pad
# 这种加固逻辑能强迫运行时保持至少 1 个点的长度，避免 DML 崩溃
if end <= start:
    end = start + 1
    
hidden_state = hidden_state[..., start : end]
```

---

## 总结

1.  **符号完整性**：在 Dynamo 导出流程中，计算维度相关的变量时，严禁离开“整数算术”领域。
2.  **避免零维**：DML 对 `Shape` 中的 `0` 极其敏感。在做任何基于计算得出的切片（Slice）时，必须设置下限保护，哪怕切出一段全 0 的静默信号，也比程序报错强。
3.  **Profiling 是利器**：当 DML 报 `80070057` 却难以定位时，开启 ONNX Runtime Profiling 观察报错算子的 `input_type_shape` 是否包含 `0` 是最有效的查错手段。
