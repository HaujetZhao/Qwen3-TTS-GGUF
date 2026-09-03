# 10 - Codec Encoder FP16 溢出导致静音帧 coarse 码塌 0

## 现象

GUI 工具页 WAV → JSON → WAV roundtrip，音频静音段（开头/结尾）出现可闻杂音。
解码器无责：ONNX 解码器与官方 PyTorch 解码器对同一 codes 输出完全一致。
直接观察像是"只有头尾坏"，实际是**所有静音帧都坏**（头尾恰好是静音）。

## 根因：torch.cdist 的 ONNX 展开式在 fp16 下溢出

`MimiEuclideanCodebook.quantize` 用 `torch.cdist` 求最近邻。ONNX 导出时 cdist
被降级为代数展开式：

```
ArgMin ← Sqrt ← Max( ‖x‖² − 2x·eᵀ + ‖e‖² , 0 )
                   ↑ ReduceSum(Pow(x,2))
```

"先减后平方"（每项小）变成了"先平方后加减"（单项 ‖x‖² 很大）。本模型中
**静音帧的量化器输入范数反而更大**（实测 ‖x‖≈263~383，‖x‖²≈6.9万~14.7万），
超过 fp16 上限 65504 → ReduceSum 溢出为 Inf → 该帧全部码本距离为 Inf →
`ArgMin` 在全 Inf 上返回索引 **0**（死码字，解码出来是噪声）。

语音帧 ‖x‖≈200，‖x‖²≈4.0万 < 65504，不溢出，所以中间正常。
fp32 不溢出，一切正常；这不是精度损失，是溢出 bug。

定位方法：给 encoder 各层挂多输出头（探针），fp32/fp16 同头对比逐帧差异。
量化器输入误差仅 0.16%，量化后跳到 82.5% → 锁定最近邻查找。纯数值实验
（numpy 手工 cast fp16）证明 argmin 本应全对（margin 6~80 远大于 fp16 舍入），
排除精度问题，指向图级展开式的溢出。探针脚本在 `tmp/probe_export.py` /
`tmp/probe_compare.py`。

## 修复

`qwen3_tts_gguf/export/tokenizer_12hz/internal/modeling_mimi.py` 的
`MimiEuclideanCodebook.quantize`：cdist 前对 x 与 embed 同乘 α=1/16。
L2 距离同比例缩小，argmin 对正缩放不变 → fp32 语义逐位不变；
‖x‖²/256 ≈ 270，余量充足（需 ‖x‖>4096 才会再溢出）。

验证（speech-2.6-turbo.wav，43 帧 × 16 码）：

| 版本 | 与官方 fp32 coarse 一致率 | 头 2 帧 coarse |
|---|---|---|
| 修复前 fp16 | 67.4% | 0, 0 |
| 修复后 fp16 | 100% | 1995, 215（正确） |

端到端（修复 fp16 编码 → 解码）：头尾静音段 RMS 1e-5，杂音消失。
深层 acoustic RVQ 与官方 fp32 仍有 ~10% 近邻平局差异，官方自家 fp32 vs fp16
也仅 ~75% 一致，同属正常波动，不影响听感。

## 备注

- codec encoder / decoder 在 12Hz 全系列模型间通用，可直接复制。
- 16 号脚本对 codec encoder 只做 fp16 转换、不做 INT8（精度未验证）。
- `onnxruntime.transformers` 的 fp16 转换器有 bug：op_block_list 超过
  LayerNormalization/Softmax/Range 三项就可能生成重名 Cast 节点导致模型非法。
