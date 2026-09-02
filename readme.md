# Qwen3-TTS GGUF

用 llama.cpp 跑的 Qwen3-TTS，支持流式合成、声音克隆。

## 模型类型

本项目支持三种官方模型，对应三种场景：

| 源模型 | 场景 |
| :--- | :--- |
| Qwen3-TTS-12Hz-1.7B-Base | 声音克隆 |
| Qwen3-TTS-12Hz-1.7B-CustomVoice | 内置音色 + 风格指令 |
| Qwen3-TTS-12Hz-1.7B-VoiceDesign | 用自然语言设计音色 |
| Qwen3-TTS-12Hz-0.6B-Base | 声音克隆 |
| Qwen3-TTS-12Hz-0.6B-CustomVoice | 内置音色 + 风格指令 |

你想用哪个，就导出哪个，但是需要先下载官方模型才能导出。

## 性能表现

在我的 RTX 5050 上的实测数据：

- **RTX 5050 (独显)**: RTF 0.35 (实时率，1秒音频只需0.35秒生成)
- **CPU**: RTF 1.3
- **集显**: RTF 1.3

RTF < 1 就表示生成速度比实时播放还快。没有独显的电脑，很难做到 RTF < 1，因此不太可能流畅地流式播放。

显存占用：

- **Encoder** 用于从音频提取特征克隆，无需显卡加速，可节省显存
- **Talker 1.7B**，用 Q5_k 量化，载入 955MB，上下文224MB，计算50MB，共1229MB
- **Predictor 0.1B**，用 Q8_0 量化，载入 144MB，上下文5MB，计算7MB，共156MB
- **Decoder** 用 fp16 量化，DML 加速，模型 237MB，推理204MB

使用 1.7B 总共需 1.8G 显存。

用 0.6B 版可以再省 500MB 显存，但对速度的提升不大，因为**计算瓶径在于 Predictor**，每一秒的音频需要自回归 12.5*15=187.5 次，0.6B 和 1.7B 的差异仅在于 Talker。

### 上下文长度估算

每路任务在 Talker 上下文里占用的 token 数可以这样估算：

- 每秒音频占 **12.5** 个音频 token（12Hz 帧率，12.5 帧/秒）
- 目标文本按每秒生成音频 **4** 个文本 token 折算（约 4 字/秒的语速，文本 token 随音频帧逐个注入）
- 再加约 **10** 个常数控制 token（角色头、语言、说话人嵌入、TTS_BOS 等）

克隆时参考音频同样占上下文（音频 12.5 token/秒，参考文本同样折算 4 token/秒），于是单路所需上下文只看**总音频秒数**（参考 + 生成）：

```
ctx ≈ 10 + 16.5 × 总音频秒数
```

例：每路 512 ctx 时，参考音频加生成音频合计最长 `(512 - 10) / 16.5 ≈ 30` 秒；Custom Voice / Voice Design 没有参考音频，全部上下文都留给生成。超出上下文会报 Talker context overflow，批量推理时 `n_ctx_per_seq` 按这个公式的上限取即可。

### 批量显存估算

批量推理（`BatchRunner`，CUDA）时显存分两块：

**常驻部分**（权重按文件大小计）：

| 组件 | 大小 |
| :--- | :--- |
| Talker 权重 (q5_k) | 960 MB |
| Predictor 权重 (q8_0) | 144 MB |
| Decoder chunk64 | 620 MB |

**批量部分**（每轮 `clone_batch` 创建、跑完即释放，随路数 B 线性增长）：

| 组件 | 大小 |
| :--- | :--- |
| Talker KV | 0.11 MB/token，即 0.11 MB × B × ctx |
| Predictor KV | 5 MB × B |
| 计算缓冲 | 50 MB + 2 MB × B |

合计估算：`总量 ≈ 1.78 GB + 0.11 MB × B × ctx`

例：32 路、每路 512 ctx ≈ 1.78 + 1.80 ≈ **3.6 GB**。（注意：上下文 512 只能容纳 30s 总音频时长）

批量仅对 LLM 部分有加速效果（读取瓶径），生成 Codes 后，音频解码器部分（计算瓶径）多路无加速效果。

实测，在 RTX5050 上，32路 ctx512 可以做到 RTF0.055（LLM 0.043 + Decoder 0.012）

## 项目特性

- **流式合成**：大幅缩短流式实际首音延迟，最低可至 300ms 内。
- **加速推理**：对 1.7B 模型，RTX5050 可以做到 RTF0.35，AMD 显卡也可以用 Vulkan 加速
- **确定性控制**：支持独立设置 Talker 和 Predictor 的随机种子，确保输出可复现。

## Clone 原理

声音克隆的本质是 **接续说话** (In-Context Learning)。

想象一下，你正在读一段话，读到一半时，我让你接着往下读，你自然会用同样的语气和声音。Qwen3-TTS 的克隆原理也是如此：

1. **文本拼接**：把「参考文本」和「目标文本」连接起来。
2. **注入记忆**：把「参考音频」转为 spk_emb 和 codes，注入记忆，让模型以为前面的音频是它自己刚说的：
   - **嗓子（spk_emb）**：整体音色特征，告诉模型我是用什么样的嗓子在说话。
   - **音节（codes）**：具体的发音码（12.5Hz），让模型认为「我刚才确实亲口说出了这些音节」。
3. **顺势说完**：有了之前的「嗓子」和音节记忆，模型就会顺理成章地继续保持这个声音，把剩下的文字接着读完。

## Custom Voice 原理

与克隆类似，只是模型内置了一些说话人音色（spk_embd），模型的记忆中只会有嗓子，没有音节。它要根据指令和目标文本，让自己处于某种情感状态（入戏），然后用嗓子音色把目标文本读出来。

有点像是读台词：

- Clone是读了一半（参考文本），然后接着读（目标文本）。
- Custom Voice 是还没有读，酝酿一下情感，从头开始读。

因此带来了不同的特点：

- Clone 因为已经读了一半，情感基调已定，音色就更稳定可控。
- Custom Voice 因为要酝酿情感，不同的随机种子，会酝酿出不同的基调，就会有些抽卡。

因此最佳的配音实践是：用 Custom Voice 对同一段文字抽卡出最好的效果，再用 Base 模型基于这个音频克隆阅读其它文本。

## 快速开始

#### 下载模型

- [Qwen3-TTS-12Hz-1.7B-Base](https://www.modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base)
- [Qwen3-TTS-12Hz-1.7B-CustomVoice](https://www.modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
- [Qwen3-TTS-12Hz-1.7B-VoiceDesign](https://www.modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign)

- [Qwen3-TTS-12Hz-0.6B-Base](https://www.modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-0.6B-Base)
- [Qwen3-TTS-12Hz-0.6B-CustomVoice](https://www.modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)

```
pip install modelscope
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-Base
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign

modelscope download --model Qwen/Qwen3-TTS-12Hz-0.6B-Base
modelscope download --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
```

#### 依赖环境

适配版本 **llama.cpp b10621**。从 [llama.cpp Releases](https://github.com/ggml-org/llama.cpp/releases) 下载预编译二进制，将 DLL 放入 `qwen3_tts_gguf/bin/`：

| 平台 | 下载文件 |
|------|----------|
| **Windows (Vulkan)** | `llama-b10621-bin-win-vulkan-x64.zip` |
| **Windows (CUDA)** | `llama-b10621-bin-win-cuda-13.3-x64.zip`（另需 CUDA 13 运行时 `cudart64_13.dll`、`cublas64_13.dll`） |

另外还需安装 FFmpeg，用于读取音频文件。

推荐用 [uv](https://docs.astral.sh/uv/) 安装依赖：

```
uv sync --extra dml
```

N 卡可换 `--extra gpu`；跑导出脚本再加 `--extra export`。不用 uv 则 `pip install -r requirements.txt`。

#### 配置路径

打开 `export_config.py`，在里面配置好模型的导出路径：


#### 阶段一：导出小组件

```bash
python 11-Export-Codec-Encoder.py    # 编码器，用于克隆
python 12-Export-Speaker-Encoder.py  # 说话人特征提取器
python 13-Export-Decoder.py          # 解码器，核心渲染器
python 14-Export-Embeddings.py       # Embedding 权重
python 15-Copy-Tokenizer.py          # 文本分词器
python 16-Quantize-ONNX-Models.py    # 重要：将 ONNX 转为 FP16 以供 DML 加速
```

#### 阶段二：导出大师（Talker）

大师是 1.42B 的 LLM backbone，负责理解文本、生成语音骨架：

```bash
python 21-Extract-Talker-Weights.py    # 拆分并初始化权重
python 22-Prepare-Talker-Tokenizer.py  # 构造 GGUF 所需的迷你词表
python 23-Convert-Talker-GGUF.py       # 转换为 F16 的 GGUF
python 24-Quantize-Talker-GGUF.py      # 量化为 q5_k，这是推理引擎默认加载的版本
```

#### 阶段三：导出工匠（Predictor）

工匠是 142M 的小模型，负责给骨架补充细节：

```bash
python 31-Extract-Predictor-Weights.py
python 32-Prepare-Predictor-Tokenizer.py
python 33-Convert-Predictor-GGUF.py
python 34-Quantize-Predictor-GGUF.py    # 量化为 q8_0，这是推理引擎默认加载的版本
```

导完之后，`EXPORT_DIR` 里就有你需要的所有文件了。

## 推理

### GUI 模式（推荐）

```
python 52-GUI.py
```

图形界面：模型载入（LLM 设备 / ONNX 组件可选）、声音克隆 / 音色 / 设计、批量任务落盘 wav+json、模型瘦身工具。

### 脚本模式

三个示例脚本，对应三种模型：

```bash
python 41-Inference-Custom.py  # 精品音色
python 42-Inference-Design.py  # 音色设计
python 43-Inference-Base.py    # 声音克隆
```

### 交互模式（推荐）

```bash
python 51-Interactive-Clone.py
```

启动后直接打字，边推边播。

## 代码调用

```python
from qwen3_tts_gguf import TTSEngine, TTSConfig

# 初始化引擎（后台自动并行加载模型）
engine = TTSEngine(model_dir="model-base")
stream = engine.create_stream()

# 设置音色（支持 .wav 路径、.json 路径或 TTSResult 对象）
stream.set_voice("output/elaborate/sample.json")

# 配置推理参数
config = TTSConfig(
    temperature=0.8,      # 核心温度，控制随机性
    sub_temperature=0.8,  # 细节温度，控制随机性
    seed=42,           # 核心种子
    sub_seed=45,       # 细节种子
    streaming=True,    # 开启流式
)

# 流式合成
result = stream.clone("你好，世界！", config=config)
stream.join()  # 等待播完

# 保存结果
result.save("output/output.wav")
result.save("output/output.json")  # 保存 codes，下次可无损加载
```

## 可用说话人

CustomVoice 模型内置 9 个音色：

| ID | 说明 |
| :--- | :--- |
| vivian | 年轻女声，明亮利落 |
| serena | 温暖女声，柔和亲切 |
| uncle_fu | 成熟男声，沉稳低沉 |
| dylan | 北京男声，自然清晰 |
| eric | 成都男声，略带沙哑 |
| ryan | 活力男声，节奏感强 |
| aiden | 阳光美男，中频清澈 |
| ono_anna | 日语女声，俏皮轻快 |
| sohee | 韩语女声，温润动情 |

## 支持的语言

- chinese, english, japanese, korean
- german, spanish, french, russian, italian, portuguese
- beijing_dialect, sichuan_dialect

## 架构简述

用人体器官来理解：

1. **耳朵**（Encoder）: 听参考音频，提取音色特征
2. **大脑**（Talker）: 生成语音骨架 (28层, 1.42B)
3. **双手**（Predictor）: 补充细节 (8层, 142M)
4. **嘴巴**（Decoder）: 把 codes 解码成声音

推理引擎：
- Talker / Predictor: llama.cpp (GGUF格式，Vulkan/Cuda 加速)
- Encoder / Decoder: ONNX Runtime (ONNX格式，DirectML/Cuda 加速)

## 常见问题

**Q: 为什么不用官方 PyTorch？**

官方实现要大显存。llama.cpp 省资源，还能用 Vulkan/DML 加速。

**Q: 流式和离线有什么区别？**

流式边推边播，首包延迟低（~300ms）。

**Q: 怎么调质量？**

`TTSConfig(temperature=0.8, sub_temperature=0.8, seed=42, sub_seed=45)` 温度控制随机性，用种子控制稳定复现。

## 相关链接

- [Qwen3-TTS 技术报告](./Qwen3-TTS%20Technical%20Report.md)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
