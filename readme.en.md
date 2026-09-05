# Qwen3-TTS GGUF

[中文](readme.md)

Qwen3-TTS running on llama.cpp, with streaming synthesis and voice cloning.

## Model Types

Three official models are supported, each for a different scenario:

| Source model | Scenario |
| :--- | :--- |
| Qwen3-TTS-12Hz-1.7B-Base | Voice cloning |
| Qwen3-TTS-12Hz-1.7B-CustomVoice | Built-in voices + style instructions |
| Qwen3-TTS-12Hz-1.7B-VoiceDesign | Designing voices with natural language |
| Qwen3-TTS-12Hz-0.6B-Base | Voice cloning |
| Qwen3-TTS-12Hz-0.6B-CustomVoice | Built-in voices + style instructions |

Export whichever one you want to use; the official model must be downloaded first.

## Performance

Measured on my RTX 5050:

- **RTX 5050 (discrete GPU)**: RTF 0.35 (real-time factor; 1 second of audio takes 0.35 s to generate)
- **CPU**: RTF 1.3
- **Integrated GPU**: RTF 1.3

RTF < 1 means generation is faster than real-time playback. Without a discrete GPU it's hard to get RTF < 1, so smooth streaming playback is unlikely there.

VRAM usage:

- **Encoder** extracts features from audio for cloning; no GPU acceleration needed, saving VRAM
- **Talker 1.7B**: quantized to q5_k — 955 MB weights + 224 MB context + 50 MB compute = 1229 MB
- **Predictor 0.1B**: quantized to q8_0 — 144 MB weights + 5 MB context + 7 MB compute = 156 MB
- **Decoder**: fp16, DML-accelerated — 237 MB model + 204 MB inference

The 1.7B setup needs about 1.8 GB of VRAM in total.

The 0.6B model saves another ~500 MB but doesn't speed things up much, because the **compute bottleneck is the Predictor**: every second of audio requires 12.5 × 15 = 187.5 autoregressive steps, and 0.6B vs 1.7B only differs in the Talker.

### Context Length Estimation

Each task occupies this many tokens in the Talker's context:

- Every second of audio costs **12.5** audio tokens (12 Hz frame rate, 12.5 frames/second)
- Target text is budgeted at **4** text tokens per second of generated audio (~4 characters/second; text tokens are injected alongside the audio frames)
- Plus about **10** constant control tokens (role head, language, speaker embedding, TTS_BOS, etc.)

When cloning, the reference audio also occupies context (audio at 12.5 tokens/s, reference text likewise budgeted at 4 tokens/s), so the context needed per task depends only on the **total audio seconds** (reference + generated):

```
ctx ≈ 10 + 16.5 × total audio seconds
```

Example: with 512 ctx per task, the combined reference + generated audio can be at most `(512 - 10) / 16.5 ≈ 30` seconds; Custom Voice / Voice Design have no reference audio, leaving all context for generation. Exceeding the context raises a Talker context overflow; for batch inference, set `n_ctx_per_seq` to this formula's upper bound.

### Batch VRAM Estimation

Batch inference (`BatchRunner`, CUDA) splits VRAM into two parts:

**Resident part** (weights counted at file size):

| Component | Size |
| :--- | :--- |
| Talker weights (q5_k) | 960 MB |
| Predictor weights (q8_0) | 144 MB |
| Decoder chunk64 | 620 MB |

**Batch part** (created by each `clone_batch` round, freed after; grows linearly with batch size B):

| Component | Size |
| :--- | :--- |
| Talker KV | 0.11 MB/token, i.e. 0.11 MB × B × ctx |
| Predictor KV | 5 MB × B |
| Compute buffer | 50 MB + 2 MB × B |

Total estimate: `total ≈ 1.78 GB + 0.11 MB × B × ctx`

Example: 32 tasks × 512 ctx each ≈ 1.78 + 1.80 ≈ **3.6 GB**. (Note: 512 context only fits 30 s of total audio.)

Batching only accelerates the LLM part (memory-bandwidth bound); after the codes are generated, the audio decoder part (compute bound) gets no speedup from multiple streams.

Measured on an RTX 5050: 32 tasks at ctx 512 reach RTF 0.055 (LLM 0.043 + Decoder 0.012).

## Features

- **Streaming synthesis**: dramatically cuts real time-to-first-audio, down to under 300 ms.
- **Accelerated inference**: RTF 0.35 for the 1.7B model on an RTX 5050; AMD GPUs work too via Vulkan.
- **Deterministic control**: independent random seeds for the Talker and the Predictor, so output is reproducible.

## How Cloning Works

Voice cloning is essentially **continued speech** (In-Context Learning).

Imagine you're reading a passage aloud, and halfway through I ask you to keep going — you'd naturally continue with the same voice and tone. That's exactly how Qwen3-TTS clones:

1. **Text concatenation**: the "reference text" and "target text" are joined together.
2. **Memory injection**: the "reference audio" is converted into spk_emb and codes and injected into memory, making the model believe it just spoke that audio itself:
   - **Voice (spk_emb)**: the overall timbre, telling the model what kind of voice it's speaking with.
   - **Syllables (codes)**: the concrete pronunciation codes (12.5 Hz), convincing the model "I really did just say these syllables myself".
3. **Riding the momentum**: with the "voice" and syllable memory in place, the model naturally keeps the same voice and finishes reading the rest of the text.

## How Custom Voice Works

Similar to cloning, except the model has built-in speaker voices (spk_embd). Its memory contains only the voice, no syllables. It must get itself into the right emotional state (get into character) based on the instructions and target text, then read the text with that voice.

It's like reading lines from a script:

- Clone = you've already read half (the reference text) and keep going (the target text).
- Custom Voice = you haven't started yet; you psych yourself up, then read from the beginning.

This brings different characteristics:

- Clone has already set the emotional tone (halfway through), so the voice is more stable and controllable.
- Custom Voice has to build up emotion, so different random seeds produce different tones — some gacha-like luck is involved.

The best dubbing workflow is therefore: use Custom Voice to gacha-roll the best take of a passage, then use the Base model to clone that audio and read other texts.

## Getting Started

#### Download Models

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

#### Dependencies

Pinned to **llama.cpp b10621**. Download a prebuilt binary from [llama.cpp Releases](https://github.com/ggml-org/llama.cpp/releases) and put the DLLs into `qwen3_tts_gguf/bin/`:

| Platform | Download |
|------|----------|
| **Windows (Vulkan)** | `llama-b10621-bin-win-vulkan-x64.zip` |
| **Windows (CUDA)** | `llama-b10621-bin-win-cuda-13.3-x64.zip` (also needs the CUDA 13 runtime `cudart64_13.dll`, `cublas64_13.dll`) |

FFmpeg is also required, for reading audio files.

[uv](https://docs.astral.sh/uv/) is recommended for installing dependencies:

```
uv sync --extra dml
```

NVIDIA users can switch to `--extra gpu`; add `--extra export` for the export scripts. Without uv, use `pip install -r requirements.txt`.

#### Configure Paths

Open `export_config.py` and set the model export paths:

#### Stage 1: Export Small Components

```bash
python 11-Export-Codec-Encoder.py    # Encoder, used for cloning
python 12-Export-Speaker-Encoder.py  # Speaker feature extractor
python 13-Export-Decoder.py          # Decoder, the core renderer
python 14-Export-Embeddings.py       # Embedding weights
python 15-Copy-Tokenizer.py          # Text tokenizer
python 16-Quantize-ONNX-Models.py    # Important: convert ONNX to FP16 for DML acceleration
```

#### Stage 2: Export the Talker

The Talker is the 1.42B LLM backbone that understands the text and generates the speech skeleton:

```bash
python 21-Extract-Talker-Weights.py    # Split and initialize weights
python 22-Prepare-Talker-Tokenizer.py  # Build the mini vocab needed for GGUF
python 23-Convert-Talker-GGUF.py       # Convert to F16 GGUF
python 24-Quantize-Talker-GGUF.py      # Quantize to q5_k, the default loaded by the engine
```

#### Stage 3: Export the Predictor

The Predictor is a 142M mini-model that fills in the details for the skeleton:

```bash
python 31-Extract-Predictor-Weights.py
python 32-Prepare-Predictor-Tokenizer.py
python 33-Convert-Predictor-GGUF.py
python 34-Quantize-Predictor-GGUF.py    # Quantize to q8_0, the default loaded by the engine
```

Once done, `EXPORT_DIR` contains everything you need.

## Inference

### GUI Mode (Recommended)

```
python 52-GUI.py
```

Graphical interface: model loading (LLM device / optional ONNX components), voice cloning / custom voice / design, batch tasks written to wav+json, and a model-slimming tool.

### Script Mode

Three example scripts, one per model type:

```bash
python 41-Inference-Custom.py  # Built-in voices
python 42-Inference-Design.py  # Voice design
python 43-Inference-Base.py    # Voice cloning
```

### Interactive Mode (Recommended)

```bash
python 51-Interactive-Clone.py
```

Start it and just type — it synthesizes and plays as you go.

## Code Usage

```python
from qwen3_tts_gguf import TTSEngine, TTSConfig

# Initialize the engine (models load in parallel in the background)
engine = TTSEngine(model_dir="model-base")
stream = engine.create_stream()

# Set the voice (accepts a .wav path, a .json path, or a TTSResult object)
stream.set_voice("output/elaborate/sample.json")

# Configure inference parameters
config = TTSConfig(
    temperature=0.8,      # core temperature, controls randomness
    sub_temperature=0.8,  # detail temperature, controls randomness
    seed=42,           # core seed
    sub_seed=45,       # detail seed
    streaming=True,    # enable streaming
)

# Streaming synthesis
result = stream.clone("你好，世界！", config=config)
stream.join()  # wait for playback to finish

# Save the result
result.save("output/output.wav")
result.save("output/output.json")  # save codes; can be reloaded losslessly later
```

## Available Speakers

The CustomVoice model ships with 9 voices:

| ID | Description |
| :--- | :--- |
| vivian | Young female, bright and crisp |
| serena | Warm female, soft and approachable |
| uncle_fu | Mature male, steady and deep |
| dylan | Beijing male, natural and clear |
| eric | Chengdu male, slightly husky |
| ryan | Energetic male, strong rhythm |
| aiden | Sunny young male, clear mids |
| ono_anna | Japanese female, playful and lively |
| sohee | Korean female, tender and emotional |

## Supported Languages

- chinese, english, japanese, korean
- german, spanish, french, russian, italian, portuguese
- beijing_dialect, sichuan_dialect

## Architecture in a Nutshell

Think of it in terms of human organs:

1. **Ear** (Encoder): listens to the reference audio and extracts timbre features
2. **Brain** (Talker): generates the speech skeleton (28 layers, 1.42B)
3. **Hands** (Predictor): fills in the details (8 layers, 142M)
4. **Mouth** (Decoder): decodes the codes into sound

Inference engines:
- Talker / Predictor: llama.cpp (GGUF format, Vulkan/CUDA accelerated)
- Encoder / Decoder: ONNX Runtime (ONNX format, DirectML/CUDA accelerated)

## FAQ

**Q: Why not the official PyTorch implementation?**

The official implementation needs lots of VRAM. llama.cpp is lean, and works with Vulkan/DML acceleration.

**Q: Streaming vs offline?**

Streaming plays as it synthesizes, with low first-packet latency (~300 ms).

**Q: How do I tune quality?**

`TTSConfig(temperature=0.8, sub_temperature=0.8, seed=42, sub_seed=45)` — temperature controls randomness, seeds control reproducibility.

## Links

- [Qwen3-TTS Technical Report](./Qwen3-TTS%20Technical%20Report.md)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
