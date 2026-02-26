# coding=utf-8
import os
import sys
import torch
import numpy as np
import soundfile as sf
import sounddevice as sd
from pathlib import Path

# 确保路径正确：官方源码 和 我们自己的库
ROOT_DIR = Path(r"d:\qwen3-tts")
OFFICIAL_REPO_PATH = ROOT_DIR / "Qwen3-TTS-main"
if str(OFFICIAL_REPO_PATH) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_REPO_PATH))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from qwen_tts import Qwen3TTSModel
from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem
from qwen3_tts_gguf.inference.result import TTSResult
from export_config import MODEL_DIR

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    JSON_PATH = ROOT_DIR / "output" / "sample.json"
    
    if not JSON_PATH.exists():
        print(f"❌ 找不到特征文件: {JSON_PATH}")
        return

    # 1. 加载 GGUF 引擎保存出来的 JSON 特征
    print(f"📖 [Load] 正在从 {JSON_PATH.name} 加载音色和语义特征...")
    res = TTSResult.from_json(str(JSON_PATH))
    
    # 2. 加载官方模型 (推理模式)
    print(f"🚀 [Model] 正在加载官方基础模型: {MODEL_DIR}")
    qwen_model = Qwen3TTSModel.from_pretrained(
        str(MODEL_DIR),
        device_map=device,
        dtype=torch.float32,
    )

    # 3. 构造 VoiceClonePromptItem (关键：全量 16 层注入)
    # 官方 12Hz 模型 codes 应该是 (T, 16)
    ref_codes = torch.from_numpy(res.codes).to(device=device, dtype=torch.long)
    ref_spk_emb = torch.from_numpy(res.spk_emb).to(device=device, dtype=qwen_model.model.dtype)
    
    print(f"💉 [Inject] 准备注入特征:")
    print(f"   - 参考文本: {res.text}")
    print(f"   - Codes 形状: {ref_codes.shape}")
    print(f"   - 音色向量形状: {ref_spk_emb.shape}")

    prompt_item = VoiceClonePromptItem(
        ref_code=ref_codes,
        ref_spk_embedding=ref_spk_emb,
        x_vector_only_mode=False,
        icl_mode=True,       # ICL 模式（语内插值/声音克隆）
        ref_text=res.text    # 必须提供 JSON 里的参考文本以配合 codes
    )

    # 4. 执行生成 (跳过实时 Encoder，直接使用注入的 prompt_item)
    TARGET_TEXT = "我今天特别想你，想和你聊会儿。"
    print(f"🎙️  [Generate] 开始注入式克隆生成: '{TARGET_TEXT}'")
    
    wavs, sr = qwen_model.generate_voice_clone(
        text=TARGET_TEXT,
        language="Chinese",
        voice_clone_prompt=[prompt_item], # 这里是核心注入点
        do_sample=False,                 # 确定性生成
        subtalker_dosample=False
    )

    # 5. 保存与预览
    OUT_DIR = ROOT_DIR / "output" / "official_results"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUT_DIR / "clone_from_json.wav"
    
    if wavs:
        sf.write(save_path, wavs[0], sr)
        print(f"✅ [Success] 推理完成！结果已保存至: {save_path}")
        print(f"🔊 正在播放生成音频...")
        sd.play(wavs[0], sr)
        sd.wait()

if __name__ == "__main__":
    main()
