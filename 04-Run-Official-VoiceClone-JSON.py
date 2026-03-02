import os
import sys
import time
import torch
import random
import numpy as np
import soundfile as sf
import subprocess
import functools
import types
import traceback
from pathlib import Path
from export_config import MODEL_DIR
from qwen3_tts_gguf.inference.schema.result import TTSResult, Timing
from qwen3_tts_gguf.inference.capturer import OfficialCapturer

ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR / "Qwen3-TTS-main"))
from qwen_tts import Qwen3TTSModel
from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    # 使用 GPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 特征文件路径
    JSON_PATH = ROOT_DIR / "output" / "sample.json"
    if not JSON_PATH.exists():
        print(f"❌ 找不到特征文件: {JSON_PATH}")
        return

    # 1. 加载 GGUF 引擎保存出来的 JSON 特征 (作为音色注入源)
    print(f"📖 [Load] 正在从 {JSON_PATH.name} 加载音色和语义特征...")
    ref_result = TTSResult.from_json(str(JSON_PATH))
    
    # 2. 加载官方模型
    print(f"🚀 [Model] 正在负载基础模型: {MODEL_DIR}")
    try:
        # 定义数据类型
        dtype = torch.bfloat16
        
        # 载入模型
        t_load_start = time.time()
        set_seed(47)
        tts = Qwen3TTSModel.from_pretrained(
            str(MODEL_DIR),
            device_map=device,
            dtype=dtype,
        )
        t_load_end = time.time()
        print(f"模型加载完成，耗时 {t_load_end - t_load_start:.4f} 秒。")

        # --- 初始化自动捕获器 ---
        # 即使是注入模式，我们也希望捕获生成的 codes 和 audio 为新的 TTSResult
        capturer = OfficialCapturer(tts)

        # 3. 构造 16 层全量特征注入 Prompt
        # 这是官方模型实现“无损”或“强力”克隆的核心手段
        ref_codes = torch.from_numpy(ref_result.codes).to(device=device, dtype=torch.long)
        ref_spk_emb = torch.from_numpy(ref_result.spk_emb).to(device=device, dtype=dtype)
        
        print(f"💉 [Inject] 准备注入特征:")
        print(f"   - 参考文本: {ref_result.text}")
        print(f"   - Codes 形状: {ref_codes.shape}")

        prompt_item = VoiceClonePromptItem(
            ref_code=ref_codes,
            ref_spk_embedding=ref_spk_emb,
            x_vector_only_mode=False,
            icl_mode=True,       # ICL 模式（语内插值/声音克隆）
            ref_text=ref_result.text 
        )

        # 4. 执行生成
        target_text = "我今天特别想你，想跟你聊会儿。"
        print(f"\n--- [Voice Injection] 正在注入式生成音频 ---")
        
        t_infer_start = time.time()
        # 调用包装后的方法，直接返回新的 TTSResult
        res = tts.generate_voice_clone(
            text=target_text,
            language="Chinese",
            voice_clone_prompt=[prompt_item], # 注入核心
            temperature=0.8, 
            subtalker_temperature=0.8, 
        )
        t_infer_end = time.time()
        print(f"推理完成，耗时 {t_infer_end - t_infer_start:.4f} 秒。")

        # --- 保存并播放结果 ---
        output_base = f"output/clone_from_json"
        res.save_json(f"{output_base}.json")
        res.save_wav(f"{output_base}.wav")
        
        print(f"✅ 生成成功！注入生成的结果已保存至: {output_base}.*")
        res.play()

    except Exception as e:
        print(f"推理失败: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
