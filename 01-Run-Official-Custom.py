import os
import sys
import time
import torch
import random
import numpy as np
import soundfile as sf
import subprocess
from pathlib import Path
from export_config import MODEL_CUSTOM as MODEL_DIR

ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR / "Qwen3-TTS-main"))

from qwen_tts import Qwen3TTSModel

tasks = []

# 单属性控制

speaker = "Ryan"
language = None
text = 'She said she would be here by noon.'
instruct = 'spoke with a very sad and tearful voice.'
tasks.append((speaker, language, text, instruct))

speaker = "Ryan"
language = None
text = 'She said she would be here by noon.'
instruct = 'Very happy.'
tasks.append((speaker, language, text, instruct))

# speaker = "Ryan"
# language = None
# text = 'She said she would be here by noon.'
# instruct = '用特别愤怒的语气说'
# tasks.append((speaker, language, text, instruct))

# speaker = "Ryan"
# language = None
# text = 'She said she would be here by noon.'
# instruct = '请特别小声的悄悄说'
# tasks.append((speaker, language, text, instruct))

# speaker = "Ryan"
# language = None
# text = 'She said she would be here by noon.'
# instruct = '请特别小声的悄悄说'
# tasks.append((speaker, language, text, instruct))

# speaker = "Ryan"
# language = None
# text = 'She said she would be here by noon.'
# instruct = '音调低沉'
# tasks.append((speaker, language, text, instruct))



# #多属性控制

# speaker = "Vivian"
# language = None
# text = '就算你自己不想治，你也得考虑考虑别人的感受吧。我们这些朋友的感受你不在乎无所谓，那你家人呢？你家人的感受你难道一点都不在乎吗！'
# instruct = \
# """
# 性别: 女性声音.
# 音高: 女性中高音区，语调富于变化.
# 语速: 语速明快，偶有加速.
# 音量: 正常交谈音量，笑声响亮.
# 清晰度: 吐字清晰，发音标准.
# 流畅度: 表达流畅自如.
# 口音: 普通话.
# 音色质感: 音色明亮，略带爽朗.
# 情绪: 愉悦友好，伴随爽朗笑意.
# 语调: 语调上扬活泼，疑问时尤为明显.
# 性格: 外向开朗，热情健谈.
# """
# tasks.append((speaker, language, text, instruct))

# speaker = "Vivian"
# language = None
# text = '就算你自己不想治，你也得考虑考虑别人的感受吧。我们这些朋友的感受你不在乎无所谓，那你家人呢？你家人的感受你难道一点都不在乎吗！'
# instruct = """以极度悲伤、带着明显哭腔的语气，用较小的音量缓缓诉说，语速缓慢，仿佛每一个字都承载着沉重的痛楚，声音颤抖而压抑，吐字虽轻却清晰可辨，透出深藏心底的哀伤与无助。"""
# tasks.append((speaker, language, text, instruct))

# speaker = "Vivian"
# language = None
# text = '就算你自己不想治，你也得考虑考虑别人的感受吧。我们这些朋友的感受你不在乎无所谓，那你家人呢？你家人的感受你难道一点都不在乎吗！'
# instruct = """保持青年女性的声线特征，展现出一种清亮且略具紧迫感的音色，语速从平稳开始在叙述过程中逐渐加快，音量在情绪波动时增加，语调在句末调高以强调劝告的语气。"""
# tasks.append((speaker, language, text, instruct))


# # 单人多语泛化

# speaker = "Vivian"
# language = "Korean"
# text = '안녕하세요, 오늘은 어떤 용건입니까?'
# instruct = """在语速偏快的情况下流畅自然地表达,音质清亮,音调略高,吐字清晰标准,给人一种开心愉悦的感觉。"""
# tasks.append((speaker, language, text, instruct))

# speaker = "Vivian"
# language = "Japanese"
# text = 'こんにちは、本日はどのようなご用件でしょうか？'
# instruct = """A deep, rich, and solid vocal register characteristic of a middle-aged woman, with full and powerful volume. Speech is delivered at a steady pace, articulation clear and precise, with fluent and confident intonation that rises slightly at the end of sentences."""
# tasks.append((speaker, language, text, instruct))

# speaker = "Vivian"
# language = "sichuan_dialect"
# text = '我早就该下班了，就是跟你说我这事情干不完，我现在走不脱。'
# instruct = """语音应表现为直率且略显主观强势的中年女性,音色略带尖锐感,流畅表达中偶尔断句以凸显语气,情绪略带不满,音量随情感激动略有增强。"""
# tasks.append((speaker, language, text, instruct))





# # 9个音色

# speaker = "Serena"
# language = "Chinese"
# text = '其实我真的有发现，我是一个特别善于观察别人情绪的人。'
# instruct = ''
# tasks.append((speaker, language, text, instruct))

# speaker = "uncle_fu"
# language = "Chinese"
# text = '其实我真的有发现，我是一个特别善于观察别人情绪的人。'
# instruct = ''
# tasks.append((speaker, language, text, instruct))

# speaker = "Vivian"
# language = "Chinese"
# text = '其实我真的有发现，我是一个特别善于观察别人情绪的人。'
# instruct = ''
# tasks.append((speaker, language, text, instruct))

# speaker = "Aiden"
# language = "English"
# text = 'Then by the end of the movie, when Dorothy clicks her heels and says, “There’s no place like home,” I got a little bit teary, I’ll admit. You know, I don’t even know why—I just, I just felt.'
# instruct = ''
# tasks.append((speaker, language, text, instruct))

# speaker = "Ryan"
# language = "English"
# text = 'Then by the end of the movie, when Dorothy clicks her heels and says, “There’s no place like home,” I got a little bit teary, I’ll admit. You know, I don’t even know why—I just, I just felt.'
# instruct = ''
# tasks.append((speaker, language, text, instruct))

# speaker = "ono_anna"
# language = "Japanese"
# text = 'やばい、明日のプレゼン資料まだ完成してない… 助けて！'
# instruct = ''
# tasks.append((speaker, language, text, instruct))

# speaker = "Sohee"
# language = "Korean"
# text = '야, 오늘 점심에 뭐 먹을지 생각해 봤어? 근처에 새로 생긴 분식집 어때?'
# instruct = ''
# tasks.append((speaker, language, text, instruct))

# speaker = "Dylan"
# language = "beijing_dialect"
# text = '我们就在山上啊，就是其实也没什么，就是在土坡上跑来跑去，然后谁捡个那个嗯比较威风的棍儿，完了我们就就瞎打，呃要不就是什么掏个洞啊什么的。'
# instruct = ''
# tasks.append((speaker, language, text, instruct))

# speaker = "Eric"
# language = "sichuan_dialect"
# text = '你龟儿太过分了，把我的东西都搞坏了，还晓不晓得认错，硬是要把我整冒火你才安逸嗦，莫再烦老子爬球开。'
# instruct = ''
# tasks.append((speaker, language, text, instruct))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def play_audio(file_path):
    print(f"正在播放 {file_path}...")
    # Use powershell to play audio
    try:
        subprocess.run(["powershell", "-c", f"(New-Object Media.SoundPlayer '{file_path}').PlaySync();"], check=True)
    except Exception as e:
        print(f"播放音频失败: {e}")

import functools
import types

class OfficialCapturer:
    """
    打桩捕获器：拦截官方模型的内部调用以自动提取关键中间变量。
    同时包装高层 API 使其直接返回 TTSResult。
    """
    def __init__(self, tts_wrapper):
        self.tts = tts_wrapper
        self.model = tts_wrapper.model
        self.talker = self.model.talker
        
        # 缓存区
        self.last_codes = None
        self.last_spk_emb = None
        self.last_text_ids = None

        self._patch_all()

    def _patch_all(self):
        # 1. 拦截 model.generate 获取 codes 和 text_ids
        orig_gen = self.model.generate
        @functools.wraps(orig_gen)
        def hooked_gen(inner_self, *args, **kwargs):
            # 自动捕获 input_ids 并提取核心文本 Token
            input_ids = kwargs.get("input_ids") or (args[0] if args else None)
            if input_ids is not None:
                raw_ids = input_ids[0][0].tolist()
                # 官方格式启发式：[4:-5] 对应中间的文本内容
                if len(raw_ids) > 9:
                    self.last_text_ids = raw_ids[4:-5]
                else:
                    self.last_text_ids = raw_ids

            res_codes, res_hiddens = orig_gen(*args, **kwargs)
            if res_codes and len(res_codes) > 0:
                self.last_codes = res_codes[0].detach().cpu().numpy()
            return res_codes, res_hiddens
        self.model.generate = types.MethodType(hooked_gen, self.model)

        # 2. 拦截嵌入层捕获 speaker_embed (针对 CustomVoice 模式)
        talker_emb_layer = self.talker.get_input_embeddings()
        orig_emb_forward = talker_emb_layer.forward
        @functools.wraps(orig_emb_forward)
        def hooked_emb_forward(inner_layer_self, indices):
            emb = orig_emb_forward(indices)
            if self.last_spk_emb is None:
                if indices.dim() == 0 or (indices.dim() == 1 and indices.size(0) == 1):
                    self.last_spk_emb = emb.detach().cpu().to(torch.float32).numpy().flatten()
            return emb
        talker_emb_layer.forward = types.MethodType(hooked_emb_forward, talker_emb_layer)

        # 3. 拦截 talker.forward 捕捉传入的 speaker_embed (针对 Clone/Design 模式)
        orig_talker_forward = self.talker.forward
        @functools.wraps(orig_talker_forward)
        def hooked_talker_forward(inner_self, **kwargs):
            if "speaker_embed" in kwargs:
                spk_emb = kwargs["speaker_embed"]
                if spk_emb is not None:
                    self.last_spk_emb = spk_emb[0].detach().cpu().to(torch.float32).numpy().flatten()
            return orig_talker_forward(**kwargs)
        self.talker.forward = types.MethodType(hooked_talker_forward, self.talker)

        # 4. 包装高层 API，使其直接返回 TTSResult
        methods = ["generate_custom_voice", "generate_voice_design", "generate_voice_clone"]
        for m_name in methods:
            if hasattr(self.tts, m_name):
                orig_m = getattr(self.tts, m_name)
                def make_hooked(original):
                    @functools.wraps(original)
                    def hooked(inner_self, *args, **kwargs):
                        # 重置缓冲区，确保每一轮生成都是独立的，不收历史残留影响
                        self.last_codes = None
                        self.last_spk_emb = None
                        self.last_text_ids = None
                        
                        text = kwargs.get("text") or (args[0] if args else "")
                        
                        # 执行原始推理逻辑 (wavs, sr)
                        wavs, sr = original(*args, **kwargs)
                        
                        # 自动汇总并封装返回
                        return self.pull_result(text, wavs[0])
                    return hooked
                setattr(self.tts, m_name, types.MethodType(make_hooked(orig_m), self.tts))

    def pull_result(self, text, audio_wav):
        """汇总当前捕获的数据并封装为 TTSResult"""
        from qwen3_tts_gguf.inference.result import TTSResult, Timing
        res = TTSResult(
            text=text,
            spk_emb=self.last_spk_emb,
            text_ids=self.last_text_ids if self.last_text_ids is not None else [],
            codes=self.last_codes,
            audio=audio_wav,
            stats=Timing(total_steps=len(self.last_codes) if self.last_codes is not None else 0)
        )
        return res

def main():
    # 使用 GPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 检查模型文件
    if not os.path.exists(MODEL_DIR):
        print(f"错误：找不到模型路径：{MODEL_DIR}")
        return

    print(f"正在从 {MODEL_DIR} 加载模型")
    
    try:
        from qwen3_tts_gguf.inference.result import TTSResult

        print("开始加载模型...")
        # 定义数据类型
        dtype = torch.bfloat16
        
        # 载入模型
        t_load_start = time.time()
        set_seed(47)
        tts = Qwen3TTSModel.from_pretrained(
            MODEL_DIR,
            device_map=device,
            dtype=dtype,
        )
        t_load_end = time.time()
        load_time = t_load_end - t_load_start
        print(f"模型加载完成，耗时 {load_time:.4f} 秒。")

        # --- 初始化自动捕获器 ---
        # 它会自动拦截 tts 的方法并使 generate_* 返回 TTSResult
        capturer = OfficialCapturer(tts)
        
        # 连续生成多个音频
        for i, (speaker, language, text, instruct) in enumerate(tasks):
            print(f"\n--- 正在生成第 {i+1} 个音频 ---")
            
            t_infer_start = time.time()
            # 现在直接返回 TTSResult
            res = tts.generate_custom_voice(
                text=text,
                language=language,
                speaker=speaker,
                instruct=instruct, 
                temperature=0.8, 
                subtalker_temperature=0.8, 
            )
            t_infer_end = time.time()
            infer_time = t_infer_end - t_infer_start
            print(f"推理完成，耗时 {infer_time:.4f} 秒。")

            # --- 保存并播放音频 ---
            output_base = f"output/custom_{i+1}"
            res.save_json(f"{output_base}.json")
            res.save_wav(f"{output_base}.wav")
            
            print(f"音频与元数据已保存至: {output_base}.*")
            play_audio(f"{output_base}.wav")

    except Exception as e:
        print(f"推理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
