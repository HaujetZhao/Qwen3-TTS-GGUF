"""
Qwen3-TTS CustomVoice 模型，内置音色合成

同一个音色，每一次生成，会因随机种子、文本的不同，而略有不同，无法稳定

但是合成的 TTSResult 可保存为 json 或 wav，供 Base 模型用于克隆，可保持稳定的音色
"""
import time
import os
import numpy as np
from qwen3_tts_gguf.inference import TTSEngine, TTSConfig, TTSResult




tasks = []

# 单属性控制

speaker = "Ryan"
language = None
text = 'She said she would be here by noon.'
instruct = 'spoke with a very sad and tearful voice.'
tasks.append((speaker, language, text, instruct))

# speaker = "Ryan"
# language = None
# text = 'She said she would be here by noon.'
# instruct = 'Very happy.'
# tasks.append((speaker, language, text, instruct))

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



#多属性控制

speaker = "Vivian"
language = None
text = '就算你自己不想治，你也得考虑考虑别人的感受吧。我们这些朋友的感受你不在乎无所谓，那你家人呢？你家人的感受你难道一点都不在乎吗！'
instruct = \
"""
性别: 女性声音.
音高: 女性中高音区，语调富于变化.
语速: 语速明快，偶有加速.
音量: 正常交谈音量，笑声响亮.
清晰度: 吐字清晰，发音标准.
流畅度: 表达流畅自如.
口音: 普通话.
音色质感: 音色明亮，略带爽朗.
情绪: 愉悦友好，伴随爽朗笑意.
语调: 语调上扬活泼，疑问时尤为明显.
性格: 外向开朗，热情健谈.
"""
tasks.append((speaker, language, text, instruct))

speaker = "Vivian"
language = None
text = '就算你自己不想治，你也得考虑考虑别人的感受吧。我们这些朋友的感受你不在乎无所谓，那你家人呢？你家人的感受你难道一点都不在乎吗！'
instruct = """以极度悲伤、带着明显哭腔的语气，用较小的音量缓缓诉说，语速缓慢，仿佛每一个字都承载着沉重的痛楚，声音颤抖而压抑，吐字虽轻却清晰可辨，透出深藏心底的哀伤与无助。"""
tasks.append((speaker, language, text, instruct))

# speaker = "Vivian"
# language = None
# text = '就算你自己不想治，你也得考虑考虑别人的感受吧。我们这些朋友的感受你不在乎无所谓，那你家人呢？你家人的感受你难道一点都不在乎吗！'
# instruct = """保持青年女性的声线特征，展现出一种清亮且略具紧迫感的音色，语速从平稳开始在叙述过程中逐渐加快，音量在情绪波动时增加，语调在句末调高以强调劝告的语气。"""
# tasks.append((speaker, language, text, instruct))


# 单人多语泛化

speaker = "Vivian"
language = "Korean"
text = '안녕하세요, 오늘은 어떤 용건입니까?'
instruct = """在语速偏快的情况下流畅自然地表达,音质清亮,音调略高,吐字清晰标准,给人一种开心愉悦的感觉。"""
tasks.append((speaker, language, text, instruct))

speaker = "Vivian"
language = "Japanese"
text = 'こんにちは、本日はどのようなご用件でしょうか？'
instruct = """A deep, rich, and solid vocal register characteristic of a middle-aged woman, with full and powerful volume. Speech is delivered at a steady pace, articulation clear and precise, with fluent and confident intonation that rises slightly at the end of sentences."""
tasks.append((speaker, language, text, instruct))

speaker = "Vivian"
language = "sichuan_dialect"
text = '我早就该下班了，就是跟你说我这事情干不完，我现在走不脱。'
instruct = """语音应表现为直率且略显主观强势的中年女性,音色略带尖锐感,流畅表达中偶尔断句以凸显语气,情绪略带不满,音量随情感激动略有增强。"""
tasks.append((speaker, language, text, instruct))





# 9个音色

speaker = "Serena"
language = "Chinese"
text = '其实我真的有发现，我是一个特别善于观察别人情绪的人。'
instruct = ''
tasks.append((speaker, language, text, instruct))

speaker = "uncle_fu"
language = "Chinese"
text = '其实我真的有发现，我是一个特别善于观察别人情绪的人。'
instruct = ''
tasks.append((speaker, language, text, instruct))

speaker = "Vivian"
language = "Chinese"
text = '其实我真的有发现，我是一个特别善于观察别人情绪的人。'
instruct = ''
tasks.append((speaker, language, text, instruct))

speaker = "Aiden"
language = "English"
text = 'Then by the end of the movie, when Dorothy clicks her heels and says, “There’s no place like home,” I got a little bit teary, I’ll admit. You know, I don’t even know why—I just, I just felt.'
instruct = ''
tasks.append((speaker, language, text, instruct))

speaker = "Ryan"
language = "English"
text = 'Then by the end of the movie, when Dorothy clicks her heels and says, “There’s no place like home,” I got a little bit teary, I’ll admit. You know, I don’t even know why—I just, I just felt.'
instruct = ''
tasks.append((speaker, language, text, instruct))

speaker = "ono_anna"
language = "Japanese"
text = 'やばい、明日のプレゼン資料まだ完成してない… 助けて！'
instruct = ''
tasks.append((speaker, language, text, instruct))

speaker = "Sohee"
language = "Korean"
text = '야, 오늘 점심에 뭐 먹을지 생각해 봤어? 근처에 새로 생긴 분식집 어때?'
instruct = ''
tasks.append((speaker, language, text, instruct))

speaker = "Dylan"
language = "beijing_dialect"
text = '我们就在山上啊，就是其实也没什么，就是在土坡上跑来跑去，然后谁捡个那个嗯比较威风的棍儿，完了我们就就瞎打，呃要不就是什么掏个洞啊什么的。'
instruct = ''
tasks.append((speaker, language, text, instruct))

speaker = "Eric"
language = "sichuan_dialect"
text = '你龟儿太过分了，把我的东西都搞坏了，还晓不晓得认错，硬是要把我整冒火你才安逸嗦，莫再烦老子爬球开。'
instruct = ''
tasks.append((speaker, language, text, instruct))





def main():
    
    # 1. 初始化引擎
    print(f"🚀 [Custom-Inference] 正在初始化引擎")
    engine = TTSEngine(model_dir="model-custom")
    stream = engine.create_stream()
    
    # 确保输出目录存在
    os.makedirs("./output/custom", exist_ok=True)

    config = TTSConfig(
        max_steps=400, 
        temperature=0.6, 
        sub_temperature=0.6, 
        seed=42, 
        sub_seed=45,
        streaming=True,
    )

    # 遍历任务
    for i, (speaker, language, text, instruct) in enumerate(tasks):
        print(f"\n🎭 [{i+1}/{len(tasks)}] 正在合成: {speaker}")
        print(f"   文本: {text}")
        if instruct:
            print(f"   提示: {instruct}")

        
        # 如果 language 为 None，则由模型自适应或设为默认
        lang = language if language else 'chinese'
        
        result = stream.custom(
            text=text,
            speaker=speaker,
            instruct=instruct,
            language=lang,
            config=config, 
        )
        print(f"✅ 合成成功！")
        
        # 保存结果
        # 取前10个字并清理非法字符以用于文件名
        text_prefix = "".join(c for c in text.strip() if c not in r'<>:"/\|?*')[:20].strip()
        save_path = f"./output/custom/{i+1:02d}_{speaker}_{text_prefix}"
        result.save(f"{save_path}.wav")
        result.save(f"{save_path}.json")
        print(f"💾 已保存至: {save_path}.wav / .json")

        result.print_stats()
        stream.join()

    engine.shutdown()

if __name__ == "__main__":
    main()
