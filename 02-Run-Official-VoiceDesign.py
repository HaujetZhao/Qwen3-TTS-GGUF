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
from export_config import Models; MODEL_DIR = Models.design.source
from qwen3_tts_gguf.inference.schema.result import TTSResult, Timing
from qwen3_tts_gguf.inference.capturer import OfficialCapturer

ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR / "Qwen3-TTS-main"))
from qwen_tts import Qwen3TTSModel

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


tasks = []

# 声学属性控制

text = \
"""
好了各位，往后退，往后退！我有个天大的好消息要宣布：Qwen-TTS正式开源啦！
"""
instruct = \
"""
采用高亢的男性嗓音，语调随兴奋情绪不断上扬，以快速而充满活力的节奏传达信息。音量要足够响亮，近乎喊叫，以体现紧迫感。发音务必清晰精准、字字分明，让每个词都铿锵有力。整体表达需流畅自然、明亮生动，富有戏剧性，展现出外向、自信且张扬的个性，同时传递出一种威严而宏大的宣告语气，洋溢着满溢的激动之情。
"""
tasks.append((text, instruct))


text = \
"""
Nine different, exciting ways of cooking sausage. Incredible. There were three outstanding deliveries in terms of the sausage being the hero. The first dish that we want to dissect, this individual smartly combined different proteins in their sausage. Great seasoning. The blend was absolutely spot on. Congratulations. Please step forward. Natasha.
"""
instruct = \
"""
gender: Male.
pitch: Low male pitch with significant upward inflections for emphasis and excitement.
speed: Fast-paced delivery with deliberate pauses for dramatic effect.
volume: Loud and projecting, increasing notably during moments of praise and announcements.
age: Young adult to middle-aged adult.
clarity: Highly articulate and distinct pronunciation.
fluency: Very fluent speech with no hesitations.
accent: British English.
texture: Bright and clear vocal texture.
emotion: Enthusiastic and excited, especially when complimenting.
tone: Upbeat, authoritative, and performative.
personality: Confident, extroverted, and engaging.
"""
tasks.append((text, instruct))


text = \
"""
皇上啊！臣妾一片真心可昭日月，为何您竟信那毒妇谗言，将我打入冷宫？这心……比雪还凉啊……
"""
instruct = \
"""
展现出悲苦沙哑的声音质感,语速偏慢,情绪浓烈且带有哭腔,以标准普通话缓慢诉说,情感强烈,语调哀怨高亢,音高起伏大。
"""
tasks.append((text, instruct))


text = \
"""
Good one. Okay, fine, I'm just gonna leave this sock monkey here. Goodbye.
"""
instruct = \
"""
gender: Male.
pitch: Artificially high-pitched, slightly lowering after the initial laugh.
speed: Rapid during the laugh, then slowing to a deliberate pace.
volume: Loud laugh transitioning to a standard conversational level.
age: Young adult to middle-aged, performing a character voice.
clarity: Clear and distinct articulation.
fluency: Fluent delivery without hesitation.
accent: American English.
texture: Slightly strained and somewhat nasal quality.
emotion: Forced amusement shifting to feigned resignation.
tone: Initially playful, then shifts to a slightly put-upon tone.
personality: Theatrical and expressive.
"""
tasks.append((text, instruct))


text = \
"""
哥哥，你回来啦，人家等了你好久好久了，要抱抱！
"""
instruct = \
"""
体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显，营造出黏人、做作又刻意卖萌的听觉效果。
"""
tasks.append((text, instruct))


text = \
"""
Blah, blah, blah. We're all very fascinated, Whitey, but we'd like to get paid.
"""
instruct = \
"""
Speak as a sarcastic, assertive teenage girl: crisp enunciation, controlled volume, with vocal emphasis that conveys disdain and authority.
"""
tasks.append((text, instruct))


# 年龄控制

text = \
"""
把你所有的表情都藏在面具里，保持你的中性状态，不用表情，只用身体的语言，要记住，要学会藏。
"""
instruct = \
"""
性别: 男性.
音高: 男性低沉音域，音高稳定.
语速: 语速稍快，节奏紧凑.
音量: 音量洪亮，力度强劲.
年龄: 中老年.
清晰度: 发音清晰，字句有力.
流畅度: 表达流畅，一气呵成.
口音: 标准普通话.
音色质感: 嗓音浑厚，略带沙哑感.
情绪: 严肃告诫，指令明确.
语调: 命令式语调，强调果断.
性格: 权威果断，不容置喙.
"""
tasks.append((text, instruct))


text = \
"""
Older gentleman, 110, maybe 111 years old, sort of a surly Elvis thing happening with him. He smiles like this. Seen him around?
"""
instruct = \
"""
gender: Male.
pitch: Low male pitch, generally stable.
speed: Deliberate pace, slowing slightly after the initial exclamation.
volume: Starts loud, then transitions to a projected conversational volume.
age: Middle-aged adult.
clarity: High clarity with distinct pronunciation.
fluency: Highly fluent.
accent: American English.
texture: Resonant and slightly gravelly.
emotion: Initially commanding, shifting to narrative amusement.
tone: Authoritative start, moving to an engaging, descriptive tone.
personality: Confident and performative.
"""
tasks.append((text, instruct))


# 渐变控制

text = \
"""
你在干什么?有什么好看的?喂!我叫你走，你在干什么?给我走啊!
"""
instruct = \
"""
性别: 男性
音高: 男性低沉音区，偶有拔高.
语速: 初始平稳，后段因激动逐渐加快.
音量: 初始音量正常，后段逐渐提高至喊叫.
年龄: 中年男性.
清晰度: 吐字清晰，发音准确.
流畅度: 言语连贯，表达自然.
口音: 标准普通话发音.
音色质感: 音质略带粗砺，富有力量感.
情绪: 初始不耐烦，迅速转为恼怒斥责.
语调: 质问命令式，语带不悦与威慑.
性格: 急躁易怒，态度强硬.
"""
tasks.append((text, instruct))


text = \
"""
Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you-
"""
instruct = \
"""
gender: Female.
pitch: Mid-range female pitch, rising sharply with frustration.
speed: Starts measured, then accelerates rapidly during emotional outburst.
volume: Begins conversational, escalates quickly to loud and forceful.
age: Young adult to middle-aged.
clarity: High clarity and distinct articulation throughout.
fluency: Highly fluent with no significant pauses or fillers.
accent: General American English.
texture: Bright and clear vocal quality.
emotion: Shifts abruptly from neutral acceptance to intense resentment and anger.
tone: Initially accepting, becomes sharply accusatory and confrontational.
personality: Assertive and emotionally expressive when provoked.
"""
tasks.append((text, instruct))


# 拟人感

text = \
"""
我跟我闺蜜看电影就特别有画面感。就你知道吗，一紧张我就忍不住吃爆米花，吃得特别快，然后手里那杯可乐也跟着晃，差点就洒了，真的差一点点。然后旁边那个人就突然来一句，嘘——声音压得特别低。哎我当下那个情绪，既想笑又有点气，太尴尬了。
"""
instruct = \
"""
自然感的女声，语调活泼带笑意，模仿别人‘嘘’你时压低嗓音，就是平时聊天的感觉
"""
tasks.append((text, instruct))


text = \
"""
Yeah, so—uh—I’m a digital nomad, right? So… pretty much all my communication is just, like, texts and messages. And now, you know, there’s these AI agents that can, uh… reply for you? Which is—heh—convenient, sure, I guess? But also… kinda delicate, you know?
Like, you’ll type something super short—like, “Yep, sounds good”—and it’ll turn that into this whole… warm, polished paragraph. Like, way nicer than I’d ever write myself. huh… ha Seriously, I sound like a Hallmark card all of a sudden.
But then… once you outsource that… what’s the other person actually hearing? Are they hearing me… or just some… generic, friendly-bot voice? Man, that’s weird to even say out loud.
"""
instruct = \
"""
A relaxed, naturally expressive male voice in his late twenties to early thirties, with a moderately low pitch, casual speaking rate, and conversational volume; deliver lines with a light, self-deprecating tone, breaking into genuine, easygoing laughter at moments of embarrassment, while maintaining clear articulation and an overall warm, approachable clarity.
"""
tasks.append((text, instruct))


# 背景信息

text = \
"""
有些事，只要国家需要，就得有人扛起来。
我们那一代人，是背着泥土铺路的；
你们要做的，是让这条路，通向星辰大海。
"""
instruct = \
"""
角色姓名：林怀岳
音色信息：音量洪亮，音域低沉，力度感强的中年男性声音。
身份背景：某国家重点科研项目首席顾问，年近七十的资深战略科学家。曾参与国家重大科技攻关工程，历经数十年风雨，见证了从落后追赶到自主创新的艰难历程。现任国家科技咨询委员会终身荣誉委员，仍坚持在一线培养青年人才，为国家战略发展建言献策。
外貌特征：身形挺拔，两鬓斑白，眉宇间刻着岁月沉淀的坚毅。常着深色中山装或简洁正装，眼神沉静而锐利，举手投足间自带威严与从容。
性格特质：意志如钢，信念坚定，面对挑战从不退缩；胸怀家国，心系民族未来，将个人命运与国家兴衰紧密相连；严谨自律，言出必行，话语中充满责任感与历史担当；外冷内热，表面严肃，实则对后辈寄予厚望，甘为人梯。
人生信条：“我们这一代人，不是为了站在光里，而是为了把路铺到光里。”
"""
tasks.append((text, instruct))


text = \
"""
Lot being you watching. 1-866-IDLE-03 for JPL. That's 1-866-436-5703. Or text the word VOTE to 5703. Diana DeGarmo's next with more from the movies right after this brief intermission on American Idol.
"""
instruct = \
"""
Character Name: Marcus Cole
Voice Profile: A bright, agile male voice with a natural upward lift, delivering lines at a brisk, energetic pace. Pitch leans high with spark, volume projects clearly—near-shouting at peaks—to convey urgency and excitement. Speech flows seamlessly, fluently, each word sharply defined, riding a current of dynamic rhythm.
Background: Longtime broadcast booth announcer for national television, specializing in live interstitials and public engagement spots. His voice bridges segments, rallies action, and keeps momentum alive—from voter drives to entertainment news.
Presence: Late 50s, neatly groomed, dressed in a crisp shirt under studio lights. Moves with practiced ease, eyes locked on the script, energy coiled and ready.
Personality: Energetic, precise, inherently engaging. He doesn’t just read—he propels. Behind the speed is intent: to inform fast, to move people to act. Whether it’s “text VOTE to 5703” or a star-studded tease, he makes it feel immediate, vital.
"""
tasks.append((text, instruct))






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
        capturer = OfficialCapturer(tts)
        
        # 连续生成多个音频
        for i, (text, instruct) in enumerate(tasks):
            print(f"\n--- [Voice Design] 正在生成第 {i+1} 个音频 ---")
            t_infer_start = time.time()
            # 现在直接返回 TTSResult
            res = tts.generate_voice_design(
                text=text,
                instruct=instruct,
                temperature=0.8, 
                subtalker_temperature=0.8, 
            )
            t_infer_end = time.time()
            infer_time = t_infer_end - t_infer_start
            print(f"推理完成，耗时 {infer_time:.4f} 秒。")

            # --- 保存并播放音频 ---
            output_base = f"output/design_{i+1}"
            res.save_json(f"{output_base}.json")
            res.save_wav(f"{output_base}.wav")
            print(f"音频与元数据已保存至: {output_base}.*")
            res.play()
            
    except Exception as e:
        print(f"推理失败: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
