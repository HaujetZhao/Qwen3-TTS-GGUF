"""
49-Batch-Memory.py - 批量路数显存/耗时测试
用长短不一 (5~15 字) 的文本，分别以 1/2/4/8 路批量推理，
llama.cpp 的显存分配信息会写入 log/latest.log，事后从 log 中提取。
"""
import time
from qwen3_tts_gguf.inference import TTSEngine, TTSConfig
from qwen3_tts_gguf.inference.batch import BatchRunner
from qwen3_tts_gguf.inference.schema.result import TTSResult

REF_JSON = "output/elaborate/Vivian.json"

# 较长文本: 60 ~ 90 字，长度接近以减少 lockstep 尾部空转
TEXTS = [
    "人工智能的发展速度令人惊叹，从最早只能完成简单规则的棋类程序，到如今能够理解自然语言、生成图像和视频的大模型，短短几十年间，机器的能力边界被一次次推远，而这场变革显然还远未到达终点。",
    "清晨的菜市场总是热闹非凡，摊主们把新鲜的蔬菜水果摆放得整整齐齐，吆喝声此起彼伏，买菜的老人家仔细挑选着每一样食材，讨价还价之间，藏着最真实也最温暖的人间烟火气息。",
    "学习一门新的编程语言，最重要的不是背诵语法规则，而是用它去解决实际的问题，当你亲手写出第一个能跑起来的程序，看到屏幕上输出期待已久的结果时，那种成就感会推着你继续往下走。",
    "山间的徒步路线比想象中艰难，碎石坡陡峭湿滑，我们互相搀扶着往上爬，等终于登上垭口的那一刻，云海在脚下翻涌，远处的雪峰被夕阳染成金色，所有的疲惫在一瞬间都变得值得。",
    "城市的夜景从高空俯瞰别有一番风味，纵横交错的道路像发光的血管，车流缓慢涌动，写字楼里还亮着零星的灯光，每一盏灯背后大概都有 一个还在为生活或者梦想加班的人吧。",
    "做菜其实是一种很治愈的修行，从挑选食材、清洗切配，到掌握火候、调味出锅，每一步都需要耐心和专注，当你端出一盘色香味俱全的菜，看着家人朋友吃得津津有味，心里会涌起踏实的幸福。",
    "读历史最有意思的地方，在于你会发现人性千百年来几乎没有改变，古人面对的贪婪与恐惧、忠诚与背叛，和今天职场里、商场里上演的故事如出一辙，只是换了服装和道具而已。",
    "耳机里放着熟悉的老歌，窗外的雨不紧不慢地下着，这种天气最适合什么都不做，窝在沙发里发呆，让思绪随着旋律飘远，把一周积攒的疲惫和烦躁，都交给这个安静的下午慢慢消化。",
]


def main():
    engine = TTSEngine(model_dir="model-base", onnx_provider="CUDA")
    voice = TTSResult.from_json(REF_JSON)
    runner = BatchRunner(engine, n_ctx_per_seq=1024)
    cfg = TTSConfig(max_steps=400, temperature=0.6, sub_temperature=0.6,
                    seed=42, sub_seed=45, streaming=False)

    for B in [1, 4, 8, 16]:
        tasks = [(TEXTS[i % len(TEXTS)], voice, 'Chinese', False, cfg) for i in range(B)]
        t0 = time.time()
        results = runner.clone_batch(tasks)
        dt = time.time() - t0
        s = results[-1].stats
        frames = [r.codes.shape[0] for r in results]
        audio_s = sum(f for f in frames) / 12.5

        prefill_tokens = runner.last_prefill_tokens
        gen_tokens = sum(frames) * 18   # 每帧每路: Talker 1 + Predictor (2 prefill + 15 步)
        t_gen = s.gen_time
        t_dec = dt - s.prefill_time - t_gen   # 解码墙钟 = 总壁钟 - prefill - 生成 (含全部路)

        print(f"B={B}: 帧数 {frames}, 壁钟 {dt:.2f}s "
              f"(prefill {s.prefill_time*1000:.0f}ms + 生成 {t_gen:.2f}s + 解码 {t_dec:.2f}s)")
        print(f"     prefill {prefill_tokens/s.prefill_time:.0f} tok/s | 生成 {gen_tokens/t_gen:.0f} LLM tok/s | "
              f"生成 RTF {t_gen/audio_s:.3f} | 端到端 RTF {dt/audio_s:.3f}")
        n_frames = max(frames)
        prof = runner.profile
        parts = [f"{k} {v/n_frames*1000:.1f}" for k, v in prof.items()]
        print(f"     每帧剖析(ms): " + " | ".join(parts))
        print("-" * 60)

    engine.shutdown()


if __name__ == "__main__":
    main()
