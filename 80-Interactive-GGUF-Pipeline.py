import os
import ctypes
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
import onnxruntime as ort
import time
from transformers import AutoTokenizer
import qwen3_tts_gguf.nano_llama as nano_llama
from qwen3_tts_gguf import logger

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Qwen3TTS:
    """
    交互式 Qwen3-TTS GGUF 合成引擎
    设计目标: 在 ipynb 中持久化运行，避免重复加载大型模型。
    """
    def __init__(self, model_root="model", tokenizer_path="Qwen3-TTS-12Hz-1.7B-CustomVoice"):
        self.project_root = os.getcwd()
        self.model_dir = os.path.join(self.project_root, model_root)
        self.tokenizer_path = os.path.join(self.project_root, tokenizer_path)
        
        # 路径定义
        self.paths = {
            "master_gguf": os.path.join(self.model_dir, "qwen3_tts_talker.gguf"),
            "craftsman_gguf": os.path.join(self.model_dir, "qwen3_tts_craftsman_advanced.gguf"),
            "mouth_onnx": os.path.join(self.model_dir, "qwen3_tts_decoder.onnx"),
            "master_head": os.path.join(self.model_dir, "codec_head_weight.npy"),
            "text_table": os.path.join(self.model_dir, "text_embedding_projected.npy"),
            "proj_pt": os.path.join(self.model_dir, "craftsman_hf/master_to_craftsman_proj.pt")
        }
        
        print("[Engine] 正在启动初始化流程...")
        self.load_assets()
        self.init_engines()
        
    def load_assets(self):
        """加载权重表与 Tokenizer"""
        print("  - 加载权重表与 Tokenizer...")
        self.assets = {
            "master_head": np.load(self.paths["master_head"]),
            "text_table": np.load(self.paths["text_table"]),
            "emb_tables": [np.load(os.path.join(self.model_dir, f"codec_embedding_{i}.npy")) for i in range(16)],
            "proj": torch.load(self.paths["proj_pt"], map_location="cpu")
        }
        self.assets["tts_pad"] = self.assets["text_table"][151671]
        
        # 预投影加速
        proj_w = self.assets["proj"]["weight"].float()
        proj_b = self.assets["proj"]["bias"].float()
        self.assets["emb_tables_1024"] = [
            F.linear(torch.from_numpy(t).float(), proj_w, proj_b).numpy() for t in self.assets["emb_tables"]
        ]
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True, fix_mistral_regex=True)
        print("  ✅ 资产加载完成。")

    def init_engines(self):
        """初始化 GGUF 与 ONNX 引擎"""
        print("  - 正在通过 Vulkan 挂载 GPU 引擎...")
        self.m_model = nano_llama.load_model(self.paths["master_gguf"], n_gpu_layers=-1)
        self.c_model = nano_llama.load_model(self.paths["craftsman_gguf"], n_gpu_layers=-1)
        
        # 上下文初始化 (持久化)
        m_params = nano_llama.llama_context_default_params()
        m_params.n_ctx = 4096
        m_params.embeddings = True
        self.m_ctx = nano_llama.llama_init_from_model(self.m_model, m_params)
        
        # 初始化工匠上下文 (工匠仅需 Logits，无需 Embeddings 反馈，关闭以节省内存并消除告警)
        c_params = nano_llama.llama_context_default_params()
        c_params.n_ctx = 512
        c_params.embeddings = False
        self.c_ctx = nano_llama.llama_init_from_model(self.c_model, c_params)
        
        # 口腔解码器
        self.mouth_sess = ort.InferenceSession(self.paths["mouth_onnx"], providers=['CPUExecutionProvider'])
        
        # Batch 初始化
        self.m_batch = nano_llama.llama_batch_init(4096, 2048, 1)
        self.c_batch = nano_llama.llama_batch_init(32, 1024, 1)
        print("  ✅ 引擎初始化成功。环境已就绪。")

    def synthesize(self, text, speaker_id=3065, max_steps=250, verbose=False):
        """
        全动态合成入口
        text: 目标文本
        speaker_id: 3065(Vivian), 3010(傅叔)
        verbose: 是否显示详细组件耗时
        """
        if verbose: print(f"\n[Synthesizer] 目标文本: {text}")
        start_time = time.time()
        
        # 1. 编译 Prompt
        p_start = time.time()
        prompt_embeds = self._construct_prompt(text, speaker_id)
        p_time = time.time() - p_start
        
        # 2. 推理
        inf_start = time.time()
        all_codes, perf_stats = self._execute_inference(prompt_embeds, max_steps, verbose)
        inf_time = time.time() - inf_start
        
        # 3. 渲染
        r_start = time.time()
        audio_data = self._render_audio(all_codes)
        r_time = time.time() - r_start
        
        total_time = time.time() - start_time
        audio_dur = len(audio_data) / 24000.0
        rtf = total_time / audio_dur
        
        if verbose:
            print("-" * 40)
            print(f"性能分析报告 (音频长度: {audio_dur:.2f}s)")
            print(f"  1. Prompt 编译:   {p_time:.4f}s")
            print(f"  2. 大师 Prefill:  {perf_stats['prefill_time']:.4f}s")
            print(f"  3. 自回环总计:    {perf_stats['loop_time']:.4f}s")
            print(f"     └─ 大师 (Master):    {perf_stats['master_time']:.4f}s")
            print(f"     └─ 工匠 (Craftsman): {perf_stats['craftsman_time']:.4f}s")
            print(f"     └─ 反馈 (Feedback):  {perf_stats['feedback_time']:.4f}s")
            print(f"  4. 嘴巴渲染 (Mouth): {r_time:.4f}s")
            print("-" * 40)
            print(f"总端到端耗时: {total_time:.4f}s | RTF: {rtf:.4f}")
        else:
            print(f"[Done] {text[:10]}... | RTF: {rtf:.4f}")
            
        return audio_data

    # --- 内部组件 ---

    def _construct_prompt(self, text, spk_id):
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        # 固定头
        seq = [ (151644, 0), (77091, 0), (198, 0), (151671, 2154), (151671, 2156), (151671, 2055), (151671, 2157), (151671, spk_id), (151672, 2148) ]
        for tid in ids: seq.append((tid, 2148))
        seq.append((151673, 2148))
        seq.append((151671, 2149))
        
        embeds = []
        for tid, cid in seq:
            v = self.assets["text_table"][tid] + (self.assets["emb_tables"][0][cid] if cid != 0 else 0)
            embeds.append(v)
        return np.array(embeds).reshape(1, len(seq), 2048).astype(np.float32)

    def _execute_inference(self, prompt, max_steps, verbose=False):
        # 清理大师记忆，确保每一轮合成都是独立的，防止上下文污染
        nano_llama.llama_memory_clear(nano_llama.llama_get_memory(self.m_ctx), True)
        
        stats = {"master_time": 0, "craftsman_time": 0, "feedback_time": 0}
        
        # Prefill Master
        pre_start = time.time()
        n_p = prompt.shape[1]
        self.m_batch.n_tokens = n_p
        ctypes.memmove(self.m_batch.embd, np.ascontiguousarray(prompt[0]).ctypes.data, prompt[0].nbytes)
        for i in range(n_p):
            self.m_batch.pos[i], self.m_batch.pos[n_p+i], self.m_batch.pos[2*n_p+i], self.m_batch.pos[3*n_p+i] = i, i, i, 0
            self.m_batch.n_seq_id[i], self.m_batch.seq_id[i][0], self.m_batch.logits[i] = 1, 0, 1 # 全部标记输出以消除告警
        nano_llama.llama_decode(self.m_ctx, self.m_batch)
        
        m_hidden = np.ctypeslib.as_array(nano_llama.llama_get_embeddings(self.m_ctx), shape=(n_p, 2048))[-1].copy()
        cur_pos, all_codes = n_p, []
        stats["prefill_time"] = time.time() - pre_start
        
        # Loop
        loop_start = time.time()
        for step_idx in range(max_steps):
            # 1. 大师预测
            m_s = time.time()
            code_0 = np.argmax(m_hidden @ self.assets["master_head"].T)
            stats["master_time"] += (time.time() - m_s)
            

            logger.info(f'{code_0=}')
            if code_0 == 2150: 
                if verbose: print(f"  └─ 步数 {step_idx}: 获得 EOS 信号，结束生成。")
                break # 正常结束
            
            # Craftsman (15 steps)
            c_s = time.time()
            step_codes, step_emb_2048 = [code_0], [self.assets["emb_tables"][0][code_0].copy()]
            proj_assets = self.assets["proj"]
            m_h_1024 = m_hidden @ proj_assets["weight"].float().numpy().T + proj_assets["bias"].float().numpy()
            c_in = np.stack([m_h_1024, self.assets["emb_tables_1024"][0][code_0]], axis=0)
            
            nano_llama.llama_memory_clear(nano_llama.llama_get_memory(self.c_ctx), True)
            self.c_batch.n_tokens = 2
            ctypes.memmove(self.c_batch.embd, c_in.ctypes.data, c_in.nbytes)
            for j in range(2):
                self.c_batch.pos[j], self.c_batch.n_seq_id[j], self.c_batch.seq_id[j][0], self.c_batch.logits[j] = j, 1, 0, (1 if j == 1 else 0)
            nano_llama.llama_decode(self.c_ctx, self.c_batch)
            
            # 由于 embeddings=False 且仅 logits[1]=1，此处仅返回 1 组 Logits
            last_logits = np.ctypeslib.as_array(nano_llama.llama_get_logits(self.c_ctx), shape=(1, 30720))[0]
            for cs in range(1, 16):
                c = np.argmax(last_logits[(cs-1)*2048 : (cs-1)*2048 + 2048])
                step_codes.append(c)
                step_emb_2048.append(self.assets["emb_tables"][cs][c].copy())
                if cs < 15:
                    self.c_batch.n_tokens, self.c_batch.pos[0], self.c_batch.logits[0] = 1, cs+1, 1
                    ctypes.memmove(self.c_batch.embd, self.assets["emb_tables_1024"][cs][c].ctypes.data, 4096)
                    nano_llama.llama_decode(self.c_ctx, self.c_batch)
                    last_logits = np.ctypeslib.as_array(nano_llama.llama_get_logits(self.c_ctx), shape=(30720,))
            stats["craftsman_time"] += (time.time() - c_s)
            
            all_codes.append(step_codes)
            
            # Feedback to Master
            f_s = time.time()
            summed = np.sum(step_emb_2048, axis=0) + self.assets["tts_pad"].flatten()
            self.m_batch.n_tokens = 1
            ctypes.memmove(self.m_batch.embd, summed.ctypes.data, summed.nbytes)
            self.m_batch.pos[0] = self.m_batch.pos[1] = self.m_batch.pos[2] = cur_pos
            self.m_batch.pos[3], self.m_batch.logits[0], cur_pos = 0, 1, cur_pos + 1
            nano_llama.llama_decode(self.m_ctx, self.m_batch)
            m_hidden = np.ctypeslib.as_array(nano_llama.llama_get_embeddings(self.m_ctx), shape=(1, 2048))[0].copy()
            stats["feedback_time"] += (time.time() - f_s)
        else:
            print(f"  ⚠️ 熔断预警: 推理达到上限 {max_steps} 步仍未停止，已强行熔断。")
            
        stats["loop_time"] = time.time() - loop_start
        return all_codes, stats

    def _render_audio(self, codes):
        if not codes: return np.array([])
        c_in = np.array(codes)[np.newaxis, ...].astype(np.int64)
        return self.mouth_sess.run(None, {'audio_codes': c_in})[0].squeeze()

    def __del__(self):
        # 释放资源
        try:
            nano_llama.llama_batch_free(self.m_batch)
            nano_llama.llama_batch_free(self.c_batch)
            nano_llama.llama_free(self.m_ctx)
            nano_llama.llama_free(self.c_ctx)
            nano_llama.llama_model_free(self.m_model)
            nano_llama.llama_model_free(self.c_model)
        except: pass

# =========================================================================
# 交互式测试区域 (用户可自由修改的部分)
# =========================================================================
if __name__ == "__main__":
    # 1. 载入引擎 (仅需执行一次)
    tts = Qwen3TTS()
    
    # 2. 合成实验
    TARGET_TEXT = "今天天气好"
    SPEAKER_ID = 3065  # 3065: Vivian, 3010: 傅叔
    MAX_STEPS_LIMIT = 250 # 熔断参数：最大推理步数
    wav = tts.synthesize(TARGET_TEXT, speaker_id=SPEAKER_ID, max_steps=MAX_STEPS_LIMIT, verbose=True)
    sf.write("output/interactive_test1.wav", wav, 24000)

    print(f"✅ 生成成功")
