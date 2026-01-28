import os
import ctypes
import time
import numpy as np
from multiprocessing import Process, Queue
from transformers import AutoTokenizer
from . import llama
from .workers import decoder_worker_proc, speaker_worker_proc

class Qwen3TTSDoubleStreamEngine:
    """
    可导出的 Qwen3-TTS 三级流水线推理引擎。
    支持在 Jupyter Notebook 中导入使用。
    """
    SPEAKER_MAP = {
        "vivian": 3065, "serena": 3066, "uncle_fu": 3010, "ryan": 3061,
        "aiden": 2861, "ono_anna": 2873, "sohee": 2864, "eric": 2875, "dylan": 2878
    }
    LANGUAGE_MAP = {
        "chinese": 2055, "english": 2050, "japanese": 2058, "korean": 2064,
        "beijing_dialect": 2074, "sichuan_dialect": 2062, "auto": 2055
    }
    PROTOCOL = {
        "PAD": 2148, "BOS": 2149, "EOS": 2150, "BOS_TOKEN": 151672, "EOS_TOKEN": 151673,
        "THINK": 2154, "NOTHINK": 2155, "THINK_BOS": 2156, "THINK_EOS": 2157
    }

    def __init__(self, model_root="model", tokenizer_path="Qwen3-TTS-12Hz-1.7B-CustomVoice"):
        self.project_root = os.getcwd()
        self.model_dir = os.path.join(self.project_root, model_root)
        self.tokenizer_path = os.path.join(self.project_root, tokenizer_path)
        
        self.paths = {
            "master_gguf": os.path.join(self.model_dir, "qwen3_tts_talker.gguf"),
            "craftsman_gguf": os.path.join(self.model_dir, "qwen3_tts_craftsman.gguf"),
            "mouth_onnx": os.path.join(self.model_dir, "qwen3_tts_decoder.onnx"),
            "text_table": os.path.join(self.model_dir, "text_embedding_projected.npy"),
            "proj_w": os.path.join(self.model_dir, "proj_weight.npy"),
            "proj_b": os.path.join(self.model_dir, "proj_bias.npy")
        }
        
        self.load_assets()
        self.init_engines()
        
        # 启动后台流水线
        self.codes_q = Queue()
        self.pcm_q = Queue()
        self.record_q = Queue() # 新增：录制队列
        
        from .workers import wav_writer_proc
        self.dec_p = Process(target=decoder_worker_proc, args=(self.codes_q, self.pcm_q, self.paths["mouth_onnx"], self.record_q))
        self.spk_p = Process(target=speaker_worker_proc, args=(self.pcm_q,))
        self.wav_p = Process(target=wav_writer_proc, args=(self.record_q, os.path.join(self.project_root, "output/stream_debug.wav")))
        
        self.dec_p.daemon = self.spk_p.daemon = self.wav_p.daemon = True
        self.dec_p.start()
        self.spk_p.start()
        self.wav_p.start()

    def load_assets(self):
        self.assets = {
            "text_table": np.load(self.paths["text_table"]),
            "proj": {"weight": np.load(self.paths["proj_w"]), "bias": np.load(self.paths["proj_b"])},
            "emb_tables": [], "emb_tables_1024": []
        }
        for i in range(16):
            t = np.load(os.path.join(self.model_dir, f"codec_embedding_{i}.npy"))
            self.assets["emb_tables"].append(t)
            self.assets["emb_tables_1024"].append(t @ self.assets["proj"]["weight"].T + self.assets["proj"]["bias"])
        self.assets["tts_pad"] = self.assets["text_table"][151671]
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True, fix_mistral_regex=True)

    def init_engines(self):
        self.m_model = llama.load_model(self.paths["master_gguf"], n_gpu_layers=-1)
        self.c_model = llama.load_model(self.paths["craftsman_gguf"], n_gpu_layers=-1)
        
        m_params = llama.llama_context_default_params()
        m_params.n_ctx = 4096
        m_params.embeddings = True
        self.m_ctx = llama.llama_init_from_model(self.m_model, m_params)
        self.c_ctx = llama.llama_init_from_model(self.c_model, llama.llama_context_default_params())
        
        self.m_batch = llama.llama_batch_init(4096, 2048, 1)
        self.c_batch = llama.llama_batch_init(32, 1024, 1)

    def _sample(self, logits, temperature=0.9):
        if temperature <= 1e-5: return np.argmax(logits)
        logits /= temperature
        probs = np.exp(logits - np.max(logits))
        probs /= probs.sum()
        return np.random.choice(len(probs), p=probs)

    def synthesize(self, text, speaker_id=3066, language=2055, chunk_size=12, verbose=True):
        if isinstance(language, str): language = self.LANGUAGE_MAP.get(language.lower(), 2055)
        if isinstance(speaker_id, str): speaker_id = self.SPEAKER_MAP.get(speaker_id.lower(), 3066)

        ids = self.tokenizer.encode(text, add_special_tokens=False)
        p = self.PROTOCOL
        seq = [(151644, 0), (77091, 0), (198, 0), (151671, p["THINK"]), (151671, p["THINK_BOS"]), (151671, language), (151671, p["THINK_EOS"]), (151671, speaker_id), (p["BOS_TOKEN"], p["PAD"])]
        for tid in ids: seq.append((tid, p["PAD"]))
        seq.append((p["EOS_TOKEN"], p["PAD"]))
        seq.append((151671, 2149))
        
        prompt_embeds = np.array([self.assets["text_table"][tid] + (self.assets["emb_tables"][0][cid] if cid != 0 else 0) for tid, cid in seq]).reshape(1, len(seq), 2048).astype(np.float32)
        
        llama.llama_memory_clear(llama.llama_get_memory(self.m_ctx), True)
        self.m_batch.n_tokens = len(seq)
        ctypes.memmove(self.m_batch.embd, prompt_embeds.ctypes.data, prompt_embeds.nbytes)
        for i in range(len(seq)):
            self.m_batch.pos[i] = self.m_batch.pos[len(seq)+i] = self.m_batch.pos[2*len(seq)+i] = i
            self.m_batch.pos[3*len(seq)+i] = 0
            self.m_batch.n_seq_id[i], self.m_batch.seq_id[i][0], self.m_batch.logits[i] = 1, 0, 1
        llama.llama_decode(self.m_ctx, self.m_batch)
        
        m_hidden = np.ctypeslib.as_array(llama.llama_get_embeddings(self.m_ctx), shape=(len(seq), 2048))[-1].copy()
        m_logits = np.ctypeslib.as_array(llama.llama_get_logits(self.m_ctx), shape=(len(seq), 3072))[-1].copy()
        
        cur_pos, step_buffer = len(seq), []
        
        for step_idx in range(400):
            code_0 = self._sample(m_logits)
            if code_0 == 2150: break
            
            # Craftsman (略去详细计算，同 44 号逻辑)
            step_codes = [code_0]
            step_emb_2048 = [self.assets["emb_tables"][0][code_0].copy()]
            proj = self.assets["proj"]
            m_h_1024 = m_hidden @ proj["weight"].T + proj["bias"]
            c_in = np.stack([m_h_1024, self.assets["emb_tables_1024"][0][code_0]], axis=0)
            
            llama.llama_memory_clear(llama.llama_get_memory(self.c_ctx), True)
            self.c_batch.n_tokens = 2
            ctypes.memmove(self.c_batch.embd, c_in.ctypes.data, c_in.nbytes)
            for j in range(2):
                self.c_batch.pos[j], self.c_batch.n_seq_id[j], self.c_batch.seq_id[j][0], self.c_batch.logits[j] = j, 1, 0, (1 if j == 1 else 0)
            llama.llama_decode(self.c_ctx, self.c_batch)
            last_logits = np.ctypeslib.as_array(llama.llama_get_logits(self.c_ctx), shape=(1, 30720))[0]
            for cs in range(1, 16):
                c = self._sample(last_logits[(cs-1)*2048 : cs*2048])
                step_codes.append(c)
                step_emb_2048.append(self.assets["emb_tables"][cs][c].copy())
                if cs < 15:
                    self.c_batch.n_tokens, self.c_batch.pos[0], self.c_batch.logits[0] = 1, cs+1, 1
                    ctypes.memmove(self.c_batch.embd, self.assets["emb_tables_1024"][cs][c].ctypes.data, 4096)
                    llama.llama_decode(self.c_ctx, self.c_batch)
                    last_logits = np.ctypeslib.as_array(llama.llama_get_logits(self.c_ctx), shape=(30720,))
            
            step_buffer.append(step_codes)
            if len(step_buffer) >= chunk_size:
                self.codes_q.put(list(step_buffer))
                step_buffer = []
                if verbose: print(f"  └─ 步数 {step_idx+1}: 推送分片")

            # Feedback (略)
            summed = np.sum(step_emb_2048, axis=0) + self.assets["tts_pad"].flatten()
            self.m_batch.n_tokens = 1
            ctypes.memmove(self.m_batch.embd, summed.ctypes.data, summed.nbytes)
            self.m_batch.pos[0] = self.m_batch.pos[1] = self.m_batch.pos[2] = cur_pos
            self.m_batch.pos[3], self.m_batch.logits[0], cur_pos = 0, 1, cur_pos + 1
            llama.llama_decode(self.m_ctx, self.m_batch)
            m_hidden = np.ctypeslib.as_array(llama.llama_get_embeddings(self.m_ctx), shape=(1, 2048))[0].copy()
            m_logits = np.ctypeslib.as_array(llama.llama_get_logits(self.m_ctx), shape=(1, 3072))[0].copy()

        if step_buffer: self.codes_q.put(list(step_buffer))

    def shutdown(self):
        try:
            self.codes_q.put(None)
            self.dec_p.terminate()
            self.spk_p.terminate()
            self.wav_p.terminate()
        except: pass
