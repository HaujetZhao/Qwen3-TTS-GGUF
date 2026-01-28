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

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# =========================================================================
# 1. 动态输入配置 (User Configurable)
# =========================================================================
TARGET_TEXT = "你好，我是由大师和工匠联合驱动的语音模型。今天的天气真的很棒！"
SPEAKER_ID = 3065  # 3065: Vivian (Female), 3010: Uncle Fu (Male)
MAX_STEPS = 250    # 最大生成步数

# =========================================================================
# 2. 路径配置
# =========================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
SAVE_DIR = os.path.join(PROJECT_ROOT, "output")
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "Qwen3-TTS-12Hz-1.7B-CustomVoice")

os.makedirs(SAVE_DIR, exist_ok=True)

MASTER_GGUF = os.path.join(MODEL_DIR, "qwen3_tts_talker.gguf")
CRAFTSMAN_GGUF = os.path.join(MODEL_DIR, "qwen3_tts_craftsman_advanced.gguf")
MOUTH_ONNX = os.path.join(MODEL_DIR, "qwen3_tts_decoder.onnx")
MASTER_HEAD_PATH = os.path.join(MODEL_DIR, "codec_head_weight.npy")
PROJ_PT_PATH = os.path.join(MODEL_DIR, "craftsman_hf/master_to_craftsman_proj.pt")
TEXT_TABLE_PATH = os.path.join(MODEL_DIR, "text_embedding_projected.npy")

# 常量 (控制码)
EOS_TOKEN_ID = 2150
C_PAD = 2148
C_THINK = 2154
C_TBOS = 2156
C_CHINESE = 2055
C_TEOS = 2157
C_AUDIO_BOS = 2148 
C_TTS_EOS = 2148
C_CODEC_BOS = 2149

T_IM_START = 151644
T_ASSISTANT = 77091
T_NL = 198
T_THINK_T = 151671
T_AUDIO_BOS = 151672
T_TTS_EOS = 151673

# =========================================================================
# 3. 资产加载逻辑
# =========================================================================
def load_all_assets():
    print("[1/6] 正在加载核心权重与分码表...")
    start_load = time.time()
    assets = {
        "master_head": np.load(MASTER_HEAD_PATH),
        "text_table": np.load(TEXT_TABLE_PATH),
        "emb_tables": [np.load(os.path.join(MODEL_DIR, f"codec_embedding_{i}.npy")) for i in range(16)],
        "proj": torch.load(PROJ_PT_PATH, map_location="cpu")
    }
    
    # 辅助向量 (77号脚本提到的补偿项)
    # 虽然脱离了捕获数据，但 tts_pad 和 trailing_text 通常来自于固定训练输出或特定逻辑
    # 如果没有捕获，可以使用模型自带的 Pad 逻辑。这里我们动态从 text_table 提取 Pad
    assets["tts_pad"] = assets["text_table"][151671] # TTS_PAD 对应的 2048 向量
    
    # 性能优化：预投影工匠 1024 维表
    print("  正在预计算 1024 维工匠表...")
    proj_w = assets["proj"]["weight"].float()
    proj_b = assets["proj"]["bias"].float()
    assets["emb_tables_1024"] = [
        F.linear(torch.from_numpy(t).float(), proj_w, proj_b).numpy() for t in assets["emb_tables"]
    ]
    
    print("✅ 资产加载与预热完成。")
    assets["load_time"] = time.time() - start_load
    return assets

def apply_projection(hidden_2048, proj_assets):
    """2048 -> 1024 线性投影"""
    w = proj_assets["weight"].float().numpy()
    b = proj_assets["bias"].float().numpy()
    return hidden_2048 @ w.T + b

# =========================================================================
# 4. 动态输入构造
# =========================================================================
def construct_prompt(text, spk_id, assets):
    print(f"[2/6] 正在动态编译输入 Prompt...")
    start_p = time.time()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True, fix_mistral_regex=True)
    content_ids = tokenizer.encode(text, add_special_tokens=False)
    
    # 序列结构: (TextID, CodecID)
    sequence = [
        (T_IM_START, 0),         (T_ASSISTANT, 0),        (T_NL, 0),
        (T_THINK_T, C_THINK),    (T_THINK_T, C_TBOS),     (T_THINK_T, C_CHINESE),
        (T_THINK_T, C_TEOS),     (T_THINK_T, spk_id),     (T_AUDIO_BOS, C_AUDIO_BOS),
    ]
    for tid in content_ids:
        sequence.append((tid, C_PAD))
    sequence.append((T_TTS_EOS, C_TTS_EOS))
    sequence.append((T_THINK_T, C_CODEC_BOS))
    
    # 物理叠加
    text_table = assets["text_table"]
    codec_table_0 = assets["emb_tables"][0]
    
    embed_list = []
    for tid, cid in sequence:
        t_vec = text_table[tid]
        c_vec = codec_table_0[cid] if cid != 0 else np.zeros(2048, dtype=np.float32)
        embed_list.append(t_vec + c_vec)
        
        
    assets["prompt_time"] = time.time() - start_p
    return np.array(embed_list).reshape(1, len(sequence), 2048).astype(np.float32)

# =========================================================================
# 5. 主流水线
# =========================================================================
def run_end_to_end():
    print("=== [79] 端到端全动态语音合成演示 (Quwen3-TTS GGUF) ===\n")
    
    assets = load_all_assets()
    prompt_embeds = construct_prompt(TARGET_TEXT, SPEAKER_ID, assets)
    
    # 初始化 GGUF 引擎
    m_init_start = time.time()
    print("[3/6] 初始化 GGUF 模型 (Master & Craftsman)...")
    m_model = nano_llama.load_model(MASTER_GGUF, n_gpu_layers=-1)
    c_model = nano_llama.load_model(CRAFTSMAN_GGUF, n_gpu_layers=-1)
    
    # 初始化大师上下文
    m_ctx_params = nano_llama.llama_context_default_params()
    m_ctx_params.n_ctx = 4096
    m_ctx_params.embeddings = True
    m_ctx = nano_llama.llama_init_from_model(m_model, m_ctx_params)
    
    # 初始化工匠上下文
    c_ctx_params = nano_llama.llama_context_default_params()
    c_ctx_params.n_ctx = 512
    c_ctx_params.embeddings = True
    c_ctx = nano_llama.llama_init_from_model(c_model, c_ctx_params)
    
    # 初始化 ONNX Mouth
    mouth_sess = ort.InferenceSession(MOUTH_ONNX, providers=['CPUExecutionProvider'])
    m_init_end = time.time()
    init_time = m_init_end - m_init_start
    
    # -----------------------
    # 执行生成
    # -----------------------
    print("\n--- 启动生成流水线 ---")
    start_time = time.time()
    
    # A. 大师 Prefill
    prefill_start = time.time()
    n_prefill = prompt_embeds.shape[1]
    m_batch = nano_llama.llama_batch_init(4096, 2048, 1)
    m_batch.n_tokens = n_prefill
    ctypes.memmove(m_batch.embd, np.ascontiguousarray(prompt_embeds[0]).ctypes.data, prompt_embeds[0].nbytes)
    for i in range(n_prefill):
        m_batch.pos[i] = i
        m_batch.pos[n_prefill+i] = i
        m_batch.pos[2*n_prefill+i] = i
        m_batch.pos[3*n_prefill+i] = 0
        m_batch.n_seq_id[i], m_batch.seq_id[i][0], m_batch.logits[i] = 1, 0, (1 if i == n_prefill-1 else 0)
    nano_llama.llama_decode(m_ctx, m_batch)
    
    m_hidden_last = np.ctypeslib.as_array(nano_llama.llama_get_embeddings(m_ctx), shape=(n_prefill, 2048))[-1].copy()
    current_m_pos = n_prefill
    prefill_time = time.time() - prefill_start
    all_codes = []
    
    # B. 自回归循环
    loop_start = time.time()
    c_batch = nano_llama.llama_batch_init(32, 1024, 1)
    
    # 统计内部耗时
    master_total_time = 0
    craftsman_total_time = 0
    feedback_total_time = 0
    
    for step_idx in range(MAX_STEPS):
        # 1. 大师预测分码 0
        m_step_start = time.time()
        m_logits = m_hidden_last @ assets["master_head"].T
        code_0 = np.argmax(m_logits)
        master_total_time += (time.time() - m_step_start)
        
        if code_0 == EOS_TOKEN_ID: break
        
        step_codes = [code_0]
        step_emb_2048 = [assets["emb_tables"][0][code_0].copy()]
        
        # 2. 工匠执行 15 步生成
        c_step_start = time.time()
        m_hidden_1024 = apply_projection(m_hidden_last, assets["proj"])
        c_in_1024 = np.stack([m_hidden_1024, assets["emb_tables_1024"][0][code_0]], axis=0)
        
        nano_llama.llama_memory_clear(nano_llama.llama_get_memory(c_ctx), True)
        c_batch.n_tokens = 2
        ctypes.memmove(c_batch.embd, c_in_1024.ctypes.data, c_in_1024.nbytes)
        for j in range(2):
            c_batch.pos[j], c_batch.n_seq_id[j], c_batch.seq_id[j][0], c_batch.logits[j] = j, 1, 0, (1 if j == 1 else 0)
        nano_llama.llama_decode(c_ctx, c_batch)
        
        last_logits = np.ctypeslib.as_array(nano_llama.llama_get_logits(c_ctx), shape=(2, 30720))[1]
        for c_step in range(1, 16):
            code = np.argmax(last_logits[(c_step-1)*2048 : (c_step-1)*2048 + 2048])
            step_codes.append(code)
            step_emb_2048.append(assets["emb_tables"][c_step][code].copy())
            if c_step < 15:
                # 性能优化：直接查 1024 预投影表
                next_in_1024 = assets["emb_tables_1024"][c_step][code]
                c_batch.n_tokens = 1
                c_batch.pos[0] = c_step + 1
                ctypes.memmove(c_batch.embd, next_in_1024.ctypes.data, next_in_1024.nbytes)
                c_batch.logits[0] = 1
                nano_llama.llama_decode(c_ctx, c_batch)
                last_logits = np.ctypeslib.as_array(nano_llama.llama_get_logits(c_ctx), shape=(30720,))
        craftsman_total_time += (time.time() - c_step_start)
        
        all_codes.append(step_codes)
        
        # 3. 反馈给大师
        f_step_start = time.time()
        summed = np.sum(step_emb_2048, axis=0) + assets["tts_pad"].flatten()
        m_batch.n_tokens = 1
        ctypes.memmove(m_batch.embd, summed.ctypes.data, summed.nbytes)
        m_batch.pos[0] = m_batch.pos[1] = m_batch.pos[2] = current_m_pos
        m_batch.pos[3], m_batch.logits[0] = 0, 1
        nano_llama.llama_decode(m_ctx, m_batch)
        current_m_pos += 1
        m_hidden_last = np.ctypeslib.as_array(nano_llama.llama_get_embeddings(m_ctx), shape=(1, 2048))[0].copy()
        feedback_total_time += (time.time() - f_step_start)
        
    loop_end = time.time()
    
    # C. 音频解码
    print(f"\n[5/6] 渲染音频中 (帧数: {len(all_codes)})...")
    if len(all_codes) == 0: return
    decode_start = time.time()
    codes_input = np.array(all_codes)[np.newaxis, ...].astype(np.int64) # [1, T, 16]
    audio_data = mouth_sess.run(None, {'audio_codes': codes_input})[0].squeeze()
    decode_end = time.time()
    
    total_time = time.time() - start_time
    W_PATH = os.path.join(SAVE_DIR, "dynamic_gguf_output.wav")
    sf.write(W_PATH, audio_data, 24000)
    
    # D. 报告
    audio_dur = len(audio_data) / 24000.0
    print("\n--- [6/6] 分时性能报告 ---")
    print(f"音频长度: {audio_dur:.2f} 秒")
    print("-" * 30)
    print(f"1. 静态资产加载: {assets['load_time']:.4f} 秒")
    print(f"2. Prompt 编译:  {assets['prompt_time']:.4f} 秒")
    print(f"3. 模型引擎初始化: {init_time:.4f} 秒")
    print(f"4. 大师 Prefill: {prefill_time:.4f} 秒")
    print(f"5. 自回归循环总计: {loop_end - loop_start:.4f} 秒")
    print(f"   └─ 大师预测 (Master):    {master_total_time:.4f} 秒")
    print(f"   └─ 工匠预测 (Craftsman): {craftsman_total_time:.4f} 秒")
    print(f"   └─ 反馈回路 (Feedback):  {feedback_total_time:.4f} 秒")
    print(f"6. 嘴巴解码 (Mouth): {decode_end - decode_start:.4f} 秒")
    print("-" * 30)
    print(f"总端到端耗时: {total_time:.4f} 秒")
    print(f"实时率 (RTF): {total_time / audio_dur:.4f} (越低越快)")
    print(f"✅ 合成音频已保存: {W_PATH}")

    # 清理
    nano_llama.llama_batch_free(m_batch)
    nano_llama.llama_batch_free(c_batch)
    nano_llama.llama_free(m_ctx)
    nano_llama.llama_free(c_ctx)
    nano_llama.llama_model_free(m_model)
    nano_llama.llama_model_free(c_model)

if __name__ == "__main__":
    run_end_to_end()
