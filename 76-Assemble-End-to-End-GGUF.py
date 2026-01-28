import os
import ctypes
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
import onnxruntime as ort
import qwen3_tts_gguf.nano_llama as nano_llama

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 1. 路径与配置
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
CAPTURED_DIR = os.path.join(PROJECT_ROOT, "captured_assembly")
SAVE_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(SAVE_DIR, exist_ok=True)

MASTER_GGUF = os.path.join(MODEL_DIR, "qwen3_tts_talker.gguf")
CRAFTSMAN_GGUF = os.path.join(MODEL_DIR, "qwen3_tts_craftsman_advanced.gguf")
MOUTH_ONNX = os.path.join(MODEL_DIR, "qwen3_tts_decoder.onnx")
MASTER_HEAD_PATH = os.path.join(MODEL_DIR, "codec_head_weight.npy")
PROJ_PT_PATH = os.path.join(MODEL_DIR, "craftsman_hf/master_to_craftsman_proj.pt")

EOS_TOKEN_ID = 2150

def load_assets():
    print("[1/6] 正在加载模型权重与词表资产...")
    assets = {
        "master_head": np.load(MASTER_HEAD_PATH),
        "emb_tables": [np.load(os.path.join(MODEL_DIR, f"codec_embedding_{i}.npy")) for i in range(16)],
        "proj": torch.load(PROJ_PT_PATH, map_location="cpu")
    }
    # 辅助推理资产
    assets["prefill_input"] = np.load(os.path.join(CAPTURED_DIR, "prefill_input_embeds.npy")).astype(np.float32)
    assets["trailing_text"] = np.load(os.path.join(CAPTURED_DIR, "trailing_text_hidden.npy")).astype(np.float32)
    assets["tts_pad"] = np.load(os.path.join(CAPTURED_DIR, "tts_pad_embed.npy")).astype(np.float32)
    
    # 真值
    try:
        assets["official_codes"] = np.load(os.path.join(os.path.dirname(CAPTURED_DIR), "captured_full_gen", "full_generated_codes.npy"))
        print(f"  ✅ 发现官方真值 (共 {len(assets['official_codes'])} 步)")
    except Exception:
        print("  ⚠️ 未发现官方真值，将进行盲跑。")
        assets["official_codes"] = None

    # 预计算 1024 维度的工匠输入表 (加速核心：避免运行期重复投影)
    print("  正在预处理 1024 维工匠输入表...")
    proj_w = assets["proj"]["weight"].float() # [1024, 2048]
    proj_b = assets["proj"]["bias"].float()   # [1024]
    
    emb_tables_1024 = []
    for i in range(16):
        # 我们可以直接对整个表做线性变换
        table_2048 = torch.from_numpy(assets["emb_tables"][i]).float()
        table_1024 = F.linear(table_2048, proj_w, proj_b).numpy()
        emb_tables_1024.append(table_1024)
    assets["emb_tables_1024"] = emb_tables_1024
    
    print("✅ 资产加载与预投影处理完成。")
    return assets

def apply_projection(hidden_2048, proj_assets):
    """投影维度: 2048 -> 1024 (工匠专用)"""
    w = proj_assets["weight"].float().numpy()
    b = proj_assets["bias"].float().numpy()
    return hidden_2048 @ w.T + b

def run_e2e_pipeline():
    print("=== [76] 全链路语音合成 (GGUF Master + GGUF Craftsman + ONNX Mouth) ===\n")
    
    # A. 加载资产
    assets = load_assets()
    
    # B. 初始化三位核心成员
    print("[2/6] 初始化推理引擎...")
    # 大师
    m_model = nano_llama.load_model(MASTER_GGUF, n_gpu_layers=0)
    m_ctx_params = nano_llama.llama_context_default_params()
    m_ctx_params.n_ctx = 4096
    m_ctx_params.embeddings = True
    m_ctx = nano_llama.llama_init_from_model(m_model, m_ctx_params)
    m_embd_dim = 2048
    
    # 工匠
    c_model = nano_llama.load_model(CRAFTSMAN_GGUF, n_gpu_layers=0)
    c_ctx_params = nano_llama.llama_context_default_params()
    c_ctx_params.n_ctx = 512
    c_ctx_params.embeddings = True
    c_ctx = nano_llama.llama_init_from_model(c_model, c_ctx_params)
    c_embd_dim = 1024
    
    # 嘴巴 (Decoder)
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    mouth_sess = ort.InferenceSession(MOUTH_ONNX, sess_opts, providers=['CPUExecutionProvider'])
    
    # 复用的 Batch
    m_batch = nano_llama.llama_batch_init(4096, m_embd_dim, 1)
    c_batch = nano_llama.llama_batch_init(32, c_embd_dim, 1)

    # ---------------------------------------------------------
    # C. 大师 Prefill
    # ---------------------------------------------------------
    print("\n[3/6] 开始 Prefill 阶段...")
    prefill_input = assets["prefill_input"]
    n_tokens = prefill_input.shape[1]
    
    m_batch.n_tokens = n_tokens
    ctypes.memmove(m_batch.embd, np.ascontiguousarray(prefill_input[0]).ctypes.data, prefill_input[0].nbytes)
    
    for i in range(n_tokens):
        m_batch.pos[i] = i
        m_batch.pos[n_tokens + i] = i
        m_batch.pos[2 * n_tokens + i] = i
        m_batch.pos[3 * n_tokens + i] = 0
        m_batch.n_seq_id[i] = 1
        m_batch.seq_id[i][0] = 0
        m_batch.logits[i] = 1 if i == n_tokens - 1 else 0
        
    nano_llama.llama_decode(m_ctx, m_batch)
    
    # 提取 Prefill 后最后一帧隐藏层
    m_out_ptr = nano_llama.llama_get_embeddings(m_ctx)
    m_hidden_last = np.ctypeslib.as_array(m_out_ptr, shape=(n_tokens, m_embd_dim))[-1].copy()
    
    current_m_pos = n_tokens
    all_step_codes = []

    # ---------------------------------------------------------
    # D. 自回归循环
    # ---------------------------------------------------------
    print(f"\n[4/6] 进入分码生成循环 (最大 50 步)...")
    
    for step_idx in range(50):
        # 1. 大师 -> 推理 Code 0
        m_logits = m_hidden_last @ assets["master_head"].T
        code_0 = np.argmax(m_logits)
        
        if code_0 == EOS_TOKEN_ID:
            print(f"  🛑 生成到第 {step_idx} 步遇到 EOS，自回归结束。")
            break
            
        # 2. 构造工匠输入并投影
        emb_0_2048 = assets["emb_tables"][0][code_0].copy()
        c_in_2048 = np.stack([m_hidden_last, emb_0_2048], axis=0) # [2, 2048]
        c_in_1024 = apply_projection(c_in_2048, assets["proj"]) # [2, 1024]
        
        # 3. 工匠 -> 推理剩余 15 个分码
        # 复用上下文 & 清理 KV Cache
        c_mem = nano_llama.llama_get_memory(c_ctx)
        nano_llama.llama_memory_clear(c_mem, True)
        
        step_codes = [code_0]
        step_embeds_2048 = [emb_0_2048]
        
        # 工匠 Prefill (M + C0)
        c_batch.n_tokens = 2
        ctypes.memmove(c_batch.embd, c_in_1024.ctypes.data, c_in_1024.nbytes)
        for i in range(2):
            c_batch.pos[i] = i
            c_batch.n_seq_id[i] = 1
            c_batch.seq_id[i][0] = 0
            c_batch.logits[i] = 1 if i == 1 else 0
            
        nano_llama.llama_decode(c_ctx, c_batch)
        
        # 获取 Logits 并循环生成 (Table 1...15)
        c_logits_ptr = nano_llama.llama_get_logits(c_ctx)
        last_logits_all = np.ctypeslib.as_array(c_logits_ptr, shape=(2, 30720))
        last_logits = last_logits_all[1]
        
        for c_step in range(1, 16):
            logits_offset = (c_step - 1) * 2048
            code = np.argmax(last_logits[logits_offset : logits_offset + 2048])
            step_codes.append(code)
            
            codec_emb_2048 = assets["emb_tables"][c_step][code].copy()
            step_embeds_2048.append(codec_emb_2048)
            
            if c_step < 15:
                # 性能优化：直接从预计算的 1024 维表中查找，避免实时投影运算
                next_in_1024 = assets["emb_tables_1024"][c_step][code]
                
                c_batch.n_tokens = 1
                c_batch.pos[0] = c_step + 1
                ctypes.memmove(c_batch.embd, next_in_1024.ctypes.data, next_in_1024.nbytes)
                c_batch.logits[0] = 1
                nano_llama.llama_decode(c_ctx, c_batch)
                last_logits = np.ctypeslib.as_array(nano_llama.llama_get_logits(c_ctx), shape=(30720,))

        all_step_codes.append(step_codes)
        print(f"  Step {step_idx:02}: Codes {step_codes}")
        
        # 4. 反馈 -> 求和并处理补偿
        summed = np.sum(step_embeds_2048, axis=0)
        if step_idx < assets["trailing_text"].shape[1]:
            summed += assets["trailing_text"][0, step_idx]
        else:
            summed += assets["tts_pad"].flatten()
            
        # 5. 大师 -> 反馈输入并生成下一帧隐藏层
        m_batch.n_tokens = 1
        ctypes.memmove(m_batch.embd, summed.ctypes.data, summed.nbytes)
        for i in range(1):
            m_batch.pos[0] = current_m_pos
            m_batch.pos[1] = current_m_pos
            m_batch.pos[2] = current_m_pos
            m_batch.pos[3] = 0
            m_batch.logits[0] = 1
            
        nano_llama.llama_decode(m_ctx, m_batch)
        current_m_pos += 1
        
        # 更新供下一轮使用的隐藏层
        m_out_ptr = nano_llama.llama_get_embeddings(m_ctx)
        m_hidden_last = np.ctypeslib.as_array(m_out_ptr, shape=(1, m_embd_dim))[0].copy()

    # ---------------------------------------------------------
    # E. 序列核验 (如果有真值)
    # ---------------------------------------------------------
    print("\n[5/6] 执行序列核验...")
    mine = np.array(all_step_codes)
    auth = assets["official_codes"]
    
    if auth is not None:
        min_len = min(len(mine), len(auth))
        match = np.array_equal(mine[:min_len], auth[:min_len])
        if match:
             print("  ✅ 全序列完全对齐！优化后的双 GGUF 流水线精度通过。")
        else:
             print("  ❌ 序列存在不一致 (精度累积误差或逻辑分歧)。")
             diffs = (mine[:min_len] != auth[:min_len]).any(axis=1)
             first_diff = np.where(diffs)[0][0]
             print(f"  首次分歧发生在 Step {first_diff}")
    
    # ---------------------------------------------------------
    # F. 嘴巴渲染 (Codec Decoding)
    # ---------------------------------------------------------
    print(f"\n[6/6] 渲染音频中 (ONNX Decoder)...")
    if len(all_step_codes) == 0:
        print("❌ 未生成任何分码，跳过解码。")
        return
        
    codes_np = np.array(all_step_codes) # [Steps, 16]
    # 对齐 Decoder 输入形状: [1, T, 16]
    decoder_input = codes_np[np.newaxis, ...].astype(np.int64)
    print(f"  解码器输入尺寸: {decoder_input.shape}")
    
    audio_out = mouth_sess.run(None, {'audio_codes': decoder_input})[0]
    audio_data = audio_out.squeeze()
    
    WAV_PATH = os.path.join(SAVE_DIR, "gguf_e2e_output.wav")
    sf.write(WAV_PATH, audio_data, 24000)
    print(f"\n✅ 语音合成成功！波形文件存放在: {WAV_PATH}")

    # ---------------------------------------------------------
    # F. 清理现场
    # ---------------------------------------------------------
    print("[6/6] 正在释放资源...")
    nano_llama.llama_batch_free(m_batch)
    nano_llama.llama_batch_free(c_batch)
    nano_llama.llama_free(m_ctx)
    nano_llama.llama_free(c_ctx)
    nano_llama.llama_model_free(m_model)
    nano_llama.llama_model_free(c_model)
    print("DONE.")

if __name__ == "__main__":
    run_e2e_pipeline()
