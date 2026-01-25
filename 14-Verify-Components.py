import torch
import numpy as np
import onnxruntime as ort

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

# 路径设置
# 导入路径 setup 已经在 __init__ 中处理了吗？不，那是 logger 的路径。
# 但这里是脚本，需要手动添加 sys.path 才能找到 qwen3_tts_gguf
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Qwen3-TTS"))

from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model
from qwen3_tts_gguf import logger

MODEL_DIR = r'./Qwen3-TTS-12Hz-1.7B-CustomVoice'
EXPORT_DIR = r'./model'

def verify_encoder():
    logger.info(">>> 验证 Codec Encoder (ONNX vs PyTorch) <<<")
    
    # 1. 准备 PyTorch 模型
    logger.info("Loading PyTorch Encoder...")
    
    tokenizer_model_dir = os.path.join(MODEL_DIR, "speech_tokenizer")
    if os.path.exists(tokenizer_model_dir):
        load_path = tokenizer_model_dir
    else:
        load_path = MODEL_DIR

    try:
        pt_model = Qwen3TTSTokenizerV2Model.from_pretrained(load_path)
        pt_model.eval()
    except Exception as e:
        logger.error(f"Failed to load PyTorch model: {e}")
        return False

    # 2. 准备 ONNX 模型
    onnx_path = os.path.join(EXPORT_DIR, "Qwen3-Codec-Encoder.onnx")
    if not os.path.exists(onnx_path):
        logger.error(f"ONNX model not found: {onnx_path}")
        return False
    
    logger.info("Loading ONNX Encoder...")
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    # 3. 构造随机输入
    # 模拟 1秒音频, 16000Hz (或根据 config)
    sr = 16000 # 假设
    dummy_audio = torch.randn(1, sr) 
    
    # 4. PyTorch 推理
    logger.info("Running PyTorch Inference...")
    with torch.no_grad():
        # 复刻 wrapper 中的逻辑: 先 unsqueeze, 再 encode
        # 直接使用我们之前定义的 CodecEncoderExportWrapper 逻辑最好，但这里尽量保持独立验证
        # 简单起见，调用 pt_model.encoder.encode
        encoded = pt_model.encoder.encode(dummy_audio.unsqueeze(1), return_dict=True)
        pt_codes = encoded.audio_codes # [B, Q, T]
        # 根据 export 逻辑，我们取了有效 quantizers 并转置
        valid_q = pt_model.config.encoder_valid_num_quantizers
        pt_out = pt_codes[:, :valid_q, :].transpose(1, 2).numpy() # [B, T, Q]

    # 5. ONNX 推理
    logger.info("Running ONNX Inference...")
    inputs = {sess.get_inputs()[0].name: dummy_audio.numpy()}
    onnx_out = sess.run(None, inputs)[0]

    # 6. 对比
    # 离散 Code 应该完全一致，或者误差极小（如果经过了 float 中间态）
    # 但 audio_codes 是 LongTensor，应该是完全一致
    match_rate = np.mean(pt_out == onnx_out)
    logger.info(f"Shape Match: PyTorch {pt_out.shape} vs ONNX {onnx_out.shape}")
    logger.info(f"Exact Match Rate: {match_rate * 100:.2f}%")
    
    if match_rate == 1.0:
        logger.info("✅ Encoder Verdict: PERFECT MATCH")
        return True
    else:
        logger.warning("❌ Encoder Verdict: MISMATCH")
        return False

def verify_decoder():
    logger.info("\n>>> 验证 Codec Decoder (ONNX vs PyTorch) <<<")
    
    # 1. Loading Models (Reuse if possible, but load fresh for isolation)
    # 略去 PyTorch 加载，假设 verify_encoder 已经加载了，或者重新加载
    tokenizer_model_dir = os.path.join(MODEL_DIR, "speech_tokenizer")
    load_path = tokenizer_model_dir if os.path.exists(tokenizer_model_dir) else MODEL_DIR
    
    pt_model = Qwen3TTSTokenizerV2Model.from_pretrained(load_path)
    pt_model.eval()

    onnx_path = os.path.join(EXPORT_DIR, "Qwen3-Codec-Decoder.onnx")
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    # 2. 构造随机 Codes
    # Shape: [1, 100, valid_q]
    valid_q = pt_model.config.decoder_config.num_quantizers # 注意 decoder config 可能不同于 encoder valid
    dummy_codes = torch.randint(0, 1024, (1, 100, valid_q), dtype=torch.long)

    # 3. PyTorch 推理
    with torch.no_grad():
        # Wrapper 逻辑: transpose(1, 2) -> decoder(codes)
        # pt_model.decoder expects [B, Q, T]
        pt_in = dummy_codes.transpose(1, 2)
        pt_out = pt_model.decoder(pt_in).squeeze(1).numpy()
    
    # 4. ONNX 推理
    inputs = {sess.get_inputs()[0].name: dummy_codes.numpy()}
    onnx_out = sess.run(None, inputs)[0]

    # 5. 对比 (MSE)
    mse = np.mean((pt_out - onnx_out) ** 2)
    max_diff = np.max(np.abs(pt_out - onnx_out))
    
    logger.info(f"MSE Error: {mse:.6f}")
    logger.info(f"Max Diff: {max_diff:.6f}")
    
    if mse < 1e-4:
        logger.info("✅ Decoder Verdict: PASS (High Fidelity)")
        return True
    else:
        logger.warning("❌ Decoder Verdict: FAIL (High Error)")
        return False

def verify_llm_sanity():
    logger.info("\n>>> 验证 LLM (GGUF Sanity Check) <<<")
    gguf_path = os.path.join(EXPORT_DIR, "Qwen3-LLM-1.7B-F16.gguf")
    
    if not os.path.exists(gguf_path):
        logger.error(f"GGUF file not found: {gguf_path}")
        return False
        
    if not Llama:
        logger.error("llama-cpp-python not installed, cannot verify GGUF.")
        return False

    try:
        llm = Llama(model_path=gguf_path, verbose=False)
        # 简单生成测试
        output = llm("Hello", max_tokens=5)
        logger.info(f"GGUF Load & Run Success. Output: {output}")
        logger.info("✅ LLM Verdict: PASS (Loadable)")
        return True
    except Exception as e:
        logger.error(f"LLM Verification Failed: {e}")
        return False

if __name__ == "__main__":
    enc_ok = verify_encoder()
    dec_ok = verify_decoder()
    llm_ok = verify_llm_sanity()
    
    logger.info("\n=== Summary ===")
    logger.info(f"Encoder: {'OK' if enc_ok else 'FAIL'}")
    logger.info(f"Decoder: {'OK' if dec_ok else 'FAIL'}")
    logger.info(f"LLM    : {'OK' if llm_ok else 'FAIL'}")
