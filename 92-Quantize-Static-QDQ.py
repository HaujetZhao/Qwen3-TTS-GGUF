"""
92-Quantize-Static-QDQ.py
使用 QDQ 格式进行静态量化 (需要校准数据)。
QDQ 格式在 Conv 前后插入 Q/DQ 节点，避免使用 ConvInteger。
"""
import os
import time
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType

# 配置
ONNX_PATH = "onnx_export/qwen3_tts_decoder_stateful.onnx"
QUANT_PATH = "onnx_export/qwen3_tts_decoder_stateful_qdq.onnx"

TOTAL_FRAMES = 200
CHUNK_SIZE = 25
Q = 16
NUM_LAYERS = 8
NUM_HEADS = 16
HEAD_DIM = 64
CALIB_SAMPLES = 20  # 校准样本数

class StatefulCalibrationReader(CalibrationDataReader):
    """
    为状态化模型生成校准数据的迭代器。
    每次返回一个 Chunk 的输入，模拟流式推理过程。
    """
    def __init__(self, num_samples=20):
        self.num_samples = num_samples
        self.current = 0
        
        # 生成随机测试码
        self.codes_ref = np.random.randint(0, 1024, (1, TOTAL_FRAMES, Q), dtype=np.int64)
        
        # 初始化状态
        self.pre_conv = np.zeros((1, 512, 0), dtype=np.float32)
        self.latent = np.zeros((1, 1024, 0), dtype=np.float32)
        self.conv = np.zeros((1, 1024, 0), dtype=np.float32)
        self.pkv = []
        for _ in range(NUM_LAYERS):
            self.pkv.append(np.zeros((1, NUM_HEADS, 0, HEAD_DIM), dtype=np.float32))
        for _ in range(NUM_LAYERS):
            self.pkv.append(np.zeros((1, NUM_HEADS, 0, HEAD_DIM), dtype=np.float32))
        
        # 预加载模型以获取中间状态
        self.sess = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
        self.output_names = [out.name for out in self.sess.get_outputs()]
        
        # 预生成所有校准样本
        self.samples = []
        frame_idx = 0
        for i in range(num_samples):
            if frame_idx >= TOTAL_FRAMES:
                frame_idx = 0
                # 重置状态
                self.pre_conv = np.zeros((1, 512, 0), dtype=np.float32)
                self.latent = np.zeros((1, 1024, 0), dtype=np.float32)
                self.conv = np.zeros((1, 1024, 0), dtype=np.float32)
                self.pkv = [np.zeros((1, NUM_HEADS, 0, HEAD_DIM), dtype=np.float32) for _ in range(NUM_LAYERS * 2)]
            
            chunk = self.codes_ref[:, frame_idx:frame_idx+CHUNK_SIZE, :]
            is_last = np.array([1.0 if frame_idx + CHUNK_SIZE >= TOTAL_FRAMES else 0.0], dtype=np.float32)
            
            feed = {
                "audio_codes": chunk,
                "is_last": is_last,
                "pre_conv_history": self.pre_conv.copy(),
                "latent_buffer": self.latent.copy(),
                "conv_history": self.conv.copy(),
            }
            for j in range(NUM_LAYERS):
                feed[f"past_key_{j}"] = self.pkv[j].copy()
                feed[f"past_value_{j}"] = self.pkv[NUM_LAYERS + j].copy()
            
            self.samples.append(feed)
            
            # 运行一次以更新状态
            outputs = self.sess.run(self.output_names, feed)
            self.pre_conv = outputs[2]
            self.latent = outputs[3]
            self.conv = outputs[4]
            for j in range(NUM_LAYERS):
                self.pkv[j] = outputs[5 + j]
                self.pkv[NUM_LAYERS + j] = outputs[5 + NUM_LAYERS + j]
            
            frame_idx += CHUNK_SIZE
    
    def get_next(self):
        if self.current >= len(self.samples):
            return None
        sample = self.samples[self.current]
        self.current += 1
        return sample
    
    def rewind(self):
        self.current = 0

def main():
    print("="*60)
    print("📊 QDQ 格式静态量化测试")
    print("="*60)
    
    # 1. 生成校准数据
    print(f"\n🔧 生成校准数据 ({CALIB_SAMPLES} 样本)...")
    calib_reader = StatefulCalibrationReader(num_samples=CALIB_SAMPLES)
    print(f"   ✅ 校准数据准备完成")
    
    # 2. 执行静态量化
    print(f"\n🔧 执行 QDQ 静态量化...")
    print(f"   输入: {ONNX_PATH}")
    print(f"   输出: {QUANT_PATH}")
    
    t0 = time.time()
    try:
        quantize_static(
            model_input=ONNX_PATH,
            model_output=QUANT_PATH,
            calibration_data_reader=calib_reader,
            quant_format=QuantFormat.QDQ,  # 关键：使用 QDQ 格式
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QUInt8,
            # 可选：仅量化特定算子
            # op_types_to_quantize=['Conv', 'MatMul'],
        )
        print(f"✅ 量化完成！耗时: {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"❌ 量化失败: {e}")
        return
    
    # 检查文件大小
    orig_size = os.path.getsize(ONNX_PATH) / 1e6
    quant_size = os.path.getsize(QUANT_PATH) / 1e6
    print(f"   原始大小: {orig_size:.1f} MB")
    print(f"   量化后:   {quant_size:.1f} MB (压缩率: {quant_size/orig_size*100:.0f}%)")
    
    # 3. 尝试加载
    print(f"\n📦 加载量化模型...")
    try:
        sess_qdq = ort.InferenceSession(QUANT_PATH, providers=['CPUExecutionProvider'])
        print("   ✅ 模型加载成功！")
    except Exception as e:
        print(f"   ❌ 加载失败: {e}")
        return
    
    # 4. 性能对比
    print(f"\n🔥 性能对比测试...")
    
    sess_fp32 = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
    
    def run_inference(sess):
        pre_conv = np.zeros((1, 512, 0), dtype=np.float32)
        latent = np.zeros((1, 1024, 0), dtype=np.float32)
        conv = np.zeros((1, 1024, 0), dtype=np.float32)
        pkv = [np.zeros((1, NUM_HEADS, 0, HEAD_DIM), dtype=np.float32) for _ in range(NUM_LAYERS * 2)]
        output_names = [out.name for out in sess.get_outputs()]
        
        codes = np.random.randint(0, 1024, (1, TOTAL_FRAMES, Q), dtype=np.int64)
        t_total = 0
        
        for i in range(0, TOTAL_FRAMES, CHUNK_SIZE):
            chunk = codes[:, i:i+CHUNK_SIZE, :]
            is_last = np.array([1.0 if i + CHUNK_SIZE >= TOTAL_FRAMES else 0.0], dtype=np.float32)
            
            feed = {"audio_codes": chunk, "is_last": is_last,
                    "pre_conv_history": pre_conv, "latent_buffer": latent, "conv_history": conv}
            for j in range(NUM_LAYERS):
                feed[f"past_key_{j}"] = pkv[j]
                feed[f"past_value_{j}"] = pkv[NUM_LAYERS + j]
            
            ts = time.perf_counter()
            outputs = sess.run(output_names, feed)
            t_total += time.perf_counter() - ts
            
            pre_conv, latent, conv = outputs[2], outputs[3], outputs[4]
            for j in range(NUM_LAYERS):
                pkv[j] = outputs[5 + j]
                pkv[NUM_LAYERS + j] = outputs[5 + NUM_LAYERS + j]
        
        return t_total
    
    # 预热
    run_inference(sess_fp32)
    run_inference(sess_qdq)
    
    # 测试
    ITERS = 3
    t_fp32 = sum(run_inference(sess_fp32) for _ in range(ITERS)) / ITERS
    t_qdq = sum(run_inference(sess_qdq) for _ in range(ITERS)) / ITERS
    
    print("\n" + "="*50)
    print("📊 性能对比结果")
    print("="*50)
    print(f"  FP32 模型: {t_fp32*1000:.1f} ms")
    print(f"  QDQ 模型:  {t_qdq*1000:.1f} ms")
    print(f"  加速比:    {t_fp32/t_qdq:.2f}x")
    print(f"  模型压缩:  {orig_size:.1f} MB -> {quant_size:.1f} MB")
    print("="*50)

if __name__ == "__main__":
    main()
