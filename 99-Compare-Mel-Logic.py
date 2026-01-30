import torch
import numpy as np
import librosa
import os
from librosa.filters import mel as librosa_mel_fn

# --- 官方实现 (Qwen3-TTS 源代码复刻) ---

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def mel_spectrogram_official(
    y: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: int,
    fmax: int = None,
    center: bool = False,
) -> torch.Tensor:
    
    mel = librosa_mel_fn(
        sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
    )
    mel_basis = torch.from_numpy(mel).float()
    hann_window = torch.hann_window(win_size)

    # 官方手动补齐策略: (1024 - 256) // 2 = 384
    padding = (n_fft - hop_size) // 2
    y = torch.nn.functional.pad(
        y.unsqueeze(1), (padding, padding), mode="reflect"
    ).squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center, # False
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
    mel_spec = torch.matmul(mel_basis, spec)
    mel_spec = dynamic_range_compression_torch(mel_spec)
    return mel_spec

# --- 我们的实现 (qwen3_tts_gguf/predictors/encoder.py 复底复刻) ---

def extract_mel_ours(wav: np.ndarray) -> np.ndarray:
    mel_basis = librosa.filters.mel(
        sr=24000, n_fft=1024, n_mels=128, fmin=0.0, fmax=12000.0
    )
    
    # 2. 手动 Padding (对齐官方: (n_fft - hop_size) // 2 = 384)
    padding = (1024 - 256) // 2
    wav_padded = np.pad(wav, (padding, padding), mode='reflect')
    
    # 3. STFT (禁用 center 自动 Padding)
    stft = librosa.stft(
        wav_padded, n_fft=1024, hop_length=256, win_length=1024, 
        window='hann', center=False
    )
    
    magnitudes = np.sqrt(np.abs(stft)**2 + 1e-9)
    mel_spec = np.dot(mel_basis, magnitudes)
    log_mel = np.log(np.maximum(mel_spec, 1e-5))
    return log_mel.T

# --- 测试与比较 ---

def run_comparison():
    wav_path = "output/clone_source_vivian.wav"
    if not os.path.exists(wav_path):
        # 尝试备选路径
        wav_path = "output/anchor_vivian.wav"
        if not os.path.exists(wav_path):
            print(f"❌ Error: Audio file not found. Please run a test script first.")
            return
        
    print(f"🔍 Comparing Mel features for: {wav_path}")
    wav, _ = librosa.load(wav_path, sr=24000)
    
    # 提取官方特征
    wav_torch = torch.from_numpy(wav).unsqueeze(0)
    mel_off = mel_spectrogram_official(
        wav_torch, 1024, 128, 24000, 256, 1024, 0, 12000
    ).squeeze(0).numpy().T # 转置为 [T, 128]
    
    # 提取我们的特征
    mel_ours = extract_mel_ours(wav)
    
    # 对齐长度
    print(f"   Official Shape: {mel_off.shape}")
    print(f"   Our Shape:      {mel_ours.shape}")
    
    min_len = min(len(mel_off), len(mel_ours))
    mel_off_c = mel_off[:min_len]
    mel_ours_c = mel_ours[:min_len]
    
    # 计算差异
    mse = np.mean((mel_off_c - mel_ours_c)**2)
    cosine_sim = np.dot(mel_off_c.flatten(), mel_ours_c.flatten()) / (
        np.linalg.norm(mel_off_c) * np.linalg.norm(mel_ours_c)
    )
    
    print(f"\n📊 Comparison Results:")
    print(f"   - MSE: {mse:.8f}")
    print(f"   - Cosine Similarity: {cosine_sim:.8f}")
    
    if cosine_sim > 0.999:
        print("✅ [Perfect Match] Features are almost identical.")
    elif cosine_sim > 0.99:
        print("✅ [High Consistency] Features are very similar, slight deviations likely due to padding.")
    else:
        print("⚠️ [Significant Difference] Padding strategies (384 vs 512) might be impacting alignment.")

if __name__ == "__main__":
    run_comparison()
