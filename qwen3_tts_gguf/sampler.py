"""
sampler.py - 采样器逻辑
提供基于 NumPy 的 Top-K, Top-P, Temperature 采样方案。
"""
import numpy as np

def sample(logits: np.ndarray, temperature=1.0, top_p=1.0, top_k=0) -> int:
    """
    基于 NumPy 的核心采样函数。
    
    Args:
        logits: [vocab_size] 的未归一化对数概率
        temperature: 温度 (0 = greedy)
        top_p: Nucleus 采样阈值
        top_k: Top-K 采样数量
        
    Returns:
        token_id: 选中的 Token ID
    """
    # 1. Temperature 处理
    if temperature <= 1e-5:
        return int(np.argmax(logits))
    
    logits = logits.astype(np.float64) / temperature
    
    # 2. Softmax (数值稳定性优化)
    logits_max = np.max(logits)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / np.sum(exp_logits)
    
    # 3. Top-K 过滤
    if 0 < top_k < len(probs):
        top_k_indices = np.argsort(probs)[-top_k:]
        mask = np.ones_like(probs, dtype=bool)
        mask[top_k_indices] = False
        probs[mask] = 0.0
        # 如果 mask 后全为 0 (极罕见)，回退到原始分布
        if np.sum(probs) > 0:
            probs /= np.sum(probs)
            
    # 4. Top-P (Nucleus) 过滤
    if top_p < 1.0:
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)
        
        # 寻找超过阈值的切割点
        cutoff_index = np.searchsorted(cumulative_probs, top_p)
        cutoff_index = min(cutoff_index + 1, len(probs))
        
        keep_indices = sorted_indices[:cutoff_index]
        mask = np.ones_like(probs, dtype=bool)
        mask[keep_indices] = False
        probs[mask] = 0.0
        if np.sum(probs) > 0:
            probs /= np.sum(probs)
            
    # 5. 随机选择
    return int(np.random.choice(len(probs), p=probs))
