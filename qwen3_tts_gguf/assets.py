"""
assets.py - 资产管理器
负责加载词表、投影矩阵和 Codec 嵌入表。
提供针对工匠模型加速的预投影嵌入表。
"""
import os
import numpy as np
from . import logger

class AssetsManager:
    """
    负责加载和持有所有静态权重资产。
    """
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        
        # 定义关键资产路径
        self.paths = {
            "text_table": os.path.join(model_dir, "text_embedding_projected.npy"),
            "proj_w": os.path.join(model_dir, "proj_weight.npy"),
            "proj_b": os.path.join(model_dir, "proj_bias.npy")
        }
        
        self.load_all()

    def load_all(self):
        """加载所有权重资产到内存"""
        logger.info(f"[Assets] 正在从 {self.model_dir} 加载资产...")
        
        # 1. 加载文本投影表
        if not os.path.exists(self.paths["text_table"]):
            raise FileNotFoundError(f"缺失核心资产: {self.paths['text_table']}")
            
        self.text_table = np.load(self.paths["text_table"])
        self.tts_pad = self.text_table[151671] # 预存 PAD 向量
        
        # 2. 加载投影矩阵 (2048 -> 1024)
        self.proj = {
            "weight": np.load(self.paths["proj_w"]),
            "bias": np.load(self.paths["proj_b"])
        }
        
        # 3. 加载 16 组 Codec Embedding Tables
        self.emb_tables = []
        self.emb_tables_1024 = []
        
        pw = self.proj["weight"]
        pb = self.proj["bias"]
        
        for i in range(16):
            path = os.path.join(self.model_dir, f"codec_embedding_{i}.npy")
            if not os.path.exists(path):
                # 如果没有 16 组，尝试兼容某些旧版本或精简版
                logger.warning(f"[Assets] 缺失第 {i} 组 Codec 嵌入表，尝试跳过...")
                continue
                
            table = np.load(path)
            self.emb_tables.append(table)
            
            # 预投影：加速工匠模型推理。
            # 工匠模型接收的输入是经过 projection 的 1024 维向量
            # 我们提前将 2048->1024 计算好，推理时直接 O(1) 取值
            table_1024 = table @ pw.T + pb
            self.emb_tables_1024.append(table_1024)
            
        logger.info(f"✅ [Assets] 资产加载完成 (Codec 表数量: {len(self.emb_tables)})")

    def get_text_embedding(self, token_id: int) -> np.ndarray:
        return self.text_table[token_id]

    def get_codec_embedding(self, q_idx: int, code: int) -> np.ndarray:
        """获取原始 2048 维嵌入"""
        return self.emb_tables[q_idx][code]

    def get_codec_embedding_1024(self, q_idx: int, code: int) -> np.ndarray:
        """获取预投影后的 1024 维嵌入"""
        return self.emb_tables_1024[q_idx][code]
