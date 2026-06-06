# -*- coding: utf-8 -*-
"""
嵌入模型模块 —— 加载BGE-M3生成稠密向量
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
"""

import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from config import BGE_M3_PATH, BGE_M3_DIM, DEVICE


class EmbeddingModel:
    """BGE-M3 嵌入模型封装（延迟加载）"""

    def __init__(self):
        self.model_path = BGE_M3_PATH
        self.device = DEVICE
        self._model: Optional[SentenceTransformer] = None

    def _load(self):
        if self._model is not None:
            return
        print(f"[BGE-M3] 加载模型: {self.model_path}  device={self.device}")
        self._model = SentenceTransformer(self.model_path, device=self.device)
        print(f"[BGE-M3] 就绪 dim={BGE_M3_DIM}")

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """批量编码文本为稠密向量"""
        self._load()
        return np.array(
            self._model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True),
            dtype=np.float32
        )

    def encode_query(self, query: str) -> np.ndarray:
        """编码查询（加BGE检索前缀提升效果）"""
        self._load()
        prefixed = f"为这个句子生成表示以用于检索相关文章：{query}"
        emb = self._model.encode(prefixed, normalize_embeddings=True)
        return np.array([emb], dtype=np.float32)


if __name__ == "__main__":
    m = EmbeddingModel()
    v = m.encode(["测试文本"])
    print(f"向量形状: {v.shape}")
    qv = m.encode_query("注册资本是多少？")
    print(f"查询向量形状: {qv.shape}")
