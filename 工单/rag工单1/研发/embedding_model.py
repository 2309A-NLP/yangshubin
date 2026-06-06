# -*- coding: utf-8 -*-
"""
嵌入模型模块 —— 加载 BGE-M3 模型并生成文本向量
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
"""

from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from config import BGE_M3_PATH, BGE_M3_DIM, DEVICE


class EmbeddingModel:
    """
    BGE-M3 嵌入模型封装
    功能：加载模型 → 批量文本编码 → 输出稠密向量
    """

    def __init__(self, model_path: str = BGE_M3_PATH,
                 device: str = DEVICE):
        """
        初始化嵌入模型

        Args:
            model_path: BGE-M3 模型本地路径
            device: 运行设备（cuda / cpu）
        """
        self.model_path = model_path
        self.device = device
        self.model: Optional[SentenceTransformer] = None

    def load_model(self):
        """
        加载 BGE-M3 模型（延迟加载，仅在首次调用编码时执行）
        """
        if self.model is not None:
            return  # 已加载则跳过

        print(f"[Embedding] 正在加载 BGE-M3 模型（来自: {self.model_path}）")
        self.model = SentenceTransformer(self.model_path, device=self.device)
        print(f"[Embedding] 模型加载完成，向量维度: {BGE_M3_DIM}")

    def encode(self, texts: List[str],
               batch_size: int = 32,
               normalize: bool = True) -> np.ndarray:
        """
        将文本列表编码为向量矩阵

        Args:
            texts: 待编码的文本列表
            batch_size: 批处理大小
            normalize: 是否对向量进行L2归一化（Milvus内积检索建议归一化）

        Returns:
            numpy 数组，形状 (len(texts), dim)
        """
        self.load_model()
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=True
        )
        return np.array(embeddings, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """
        对单条查询文本编码（使用 query_prefix 优化检索）

        Args:
            query: 用户提问语句

        Returns:
            形状为 (1, dim) 的向量
        """
        self.load_model()
        # BGE-M3 对查询语句加前缀可提升检索效果
        prefixed_query = f"为这个句子生成表示以用于检索相关文章：{query}"
        embedding = self.model.encode(
            prefixed_query,
            normalize_embeddings=True
        )
        return np.array([embedding], dtype=np.float32)

    def get_dimension(self) -> int:
        """返回向量维度"""
        return BGE_M3_DIM


if __name__ == "__main__":
    # 自测模块：测试模型加载和编码
    emb_model = EmbeddingModel()
    test_texts = ["武汉兴图新科电子股份有限公司", "招股说明书"]
    vectors = emb_model.encode(test_texts)
    print(f"编码向量形状: {vectors.shape}")
    print(f"向量维度: {vectors.shape[1]}")

    query_vec = emb_model.encode_query("公司收入是多少？")
    print(f"查询向量形状: {query_vec.shape}")
