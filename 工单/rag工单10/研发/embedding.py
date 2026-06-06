"""
模块功能: BGE-M3 文本向量化模块
使用 sentence-transformers 加载本地 BGE-M3 模型，将文本转换为 1024 维向量
支持 GPU 推理和批处理，使用单例模式避免重复加载模型
工单编号: 人工智能NLP-RAG-金融问答系统部署
"""

import logging      # 日志记录模块
from typing import List, Optional  # 类型提示
import numpy as np  # 数值计算库，用于向量操作

# 全局模型实例缓存，实现单例模式
_model_instance = None

# 获取当前模块的日志记录器
logger = logging.getLogger("embedding")


def get_model():
    """获取 BGE-M3 模型的单例实例

    模型只在首次调用时从本地路径加载，
    后续调用复用已加载的实例，避免重复加载耗费内存。

    Returns:
        SentenceTransformer 模型对象，加载失败时返回 None
    """
    global _model_instance
    # 如果模型已加载，直接返回缓存实例
    if _model_instance is not None:
        return _model_instance
    # 首次加载模型
    try:
        from sentence_transformers import SentenceTransformer
        from app.config import config
        # 根据配置决定使用 GPU 还是 CPU
        device: str = "cuda" if config.USE_GPU else "cpu"
        logger.info(f"正在加载 BGE-M3 模型，设备: {device}")
        # 从本地路径加载模型（Windows 桌面 /mnt/c/...）
        model_path: str = config.BGE_MODEL_PATH
        _model_instance = SentenceTransformer(
            model_name_or_path=model_path,
            device=device,
        )
        logger.info("BGE-M3 模型加载成功")
        return _model_instance
    except Exception as e:
        # 模型加载失败时记录错误
        logger.error(f"BGE-M3 模型加载失败: {e}")
        return None


def generate_embeddings(texts: List[str], batch_size: int = None) -> Optional[np.ndarray]:
    """将文本列表批量向量化为 numpy 数组

    Args:
        texts: 待向量化的文本列表
        batch_size: 批处理大小，默认使用配置中的安全值 (2)

    Returns:
        形状为 (len(texts), 1024) 的浮点数组，失败时返回 None
    """
    from app.config import config
    # 使用配置中的默认批处理大小
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    # 输入为空时直接返回
    if not texts:
        logger.warning("输入文本列表为空，无法生成向量")
        return None
    # 获取模型实例
    model = get_model()
    if model is None:
        logger.error("模型未加载，无法生成向量")
        return None
    try:
        # 执行向量化编码，返回归一化的 numpy 数组
        logger.info(f"开始向量化: {len(texts)} 条文本, 批大小 {batch_size}")
        embeddings: np.ndarray = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # L2 归一化，方便后续余弦相似度计算
        )
        logger.info(f"向量化完成: 形状 {embeddings.shape}")
        return embeddings
    except Exception as e:
        # 向量化过程出错
        logger.error(f"向量化过程出错: {e}")
        return None


def embed_query(query: str) -> Optional[np.ndarray]:
    """对单条查询文本进行向量化

    用于 RAG 检索阶段，将用户的问题转换为查询向量，
    然后在 Milvus 中执行相似度搜索。

    Args:
        query: 用户输入的问题文本

    Returns:
        形状为 (1024,) 的查询向量，失败时返回 None
    """
    # 调用批量接口处理单条文本
    embeddings = generate_embeddings([query], batch_size=1)
    if embeddings is not None and len(embeddings) > 0:
        # 返回第一个（也是唯一一个）向量
        return embeddings[0]
    return None


def compute_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """计算查询向量与文档向量之间的余弦相似度

    向量已经过 L2 归一化，直接点积即可得到余弦相似度。

    Args:
        query_vec: 查询向量，形状 (1024,)
        doc_vecs: 文档向量矩阵，形状 (n, 1024)

    Returns:
        相似度分数数组，形状 (n,)
    """
    # 归一化后的向量点积 = 余弦相似度
    scores: np.ndarray = np.dot(doc_vecs, query_vec)
    return scores
