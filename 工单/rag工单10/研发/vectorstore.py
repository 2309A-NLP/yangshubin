"""
模块功能: Milvus 向量数据库客户端模块
封装对 Milvus 服务的连接、集合管理、向量插入和相似度检索操作
使用 pymilvus 库与 Milvus 服务通过 gRPC 协议交互
工单编号: 人工智能NLP-RAG-金融问答系统部署
"""

import logging              # 日志记录模块
from typing import List, Dict, Optional  # 类型提示
import numpy as np          # 数值计算库，处理向量数据

# 获取当前模块的日志记录器
logger = logging.getLogger("vectorstore")


class MilvusClient:
    """Milvus 向量数据库客户端封装类，提供增删查改接口"""

    def __init__(self):
        """初始化客户端，延迟连接和集合初始化"""
        from app.config import config
        self.config = config       # 保存全局配置引用
        self.connected: bool = False  # 标记是否已连接
        self.collection = None     # Milvus 集合对象引用

    def connect(self) -> bool:
        """连接到 Milvus 服务，创建或加载指定名称的向量集合

        Returns:
            连接成功返回 True，失败返回 False
        """
        try:
            from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType
            # 建立与 Milvus 的 gRPC 连接
            connections.connect("default",
                                host=self.config.MILVUS_HOST,
                                port=self.config.MILVUS_PORT,
                                timeout=self.config.MILVUS_TIMEOUT)
            logger.info(f"Milvus 连接成功: {self.config.MILVUS_HOST}:{self.config.MILVUS_PORT}")
            # 检查集合是否已存在
            coll_name: str = self.config.MILVUS_COLLECTION
            if utility.has_collection(coll_name):
                # 集合已存在，直接加载到内存
                self.collection = Collection(name=coll_name)
                self.collection.load()
                logger.info(f"加载已有集合: {coll_name}")
            else:
                # 集合不存在，定义字段结构并创建
                fields: List[FieldSchema] = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.config.VECTOR_DIM),
                    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=255),
                ]
                # 创建集合并建立 IVF_FLAT 索引
                schema = CollectionSchema(fields, "RAG 文档向量集合")
                self.collection = Collection(name=coll_name, schema=schema)
                self.collection.create_index("embedding",
                    {"metric_type": "IP", "index_type": "IVF_FLAT", "params": {"nlist": 128}})
                self.collection.load()
                logger.info(f"创建新集合: {coll_name}")
            self.connected = True
            return True
        except Exception as e:
            logger.error(f"Milvus 连接失败: {e}")
            self.connected = False
            return False

    def insert_embeddings(self, texts: List[str], filenames: List[str],
                          embeddings: np.ndarray) -> int:
        """将文本及其对应的向量批量插入 Milvus 集合

        Args:
            texts: 文本内容列表
            filenames: 来源文件名列表
            embeddings: 向量数组，形状 (n, 1024)

        Returns:
            成功插入的记录数量
        """
        if not self.connected or self.collection is None:
            logger.error("Milvus 未连接，无法插入")
            return 0
        try:
            # 构造插入实体数据
            entities: List[list] = [embeddings.tolist(), texts, filenames]
            result = self.collection.insert(entities)
            self.collection.flush()
            count: int = len(result.primary_keys)
            logger.info(f"向量入库成功: {count} 条")
            return count
        except Exception as e:
            logger.error(f"向量入库失败: {e}")
            return 0

    def search(self, query_vector: List[float], top_k: int = None) -> List[Dict]:
        """在 Milvus 中执行向量相似度搜索

        使用内积距离（IP，即余弦相似度）查找最相似的文档块。

        Args:
            query_vector: 查询向量，形状 (1024,)
            top_k: 返回的最相似结果数，默认从配置读取 (8)

        Returns:
            搜索结果列表，每项包含 text, filename, score 字段
        """
        if not self.connected or self.collection is None:
            logger.error("Milvus 未连接，无法搜索")
            return []
        if top_k is None:
            top_k = self.config.TOP_K
        try:
            results = self.collection.search(
                data=[query_vector],
                anns_field="embedding",
                param={"metric_type": "IP", "params": {"nprobe": 16}},
                limit=top_k,
                output_fields=["text", "filename"],
            )
            # 解析搜索结果，按分数阈值过滤低质量结果
            hits: List[Dict] = []
            for r in results[0]:
                if r.score >= self.config.SCORE_THRESHOLD:
                    hits.append({
                        "text": r.entity.get("text"),
                        "filename": r.entity.get("filename"),
                        "score": round(float(r.score), 4),
                    })
            logger.info(f"向量搜索完成: 返回 {len(hits)} 条结果")
            return hits
        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            return []

    def close(self):
        """关闭与 Milvus 的连接，释放系统资源"""
        if self.connected:
            try:
                from pymilvus import connections
                connections.disconnect("default")
                self.connected = False
                logger.info("Milvus 连接已关闭")
            except Exception:
                pass


def store_embeddings(data_dir: str) -> int:
    """一站式函数: 从数据目录完成 PDF → 分块 → 向量化 → 入库全流程

    Args:
        data_dir: PDF 文件所在目录

    Returns:
        成功入库的记录数
    """
    from app.document_loader import load_documents
    from app.text_splitter import split_text
    from app.embedding import generate_embeddings
    # 加载 PDF 文档
    docs = load_documents(data_dir)
    if not docs:
        return 0
    # 对每篇文档分块，收集所有文本和文件名
    all_texts: List[str] = []
    all_filenames: List[str] = []
    for doc in docs:
        for chunk in split_text(doc["text"]):
            all_texts.append(chunk)
            all_filenames.append(doc["filename"])
    # 批量向量化
    embeddings = generate_embeddings(all_texts)
    if embeddings is None:
        return 0
    # 连接 Milvus 并插入
    client = MilvusClient()
    if not client.connect():
        return 0
    count = client.insert_embeddings(all_texts, all_filenames, embeddings)
    client.close()
    return count
