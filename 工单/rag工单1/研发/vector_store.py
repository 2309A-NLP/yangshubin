# -*- coding: utf-8 -*-
"""
向量存储模块 —— 使用 MilvusClient 管理 PDF 文本块的向量索引
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
"""

from typing import List, Dict, Optional
from pymilvus import MilvusClient, DataType

from config import MILVUS_URI, MILVUS_COLLECTION, BGE_M3_DIM


class VectorStore:
    """
    Milvus 向量数据库操作封装
    功能：创建集合 → 插入向量数据 → 相似检索 → 清空集合
    """

    def __init__(self, uri: str = MILVUS_URI,
                 collection_name: str = MILVUS_COLLECTION,
                 dim: int = BGE_M3_DIM):
        """
        初始化向量数据库客户端

        Args:
            uri: Milvus 服务地址（如 http://localhost:19530）
            collection_name: 集合名称
            dim: 向量维度
        """
        self.uri = uri
        self.collection_name = collection_name
        self.dim = dim
        self.client = MilvusClient(uri=uri)

    def create_collection(self, drop_if_exists: bool = False):
        """
        创建集合（包含主键id、文本字段、向量字段）

        Args:
            drop_if_exists: 如果集合已存在，是否删除重建
        """
        # 检查集合是否存在
        if self.client.has_collection(self.collection_name):
            if drop_if_exists:
                self.client.drop_collection(self.collection_name)
                print(f"[Milvus] 已删除旧集合: {self.collection_name}")
            else:
                print(f"[Milvus] 集合已存在: {self.collection_name}")
                return

        # 创建集合
        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=True
        )
        # 添加主键字段
        schema.add_field(
            field_name="id", datatype=DataType.INT64,
            is_primary=True, auto_id=True
        )
        # 添加文本字段，存储原始块内容
        schema.add_field(
            field_name="text", datatype=DataType.VARCHAR, max_length=2048
        )
        # 添加元数据字段（可用于存储页码、位置等）
        schema.add_field(
            field_name="metadata", datatype=DataType.VARCHAR, max_length=512
        )
        # 添加向量字段
        schema.add_field(
            field_name="embedding", datatype=DataType.FLOAT_VECTOR,
            dim=self.dim
        )

        # 创建索引参数（IVF_FLAT 是常用的近似检索索引）
        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            metric_type="IP",  # 内积检索（向量已归一化，等价于余弦相似度）
            index_type="IVF_FLAT",
            params={"nlist": 128}
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )
        print(f"[Milvus] 集合创建成功: {self.collection_name}")

    def insert_embeddings(self, chunks: List[Dict],
                          embeddings: List[List[float]]) -> int:
        """
        将文本块及其向量插入 Milvus

        Args:
            chunks: 文本块列表（含 text, chunk_id, start_pos 等字段）
            embeddings: 对应的向量列表，顺序与 chunks 一致

        Returns:
            插入的数据条数
        """
        data = []
        for chunk, emb in zip(chunks, embeddings):
            data.append({
                "text": chunk["text"],
                "metadata": f"chunk_id={chunk.get('chunk_id', '')},"
                            f"start={chunk.get('start_pos', 0)},"
                            f"end={chunk.get('end_pos', 0)}",
                "embedding": emb
            })

        result = self.client.insert(
            collection_name=self.collection_name,
            data=data
        )
        print(f"[Milvus] 成功插入 {len(data)} 条数据")
        return len(data)

    def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict]:
        """
        在 Milvus 中检索与查询向量最相似的文本块

        Args:
            query_vector: 查询向量（列表形式）
            top_k: 返回的最相似结果数量

        Returns:
            检索结果列表，每项包含 text、metadata、distance 等
        """
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=top_k,
            output_fields=["text", "metadata"],
            metric_type="IP"  # 内积检索
        )

        # 解析检索结果
        parsed = []
        if results and len(results) > 0:
            for hit in results[0]:
                parsed.append({
                    "id": hit.get("id"),
                    "text": hit.get("entity", {}).get("text", ""),
                    "metadata": hit.get("entity", {}).get("metadata", ""),
                    "score": hit.get("distance", 0.0)
                })

        return parsed

    def get_collection_stats(self) -> Dict:
        """获取集合统计信息"""
        stats = {}
        if self.client.has_collection(self.collection_name):
            stats["collection_name"] = self.collection_name
            stats["num_entities"] = self.client.get_collection_stats(
                self.collection_name
            ).get("row_count", 0)
        else:
            stats["error"] = "集合不存在"
        return stats

    def drop_collection(self):
        """删除集合"""
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
            print(f"[Milvus] 已删除集合: {self.collection_name}")

    def close(self):
        """关闭连接"""
        self.client.close()


if __name__ == "__main__":
    # 自测模块：测试连接和集合创建
    store = VectorStore()
    print(f"连接状态: {store.client}")
    store.create_collection(drop_if_exists=True)
    stats = store.get_collection_stats()
    print(f"集合统计: {stats}")
    store.close()
