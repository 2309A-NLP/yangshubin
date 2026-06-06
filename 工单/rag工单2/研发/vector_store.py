# -*- coding: utf-8 -*-
"""
向量存储模块 —— Milvus 建表、插入、检索
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
"""

from typing import List, Dict
from pymilvus import MilvusClient, DataType
from config import MILVUS_URI, MILVUS_COLLECTION, BGE_M3_DIM


class VectorStore:
    """Milvus 向量数据库操作"""

    def __init__(self):
        self.uri = MILVUS_URI
        self.collection = MILVUS_COLLECTION
        self.dim = BGE_M3_DIM
        self.client = MilvusClient(uri=self.uri)

    def create_collection(self, drop_if_exists: bool = False):
        """创建集合（含text/metadata字段 + 向量字段）"""
        if self.client.has_collection(self.collection):
            if drop_if_exists:
                self.client.drop_collection(self.collection)
                print(f"[Milvus] 删除旧集合: {self.collection}")
            else:
                print(f"[Milvus] 集合已存在: {self.collection}")
                return

        schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("text", DataType.VARCHAR, max_length=8192)
        schema.add_field("metadata", DataType.VARCHAR, max_length=512)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self.dim)

        index_params = MilvusClient.prepare_index_params()
        index_params.add_index("embedding", metric_type="IP", index_type="IVF_FLAT", params={"nlist": 128})

        self.client.create_collection(self.collection, schema=schema, index_params=index_params)
        print(f"[Milvus] 集合创建成功: {self.collection}")

    def insert(self, chunks: List[Dict], embeddings: List[List[float]]) -> int:
        """插入文本块+向量"""
        data = [{
            "text": c["text"],
            "metadata": f"chunk_id={c.get('chunk_id','')}",
            "embedding": emb
        } for c, emb in zip(chunks, embeddings)]

        self.client.insert(self.collection, data)
        print(f"[Milvus] 插入 {len(data)} 条")
        return len(data)

    def search(self, query_vector: List[float], top_k: int = 10) -> List[Dict]:
        """向量检索"""
        results = self.client.search(
            self.collection, [query_vector], limit=top_k,
            output_fields=["text", "metadata"], metric_type="IP"
        )
        parsed = []
        if results:
            for hit in results[0]:
                parsed.append({
                    "id": hit.get("id"),
                    "text": hit.get("entity", {}).get("text", ""),
                    "score": hit.get("distance", 0.0)
                })
        return parsed

    def count(self) -> int:
        """获取数据条数"""
        if not self.client.has_collection(self.collection):
            return 0
        stats = self.client.get_collection_stats(self.collection)
        return stats.get("row_count", 0)

    def drop(self):
        """删除集合"""
        if self.client.has_collection(self.collection):
            self.client.drop_collection(self.collection)

    def close(self):
        self.client.close()
