"""\nmilvus_handler.py - RAG工单9 Milvus向量库模块\n工单编号: 人工智能NLP-RAG-Graph RAG 优化任务
需求: 检索层面 — 向量存储与相似度搜索
功能: Milvus集合管理(创建/插入/搜索/删除/IP内积搜索)
"""

import logging, json, time
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '设计'))
from config import MILVUS_HOST, MILVUS_PORT, MILVUS_COLL, MILVUS_DIM, OUTPUT_DIR, LOG_FMT, LOG_DATEFMT

logging.basicConfig(format=LOG_FMT, datefmt=LOG_DATEFMT, level=logging.INFO)
logger = logging.getLogger("milvus_handler")


class MilvusManager:
    """Milvus向量库管理器，封装连接创建和集合操作"""
    def __init__(self):
        self.client = None
        self.collection = None

    def connect(self):
        """连接到Milvus服务，使用pymilvus的连接器"""
        if self.client:
            return
        from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType
        logger.info(f"连接Milvus: {MILVUS_HOST}:{MILVUS_PORT}")
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
        logger.info("Milvus连接成功")
        self._Collection = Collection
        self._CollectionSchema = CollectionSchema
        self._FieldSchema = FieldSchema
        self._DataType = DataType

    def _ensure_collection(self):
        """确保集合存在，不存在则创建并建立IVF_FLAT索引"""
        from pymilvus import utility
        if utility.has_collection(MILVUS_COLL):
            self.collection = self._Collection(MILVUS_COLL)
            self.collection.load()
            return
        logger.info(f"创建集合: {MILVUS_COLL}")
        fields = [
            self._FieldSchema(name="id", dtype=self._DataType.INT64, is_primary=True, auto_id=True),
            self._FieldSchema(name="embedding", dtype=self._DataType.FLOAT_VECTOR, dim=MILVUS_DIM),
            self._FieldSchema(name="content", dtype=self._DataType.VARCHAR, max_length=8000),
            self._FieldSchema(name="source_pdf", dtype=self._DataType.VARCHAR, max_length=200),
            self._FieldSchema(name="page_num", dtype=self._DataType.INT64),
            self._FieldSchema(name="chunk_index", dtype=self._DataType.INT64),
        ]
        schema = self._CollectionSchema(fields, description=f"{MILVUS_COLL}-GraphRAG年报")
        self.collection = self._Collection(MILVUS_COLL, schema)
        index_params = {"metric_type": "IP", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
        self.collection.create_index("embedding", index_params)
        self.collection.load()
        logger.info("集合创建完成")

    def insert(self, vectors, texts, metas):
        """插入向量和文本到Milvus集合中"""
        self.connect()
        self._ensure_collection()
        logger.info(f"插入 {len(vectors)} 条数据...")
        start = time.time()
        entities = [
            vectors,
            texts,
            [m["source_pdf"] for m in metas],
            [m["page_num"] for m in metas],
            [m["index"] for m in metas],
        ]
        insert_result = self.collection.insert(entities)
        self.collection.flush()
        logger.info(f"插入完成! {time.time() - start:.2f}秒, {len(insert_result.primary_keys)} 条")
        return len(insert_result.primary_keys)

    def search(self, query_vector, top_k=10, output_fields=None):
        """
        搜索最相似的文本块（向量相似度检索）
        返回: list of dicts，含score, content, source_pdf, page_num等信息
        """
        self.connect()
        self._ensure_collection()
        if output_fields is None:
            output_fields = ["content", "source_pdf", "page_num", "chunk_index"]
        search_params = {"metric_type": "IP", "params": {"nprobe": 16}}
        start = time.time()
        results = self.collection.search(
            data=[query_vector], anns_field="embedding",
            param=search_params, limit=top_k, output_fields=output_fields)
        logger.info(f"搜索完成! {time.time() - start:.3f}秒")
        hits = []
        for hits_list in results:
            for hit in hits_list:
                hits.append({
                    "score": hit.score,
                    "content": hit.entity.get("content"),
                    "source_pdf": hit.entity.get("source_pdf"),
                    "page_num": hit.entity.get("page_num"),
                    "chunk_index": hit.entity.get("chunk_index"),
                })
        return hits

    def drop(self):
        """删除集合，用于重建时清除旧数据"""
        self.connect()
        from pymilvus import utility
        if utility.has_collection(MILVUS_COLL):
            utility.drop_collection(MILVUS_COLL)
            logger.info(f"集合已删除: {MILVUS_COLL}")

    def close(self):
        """关闭Milvus连接"""
        from pymilvus import connections
        connections.disconnect("default")


def create_index_and_insert(vectors, texts, metas):
    """便捷函数：重建Milvus集合并插入数据"""
    mgr = MilvusManager()
    mgr.drop()
    count = mgr.insert(vectors, texts, metas)
    mgr.close()
    return count


if __name__ == "__main__":
    """单独测试Milvus连接"""
    mgr = MilvusManager()
    mgr.connect()
    mgr._ensure_collection()
    print(f"集合{MILVUS_COLL}就绪")
    mgr.close()
