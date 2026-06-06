"""
milvus_handler.py - RAG工单5 Milvus向量数据库模块
工单编号: 人工智能NLP-RAG-Query理解优化任务
完成需求: Milvus向量数据库的增删改查，支持向量相似度搜索和标量过滤
功能说明: 连接管理、集合创建、批量插入、向量搜索、集合删除
"""

import logging  # 日志
import time     # 计时

# 导入配置
from config import (
    MILVUS_HOST, MILVUS_PORT, MILVUS_COLLECTION, MILVUS_DIMENSION,
    OUTPUT_DIR, LOG_FORMAT, LOG_DATE_FORMAT
)

# 设置日志
logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT, level=logging.INFO)
logger = logging.getLogger("milvus_handler")


class MilvusHandler:
    """
    Milvus向量数据库操作封装
    提供连接管理、集合CRUD、批量数据插入和向量搜索
    使用延迟连接策略，避免启动时阻塞
    """

    def __init__(self):
        """初始化，client和collection设为None，等待连接"""
        self.client = None
        self.collection = None
        self._is_connected = False

    def connect(self):
        """连接到Milvus服务（延迟连接）"""
        if self._is_connected:
            return
        logger.info(f"连接Milvus: {MILVUS_HOST}:{MILVUS_PORT}")
        try:
            from pymilvus import connections
            connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
            self._is_connected = True
            logger.info("Milvus连接成功!")
        except Exception as e:
            logger.error(f"连接失败: {e}")
            raise

    def create_collection(self, drop_if_exists=False):
        """创建或获取集合，定义字段结构和向量索引"""
        self.connect()
        try:
            from pymilvus import utility, Collection, CollectionSchema, FieldSchema, DataType

            # 检查集合是否已存在
            if utility.has_collection(MILVUS_COLLECTION):
                if drop_if_exists:
                    # 删除重建
                    utility.drop_collection(MILVUS_COLLECTION)
                    logger.info(f"已删除旧集合: {MILVUS_COLLECTION}")
                else:
                    # 使用已有集合
                    self.collection = Collection(MILVUS_COLLECTION)
                    logger.info(f"使用已有集合: {MILVUS_COLLECTION}")
                    return

            # 定义集合字段：主键、向量、文本、元数据
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=MILVUS_DIMENSION),
                FieldSchema(name="chunk_index", dtype=DataType.INT64),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="content_length", dtype=DataType.INT64),
                FieldSchema(name="page_num", dtype=DataType.INT64),
                FieldSchema(name="source_pdf", dtype=DataType.VARCHAR, max_length=255),
            ]
            schema = CollectionSchema(fields=fields, description="RAG工单5 - Query理解优化")
            self.collection = Collection(name=MILVUS_COLLECTION, schema=schema)

            # 创建IVF_FLAT向量索引，加速ANN搜索
            index_params = {
                "metric_type": "IP",          # 内积相似度（配合归一化向量）
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024},
            }
            self.collection.create_index(field_name="vector", index_params=index_params)
            self.collection.load()  # 加载到内存
            logger.info(f"集合创建成功: {MILVUS_COLLECTION}")

        except Exception as e:
            logger.error(f"创建集合失败: {e}")
            raise

    def insert(self, vectors, texts, metas):
        """批量插入向量和文本数据，分批每批100条"""
        self.connect()
        if not self.collection:
            self.create_collection()

        logger.info(f"插入 {len(vectors)} 条数据")

        # 从元数据中提取字段
        indices = [m["index"] for m in metas]
        page_nums = [m["page_num"] for m in metas]
        source_pdfs = [m["source_pdf"] for m in metas]
        content_lengths = [len(t) for t in texts]

        # 分批插入，每批100条防内存溢出
        batch_size = 100
        all_ids = []
        for i in range(0, len(vectors), batch_size):
            end = min(i + batch_size, len(vectors))
            # 准备插入数据实体
            entities = [
                vectors[i:end],            # vector 列
                indices[i:end],            # chunk_index 列
                texts[i:end],              # content 列
                content_lengths[i:end],    # content_length 列
                page_nums[i:end],          # page_num 列
                source_pdfs[i:end],        # source_pdf 列
            ]
            result = self.collection.insert(entities)
            all_ids.extend(result.primary_keys)
            logger.info(f"已插入 {i+1}-{end}/{len(vectors)}")

        # 刷新确保数据持久化
        self.collection.flush()
        logger.info(f"插入完成! 共 {len(all_ids)} 条")
        return all_ids

    def search(self, query_vector, top_k=5):
        """向量相似度搜索"""
        self.connect()
        # 确保collection已加载
        if self.collection:
            self.collection.load()
        else:
            from pymilvus import Collection
            self.collection = Collection(MILVUS_COLLECTION)
            self.collection.load()

        # 设置搜索参数
        search_params = {"metric_type": "IP", "params": {"nprobe": 16}}
        results = self.collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            output_fields=["chunk_index", "content", "page_num", "source_pdf"],
        )

        # 解析搜索结果
        search_results = []
        for hits in results:
            for hit in hits:
                try:
                    # 兼容不同pymilvus版本API
                    fields = hit.fields if hasattr(hit, 'fields') else hit.entity
                    search_results.append({
                        "id": hit.id,
                        "distance": hit.distance,
                        "chunk_index": fields.get("chunk_index", 0),
                        "content": fields.get("content", ""),
                        "page_num": fields.get("page_num", 0),
                        "source_pdf": fields.get("source_pdf", ""),
                    })
                except Exception:
                    # 降级处理：直接转为字符串
                    search_results.append({
                        "id": hit.id, "distance": hit.distance,
                        "chunk_index": 0, "content": str(hit),
                        "page_num": 0, "source_pdf": "",
                    })
        logger.info(f"搜索完成，返回 {len(search_results)} 条")
        return search_results

    def drop(self):
        """删除整个集合（清空数据）"""
        self.connect()
        try:
            from pymilvus import utility
            if utility.has_collection(MILVUS_COLLECTION):
                utility.drop_collection(MILVUS_COLLECTION)
                logger.info(f"集合已删除: {MILVUS_COLLECTION}")
                self.collection = None
        except Exception as e:
            logger.error(f"删除失败: {e}")


if __name__ == "__main__":
    """单独测试Milvus连接和集合创建"""
    h = MilvusHandler()
    h.create_collection(drop_if_exists=True)
    print("集合创建成功!")
