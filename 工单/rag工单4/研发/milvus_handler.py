"""
milvus_handler.py - RAG工单4 Milvus向量数据库操作模块
工单编号: 人工智能NLP-RAG-图像内容解析及检索优化
功能: 封装Milvus向量数据库的增删改查操作，
      支持向量相似度搜索和标量过滤
"""

import logging
import json
import os
import time

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
    提供连接管理、集合操作、数据插入和向量搜索功能
    """
    
    def __init__(self):
        """初始化Milvus连接（延迟连接）"""
        self.client = None
        self.collection = None
        self._is_connected = False
    
    def connect(self):
        """
        连接Milvus服务
        使用pymilvus 2.x版本API
        """
        if self._is_connected:
            return
        
        logger.info(f"连接Milvus: {MILVUS_HOST}:{MILVUS_PORT}")
        
        try:
            from pymilvus import connections, utility
            
            # 建立连接
            connections.connect(
                alias="default",
                host=MILVUS_HOST,
                port=MILVUS_PORT,
            )
            
            self._is_connected = True
            logger.info("Milvus连接成功!")
            
        except Exception as e:
            logger.error(f"Milvus连接失败: {e}")
            raise
    
    def create_collection(self, drop_if_exists=False):
        """
        创建集合（表）
        参数:
            drop_if_exists: 如果集合已存在是否删除重建
        """
        self.connect()
        
        try:
            from pymilvus import utility, Collection, CollectionSchema, FieldSchema, DataType
            
            # 检查集合是否已存在
            if utility.has_collection(MILVUS_COLLECTION):
                if drop_if_exists:
                    # 删除旧集合
                    utility.drop_collection(MILVUS_COLLECTION)
                    logger.info(f"已删除旧集合: {MILVUS_COLLECTION}")
                else:
                    # 直接使用已有集合
                    self.collection = Collection(MILVUS_COLLECTION)
                    logger.info(f"使用已有集合: {MILVUS_COLLECTION}")
                    return
            
            # 定义字段结构
            # id: 主键（自增）
            # vector: 稠密向量（BGE-M3输出1024维）
            # chunk_index: 块索引
            # content: 文本内容
            # page_num: 页码
            # source_pdf: 来源PDF
            # has_image: 是否包含图片
            # image_file: 关联图片文件名
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=MILVUS_DIMENSION),
                FieldSchema(name="chunk_index", dtype=DataType.INT64),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="content_length", dtype=DataType.INT64),
                FieldSchema(name="page_num", dtype=DataType.INT64),
                FieldSchema(name="source_pdf", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="has_image", dtype=DataType.BOOL),
                FieldSchema(name="image_file", dtype=DataType.VARCHAR, max_length=255),
            ]
            
            # 创建集合schema
            schema = CollectionSchema(
                fields=fields,
                description=f"RAG工单4 - PDF图像内容解析及检索优化",
            )
            
            # 创建集合
            self.collection = Collection(
                name=MILVUS_COLLECTION,
                schema=schema,
                using="default",
            )
            
            # 创建向量索引（IVF_FLAT，加速搜索）
            index_params = {
                "metric_type": "IP",        # 内积相似度（向量已归一化）
                "index_type": "IVF_FLAT",   # 索引类型
                "params": {"nlist": 1024},  # 聚类中心数
            }
            self.collection.create_index(
                field_name="vector",
                index_params=index_params,
            )
            
            logger.info(f"集合创建成功: {MILVUS_COLLECTION}")
            
            # 加载集合到内存
            self.collection.load()
            
        except Exception as e:
            logger.error(f"创建集合失败: {e}")
            raise
    
    def insert_data(self, vectors, chunk_texts, chunk_metas):
        """
        批量插入向量和元数据
        参数:
            vectors: 稠密向量列表 [[0.1, 0.2, ...], ...]
            chunk_texts: 文本内容列表 ["内容1", "内容2", ...]
            chunk_metas: 元数据列表 [{"chunk_index": 0, ...}, ...]
        返回:
            list: 插入数据的ID列表
        """
        self.connect()
        
        # 确保集合存在
        if not self.collection:
            self.create_collection()
        
        logger.info(f"开始插入 {len(vectors)} 条数据到Milvus")
        
        # 准备插入数据
        chunk_indices = [m["chunk_index"] for m in chunk_metas]
        page_nums = [m["page_num"] for m in chunk_metas]
        source_pdfs = [m["source_pdf"] for m in chunk_metas]
        has_images = [m["has_image"] for m in chunk_metas]
        image_files = [m.get("image_file", "") for m in chunk_metas]
        content_lengths = [len(t) for t in chunk_texts]
        
        # 分批插入（每批100条）
        batch_size = 100
        all_ids = []
        
        for i in range(0, len(vectors), batch_size):
            end = min(i + batch_size, len(vectors))
            
            entities = [
                vectors[i:end],                  # vector
                chunk_indices[i:end],            # chunk_index
                chunk_texts[i:end],              # content
                content_lengths[i:end],          # content_length
                page_nums[i:end],                # page_num
                source_pdfs[i:end],              # source_pdf
                has_images[i:end],               # has_image
                image_files[i:end],              # image_file
            ]
            
            # 插入数据
            insert_result = self.collection.insert(entities)
            all_ids.extend(insert_result.primary_keys)
            
            logger.info(f"已插入 {i+1}-{end}/{len(vectors)} 条")
        
        # 刷新数据
        self.collection.flush()
        
        logger.info(f"数据插入完成! 共 {len(all_ids)} 条")
        return all_ids
    
    def search(self, query_vector, top_k=5):
        """
        向量相似度搜索
        参数:
            query_vector: 查询向量（列表形式）
            top_k: 返回最相似的k条结果
        返回:
            list: [{
                "id": int,
                "distance": float,
                "chunk_index": int,
                "content": str,
                "page_num": int,
                "source_pdf": str,
                "has_image": bool,
                "image_file": str,
            }, ...]
        """
        self.connect()
        
        # 确保集合已加载
        if self.collection:
            self.collection.load()
        else:
            from pymilvus import Collection
            self.collection = Collection(MILVUS_COLLECTION)
            self.collection.load()
        
        logger.info(f"搜索 Top-{top_k} 相似向量")
        
        # 搜索参数
        search_params = {
            "metric_type": "IP",
            "params": {"nprobe": 16},
        }
        
        # 执行搜索
        results = self.collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            output_fields=[
                "chunk_index", "content", "content_length",
                "page_num", "source_pdf", "has_image", "image_file"
            ],
        )
        
        # 解析搜索结果
        search_results = []
        for hits in results:
            for hit in hits:
                search_results.append({
                    "id": hit.id,
                    "distance": hit.distance,
                    "chunk_index": hit.entity.get("chunk_index"),
                    "content": hit.entity.get("content"),
                    "content_length": hit.entity.get("content_length"),
                    "page_num": hit.entity.get("page_num"),
                    "source_pdf": hit.entity.get("source_pdf"),
                    "has_image": hit.entity.get("has_image"),
                    "image_file": hit.entity.get("image_file"),
                })
        
        logger.info(f"搜索完成，找到 {len(search_results)} 条结果")
        return search_results
    
    def drop_collection(self):
        """删除集合"""
        self.connect()
        
        try:
            from pymilvus import utility
            if utility.has_collection(MILVUS_COLLECTION):
                utility.drop_collection(MILVUS_COLLECTION)
                logger.info(f"集合已删除: {MILVUS_COLLECTION}")
                self.collection = None
        except Exception as e:
            logger.error(f"删除集合失败: {e}")


if __name__ == "__main__":
    """单独测试Milvus功能"""
    handler = MilvusHandler()
    handler.create_collection(drop_if_exists=True)
    print("Milvus集合创建成功!")
