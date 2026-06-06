"""
向量存储模块 - Milvus 向量数据库操作
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

本模块封装了基于 Milvus 向量数据库的存储和检索操作，是 RAG 系统的核心存储层。
主要功能：
1. 连接管理：自动连接 Milvus 服务，支持自定义主机、端口和集合名称
2. 集合管理：创建/删除集合、定义 Schema（包含文本、向量、元数据等字段）
3. 向量索引：创建 IVF_FLAT 等索引类型，优化相似度搜索性能
4. 数据插入：批量插入文本块及其对应的嵌入向量
5. 语义检索：基于向量相似度的 Top-K 检索，返回匹配的文本块及元数据
6. 集合统计：查询实体数量等统计信息

核心类：
- MilvusVectorStore: 封装所有 Milvus 操作的主类

辅助函数：
- create_vector_store(): 工厂函数，创建 MilvusVectorStore 实例
"""
import os    # 导入操作系统接口模块，用于环境变量和路径操作
import time    # 导入时间模块，用于操作耗时统计
import json    # 导入 JSON 模块，用于元数据的序列化和反序列化
import numpy as np    # 导入 NumPy 数值计算库，用于处理向量数据
from pymilvus import (    # 从 pymilvus 导入 Milvus 客户端组件
    connections, Collection, CollectionSchema,    # 连接管理器、集合对象、集合 Schema
    FieldSchema, DataType, utility,    # 字段 Schema 定义、数据类型、工具函数
)
from config import (    # 从配置模块导入 Milvus 相关配置
    MILVUS_HOST, MILVUS_PORT, MILVUS_COLLECTION,    # Milvus 主机地址、端口、默认集合名称
    MILVUS_INDEX_TYPE, MILVUS_METRIC_TYPE,    # 索引类型（如 IVF_FLAT）和距离度量方式（如 L2/COSINE）
    MILVUS_NLIST, MILVUS_NPROBE,    # 索引参数 nlist（聚类数）和搜索参数 nprobe（探测的聚类数）
    EMBEDDING_DIM, OUTPUT_DIR, log,    # 向量维度、输出目录、日志函数
)


class MilvusVectorStore:
    """Milvus 向量数据库封装"""    # 提供 Milvus 连接、集合管理、数据插入和检索的完整封装

    def __init__(self, host: str = MILVUS_HOST, port: str = MILVUS_PORT,
                 collection_name: str = MILVUS_COLLECTION,
                 dim: int = EMBEDDING_DIM):
        """
        初始化 MilvusVectorStore 实例

        Args:
            host: Milvus 服务主机地址，默认从配置中读取
            port: Milvus 服务端口，默认从配置中读取
            collection_name: 集合名称，默认从配置中读取
            dim: 向量维度，默认从配置中读取
        """
        self.host = host    # 保存 Milvus 主机地址到实例变量
        self.port = port    # 保存 Milvus 端口到实例变量
        self.collection_name = collection_name    # 保存集合名称到实例变量
        self.dim = dim    # 保存向量维度到实例变量
        self.collection = None    # 集合对象初始化为 None，连接后赋值
        self._connect()    # 初始化时自动连接 Milvus 服务

    def _connect(self):
        """连接 Milvus 服务"""    # 建立到 Milvus 服务器的 TCP 连接
        log(f"连接 Milvus: {self.host}:{self.port}", "MILVUS")    # 记录连接日志
        try:    # 捕获连接异常
            connections.connect(alias="default", host=self.host, port=self.port)    # 使用默认别名连接到 Milvus
            log("Milvus 连接成功", "MILVUS")    # 记录连接成功日志
        except Exception as e:    # 如果连接失败
            log(f"Milvus 连接失败: {e}", "ERROR")    # 记录错误日志
            raise    # 重新抛出异常，让调用方处理

    def _create_schema(self) -> CollectionSchema:
        """创建集合 Schema"""    # 定义 Milvus 集合的字段结构
        fields = [    # 定义所有字段列表
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),    # 主键 ID，自动生成
            FieldSchema(name="chunk_index", dtype=DataType.INT64),    # 文本块索引号
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),    # 文本内容，最大长度 65535 字符
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),    # 向量嵌入字段，维度由配置决定
            FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=32),    # 来源类型（如 "text" 或 "table"）
            FieldSchema(name="section_title", dtype=DataType.VARCHAR, max_length=512),    # 所属章节标题
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535),    # 附加元数据（JSON 字符串）
        ]
        schema = CollectionSchema(    # 创建集合 Schema 对象
            fields=fields,    # 传入字段定义
            description="招股说明书RAG - 表格解析与检索优化",    # 集合描述信息
        )
        return schema    # 返回创建好的 Schema

    def create_collection(self, drop_if_exists: bool = True):
        """创建集合"""    # 创建或重新创建 Milvus 集合
        if utility.has_collection(self.collection_name):    # 检查集合是否已存在
            if drop_if_exists:    # 如果允许删除已有集合
                utility.drop_collection(self.collection_name)    # 删除已有集合
                log(f"删除已有集合: {self.collection_name}", "MILVUS")    # 记录删除日志
            else:    # 如果不允许删除
                self.collection = Collection(self.collection_name)    # 直接使用已有集合
                log(f"使用已有集合: {self.collection_name}", "MILVUS")    # 记录使用日志
                return    # 返回，无需创建

        schema = self._create_schema()    # 调用 Schema 创建方法
        self.collection = Collection(    # 创建新的集合
            name=self.collection_name,    # 集合名称
            schema=schema,    # 字段 Schema
            using="default",    # 使用默认连接别名
        )
        log(f"创建集合: {self.collection_name}", "MILVUS")    # 记录创建日志

        # 创建索引    # 为新集合创建向量索引
        self._create_index()    # 调用索引创建方法

    def _create_index(self):
        """创建向量索引"""    # 为向量字段创建索引以加速检索
        if not self.collection:    # 如果集合对象不存在
            return    # 直接返回

        index_params = {    # 定义索引参数
            "metric_type": MILVUS_METRIC_TYPE,    # 距离度量方式（如 L2 欧氏距离或 IP 内积）
            "index_type": MILVUS_INDEX_TYPE,    # 索引类型（如 IVF_FLAT、HNSW 等）
            "params": {"nlist": MILVUS_NLIST},    # nlist 参数：聚类中心数量
        }
        self.collection.create_index(    # 在集合上创建索引
            field_name="embedding",    # 对向量字段创建索引
            index_params=index_params,    # 传入索引参数
        )
        log(f"创建索引: {MILVUS_INDEX_TYPE} (nlist={MILVUS_NLIST})", "MILVUS")    # 记录索引创建日志
        self.collection.load()    # 将集合加载到内存，使索引生效

    def insert(self, chunks: list, embeddings: np.ndarray):
        """
        插入向量和文本数据

        Args:
            chunks: [{chunk_index, text, type, section_title, ...}, ...]
            embeddings: numpy array (n, dim)
        """
        if not chunks or len(embeddings) == 0:    # 如果没有数据需要插入
            log("没有数据可插入", "WARN")    # 记录警告日志
            return    # 直接返回

        if len(chunks) != len(embeddings):    # 检查文本块数量和向量数量是否一致
            raise ValueError(f"chunks 数量 ({len(chunks)}) 与 embeddings 数量 ({len(embeddings)}) 不一致")    # 不一致则抛出异常

        # 构建数据    # 从 chunks 列表中提取各个字段的值
        chunk_indices = [c.get("chunk_index", i) for i, c in enumerate(chunks)]    # 提取每个文本块的索引，默认使用枚举序号
        texts = [c["text"] for c in chunks]    # 提取文本内容
        source_types = [c.get("type", "text") for c in chunks]    # 提取来源类型，默认 "text"
        section_titles = [c.get("section_title", "") for c in chunks]    # 提取章节标题，默认为空
        metadatas = []    # 初始化元数据列表
        for c in chunks:    # 遍历每个文本块
            meta = {}    # 初始化元数据字典
            if "table_index" in c:    # 如果文本块属于某个表格
                meta["table_index"] = c["table_index"]    # 保存表格索引
            if "metadata" in c:    # 如果已有元数据字段
                meta.update(c["metadata"])    # 合并到元数据字典
            metadatas.append(json.dumps(meta, ensure_ascii=False))    # 将元数据序列化为 JSON 字符串

        entities = [    # 构建 Milvus 插入数据实体，顺序与 Schema 中非自增字段对应
            chunk_indices,    # 文本块索引列表
            texts,    # 文本内容列表
            embeddings.tolist(),    # 将 NumPy 数组转为 Python 列表
            source_types,    # 来源类型列表
            section_titles,    # 章节标题列表
            metadatas,    # 元数据 JSON 字符串列表
        ]

        start = time.time()    # 记录插入开始时间
        insert_result = self.collection.insert(entities)    # 执行批量插入操作
        self.collection.flush()    # 刷新缓冲区，确保数据持久化
        elapsed = time.time() - start    # 计算插入耗时

        log(f"插入 {len(chunks)} 条数据 (耗时 {elapsed:.2f}s)", "MILVUS")    # 记录插入结果日志
        return insert_result    # 返回插入结果对象

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list:
        """
        向量检索

        Returns:
            [{id, chunk_index, text, source_type, section_title, metadata, score}, ...]
        """
        if self.collection is None:    # 如果集合对象尚未初始化
            self.collection = Collection(self.collection_name)    # 通过集合名称获取集合对象
            self.collection.load()    # 加载集合到内存

        search_params = {    # 定义搜索参数
            "metric_type": MILVUS_METRIC_TYPE,    # 距离度量方式，需与索引一致
            "params": {"nprobe": MILVUS_NPROBE},    # nprobe 参数：搜索时探测的聚类数量
        }

        results = self.collection.search(    # 执行向量相似度搜索
            data=[query_vector.tolist()],    # 将查询向量转为列表（批量搜索时用列表包裹）
            anns_field="embedding",    # 在 embedding 向量字段上进行搜索
            param=search_params,    # 搜索参数
            limit=top_k,    # 返回前 Top-K 个最相似的结果
            output_fields=["chunk_index", "text", "source_type", "section_title", "metadata"],    # 需要返回的额外字段
        )

        output = []    # 初始化输出结果列表
        for hits in results:    # 遍历每个查询向量的搜索结果
            for hit in hits:    # 遍历每个命中结果
                meta = {}    # 初始化元数据字典
                try:    # 尝试解析元数据 JSON
                    meta = json.loads(hit.entity.get("metadata") or "{}")    # 从命中的实体中提取 metadata 字段并反序列化
                except json.JSONDecodeError:    # 如果 JSON 解析失败
                    pass    # 忽略，保留空字典
                output.append({    # 构建格式化的结果条目
                    "id": hit.id,    # 实体的主键 ID
                    "chunk_index": hit.entity.get("chunk_index"),    # 文本块索引
                    "text": hit.entity.get("text"),    # 文本内容
                    "source_type": hit.entity.get("source_type"),    # 来源类型
                    "section_title": hit.entity.get("section_title"),    # 章节标题
                    "metadata": meta,    # 反序列化后的元数据字典
                    "score": hit.score,    # 相似度得分
                })

        return output    # 返回格式化的搜索结果列表

    def delete_collection(self):
        """删除集合"""    # 从 Milvus 中删除指定集合及其所有数据
        if utility.has_collection(self.collection_name):    # 检查集合是否存在
            utility.drop_collection(self.collection_name)    # 删除集合
            log(f"删除集合: {self.collection_name}", "MILVUS")    # 记录删除日志

    def get_collection_stats(self) -> dict:
        """获取集合统计信息"""    # 查询集合中的实体数量和基本统计信息
        if self.collection is None:    # 如果集合对象未初始化
            self.collection = Collection(self.collection_name)    # 通过名称获取集合对象
        self.collection.load()    # 确保集合已加载
        num_entities = self.collection.num_entities    # 获取集合中的实体数量
        log(f"集合 {self.collection_name} 共有 {num_entities} 条数据", "MILVUS")    # 记录统计日志
        return {"collection": self.collection_name, "num_entities": num_entities}    # 返回统计信息字典


def create_vector_store() -> MilvusVectorStore:
    """工厂函数"""    # 提供 MilvusVectorStore 实例的工厂方法，简化创建流程
    return MilvusVectorStore()    # 使用默认配置创建并返回 MilvusVectorStore 实例


if __name__ == "__main__":
    # 测试    # 主程序入口：测试向量存储模块功能
    store = create_vector_store()    # 创建 Milvus 向量存储实例
    store.create_collection(drop_if_exists=True)    # 创建新集合（删除已存在的集合）
    stats = store.get_collection_stats()    # 获取集合统计信息
    print(stats)    # 打印统计结果
