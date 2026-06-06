"""
embedder.py - RAG工单5 向量嵌入模块
工单编号: 人工智能NLP-RAG-Query理解优化任务
完成需求: 使用BGE-M3模型将文本块转换为稠密向量，支持FP16半精度
功能说明: 延迟加载模型，batch=2+seq=1024+FP16安全配置，分批编码
"""

import logging  # 日志记录
import json     # 保存向量信息
import os       # 文件路径
import time     # 计时

# 导入配置项
from config import (
    BGE_M3_MODEL_PATH, BGE_M3_BATCH_SIZE, BGE_M3_MAX_LENGTH,
    BGE_M3_DEVICE, BGE_M3_USE_FP16, OUTPUT_DIR,
    LOG_FORMAT, LOG_DATE_FORMAT
)

# 设置日志记录器
logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT, level=logging.INFO)
logger = logging.getLogger("embedder")


class BgeM3Embedder:
    """
    BGE-M3嵌入模型封装类
    支持文本向量化，FP16半精度减少显存占用
    使用延迟加载：首次调用encode时才加载模型
    """

    def __init__(self):
        """初始化，模型先设为None，等待延迟加载"""
        self.model = None
        self._is_loaded = False

    def load_model(self):
        """
        加载BGE-M3模型（使用sentence-transformers，兼容性更好）
        """
        if self._is_loaded:
            return

        logger.info(f"加载BGE-M3模型: {BGE_M3_MODEL_PATH}")
        start = time.time()

        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                BGE_M3_MODEL_PATH,
                device=BGE_M3_DEVICE,
                trust_remote_code=True,
            )
            if BGE_M3_USE_FP16 and BGE_M3_DEVICE == "cuda":
                self.model.half()
                logger.info("已启用FP16半精度推理")

            self._is_loaded = True
            logger.info(f"BGE-M3加载完成! 耗时: {time.time()-start:.1f}秒")

        except ImportError:
            logger.error("请安装: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"加载失败: {e}")
            raise

    def encode(self, texts):
        """
        批量编码文本为稠密向量（sentence-transformers版本）
        返回: dict: {"dense_vecs": ndarray}
        """
        self.load_model()
        logger.info(f"编码 {len(texts)} 条文本")

        start = time.time()
        embeddings = self.model.encode(
            sentences=texts,
            batch_size=BGE_M3_BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        elapsed = time.time() - start
        logger.info(f"编码完成! 耗时: {elapsed:.2f}秒")
        return {"dense_vecs": embeddings}

    def encode_query(self, query):
        """编码单条查询文本"""
        return self.encode([query])


def create_embeddings(chunks):
    """
    为所有文本块创建向量嵌入
    参数:
        chunks: text_chunker生成的文本块列表
    返回:
        dict: 包含dense_vectors, chunk_texts, chunk_metas
    """
    embedder = BgeM3Embedder()
    logger.info(f"为 {len(chunks)} 个文本块生成向量")

    # 提取所有chunk的文本内容
    texts = [c["content"] for c in chunks]

    # 分批处理，每批50条文本，减少内存峰值
    batch_size = 50
    all_dense_vecs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(texts) - 1) // batch_size + 1
        logger.info(f"编码批次 {batch_num}/{total_batches}")
        result = embedder.encode(batch)
        # 将numpy数组转为list，便于JSON序列化
        all_dense_vecs.extend(result["dense_vecs"].tolist())

    # 准备chunk元数据列表
    chunk_metas = [{
        "index": c["index"],
        "source_pdf": c["source_pdf"],
        "page_num": c["page_num"],
    } for c in chunks]

    # 组装最终结果
    result = {
        "dense_vectors": all_dense_vecs,  # 稠密向量列表
        "chunk_texts": texts,              # 原始文本
        "chunk_metas": chunk_metas,        # 元数据
    }

    # 保存向量统计信息到文件
    output_path = os.path.join(OUTPUT_DIR, "embeddings.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "vector_count": len(all_dense_vecs),
            "vector_dim": len(all_dense_vecs[0]) if all_dense_vecs else 0,
        }, f, ensure_ascii=False, indent=2)

    logger.info(f"向量嵌入完成! 共 {len(all_dense_vecs)} 个向量")
    return result


if __name__ == "__main__":
    """单独测试嵌入功能"""
    embedder = BgeM3Embedder()
    texts = ["武汉兴图新科注册资本是多少？", "力源信息发行股数多少？"]
    result = embedder.encode(texts)
    print(f"向量形状: {result['dense_vecs'].shape}")
