"""
embedder.py - RAG工单4 向量嵌入模块
工单编号: 人工智能NLP-RAG-图像内容解析及检索优化
功能: 使用BGE-M3模型将文本块转换为向量嵌入，
      支持稠密检索和稀疏检索
"""

import logging
import json
import os
import time

# 导入配置
from config import (
    BGE_M3_MODEL_PATH, BGE_M3_BATCH_SIZE, BGE_M3_MAX_SEQ_LENGTH,
    BGE_M3_DEVICE, BGE_M3_USE_FP16, OUTPUT_DIR,
    LOG_FORMAT, LOG_DATE_FORMAT
)

# 设置日志
logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT, level=logging.INFO)
logger = logging.getLogger("embedder")


class BgeM3Embedder:
    """
    BGE-M3嵌入模型封装
    支持文本向量化，采用FP16半精度减少显存占用
    """
    
    def __init__(self):
        """初始化BGE-M3模型（延迟加载）"""
        self.model = None
        self.tokenizer = None
        self._is_loaded = False
    
    def load_model(self):
        """
        加载BGE-M3模型（使用sentence-transformers，兼容性更好）
        """
        if self._is_loaded:
            return

        logger.info(f"正在加载BGE-M3模型: {BGE_M3_MODEL_PATH}")
        start_time = time.time()

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
            elapsed = time.time() - start_time
            logger.info(f"BGE-M3模型加载完成! 耗时: {elapsed:.2f}秒")

        except ImportError:
            logger.error("请安装 sentence-transformers: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"加载BGE-M3模型失败: {e}")
            raise
    
    def encode_texts(self, texts):
        """
        批量编码文本为向量（sentence-transformers版本）
        返回: dict: {"dense_vecs": numpy.ndarray}
        """
        self.load_model()

        logger.info(f"编码 {len(texts)} 条文本，batch_size={BGE_M3_BATCH_SIZE}")
        start_time = time.time()

        try:
            embeddings = self.model.encode(
                sentences=texts,
                batch_size=BGE_M3_BATCH_SIZE,
                show_progress_bar=True,
                normalize_embeddings=True,
            )

            elapsed = time.time() - start_time
            logger.info(f"编码完成! 耗时: {elapsed:.2f}秒")

            return {"dense_vecs": embeddings}

        except Exception as e:
            logger.error(f"编码文本失败: {e}")
            raise

    def encode_query(self, query):
        """
        编码单条查询文本
        返回: dict: {"dense_vecs": numpy.ndarray}
        """
        result = self.encode_texts([query])
        return {"dense_vecs": result["dense_vecs"]}
    
    def compute_scores(self, query_vecs, doc_vecs):
        """
        计算查询和文档之间的相似度分数
        参数:
            query_vecs: 查询向量
            doc_vecs: 文档向量
        返回:
            tuple: (dense_score, lexical_score, hybrid_score)
        """
        self.load_model()
        
        # 计算稠密相似度（余弦相似度）
        # 向量已归一化，内积=余弦相似度
        dense_scores = (query_vecs["dense_vecs"] @ doc_vecs["dense_vecs"].T).flatten()
        
        # 计算稀疏相似度
        sparse_scores = self.model.compute_lexical_matching_score(
            query_vecs["lexical_weights"],
            doc_vecs["lexical_weights"]
        )
        
        # 混合分数（稠密 + 稀疏加权）
        hybrid_scores = 0.5 * dense_scores + 0.5 * sparse_scores
        
        return dense_scores, sparse_scores, hybrid_scores


def create_embeddings_for_chunks(chunks):
    """
    为所有文本块创建向量嵌入
    参数:
        chunks: text_chunker生成的文本块列表
    返回:
        dict: {
            "dense_vectors": list,  # 稠密向量列表
            "chunk_texts": list,     # 对应的文本
            "chunk_metas": list,     # 对应的元数据
        }
    """
    embedder = BgeM3Embedder()
    
    logger.info(f"开始为 {len(chunks)} 个文本块生成向量嵌入")
    
    # 提取所有文本内容
    texts = [chunk["content"] for chunk in chunks]
    
    # 分批处理防止内存溢出
    batch_size = 50
    all_dense_vecs = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        logger.info(f"编码批次 {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1} ({len(batch_texts)}条)")
        
        # 编码当前批次
        embeddings = embedder.encode_texts(batch_texts)
        
        # 收集稠密向量
        all_dense_vecs.extend(embeddings["dense_vecs"].tolist())
    
    # 准备元数据
    chunk_texts = texts
    chunk_metas = [{
        "chunk_index": c["chunk_index"],
        "source_pdf": c["source_pdf"],
        "page_num": c["page_num"],
        "has_image": c["has_image"],
        "image_file": c["image_file"],
    } for c in chunks]
    
    result = {
        "dense_vectors": all_dense_vecs,
        "chunk_texts": chunk_texts,
        "chunk_metas": chunk_metas,
    }
    
    # 保存向量到文件备份
    output_path = os.path.join(OUTPUT_DIR, "embeddings.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "vector_dim": len(all_dense_vecs[0]) if all_dense_vecs else 0,
            "vector_count": len(all_dense_vecs),
            "chunk_texts_preview": [t[:100] + "..." if len(t) > 100 else t for t in chunk_texts],
            "chunk_metas": chunk_metas,
        }, f, ensure_ascii=False, indent=2)
    
    logger.info(f"向量嵌入完成! 共 {len(all_dense_vecs)} 个向量，维度 {len(all_dense_vecs[0]) if all_dense_vecs else 0}")
    return result


if __name__ == "__main__":
    """单独测试嵌入功能"""
    embedder = BgeM3Embedder()
    test_texts = ["武汉力源信息技术股份有限公司本次发行股数是多少？", "组织结构图中销售部有哪几个部门？"]
    result = embedder.encode_texts(test_texts)
    print(f"稠密向量形状: {result['dense_vecs'].shape}")
    print(f"稀疏权重数量: {len(result['lexical_weights'])}")
