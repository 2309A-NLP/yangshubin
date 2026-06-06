"""\nembedder.py - RAG工单9 向量嵌入模块\n工单编号: 人工智能NLP-RAG-Graph RAG 优化任务
需求: 检索层面 — BGE-M3文本向量化(FP16+batch=2适配RTX5060)
功能: 批量编码CCF年报文本为稠密向量，分批处理防止显存溢出
"""

import logging, json, time, os
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '设计'))
from config import BGE_M3_PATH, BGE_M3_BATCH, BGE_M3_MAX_LEN, BGE_M3_DEVICE, BGE_M3_FP16, OUTPUT_DIR, LOG_FMT, LOG_DATEFMT

logging.basicConfig(format=LOG_FMT, datefmt=LOG_DATEFMT, level=logging.INFO)
logger = logging.getLogger("embedder")


class BgeM3Embedder:
    """BGE-M3嵌入模型封装，支持延迟加载和批量编码"""
    def __init__(self):
        self.model = None
        self._loaded = False

    def load(self):
        """延迟加载BGE-M3模型（sentence-transformers兼容新版transformers）"""
        if self._loaded:
            return
        logger.info(f"加载BGE-M3: {BGE_M3_PATH}")
        start = time.time()
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(BGE_M3_PATH, device=BGE_M3_DEVICE)
            if BGE_M3_FP16:
                self.model.half()
            self._loaded = True
            logger.info(f"BGE-M3加载完成! {time.time() - start:.1f}秒")
        except Exception as e:
            logger.error(f"加载失败: {e}")
            raise

    def encode(self, texts):
        """批量编码文本为稠密向量"""
        self.load()
        logger.info(f"编码 {len(texts)} 条文本")
        start = time.time()
        vecs = self.model.encode(
            texts, batch_size=BGE_M3_BATCH,
            show_progress_bar=True,
            normalize_embeddings=True)
        logger.info(f"编码完成! {time.time() - start:.2f}秒")
        return {"dense_vecs": vecs}

    def encode_query(self, query):
        """编码单条查询文本"""
        return self.encode([query])


def create_embeddings(chunks):
    """
    为所有文本块创建向量嵌入
    参数:
        chunks: 文本块列表
    返回:
        dict: 包含dense_vectors, chunk_texts, chunk_metas
    """
    embedder = BgeM3Embedder()
    texts = [c["content"] for c in chunks]
    # 分批处理，每批50条
    batch_size = 50
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        logger.info(f"批次 {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}")
        result = embedder.encode(batch)
        all_vecs.extend(result["dense_vecs"].tolist())
    metas = [{"index": c["index"], "source_pdf": c["source_pdf"],
              "page_num": c["page_num"]} for c in chunks]
    result_data = {
        "dense_vectors": all_vecs,
        "chunk_texts": texts,
        "chunk_metas": metas,
    }
    out_path = os.path.join(OUTPUT_DIR, "embeddings.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"count": len(all_vecs), "dim": len(all_vecs[0]) if all_vecs else 0},
                  f, ensure_ascii=False, indent=2)
    logger.info(f"向量化完成! {len(all_vecs)} 个向量")
    return result_data


if __name__ == "__main__":
    """单独测试向量化功能"""
    e = BgeM3Embedder()
    r = e.encode(["平安银行2019年盈利情况如何？"])
    print(f"向量形状: {r['dense_vecs'].shape}")
