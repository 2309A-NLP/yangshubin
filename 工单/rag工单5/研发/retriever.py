"""
retriever.py - RAG工单5 检索与重排序模块
工单编号: 人工智能NLP-RAG-Query理解优化任务
完成需求: Milvus向量检索 + BGE-Reranker重排序，返回最相关文档片段
功能说明: 先生成查询向量→Milvus粗筛→Reranker精排→返回TOP_K结果
"""

import logging  # 日志
import time     # 计时

# 导入配置
from config import (
    TOP_K, RERANK_TOP_K, BGE_RERANKER_PATH, BGE_RERANKER_DEVICE,
    OUTPUT_DIR, LOG_FORMAT, LOG_DATE_FORMAT
)

# 设置日志
logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT, level=logging.INFO)
logger = logging.getLogger("retriever")


# ========== 公司名识别规则 ==========
_COMPANY_KEYWORDS = {
    "招股说明书1.pdf": ["武汉兴图新科", "兴图新科", "兴图", "军用领域"],
    "招股说明书2.pdf": ["武汉力源信息", "力源信息", "力源", "赵马克", "IC市场"],
}


def _extract_company(question: str) -> str:
    """从问题中识别目标公司，返回对应的PDF文件名"""
    for pdf_name, keywords in _COMPANY_KEYWORDS.items():
        for kw in keywords:
            if kw in question:
                return pdf_name
    return ""


class Retriever:
    """
    检索器：向量检索 + 重排序两阶段
    先通过Milvus进行向量相似度粗筛（召回TOP_K个候选）
    再用BGE-Reranker进行语义精排（保留RERANK_TOP_K个）
    """

    def __init__(self, embedder, milvus):
        """
        初始化检索器
        参数:
            embedder: BgeM3Embedder实例，用于编码查询
            milvus: MilvusHandler实例，用于向量搜索
        """
        self.embedder = embedder  # 嵌入模型
        self.milvus = milvus      # 向量数据库
        self.reranker = None      # 重排序模型（延迟加载）

    def _load_reranker(self):
        """BGE-Reranker已禁用（transformers版本不兼容），直接跳过"""
        self.reranker = "skip"

    def retrieve(self, question, top_k=None, rerank_top_k=None):
        """
        完整检索流程：向量搜索 → 重排序
        参数:
            question: 用户问题（已重写后的独立问题）
            top_k: 向量检索返回候选数
            rerank_top_k: 重排序后保留结果数
        返回:
            dict: {"question": str, "results": list, "search_time": float}
        """
        start = time.time()
        top_k = top_k or TOP_K            # 默认取5个候选
        rerank_top_k = rerank_top_k or RERANK_TOP_K  # 默认保留3个

        logger.info(f"检索: {question[:40]}...")

        # 第一步：使用BGE-M3将问题编码为向量
        query_emb = self.embedder.encode_query(question)
        query_vec = query_emb["dense_vecs"][0].tolist()

        # 第二步：在Milvus中进行向量相似度搜索
        milvus_results = self.milvus.search(query_vec, top_k=top_k)
        if not milvus_results:
            # Milvus返回空结果，提前返回
            logger.warning("Milvus未找到相关结果")
            return {
                "question": question,
                "results": [],
                "search_time": time.time() - start
            }

        # 第二步半：公司名过滤（双PDF关键优化）
        company = _extract_company(question)
        if company:
            filtered = [r for r in milvus_results if company in r.get("source_pdf", "")]
            if filtered:
                logger.info(f"公司过滤: '{company}' → {len(filtered)}/{len(milvus_results)}")
                milvus_results = filtered

        # 第三步：使用BGE-Reranker对粗筛结果进行重排序
        self._load_reranker()
        if self.reranker and self.reranker != "skip":
            # 构造(问题, 文档)对，输入重排序模型
            pairs = [(question, r["content"]) for r in milvus_results]
            scores = self.reranker.compute_score(pairs)
            # 为每条结果添加重排序分数
            for i, r in enumerate(milvus_results):
                r["rerank_score"] = (
                    float(scores[i]) if isinstance(scores, list)
                    else float(scores)
                )
            # 按重排序分数降序排列
            milvus_results.sort(key=lambda x: x["rerank_score"], reverse=True)
            milvus_results = milvus_results[:rerank_top_k]
        else:
            # 没有重排序模型，直接按Milvus距离降序
            milvus_results.sort(key=lambda x: x["distance"], reverse=True)

        # 格式化最终结果
        results = [{
            "content": r["content"],                              # 文档内容
            "page_num": r["page_num"],                            # 来源页码
            "score": r.get("rerank_score", r["distance"]),        # 最终分数
            "distance": r["distance"],                            # 原始距离
            "source_pdf": r["source_pdf"],                        # 来源PDF
        } for r in milvus_results]

        elapsed = time.time() - start
        logger.info(f"检索完成! 耗时: {elapsed:.2f}秒, 返回 {len(results)} 条")
        return {
            "question": question,
            "results": results,
            "search_time": elapsed
        }


if __name__ == "__main__":
    """单独测试检索功能"""
    print("此模块需要embedder和milvus实例，请通过main.py调用")
