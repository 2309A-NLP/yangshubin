"""
retriever.py - RAG工单4 检索与重排序模块
工单编号: 人工智能NLP-RAG-图像内容解析及检索优化
功能: 接收用户问题，先通过Milvus向量检索获取候选文档，
      再使用BGE-Reranker对结果重排序，返回最相关上下文
"""

import logging
import json
import os
import time

# 导入配置
from config import (
    TOP_K, RERANK_TOP_K,
    BGE_RERANKER_PATH, BGE_RERANKER_DEVICE,
    OUTPUT_DIR, LOG_FORMAT, LOG_DATE_FORMAT
)

# 设置日志
logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT, level=logging.INFO)
logger = logging.getLogger("retriever")


# ========== 公司名识别规则 ==========
_COMPANY_RULES = {
    "招股说明书1.pdf": ["武汉兴图新科", "兴图新科", "兴图", "军用领域"],
    "招股说明书2.pdf": ["武汉力源信息", "力源信息", "力源", "赵马克", "IC市场"],
}


def _extract_company_filter(question: str) -> str:
    """从问题中识别目标公司，返回对应的PDF文件名（用于Milvus过滤）"""
    for pdf_name, keywords in _COMPANY_RULES.items():
        for kw in keywords:
            if kw in question:
                return pdf_name
    return ""


class RetrievalPipeline:
    """
    检索流水线：向量检索 + 重排序
    先通过Milvus粗筛Top-K，再用Reranker精排
    """
    
    def __init__(self, embedder, milvus_handler):
        """
        初始化检索流水线
        参数:
            embedder: BgeM3Embedder实例
            milvus_handler: MilvusHandler实例
        """
        self.embedder = embedder
        self.milvus = milvus_handler
        self.reranker = None
    
    def _load_reranker(self):
        """BGE-Reranker已禁用（transformers版本不兼容），直接跳过"""
        self.reranker = "skip"
    
    def retrieve(self, question, top_k=None, rerank_top_k=None):
        """
        完整检索流程：向量搜索 → 重排序
        参数:
            question: 用户问题
            top_k: 向量检索返回候选数
            rerank_top_k: 重排序后保留数
        返回:
            dict: {
                "question": str,
                "results": [{
                    "content": str,
                    "page_num": int,
                    "has_image": bool,
                    "image_file": str,
                    "score": float,       # 重排序分数
                    "distance": float,    # 向量检索分数
                }, ...],
                "search_time": float,
            }
        """
        start_time = time.time()
        top_k = top_k or TOP_K
        rerank_top_k = rerank_top_k or RERANK_TOP_K
        
        logger.info(f"检索问题: {question}")
        
        # ---- 第一步：生成问题向量 ----
        logger.info("生成问题向量...")
        query_emb = self.embedder.encode_query(question)
        query_vector = query_emb["dense_vecs"][0].tolist()
        
        # ---- 第二步：Milvus向量搜索 ----
        logger.info(f"Milvus搜索 Top-{top_k}...")
        milvus_results = self.milvus.search(query_vector, top_k=top_k)
        
        if not milvus_results:
            logger.warning("Milvus未找到相关结果")
            return {
                "question": question,
                "results": [],
                "search_time": time.time() - start_time,
            }

        # ---- 第二步半：公司名过滤（关键优化） ----
        company_filter = _extract_company_filter(question)
        if company_filter:
            filtered = [r for r in milvus_results if company_filter in r.get("source_pdf", "")]
            if filtered:
                logger.info(f"公司名过滤: '{company_filter}' → {len(filtered)}/{len(milvus_results)} 条")
                milvus_results = filtered
            else:
                logger.warning(f"公司名过滤后无结果，使用原始结果")

        # ---- 第三步：BGE-Reranker重排序 ----
        self._load_reranker()
        
        if self.reranker and self.reranker != "skip":
            logger.info(f"BGE-Reranker重排序 {len(milvus_results)} 条结果...")
            
            # 构建(query, doc)对
            pairs = [(question, r["content"]) for r in milvus_results]
            
            # 计算相关性分数
            rerank_scores = self.reranker.compute_score(pairs)
            
            # 合并分数并排序
            for i, r in enumerate(milvus_results):
                score = rerank_scores[i] if isinstance(rerank_scores, list) else rerank_scores
                r["rerank_score"] = float(score)
            
            # 按重排序分数降序排列
            milvus_results.sort(key=lambda x: x["rerank_score"], reverse=True)
            
            # 取前rerank_top_k个
            milvus_results = milvus_results[:rerank_top_k]
        else:
            # 没有重排序，直接按向量距离排序
            milvus_results.sort(key=lambda x: x["distance"], reverse=True)
        
        # ---- 第四步：格式化结果 ----
        results = []
        for r in milvus_results:
            results.append({
                "content": r["content"],
                "page_num": r["page_num"],
                "has_image": r["has_image"],
                "image_file": r["image_file"],
                "score": r.get("rerank_score", r["distance"]),
                "distance": r["distance"],
                "source_pdf": r["source_pdf"],
            })
        
        elapsed = time.time() - start_time
        logger.info(f"检索完成! 耗时: {elapsed:.2f}秒, 返回 {len(results)} 条结果")
        
        return {
            "question": question,
            "results": results,
            "search_time": elapsed,
        }


def save_search_result(result):
    """
    保存检索结果到文件
    参数:
        result: retrieve()的返回结果
    """
    # 准备可序列化的数据
    save_data = {
        "question": result["question"],
        "search_time": result["search_time"],
        "result_count": len(result["results"]),
        "results": [{
            "page_num": r["page_num"],
            "has_image": r["has_image"],
            "score": float(r["score"]),
            "content_preview": r["content"][:200] + "..." if len(r["content"]) > 200 else r["content"],
        } for r in result["results"]],
    }
    
    # 保存到文件
    output_path = os.path.join(OUTPUT_DIR, "retrieval_result.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"检索结果已保存: {output_path}")


if __name__ == "__main__":
    """单独测试检索功能"""
    print("此模块需要embedder和milvus实例，请通过run.py调用")
