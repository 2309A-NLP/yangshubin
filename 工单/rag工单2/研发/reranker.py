# -*- coding: utf-8 -*-
"""
重排序模块 —— 用 BGE-Reranker 对检索结果重新排序
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
"""

from typing import List, Dict, Optional

from config import BGE_RERANKER_PATH, TOP_K_RERANK, TOP_K_FINAL


class Reranker:
    """
    重排序器：对检索结果进行精排，提升Top-N准确率
    如果reranker模型不存在，回退到按原始分数排序
    """

    def __init__(self):
        self._model = None
        self._available = False
        self._init_model()

    def _init_model(self):
        """尝试加载reranker模型"""
        if not BGE_RERANKER_PATH:
            print("[Reranker] 未配置reranker模型路径，跳过重排序")
            return
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(BGE_RERANKER_PATH)
            self._available = True
            print(f"[Reranker] 加载成功: {BGE_RERANKER_PATH}")
        except Exception as e:
            print(f"[Reranker] 加载失败: {e}，跳过重排序")

    def rerank(self, query: str, results: List[Dict],
               top_k: int = None) -> List[Dict]:
        """
        对检索结果重排序

        Args:
            query: 用户原始问题
            results: 检索结果列表（含text字段）
            top_k: 返回条数

        Returns:
            重排序后的结果列表（按相关性从高到低）
        """
        if top_k is None:
            top_k = TOP_K_FINAL

        if not self._available or not results:
            # 回退：按已有分数排序
            results.sort(key=lambda x: x.get("rrf_score", x.get("score", 0)), reverse=True)
            return results[:top_k]

        # 构建(query, doc)对
        pairs = [(query, r.get("text", "")) for r in results]
        scores = self._model.predict(pairs)

        # 按reranker得分重排
        scored = list(zip(results, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        reranked = []
        for r, s in scored[:top_k]:
            r["rerank_score"] = float(s)
            reranked.append(r)

        return reranked


if __name__ == "__main__":
    r = Reranker()
    print(f"重排序可用: {r._available}")
