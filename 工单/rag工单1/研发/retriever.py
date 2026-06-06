# -*- coding: utf-8 -*-
"""
检索器模块 —— 整合嵌入模型 + Milvus 向量检索 + 查询理解
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
"""

from typing import List, Dict, Optional
import numpy as np

from config import TOP_K
from embedding_model import EmbeddingModel
from vector_store import VectorStore
from query_processor import QueryProcessor


class Retriever:
    """
    检索器：将用户问题转换为向量并在 Milvus 中搜索相关文档块
    流程：问题 → 查询理解 → 编码 → 向量检索 → 返回候选文本
    """

    def __init__(self):
        """初始化嵌入模型、向量数据库和查询理解模块"""
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore()
        self.query_processor = QueryProcessor()

    def retrieve(self, question: str, top_k: int = None) -> List[Dict]:
        """
        完整检索流程：分析问题 → 编码 → 搜索 → 返回结果

        Args:
            question: 用户提问
            top_k: 返回的最相似文本块数量（默认使用配置值）

        Returns:
            检索结果列表，每项包含 text（文档原文）、score（相似度）、metadata
        """
        if top_k is None:
            top_k = TOP_K

        # 第一步：查询理解（意图识别 + 消歧 + 关键词提取）
        analysis = self.query_processor.analyze_query(question)
        query_text = analysis.get("disambiguated_query", question)
        keywords = analysis.get("keywords", [])

        print(f"[检索] 原始问题: {question}")
        print(f"[检索] 消歧后查询: {query_text}")
        print(f"[检索] 关键词: {keywords}")

        # 第二步：使用 BGE-M3 对查询文本编码
        query_vector = self.embedding_model.encode_query(query_text)
        query_list = query_vector[0].tolist()

        # 第三步：在 Milvus 中检索最相似的文本块
        results = self.vector_store.search(
            query_vector=query_list,
            top_k=top_k
        )

        # 如果关键词检索结果不理想，尝试用关键词子集再搜一次
        if keywords and len(results) < top_k:
            keyword_query = " ".join(keywords[:3])
            kv = self.embedding_model.encode_query(keyword_query)
            extra_results = self.vector_store.search(
                query_vector=kv[0].tolist(),
                top_k=top_k - len(results)
            )
            # 合并去重
            existing_ids = {r["id"] for r in results}
            for r in extra_results:
                if r["id"] not in existing_ids:
                    results.append(r)

        print(f"[检索] 找到 {len(results)} 个相关文本块")
        return results

    def retrieve_context(self, question: str, top_k: int = None) -> str:
        """
        检索并格式化上下文文本（供 LLM 生成答案时使用）

        Args:
            question: 用户提问
            top_k: 返回的文本块数量

        Returns:
            拼接后的上下文字符串，包含文本内容和来源
        """
        results = self.retrieve(question, top_k)
        context_parts = []

        for i, r in enumerate(results, start=1):
            context_parts.append(
                f"[文档片段 {i}]（相似度: {r['score']:.4f}）\n{r['text']}"
            )

        return "\n\n".join(context_parts)


if __name__ == "__main__":
    # 自测模块：测试检索流程
    retriever = Retriever()

    test_questions = [
        "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
        "公司注册资本是多少？",
        "法定代表人是谁？"
    ]

    for q in test_questions:
        print(f"\n{'=' * 60}")
        print(f"问题: {q}")
        print("=" * 60)
        context = retriever.retrieve_context(q, top_k=3)
        print(context[:500])
