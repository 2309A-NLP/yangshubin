# -*- coding: utf-8 -*-
"""
LLM问答模块 —— DeepSeek API，支持RAG模式和纯LLM模式
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
"""

from openai import OpenAI
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, LLM_TIMEOUT
from query_expander import QueryExpander
from hybrid_retriever import HybridRetriever
from reranker import Reranker
from config import TOP_K_RETRIEVAL


class LLMQA:
    """问答引擎：检索 → 重排序 → 生成回答"""

    def __init__(self, retriever=None):
        self.client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        self.model = DEEPSEEK_MODEL
        self.expander = QueryExpander()
        self.reranker = Reranker()
        # 支持注入预热好的检索器，避免重复加载
        self.retriever = retriever if retriever is not None else HybridRetriever()
        self.embedder = self.retriever.embedder

    def _retrieve_context(self, question: str) -> tuple:
        """检索+重排序，返回(上下文字符串, 来源列表)"""
        # 1. 查询扩展
        queries = self.expander.expand(question)

        # 2. 多查询混合检索
        all_fused = {}
        for q in queries:
            fused, _, _ = self.retriever.retrieve(q, top_k=TOP_K_RETRIEVAL)
            for r in fused:
                rid = r.get("id")
                score = r.get("rrf_score", r.get("score", 0))
                if rid is not None and (rid not in all_fused or score > all_fused[rid].get("rrf_score", 0)):
                    all_fused[rid] = r

        candidates = sorted(all_fused.values(),
                            key=lambda x: x.get("rrf_score", x.get("score", 0)),
                            reverse=True)

        # 3. 可选重排序
        from config import USE_RERANKER
        if USE_RERANKER:
            reranked = self.reranker.rerank(question, candidates, top_k=5)
        else:
            reranked = candidates[:5]

        # 4. 拼接上下文
        parts = []
        for i, r in enumerate(reranked, 1):
            score = r.get("rerank_score", r.get("rrf_score", r.get("score", 0)))
            parts.append(f"[文档{i}]({score:.4f})\n{r.get('text', '')}")
            r["display_score"] = round(score, 4)

        return "\n\n".join(parts), reranked

    def answer_with_context(self, question: str) -> tuple:
        """RAG模式：检索文档 + LLM生成回答，返回(回答, 来源)"""
        context, sources = self._retrieve_context(question)

        system = "你是一个基于PDF文档的智能问答助手。请根据提供的文档内容回答问题。如果文档中找不到答案，请说'根据文档无法确认'。引用数据时要准确。"
        user = f"文档内容：\n{context}\n\n问题：{question}"

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=0.3, max_tokens=1000, timeout=LLM_TIMEOUT
            )
            return resp.choices[0].message.content.strip(), sources
        except Exception as e:
            return f"[LLM异常] {e}", sources

    def answer_without_context(self, question: str) -> str:
        """纯LLM模式"""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": question}],
                temperature=0.3, max_tokens=1000, timeout=LLM_TIMEOUT
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"[LLM异常] {e}"
