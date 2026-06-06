# -*- coding: utf-8 -*-
"""
查询扩展模块 —— 用LLM生成相关查询词，提升检索召回率
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
"""

from openai import OpenAI
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL


class QueryExpander:
    """
    查询扩展：对用户问题进行重写和扩展
    策略：同义词替换 + 相关词补充 + 子问题分解
    """

    def __init__(self):
        self.client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        self.model = DEEPSEEK_MODEL

    def expand(self, question: str) -> list:
        """
        扩展用户问题，返回少量高质量查询变体

        Args:
            question: 用户原始问题

        Returns:
            [原始问题, 1个精简版查询]
        """
        prompt = f"""请将以下问题改写为更精简的检索查询，去掉冗余词，保留核心信息。只返回改写后的查询，不要其他文字。

问题：{question}"""

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100
            )
            variant = resp.choices[0].message.content.strip()
            all_queries = [question]
            if variant and variant != question:
                all_queries.append(variant)
            return all_queries[:2]  # 最多2个

        except Exception as e:
            return [question]

    def extract_keywords(self, question: str) -> list:
        """提取关键搜索词"""
        prompt = f"从以下问题中提取3-5个最重要的搜索关键词，用空格分隔，只返回关键词：\n{question}"
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=100
            )
            words = resp.choices[0].message.content.strip().split()
            return [w.strip() for w in words if w.strip()]
        except Exception:
            return question.split()[:5]


if __name__ == "__main__":
    qe = QueryExpander()
    questions = [
        "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
        "公司法定代表人是谁？"
    ]
    for q in questions:
        print(f"\n原始: {q}")
        expanded = qe.expand(q)
        print(f"扩展: {expanded}")
        kw = qe.extract_keywords(q)
        print(f"关键词: {kw}")
