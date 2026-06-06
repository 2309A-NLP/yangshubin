# -*- coding: utf-8 -*-
"""
查询理解模块 —— 使用 LLM 对用户问题进行意图识别、消歧和分解
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
"""

import json
from typing import Dict, List
from openai import OpenAI

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL


class QueryProcessor:
    """
    查询理解：对用户提问进行预处理
    功能：
      1. 意图识别 —— 判断问题是事实型、分析型还是模糊型
      2. 消歧 —— 处理缩写、多义词，给出明确的搜索关键词
      3. 问题分解 —— 将复杂问题拆解为多个子问题
    """

    def __init__(self):
        """初始化 DeepSeek 客户端"""
        self.client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL
        )
        self.model = DEEPSEEK_MODEL

    def analyze_query(self, question: str) -> Dict:
        """
        综合分析用户问题：意图识别 + 消歧 + 生成关键词

        Args:
            question: 用户原始提问

        Returns:
            {
                "intent": "factual" | "analytical" | "ambiguous",
                "disambiguated_query": "消歧后的查询文本",
                "keywords": ["关键", "搜索", "词"],
                "sub_questions": ["子问题1", ...]  # 如果复杂则分解
            }
        """
        prompt = f"""你是一个专业的查询分析助手。请分析以下用户问题，返回严格JSON格式的结果。

问题：{question}

请分析：
1. intent: 意图类型。factual（事实查询）/ analytical（分析型）/ ambiguous（模糊需要消歧）
2. disambiguated_query: 消歧后的明确查询文本（如果是明确的直接复制原问题）
3. keywords: 3-5个最核心的搜索关键词（用于向量检索）
4. sub_questions: 如果问题包含多个方面，拆分成子问题列表，否则为[""]

只返回JSON，不要其他文字。"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500
        )

        content = response.choices[0].message.content.strip()
        # 尝试提取JSON
        try:
            # 去除可能的 markdown 代码标记
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                content = content.rsplit("```", 1)[0]
            result = json.loads(content)
        except json.JSONDecodeError:
            # 如果解析失败，使用默认值
            result = {
                "intent": "factual",
                "disambiguated_query": question,
                "keywords": question.split(),
                "sub_questions": [question]
            }

        return result


if __name__ == "__main__":
    # 自测模块
    qp = QueryProcessor()
    test_questions = [
        "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
        "公司参与制定了哪个技术标准？",
        "上下游涉及哪些企业？"
    ]
    for q in test_questions:
        print(f"\n问题: {q}")
        result = qp.analyze_query(q)
        print(f"分析结果: {json.dumps(result, ensure_ascii=False, indent=2)}")
