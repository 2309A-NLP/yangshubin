# -*- coding: utf-8 -*-
"""
LLM问答模块 —— 使用 DeepSeek API 生成回答
支持两种模式：RAG模式（带上下文）和 纯LLM模式（无上下文）
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
"""

from typing import Optional
from openai import OpenAI

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, LLM_TIMEOUT


class LLMQA:
    """
    LLM 问答引擎
    功能：
      1. RAG 模式：结合检索到的文档上下文生成回答
      2. 纯 LLM 模式：仅凭模型自身知识回答（用于对比评估）
    """

    def __init__(self):
        """初始化 DeepSeek 客户端"""
        self.client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL
        )
        self.model = DEEPSEEK_MODEL

    def answer_with_context(self, question: str, context: str) -> str:
        """
        RAG 模式：基于检索到的文档上下文生成回答

        Args:
            question: 用户提问
            context: 检索到的文档片段（由 Retriever 提供）

        Returns:
            LLM 生成的回答文本
        """
        system_prompt = """你是一个基于PDF文档的智能问答助手。请根据提供的文档内容回答用户问题。

注意事项：
1. 仅根据提供的文档内容回答，不要编造信息
2. 如果文档内容不足以回答，请明确说明"根据提供的文档无法确认"
3. 回答要简洁、准确，直接引用文档中的数据
4. 如果文档中包含表格或数字数据，请准确引用"""
        user_prompt = f"""以下是相关文档内容：

{context}

---

请根据以上文档内容回答以下问题：
{question}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1000,
                timeout=LLM_TIMEOUT
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[LLM请求异常] {str(e)}"

    def answer_without_context(self, question: str) -> str:
        """
        纯 LLM 模式：仅凭模型自身知识回答（不提供文档上下文）

        Args:
            question: 用户提问

        Returns:
            LLM 生成的回答文本
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个通用问答助手。请基于你的知识回答用户问题。"},
                    {"role": "user", "content": question}
                ],
                temperature=0.3,
                max_tokens=1000,
                timeout=LLM_TIMEOUT
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[LLM请求异常] {str(e)}"


if __name__ == "__main__":
    # 自测模块
    llm = LLMQA()

    print("=== RAG 模式测试 ===")
    context = """
[文档片段 1]（相似度: 0.8567）
报告期内，公司来自军用领域的收入分别为 5,236.48万元、6,891.25万元、...
"""
    answer = llm.answer_with_context("公司来自军用领域的收入是多少？", context)
    print(f"回答: {answer[:200]}")

    print("\n=== 纯LLM模式测试 ===")
    answer2 = llm.answer_without_context("公司来自军用领域的收入是多少？")
    print(f"回答: {answer2[:200]}")
