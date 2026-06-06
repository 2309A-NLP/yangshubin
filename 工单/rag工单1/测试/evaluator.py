# -*- coding: utf-8 -*-
"""
RAG评估模块 —— 对比 RAG 模式 与 纯LLM 模式的回答质量
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
"""

import json
import os
import sys
import time
from typing import Dict, List, Optional
from openai import OpenAI

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL


class RAGEvaluator:
    """
    RAG 系统评估器
    评估维度：
      1. 准确率：回答是否基于文档内容
      2. 完整性：是否覆盖了问题的所有方面
      3. 相关性：回答是否紧贴问题
    输出 RAG vs 纯LLM 的对比报告
    """

    def __init__(self):
        """初始化评估器（使用 LLM 作为评审）"""
        self.client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL
        )
        self.model = DEEPSEEK_MODEL

    def evaluate_answer(self, question: str,
                        rag_answer: str,
                        llm_answer: str) -> Dict:
        """
        使用 LLM 评估两种模式的回答质量

        Args:
            question: 原始问题
            rag_answer: RAG 模式（有文档上下文）生成的回答
            llm_answer: 纯 LLM 模式生成的回答

        Returns:
            评估结果字典，包含各项评分和对比结论
        """
        prompt = f"""你是一个RAG系统评估专家。请对以下两种回答模式进行对比评估。

## 问题
{question}

## RAG模式回答（基于PDF文档）
{rag_answer}

## 纯LLM模式回答（无文档，仅凭模型知识）
{llm_answer}

请从以下三个维度评分（1-5分），并给出评估理由：

1. 准确率：回答中的数据是否正确、是否有幻觉（编造）？
2. 完整性：是否完整回答了问题的所有方面？
3. 相关性：回答是否紧扣问题，不跑题？

请严格按以下JSON格式返回：
{{
    "rag": {{
        "accuracy": 5,
        "completeness": 5,
        "relevance": 5,
        "avg_score": 5.0,
        "comment": "评估说明"
    }},
    "pure_llm": {{
        "accuracy": 3,
        "completeness": 4,
        "relevance": 5,
        "avg_score": 4.0,
        "comment": "评估说明"
    }},
    "conclusion": "哪种模式更好，为什么"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=800
            )
            content = response.choices[0].message.content.strip()
            # 处理可能的 markdown 代码块
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                content = content.rsplit("```", 1)[0]
            return json.loads(content)
        except (json.JSONDecodeError, Exception) as e:
            return {
                "error": f"评估解析失败: {str(e)}",
                "rag": {"accuracy": 0, "completeness": 0, "relevance": 0, "avg_score": 0},
                "pure_llm": {"accuracy": 0, "completeness": 0, "relevance": 0, "avg_score": 0},
                "conclusion": "评估失败"
            }

    def run_evaluation(self,
                       questions: List[Dict],
                       rag_func,
                       llm_func) -> List[Dict]:
        """
        批量评估一组问题

        Args:
            questions: 问题列表，格式 [{"id": 260, "question": "..."}, ...]
            rag_func:   RAG模式的回答函数，接收 question 返回 (answer, context)
            llm_func:   纯LLM模式的回答函数，接收 question 返回 answer

        Returns:
            每条问题一条评估记录，含对比结果
        """
        results = []

        for item in questions:
            q = item["question"]
            qid = item.get("id", "unknown")

            print(f"\n[评估] 正在处理问题 #{qid}: {q[:40]}...")

            # 记录RAG模式的响应时间
            start = time.time()
            rag_answer, context = rag_func(q)
            rag_time = time.time() - start

            # 记录纯LLM模式的响应时间
            start = time.time()
            llm_answer = llm_func(q)
            llm_time = time.time() - start

            # 评估回答质量
            evaluation = self.evaluate_answer(q, rag_answer, llm_answer)

            record = {
                "question_id": qid,
                "question": q,
                "rag_answer": rag_answer,
                "rag_time": round(rag_time, 3),
                "llm_answer": llm_answer,
                "llm_time": round(llm_time, 3),
                "evaluation": evaluation
            }
            results.append(record)

            print(f"  RAG耗时: {rag_time:.2f}s | 纯LLM耗时: {llm_time:.2f}s")

        return results

    def print_report(self, results: List[Dict]):
        """
        打印评估报告摘要

        Args:
            results: run_evaluation 的返回结果
        """
        print("\n" + "=" * 60)
        print("                RAG 系统评估报告")
        print("=" * 60)

        total_rag = {"accuracy": 0, "completeness": 0, "relevance": 0}
        total_llm = {"accuracy": 0, "completeness": 0, "relevance": 0}
        count = 0

        for r in results:
            ev = r.get("evaluation", {})
            if "error" in ev:
                continue

            rag_scores = ev.get("rag", {})
            llm_scores = ev.get("pure_llm", {})

            for k in total_rag:
                total_rag[k] += rag_scores.get(k, 0)
                total_llm[k] += llm_scores.get(k, 0)
            count += 1

        if count > 0:
            print(f"\n评价问题数量: {count}")
            print(f"\n--- RAG 模式 ---")
            for k in total_rag:
                print(f"  {k}: {total_rag[k] / count:.2f} / 5")
            print(f"  平均分: {sum(total_rag.values()) / (count * 3):.2f} / 5")
            print(f"  平均耗时: {sum(r['rag_time'] for r in results) / count:.3f}s")

            print(f"\n--- 纯 LLM 模式 ---")
            for k in total_llm:
                print(f"  {k}: {total_llm[k] / count:.2f} / 5")
            print(f"  平均分: {sum(total_llm.values()) / (count * 3):.2f} / 5")
            print(f"  平均耗时: {sum(r['llm_time'] for r in results) / count:.3f}s")

            # 打印对比结论
            print(f"\n--- 各项结论 ---")
            for r in results:
                ev = r.get("evaluation", {})
                conclusion = ev.get("conclusion", "")
                if conclusion:
                    print(f"问题 #{r['question_id']}: {conclusion[:100]}")


# 工单要求的测试问题列表
TEST_QUESTIONS = [
    {"id": 260, "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？"},
    {"id": 95, "question": "武汉兴图新科电子股份有限公司参与制定了哪个技术标准？"},
    {"id": 33, "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入占主营业务收入的比重分别是多少？"},
    {"id": 34, "question": "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的上游涉及哪些企业？"},
    {"id": 957, "question": "武汉兴图新科电子股份有限公司在哪个领域已经成为重要供应商？"},
    {"id": 793, "question": "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的下游主要包括哪些行业？"},
    {"id": 795, "question": "武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖？"},
    {"id": 543, "question": "武汉兴图新科电子股份有限公司注册资本是多少？"},
    {"id": 531, "question": "武汉兴图新科电子股份有限公司法定代表人是谁？"},
    {"id": 207, "question": "武汉兴图新科电子股份有限公司计划使用本次发行募集资金的多少用于补充流动资金？"}
]


if __name__ == "__main__":
    print("=" * 60)
    print("  RAG 评估模块 - 正在运行 10 道题评估...")
    print("=" * 60)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from retriever import Retriever
    from llm_qa import LLMQA

    retriever = Retriever()
    llm = LLMQA()

    def rag_func(question):
        context = retriever.retrieve_context(question)
        answer = llm.answer_with_context(question, context)
        return answer, context

    def llm_func(question):
        return llm.answer_without_context(question)

    evaluator = RAGEvaluator()
    results = evaluator.run_evaluation(TEST_QUESTIONS, rag_func, llm_func)
    evaluator.print_report(results)
