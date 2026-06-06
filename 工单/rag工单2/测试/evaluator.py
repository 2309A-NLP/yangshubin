# -*- coding: utf-8 -*-
"""
RAG评估模块 —— 对比优化前后的检索准确率
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
"""

import time
from typing import List, Dict
from openai import OpenAI

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL


# ============================================================
# 测试问题列表（同工单1）
# ============================================================
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
    {"id": 207, "question": "武汉兴图新科电子股份有限公司计划使用本次发行募集资金的多少用于补充流动资金？"},
]


class Evaluator:
    """
    评估器：运行测试问题，对比RAG与纯LLM的回答质量
    评估维度：准确率、响应时间、答案完整性
    """

    def __init__(self):
        self.client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        self.model = DEEPSEEK_MODEL

    def _llm_judge(self, question: str, rag_answer: str, llm_answer: str) -> Dict:
        """用LLM评估两种模式的回答"""
        prompt = f"""评估两种回答模式的准确性（1-5分），返回JSON格式：
问题：{question}

RAG模式（基于文档）：{rag_answer}
纯LLM模式（仅知识）：{llm_answer}

评估维度：
- accuracy_rag: RAG回答准确吗？有幻觉吗？
- accuracy_llm: LLM回答准确吗？有幻觉吗？
- completeness_rag: RAG回答完整吗？
- completeness_llm: LLM回答完整吗？
- has_hallucination_rag: RAG有幻觉吗？true/false
- has_hallucination_llm: LLM有幻觉吗？true/false
- which_better: "rag" 或 "llm" 或 "tie"

返回严格JSON，不要其他文字。"""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=300
            )
            import json
            content = resp.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0]
            return json.loads(content)
        except Exception:
            return {"which_better": "unknown", "accuracy_rag": 0, "accuracy_llm": 0}

    def run(self, rag_fn, llm_fn) -> List[Dict]:
        """运行评估"""
        results = []
        correct_rag, correct_llm = 0, 0

        print(f"{'='*60}")
        print(f"  RAG 系统评估 - {len(TEST_QUESTIONS)} 道题")
        print(f"{'='*60}")

        for item in TEST_QUESTIONS:
            qid, q = item["id"], item["question"]

            # RAG模式
            t0 = time.time()
            rag_ans, sources = rag_fn(q)
            t_rag = time.time() - t0

            # 纯LLM模式
            t0 = time.time()
            llm_ans = llm_fn(q)
            t_llm = time.time() - t0

            # 评估
            judge = self._llm_judge(q, rag_ans, llm_ans)
            is_correct_rag = judge.get("accuracy_rag", 0) >= 4
            is_correct_llm = judge.get("accuracy_llm", 0) >= 4
            if is_correct_rag:
                correct_rag += 1
            if is_correct_llm:
                correct_llm += 1

            results.append({
                "id": qid, "question": q[:50],
                "rag_answer": rag_ans[:100],
                "rag_time": round(t_rag, 3),
                "llm_answer": llm_ans[:100],
                "llm_time": round(t_llm, 3),
                "rag_correct": is_correct_rag,
                "llm_correct": is_correct_llm,
                "judge": judge
            })

            winner = judge.get("which_better", "?")
            print(f"  #{qid:3d} RAG={'✓' if is_correct_rag else '✗'} "
                  f"LLM={'✓' if is_correct_llm else '✗'} "
                  f"| {t_rag:.1f}s/{t_llm:.1f}s | 胜者:{winner}")

        # 输出报告
        print(f"\n{'='*60}")
        print(f"  📊 评估报告")
        print(f"{'='*60}")
        print(f"  RAG准确率:   {correct_rag}/{len(TEST_QUESTIONS)} = {correct_rag/len(TEST_QUESTIONS)*100:.1f}%")
        print(f"  纯LLM准确率: {correct_llm}/{len(TEST_QUESTIONS)} = {correct_llm/len(TEST_QUESTIONS)*100:.1f}%")
        rag_times = [r["rag_time"] for r in results]
        llm_times = [r["llm_time"] for r in results]
        print(f"  RAG平均耗时:   {sum(rag_times)/len(rag_times):.2f}s")
        print(f"  纯LLM平均耗时: {sum(llm_times)/len(llm_times):.2f}s")
        print(f"{'='*60}")

        return results


if __name__ == "__main__":
    from llm_qa import LLMQA

    qa = LLMQA()

    def rag_fn(q):
        ans, _ = qa.answer_with_context(q)
        return ans, []

    def llm_fn(q):
        return qa.answer_without_context(q)

    ev = Evaluator()
    ev.run(rag_fn, llm_fn)
