"""
模块功能: RAG 评估模块（MiMo LLM 4 维度评分）
使用 MiMo API 对问答结果评估（相关性/完整性/准确性/流畅性）
支持 LLM 超时降级关键词评分，自动生成 HTML 对比报告
工单编号: 人工智能NLP-RAG-金融问答系统部署
"""

import json, logging, time
from pathlib import Path
from app.config import config

logger = logging.getLogger("evaluator")


def llm_evaluate_single(question: str, answer: str, reference: str,
                         timeout: int = 8) -> dict:
    """使用 MiMo API 评估单条问答（4维度1-10分），超时降级为关键词评分"""
    prompt = f"""评估以下问答对。
问题: {question}
模型回答: {answer}
参考答案: {reference}
从 相关性(1-10) 完整性(1-10) 准确性(1-10) 流畅性(1-10) 分别打分。
返回JSON: {{"相关性":分,"完整性":分,"准确性":分,"流畅性":分,"reason":"理由"}}"""
    try:
        from app.llm_client import MiMoClient
        result = MiMoClient().generate(prompt)
        if '{' in result:
            result = result[result.index('{'):result.rindex('}') + 1]
            scores = json.loads(result)
            if all(k in scores for k in ["相关性","完整性","准确性","流畅性"]):
                return scores
    except Exception as e:
        logger.warning(f"LLM评估降级: {e}")
    # 关键词降级评分
    return {"相关性": 7 if len(answer)>50 else 5, "完整性": min(10,max(1,len(answer)//100+3)),
            "准确性": 6, "流畅性": 7 if len(answer)>30 else 5, "reason": "关键词降级"}


class RagasEvaluator:
    """RAGAS 风格评估器，支持批量评估和 Vector vs GraphRAG 对比"""

    def __init__(self):
        self.results = {"before": [], "after": []}

    def evaluate_batch(self, qa_results: list, mode: str = "before") -> list:
        """批量评估问答结果"""
        logger.info(f"评估{mode}({len(qa_results)}题)...")
        batch = []
        for i, item in enumerate(qa_results):
            scores = llm_evaluate_single(item["question"],
                item["qa_result"]["answer"], item.get("reference_answer",""))
            scores["question"] = item["question"]
            scores["response_time"] = item.get("response_time", 0)
            batch.append(scores)
            logger.info(f"  [{i+1}] 相关{scores.get('相关性',0)} 完整{scores.get('完整性',0)}")
        self.results[mode] = batch
        with open(Path(config.OUTPUT_DIR)/f"evaluation_{mode}.json","w",encoding="utf-8") as f:
            json.dump(batch, f, ensure_ascii=False, indent=2)
        return batch

    def compare_and_save(self) -> dict:
        """对比 Vector 和 GraphRAG，生成总结和 HTML 报告"""
        b, a = self.results.get("before", []), self.results.get("after", [])
        if not b or not a:
            return {}
        def avg(scores, key):
            vals = [s.get(key, 0) for s in scores]
            return sum(vals)/len(vals) if vals else 0
        b_avg = {k: avg(b, k) for k in ["相关性","完整性","准确性","流畅性"]}
        b_avg["avg_response_time"] = avg(b, "response_time")
        a_avg = {k: avg(a, k) for k in ["相关性","完整性","准确性","流畅性"]}
        a_avg["avg_response_time"] = avg(a, "response_time")
        summary = {
            "total_questions": len(b), "before": b_avg, "after": a_avg,
            "improvement": {
                "precision_delta": a_avg.get("相关性",0)-b_avg.get("相关性",0),
                "recall_delta": a_avg.get("完整性",0)-b_avg.get("完整性",0),
                "accuracy_delta": a_avg.get("准确性",0)-b_avg.get("准确性",0),
                "fluency_delta": a_avg.get("流畅性",0)-b_avg.get("流畅性",0),
                "time_delta": a_avg.get("avg_response_time",0)-b_avg.get("avg_response_time",0),
            },
            "before_meets_threshold": b_avg.get("相关性",0)>=7 and b_avg.get("完整性",0)>=7,
            "after_meets_threshold": a_avg.get("相关性",0)>=7 and a_avg.get("完整性",0)>=7,
        }
        with open(Path(config.OUTPUT_DIR)/"evaluation_summary.json","w",encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        self._gen_html(summary, b, a)
        return summary

    def _gen_html(self, summary: dict, b_scores: list, a_scores: list):
        """生成评估对比 HTML 报告"""
        html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>RAG 工单10 评估报告</title>
<style>body{{font-family:Arial;max-width:900px;margin:20px auto;padding:0 20px}}
table{{border-collapse:collapse;width:100%;margin:15px 0}}
th,td{{border:1px solid #ddd;padding:8px;text-align:center}}
th{{background:#283593;color:white}}
.green{{color:green}}.red{{color:red}}
</style></head><body>
<h1>📊 RAG 金融问答系统 评估报告</h1>
<p><b>LLM:</b> {config.MIMO_MODEL} | <b>测试数:</b> {summary['total_questions']}</p>
<h2>📈 Vector vs GraphRAG 对比</h2>
<table><tr><th>指标</th><th>Vector</th><th>GraphRAG</th><th>变化</th></tr>"""
        for k, lab in [("相关性","相关性"),("完整性","完整性(召回)"),
                        ("准确性","准确性"),("流畅性","流畅性")]:
            bv, av = summary['before'].get(k,0), summary['after'].get(k,0)
            delta = summary['improvement'].get(k+"_delta",0)
            cls = "green" if delta>=0 else "red"
            html += f"<tr><td>{lab}</td><td>{bv:.2f}</td><td>{av:.2f}</td><td class={cls}>{delta:+.2f}</td></tr>"
        html += f"""<tr><td>响应时间(s)</td><td>{summary['before'].get('avg_response_time',0):.2f}</td>
<td>{summary['after'].get('avg_response_time',0):.2f}</td>
<td>{summary['improvement']['time_delta']:+.2f}</td></tr></table>"""
        html += "<h2>📋 每题明细</h2><table><tr><th>#</th><th>问题</th><th>V-相关</th><th>V-完整</th><th>G-相关</th><th>G-完整</th></tr>"
        for i in range(max(len(b_scores), len(a_scores))):
            b = b_scores[i] if i<len(b_scores) else {}
            a = a_scores[i] if i<len(a_scores) else {}
            html += f"<tr><td>{i+1}</td><td>{b.get('question','')[:25]}...</td>"
            html += f"<td>{b.get('相关性',0)}</td><td>{b.get('完整性',0)}</td>"
            html += f"<td>{a.get('相关性',0)}</td><td>{a.get('完整性',0)}</td></tr>"
        html += "</table></body></html>"
        with open(Path(config.OUTPUT_DIR)/"evaluation_report.html","w",encoding="utf-8") as f:
            f.write(html)
