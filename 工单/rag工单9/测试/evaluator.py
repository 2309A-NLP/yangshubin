"""
evaluator.py - RAG工单9 评估模块(真实LLM评估版)
工单编号: 人工智能NLP-RAG-Graph RAG 优化任务
需求: 评测层面优化 — 用MiMo LLM真实评估context precision和context recall
功能: LLM判断chunk相关性 → 精确precision/recall → 优化前后对比报告
"""
import logging, json, time, re
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '设计'))
from config import OUTPUT_DIR, MIMO_API_KEY, MIMO_BASE_URL, LOG_FMT, LOG_DATEFMT

# 评估用mimo-v2.5（文本模型，直接返回结果无推理过程）
from config import MIMO_MODEL as EVAL_MODEL
EVAL_TIMEOUT = 45

logging.basicConfig(format=LOG_FMT, datefmt=LOG_DATEFMT, level=logging.INFO)
logger = logging.getLogger("evaluator")
_cache = {}


def _llm_call(prompt, max_tokens=50, timeout=EVAL_TIMEOUT):
    """调用MiMo做评估判断"""
    try:
        from openai import OpenAI
        c = OpenAI(api_key=MIMO_API_KEY, base_url=MIMO_BASE_URL, timeout=timeout)
        r = c.chat.completions.create(
            model=EVAL_MODEL,
            messages=[
                {"role": "system", "content": "你是评估器。只回答数字，不要任何解释。"},
                {"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=max_tokens)
        text = (r.choices[0].message.content or "").strip()
        # 兼容reasoning模型
        if not text:
            extra = r.choices[0].message.model_extra or {}
            text = extra.get("reasoning_content", "").strip()
        return text
    except Exception as e:
        logger.warning(f"LLM评估调用失败: {e}")
        return ""


def _extract_score(text):
    """从LLM输出中提取0或1的判断结果"""
    if not text:
        return 0.0
    # 策略1: 找末尾独立的1或0（前后是空格/换行/标点）
    for m in re.finditer(r'(?<![\d.])\b([01])\b(?![\d.])', text):
        pass  # 取最后一个
    last = None
    for m in re.finditer(r'(?<![\d.])\b([01])\b(?![\d.])', text):
        last = m.group(1)
    if last is not None:
        return float(last)
    # 策略2: 看关键词
    if "有帮助" in text or "相关" in text:
        return 1.0
    if "无帮助" in text or "不相关" in text:
        return 0.0
    # 策略3: 取最后一个数字
    matches = re.findall(r'(\d+\.?\d*)', text)
    if matches:
        val = float(matches[-1])
        return min(1.0, max(0.0, val))
    return 0.0


class RagasEvaluator:
    """RAGAS评估器，真实LLM评估版"""
    def __init__(self):
        self.results = {"before": [], "after": []}
        self.summary = {}

    def evaluate_context_precision(self, question, chunks, ref_answer):
        """
        真实精度评估：用LLM判断每个检索到的chunk是否对回答问题有帮助
        precision = 有帮助的chunk数 / 总chunk数
        """
        if not chunks:
            return 0.0
        key = f"p|{question[:60]}|{len(chunks)}"
        if key in _cache:
            return _cache[key]
        relevant_count = 0
        checked = 0
        for i, chunk in enumerate(chunks[:5]):  # 只评前5个
            content = chunk.get("content", "")[:400]
            prompt = f"问题:{question[:80]}\n文档:{content[:200]}\n文档对回答问题有帮助吗？只回答1或0。"
            resp = _llm_call(prompt, max_tokens=500)
            score = _extract_score(resp)
            if score >= 0.5:
                relevant_count += 1
            checked += 1

        precision = relevant_count / checked if checked > 0 else 0.0
        _cache[key] = round(precision, 4)
        return _cache[key]

    def evaluate_context_recall(self, question, chunks, ref_answer):
        """
        真实召回评估：用LLM判断参考答案中的关键信息是否被检索结果覆盖
        """
        if not ref_answer or not chunks:
            return 0.0
        key = f"r|{question[:60]}|{ref_answer[:60]}"
        if key in _cache:
            return _cache[key]
        ctx = "\n".join(c.get("content", "")[:300] for c in chunks[:5])
        # 一步到位：直接判断覆盖比例
        check_prompt = f"检索结果:{ctx[:500]}\n参考答案:{ref_answer[:200]}\n参考答案的关键信息在检索结果中覆盖了多少比例？只回答0到1的小数。"
        resp = _llm_call(check_prompt, max_tokens=500)
        recall = _extract_score(resp)
        _cache[key] = round(max(0.0, min(1.0, recall)), 4)
        return _cache[key]

    def evaluate_batch(self, qdata, mode_label="before"):
        """批量评估"""
        _cache.clear()
        items = []
        for i, qd in enumerate(qdata):
            prec = self.evaluate_context_precision(
                qd["question"], qd.get("retrieved_chunks", []), qd.get("reference_answer", ""))
            rec = self.evaluate_context_recall(
                qd["question"], qd.get("retrieved_chunks", []), qd.get("reference_answer", ""))
            items.append({
                "question_id": i + 1, "question": qd["question"][:80],
                "context_precision": round(prec, 4), "context_recall": round(rec, 4),
                "response_time": qd.get("response_time", 0)})
            logger.info(f"  Q{i+1}: 精度={prec:.3f} 召回={rec:.3f}")
        self.results[mode_label] = items
        p = [i["context_precision"] for i in items]
        r = [i["context_recall"] for i in items]
        return {
            "items": items,
            "avg_precision": round(sum(p) / len(p), 4) if p else 0,
            "avg_recall": round(sum(r) / len(r), 4) if r else 0,
            "avg_response_time": round(sum(i["response_time"] for i in items) / len(items), 2) if items else 0}

    def compare_and_save(self):
        """对比分析并保存报告"""
        bf, af = self.results["before"], self.results["after"]
        if not bf or not af:
            return
        b = {"avg_precision": sum(x["context_precision"] for x in bf) / len(bf),
             "avg_recall": sum(x["context_recall"] for x in bf) / len(bf),
             "avg_response_time": sum(x["response_time"] for x in bf) / len(bf)}
        a = {"avg_precision": sum(x["context_precision"] for x in af) / len(af),
             "avg_recall": sum(x["context_recall"] for x in af) / len(af),
             "avg_response_time": sum(x["response_time"] for x in af) / len(af)}
        self.summary = {
            "before": {k: round(v, 4) for k, v in b.items()},
            "after": {k: round(v, 4) for k, v in a.items()},
            "improvement": {
                "precision_delta": round(a["avg_precision"] - b["avg_precision"], 4),
                "recall_delta": round(a["avg_recall"] - b["avg_recall"], 4),
                "time_delta": round(a["avg_response_time"] - b["avg_response_time"], 2)},
            "thresholds": {"context_precision": 0.8, "context_recall": 0.9, "response_time": 3.0},
            "total_questions": len(bf),
            "before_meets_threshold": b["avg_precision"] >= 0.8 and b["avg_recall"] >= 0.9,
            "after_meets_threshold": a["avg_precision"] >= 0.8 and a["avg_recall"] >= 0.9}
        self._save_html(b, a)
        logger.info(f"优化前: 精度{b['avg_precision']:.4f} 召回{b['avg_recall']:.4f} 响应{b['avg_response_time']:.2f}s")
        logger.info(f"优化后: 精度{a['avg_precision']:.4f} 召回{a['avg_recall']:.4f} 响应{a['avg_response_time']:.2f}s")
        return self.summary

    def _save_html(self, b, a):
        """保存HTML+JSON报告"""
        import time as _t
        json.dump({"summary": self.summary, "details": self.results},
                  open(str(OUTPUT_DIR / "evaluation_report.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        imp = self.summary["improvement"]
        rows = "".join(
            f'<div class="item"><h3>Q{bi["question_id"]}</h3>'
            f'<p>优化前: precision={bi["context_precision"]:.3f} recall={bi["context_recall"]:.3f}</p>'
            f'<p>优化后: precision={ai["context_precision"]:.3f} recall={ai["context_recall"]:.3f}</p></div>'
            for bi, ai in zip(self.results.get("before", []), self.results.get("after", [])))
        html = (
            f'<!DOCTYPE html><html><head><meta charset="utf-8"><title>GraphRAG优化报告</title>'
            f'<style>body{{font-family:"Microsoft YaHei",sans-serif;max-width:1000px;margin:20px auto;padding:20px}}'
            f'.metrics{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}}'
            f'.metric{{background:#f5f5f5;padding:15px;border-radius:8px;text-align:center}}'
            f'.metric b{{font-size:24px;color:#4CAF50;display:block}}</style></head><body>'
            f'<h1>📊 GraphRAG优化报告</h1>'
            f'<p>工单: {__import__("config").WO_ID} | 评估: MiMo LLM真实评估({EVAL_MODEL})</p>'
            f'<div class="metrics">'
            f'<div class="metric"><b>{b.get("avg_precision",0):.4f}</b><span>优化前精度</span></div>'
            f'<div class="metric"><b>{a.get("avg_precision",0):.4f}</b><span>优化后精度 <span style="color:{"green" if imp["precision_delta"]>=0 else "red"}>({imp["precision_delta"]:+.4f})</span></span></div>'
            f'<div class="metric"><b>{b.get("avg_recall",0):.4f}</b><span>优化前召回</span></div>'
            f'<div class="metric"><b>{a.get("avg_recall",0):.4f}</b><span>优化后召回 <span style="color:{"green" if imp["recall_delta"]>=0 else "red"}>({imp["recall_delta"]:+.4f})</span></span></div>'
            f'<div class="metric"><b>{b.get("avg_response_time",0):.2f}s</b><span>优化前响应</span></div>'
            f'<div class="metric"><b>{a.get("avg_response_time",0):.2f}s</b><span>优化后响应 <span style="color:{"green" if imp["time_delta"]<=0 else "red"}>({imp["time_delta"]:+.2f}s)</span></span></div>'
            f'</div><h2>阈值: precision≥0.8 recall≥0.9</h2>'
            f'<p>优化前: {"✅" if self.summary.get("before_meets_threshold") else "❌"} | 优化后: {"✅" if self.summary.get("after_meets_threshold") else "❌"}</p>'
            f'{rows}<p style="color:#999">生成:{_t.strftime("%Y-%m-%d %H:%M:%S")}</p></body></html>')
        with open(str(OUTPUT_DIR / "evaluation_report.html"), "w", encoding="utf-8") as f:
            f.write(html)


if __name__ == "__main__":
    ev = RagasEvaluator()
    r = ev.evaluate_context_recall("平安银行营收？", [{"content": "平安银行2019年营收1379亿元"}], "平安银行营收1379亿元增长18%")
    print(f"召回测试: {r:.3f}")
