"""\nqa_generator.py - RAG工单9 MiMo问答生成模块(快速版)\n工单编号: 人工智能NLP-RAG-Graph RAG 优化任务
需求: 生成层面优化 — 用mimo-v2.5文本模型快速生成金融年报答案
功能: 构建检索增强提示词 → MiMo API(文本模型) → 返回答案+响应时间
"""
import logging, time
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '设计'))
from config import MIMO_API_KEY, MIMO_BASE_URL, MIMO_MODEL, MIMO_TIMEOUT, MIMO_MAX_TOKENS, LOG_FMT, LOG_DATEFMT

logging.basicConfig(format=LOG_FMT, datefmt=LOG_DATEFMT, level=logging.INFO)
logger = logging.getLogger("qa_generator")


def build_prompt(question, retrieved_chunks):
    """构建问答提示词：拼接检索到的上下文"""
    parts = []
    for i, c in enumerate(retrieved_chunks):
        parts.append(f"[{i+1}]来源:{c.get('source_pdf','未知')}(第{c.get('page_num',0)}页)\n{c['content'][:600]}")
    ctx = "\n\n".join(parts)
    return (f"你是一个金融年报分析助手。基于以下文档回答问题。\n"
            f"【文档内容】\n{ctx}\n\n【问题】\n{question}\n\n请基于文档准确回答。如无相关信息，请如实说明。\n支持中英文提问，用提问语言回答。")


class QAGenerator:
    """小米MiMo API问答生成器（mimo-v2.5文本模型，快速响应）"""
    def __init__(self):
        self.client = None

    def _ensure_client(self):
        if self.client:
            return
        from openai import OpenAI
        self.client = OpenAI(api_key=MIMO_API_KEY, base_url=MIMO_BASE_URL)

    def generate(self, question, retrieved_chunks, use_graph=True):
        """生成回答，use_graph仅影响日志标签"""
        self._ensure_client()
        prompt = build_prompt(question, retrieved_chunks)
        start = time.time()
        mode = "GraphRAG" if use_graph else "VectorOnly"
        try:
            resp = self.client.chat.completions.create(
                model=MIMO_MODEL, timeout=MIMO_TIMEOUT,  # mimo-v2.5 文本模型(快速)
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=1024)
            answer = (resp.choices[0].message.content or "").strip()
            if not answer:
                answer = "【模型返回为空】"
            elapsed = time.time() - start
            logger.info(f"[{mode}]回答完成! {elapsed:.2f}秒, {len(answer)}字")
        except Exception as e:
            logger.error(f"API调用失败: {e}")
            answer = "【API调用失败】"
            elapsed = time.time() - start
        return {
            "answer": answer,
            "source_chunks": [{"source_pdf": c.get("source_pdf"), "page_num": c.get("page_num"),
                               "score": c.get("score", 0)} for c in retrieved_chunks],
            "response_time": round(elapsed, 2), "mode": mode,
            "context_count": len(retrieved_chunks),
        }


if __name__ == "__main__":
    qa = QAGenerator()
    tc = [{"content": "平安银行2019年营收1379亿元增长18.2%", "source_pdf": "平安银行_2019_年报.pdf", "page_num": 5, "score": 0.92}]
    print(qa.generate("平安银行2019年营收多少？", tc)["answer"][:200])
