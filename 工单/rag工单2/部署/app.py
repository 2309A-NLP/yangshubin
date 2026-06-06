# -*- coding: utf-8 -*-
"""
Flask Web应用 —— 工单2优化版RAG问答系统
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
"""

import os
import sys
import time
import socket
from datetime import datetime
from flask import Flask, render_template, request, jsonify

# ===== 路径桥接：将所有子目录加入 sys.path =====
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ["设计", "研发", "测试", "优化", "部署"]:
    _p = os.path.join(_ROOT, _sub)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

app = Flask(__name__, template_folder=os.path.join(_ROOT, "研发", "templates"))

_qa = None

def get_qa():
    global _qa
    if _qa is None:
        print("[系统] 初始化LLMQA引擎...")
        from llm_qa import LLMQA
        _qa = LLMQA()
        # 预热：加载BGE-M3 + 加载BM25缓存
        print("[系统] 预热中（加载BGE-M3 + BM25缓存）...")
        _qa.retriever.warm_up()
        print("[系统] 就绪")
    return _qa


def log(msg, step="▸"):
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] [{step}] {msg}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")
    compare_mode = data.get("compare_mode", False)
    llm_only = data.get("llm_only", False)

    if not question:
        return jsonify({"error": "请输入问题"}), 400

    t0 = time.time()
    log(f"问题: {question}", "📥")

    try:
        qa = get_qa()

        if llm_only:
            log("纯LLM模式", "🤖")
            rag_answer = qa.answer_without_context(question)
            sources = []
        else:
            # RAG流程：查询扩展→混合检索→重排序→生成（统一在llm_qa中完成）
            log("RAG模式：检索+生成...", "🔍")
            rag_answer, raw_sources = qa.answer_with_context(question)
            sources = [{"text": s.get("text", "")[:300],
                        "score": s.get("display_score", s.get("rerank_score", s.get("rrf_score", s.get("score", 0)))),
                        "id": s.get("id")} for s in raw_sources]
            log(f"    检索到{len(sources)}个片段", "🔍")

        llm_answer = ""
        evaluation = {}
        if compare_mode:
            llm_answer = qa.answer_without_context(question)
            from evaluator import Evaluator
            ev = Evaluator()
            evaluation = ev._llm_judge(question, rag_answer, llm_answer)

        elapsed = time.time() - t0
        log(f"✅ 完成 ({elapsed:.1f}s)", "📊")
        return jsonify({
            "rag_answer": rag_answer, "llm_answer": llm_answer,
            "sources": sources, "response_time": f"{elapsed:.1f}秒",
            "compare": compare_mode, "evaluation": evaluation
        })

    except Exception as e:
        import traceback
        log(f"错误: {traceback.format_exc()}", "❌")
        return jsonify({
            "error": str(e), "rag_answer": f"系统异常: {str(e)}",
            "sources": [], "response_time": f"{time.time()-t0:.1f}秒"
        })


@app.route("/status")
def status():
    try:
        qa = get_qa()
        count = qa.retriever.store.count()
        return jsonify({
            "status": "running", "milvus": f"rag_pdf_qa_v2 ({count}条)",
            "model": "BGE-M3(+BM25+RRF)", "llm": "DeepSeek"
        })
    except Exception as e:
        return jsonify({"status": f"error: {e}"})


if __name__ == "__main__":
    # 启动前先预热（加载BGE-M3 + BM25缓存）
    print("[系统] 正在预热（加载BGE-M3模型 + BM25缓存）...")
    from hybrid_retriever import HybridRetriever
    _hr = HybridRetriever()
    _hr.warm_up()
    # 注入预热好的检索器到LLMQA
    from llm_qa import LLMQA
    _qa = LLMQA(retriever=_hr)
    print("[系统] 预热完成，所有模型就绪，启动Flask...")

    host_ip = "127.0.0.1"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        host_ip = s.getsockname()[0]
        s.close()
    except: pass

    print("="*50)
    print("  RAG 问答系统 v2（优化版）")
    print("  工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化")
    print("="*50)
    print(f"  http://localhost:5000")
    print(f"  BGE-M3+BM25混合检索 + Reranker重排序")
    print(f"  首次提问时加载模型，请稍候")
    print("="*50)
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
