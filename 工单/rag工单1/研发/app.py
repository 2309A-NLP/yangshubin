# -*- coding: utf-8 -*-
"""
Flask Web应用 —— RAG问答系统交互界面
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
说明：前端页面代码无行数限制，HTML模板在 templates/index.html
"""

import os
import sys
import time
import socket
from datetime import datetime
from flask import Flask, render_template, request, jsonify

# 添加当前目录到系统路径，确保能导入各模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
)

# 全局变量：延迟初始化（只在第一次请求时加载）
_retriever = None
_llm_qa = None


def get_retriever():
    """延迟加载检索器（首次调用时初始化BGE-M3模型）"""
    global _retriever
    if _retriever is None:
        print("[系统] 正在加载 BGE-M3 模型和检索器...")
        from retriever import Retriever
        _retriever = Retriever()
        print("[系统] 检索器加载完成")
    return _retriever


def get_llm():
    """延迟加载LLM引擎"""
    global _llm_qa
    if _llm_qa is None:
        print("[系统] 初始化 DeepSeek LLM 引擎...")
        from llm_qa import LLMQA
        _llm_qa = LLMQA()
        print("[系统] LLM引擎加载完成")
    return _llm_qa


def log(msg: str, step: str = "▸"):
    """带时间戳的日志"""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  [{ts}] [{step}] {msg}")


# ============================================================
# 路由：首页
# ============================================================
@app.route("/")
def index():
    """返回 Web 首页（调用独立的HTML模板）"""
    return render_template("index.html")


# ============================================================
# 路由：问答 API
# ============================================================
@app.route("/ask", methods=["POST"])
def ask():
    """
    处理用户提问
    请求体: {"question": "...", "compare_mode": true/false, "llm_only": true/false}
    返回: RAG回答/纯LLM回答/来源/评估
    """
    data = request.get_json()
    question = data.get("question", "")
    compare_mode = data.get("compare_mode", False)
    llm_only = data.get("llm_only", False)

    if not question:
        return jsonify({"error": "请输入问题"}), 400

    start_time = time.time()
    log(f"收到问题: {question}", "📥")

    try:
        # 延迟加载模型和引擎
        log("正在加载模型和引擎...", "⚙️")
        retriever = get_retriever()
        llm_qa = get_llm()
        log("模型就绪", "⚙️")

        # ----- RAG 模式 -----
        rag_answer = ""
        sources = []

        if not llm_only:
            log("Step 1/4: 查询理解（意图识别+消歧）", "①")
            from query_processor import QueryProcessor
            qp = QueryProcessor()
            analysis = qp.analyze_query(question)
            log(f"  意图: {analysis.get('intent', 'unknown')}", "①")
            log(f"  关键词: {analysis.get('keywords', [])}", "①")

            log("Step 2/4: BGE-M3编码查询向量", "②")
            query_vec = retriever.embedding_model.encode_query(question)
            log(f"  向量维度: {query_vec.shape[1]}", "②")

            log("Step 3/4: Milvus向量检索", "③")
            sources_raw = retriever.vector_store.search(
                query_vector=query_vec[0].tolist(),
                top_k=5
            )
            log(f"  检索到 {len(sources_raw)} 个文档片段", "③")
            for i, s in enumerate(sources_raw, 1):
                log(f"  片段{i}: 得分={s.get('score', 0):.4f}", "③")

            # 格式化来源
            for r in sources_raw:
                sources.append({
                    "text": r.get("text", "")[:300],
                    "score": r.get("score", 0.0)
                })

            # 拼接上下文
            context_parts = []
            for i, r in enumerate(sources_raw, 1):
                context_parts.append(
                    f"[文档片段 {i}]（相似度: {r['score']:.4f}）\n{r['text']}"
                )
            context = "\n\n".join(context_parts)
            log(f"  上下文总长度: {len(context)} 字符", "③")

            log("Step 4/4: DeepSeek生成RAG回答", "④")
            rag_answer = llm_qa.answer_with_context(question, context)
            log(f"  回答长度: {len(rag_answer)} 字符", "④")
        else:
            log("纯LLM模式（不检索文档）", "🤖")
            rag_answer = llm_qa.answer_without_context(question)

        # ----- 对比模式 -----
        llm_answer = ""
        evaluation = {}

        if compare_mode:
            llm_answer = llm_qa.answer_without_context(question)
            if llm_only:
                rag_answer, llm_answer = llm_answer, rag_answer

            from evaluator import RAGEvaluator
            evaluator = RAGEvaluator()
            evaluation = evaluator.evaluate_answer(question, rag_answer, llm_answer)

        elapsed = time.time() - start_time
        log(f"✅ 总耗时: {elapsed:.2f}秒", "📊")

        return jsonify({
            "rag_answer": rag_answer,
            "llm_answer": llm_answer,
            "sources": sources,
            "response_time": f"{elapsed:.2f}秒",
            "compare": compare_mode,
            "evaluation": evaluation
        })

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"[错误] {error_detail}")
        return jsonify({
            "error": str(e),
            "error_detail": error_detail,
            "rag_answer": f"系统异常: {str(e)}",
            "sources": [],
            "response_time": f"{time.time() - start_time:.2f}秒",
            "compare": compare_mode,
            "evaluation": {}
        })


# ============================================================
# 路由：查看系统状态
# ============================================================
@app.route("/status")
def status():
    """返回系统运行状态"""
    result = {
        "status": "starting",
        "model": "BGE-M3",
        "llm": "DeepSeek",
        "milvus": {}
    }

    try:
        retriever = get_retriever()
        from pymilvus import MilvusClient
        client = MilvusClient(uri=retriever.vector_store.uri)
        if client.has_collection(retriever.vector_store.collection_name):
            stats = client.get_collection_stats(retriever.vector_store.collection_name)
            count = stats.get("row_count", 0)
            result["milvus"] = {"collection": retriever.vector_store.collection_name, "count": count}
        else:
            result["milvus"] = {"error": "集合不存在"}
        client.close()
        result["status"] = "running"
    except Exception as e:
        result["status"] = f"error: {str(e)}"

    return jsonify(result)


# ============================================================
# 启动应用
# ============================================================
if __name__ == "__main__":
    # 获取 WSL 的真实局域网 IP
    host_ip = "127.0.0.1"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        host_ip = s.getsockname()[0]
        s.close()
    except Exception:
        pass

    print("=" * 50)
    print("  RAG PDF 问答系统 - 启动中...")
    print("  工单编号：人工智能NLP-RAG-基于PDF文档的问答系统")
    print("=" * 50)
    print(f"  ✅ 请在 Windows 浏览器打开:")
    print(f"     http://localhost:5000")
    print(f"  📌 BGE-M3 路径: {get_retriever().embedding_model.model_path}")
    print(f"  📌 BGE-M3 + DeepSeek + Milvus")
    print(f"  📌 首次提问时会加载模型，请耐心等待")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
