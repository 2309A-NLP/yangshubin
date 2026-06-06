"""
app.py - RAG工单4 Flask Web应用
工单编号: 人工智能NLP-RAG-图像内容解析及检索优化
功能: 提供Web界面，支持用户输入问题并展示检索答案，
      包含答案置信度、来源页码、响应时间等信息
      采用延迟加载（Lazy Loading）避免启动时卡死
"""

import logging
import json
import os
import sys
import time
# ===== 路径桥接 =====
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ["设计", "研发", "测试", "优化", "部署"]:
    _p = os.path.join(_ROOT, _sub)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import (
    OUTPUT_DIR, PROJECT_DIR, WORK_ORDER_ID, TEST_QUESTIONS,
    LOG_FORMAT, LOG_DATE_FORMAT
)

logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT, level=logging.INFO)
logger = logging.getLogger("app")

# 延迟加载
_flask_app = None
_retrieval_pipeline = None
_embedder = None
_milvus_handler = None


def get_flask_app():
    """获取Flask应用实例（单例模式，延迟初始化）"""
    global _flask_app

    if _flask_app is not None:
        return _flask_app

    logger.info("初始化Flask应用...")

    from flask import Flask, request, jsonify, render_template

    app = Flask(__name__, template_folder=str(_ROOT / "研发" / "templates"))

    @app.route("/")
    def index():
        return render_template("index.html", work_order_id=WORK_ORDER_ID)

    @app.route("/ask", methods=["POST"])
    def ask():
        data = request.get_json(force=True)
        question = data.get("question", "").strip()
        lang = data.get("lang", "auto")

        if not question:
            return jsonify({"error": "请输入问题"}), 400

        logger.info(f"收到问题: {question[:50]}...")

        try:
            pipeline = get_retrieval_pipeline()
            retrieval_result = pipeline.retrieve(question)

            if retrieval_result["results"]:
                from qa_generator import generate_answer
                answer = generate_answer(question, retrieval_result["results"], lang=lang)
            else:
                answer = {
                    "question": question,
                    "answer": "未找到相关的文档内容，请尝试其他问题。",
                    "confidence": "low",
                    "sources": [],
                    "response_time": retrieval_result["search_time"],
                }

            answer["search_time"] = round(retrieval_result["search_time"], 2)
            answer["source_count"] = len(retrieval_result["results"])
            return jsonify(answer)

        except Exception as e:
            logger.error(f"问答处理失败: {e}")
            return jsonify({
                "question": question,
                "answer": f"处理出错: {str(e)}",
                "confidence": "low",
                "sources": [],
                "response_time": 0,
            }), 500

    @app.route("/status")
    def status():
        global _retrieval_pipeline
        return jsonify({
            "status": "running" if _retrieval_pipeline is not None else "ready",
            "work_order": WORK_ORDER_ID,
            "model_loaded": _retrieval_pipeline is not None,
        })

    @app.route("/batch_test", methods=["POST"])
    def batch_test():
        """批量测试全部16题 - 只检索不调LLM，秒出结果"""
        try:
            pipeline = get_retrieval_pipeline()
            test_questions = TEST_QUESTIONS
            results = []

            for q in test_questions:
                # 只做检索，不调MiMo API
                ret = pipeline.retrieve(q["question"])
                sources = [{
                    "page_num": r["page_num"],
                    "has_image": r.get("has_image", False),
                    "content_preview": r["content"][:150] + "..." if len(r["content"]) > 150 else r["content"],
                    "score": float(r.get("score", 0)),
                } for r in ret["results"]]

                results.append({
                    "id": q["id"],
                    "question": q["question"],
                    "sources": sources,
                    "source_count": len(sources),
                    "search_time": round(ret["search_time"], 2),
                })

            return jsonify({"results": results, "total": len(results), "mode": "retrieval_only"})

        except Exception as e:
            logger.error(f"批量测试失败: {e}")
            return jsonify({"error": str(e)}), 500

    _flask_app = app
    logger.info("Flask应用初始化完成")
    return app


def get_retrieval_pipeline():
    """获取检索流水线实例（延迟加载）"""
    global _retrieval_pipeline, _embedder, _milvus_handler

    if _retrieval_pipeline is not None:
        return _retrieval_pipeline

    logger.info("首次请求，正在加载模型和向量数据库...")
    start_time = time.time()

    from embedder import BgeM3Embedder
    from milvus_handler import MilvusHandler
    from retriever import RetrievalPipeline

    logger.info("加载BGE-M3嵌入模型...")
    _embedder = BgeM3Embedder()
    _embedder.load_model()

    logger.info("连接Milvus向量数据库...")
    _milvus_handler = MilvusHandler()
    _milvus_handler.connect()
    _milvus_handler.create_collection()

    _retrieval_pipeline = RetrievalPipeline(_embedder, _milvus_handler)

    elapsed = time.time() - start_time
    logger.info(f"模型加载完成! 总耗时: {elapsed:.2f}秒")
    return _retrieval_pipeline


if __name__ == "__main__":
    app = get_flask_app()
    logger.info("启动Flask开发服务器: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)
