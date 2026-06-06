"""
app.py - RAG工单5 Flask Web应用
工单编号: 人工智能NLP-RAG-Query理解优化任务
完成需求: 提供多轮对话Web界面，支持指代消解+省略补全+历史回顾
功能说明: Flask后端API（/ask问答 /reset重置 /history历史 /status状态）
"""

import logging  # 日志
import json     # JSON解析请求和响应
import time     # 计时

# 导入配置
from config import OUTPUT_DIR, PROJECT_DIR, WORK_ORDER_ID

# 设置日志
logging.basicConfig(
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("app")

# 全局实例（延迟加载，避免Flask启动时卡住）
_flask_app = None
_retriever = None
_embedder = None
_milvus = None
_dialogue_mgr = None


def get_flask_app():
    """
    获取Flask应用实例（单例模式，延迟初始化）
    返回: Flask应用对象
    """
    global _flask_app
    if _flask_app is not None:
        return _flask_app

    logger.info("初始化Flask应用...")
    from flask import Flask, request, jsonify, render_template
    app = Flask(__name__, template_folder=str(_ROOT / "研发" / "templates"))

    @app.route("/")
    def index():
        """渲染首页：传入工单编号"""
        return render_template("index.html", work_order_id=WORK_ORDER_ID)

    @app.route("/ask", methods=["POST"])
    def ask():
        """
        多轮问答接口
        POST参数: {"question": str, "session_id": str}
        返回: {"answer","confidence","rewritten_question","sources",...}
        """
        data = request.get_json(force=True)
        question = data.get("question", "").strip()
        session_id = data.get("session_id", "default")

        if not question:
            return jsonify({"error": "请输入问题"}), 400

        logger.info(f"[{session_id}] 收到: {question[:40]}...")

        try:
            # 获取检索器和对话管理器
            retriever = get_retriever()
            dm = get_dialogue_manager()

            # 获取历史对话，进行Query重写
            history = dm.get_history(session_id)
            from query_rewriter import rewrite_query
            rewritten = rewrite_query(history, question)

            # 检索相关文档
            search_result = retriever.retrieve(rewritten)

            # 生成答案
            if search_result["results"]:
                from qa_generator import generate_answer
                answer = generate_answer(rewritten, search_result["results"])
            else:
                answer = {
                    "question": rewritten,
                    "answer": "未找到相关的文档内容，请尝试其他问题。",
                    "confidence": "low",
                    "sources": [],
                    "response_time": search_result["search_time"],
                }

        except Exception as e:
            import traceback
            logger.error(f"问答处理失败: {e}\n{traceback.format_exc()}")
            return jsonify({
                "question": question, "answer": f"处理出错: {str(e)}",
                "confidence": "low", "sources": [], "response_time": 0,
            }), 500

        # 将本轮对话加入历史
        dm.add_turn(session_id, question, answer["answer"])

        # 合并信息
        answer["original_question"] = question
        answer["rewritten_question"] = rewritten
        answer["search_time"] = round(search_result["search_time"], 2)
        answer["session_id"] = session_id
        answer["turn_count"] = dm.get_session(session_id)["turn_count"]
        answer["source_count"] = len(answer.get("sources", []))

        return jsonify(answer)

    @app.route("/reset", methods=["POST"])
    def reset_session():
        """重置对话会话"""
        data = request.get_json(force=True)
        session_id = data.get("session_id", "default")
        dm = get_dialogue_manager()
        dm.clear_session(session_id)
        dm.create_session(session_id)
        return jsonify({"status": "ok", "session_id": session_id})

    @app.route("/history", methods=["GET"])
    def get_history_route():
        """获取当前会话的对话历史"""
        session_id = request.args.get("session_id", "default")
        dm = get_dialogue_manager()
        history = dm.get_history(session_id)
        return jsonify({"history": history, "session_id": session_id})

    @app.route("/status")
    def status():
        """查看系统运行状态"""
        return jsonify({
            "status": "running",
            "work_order": WORK_ORDER_ID,
            "model_loaded": _retriever is not None,
        })

    _flask_app = app
    logger.info("Flask应用初始化完成")
    return app


def get_retriever():
    """
    获取检索器实例（延迟加载模型）
    首次调用时加载BGE-M3 + 连接Milvus + 创建检索器
    """
    global _retriever, _embedder, _milvus
    if _retriever is not None:
        return _retriever

    logger.info("首次请求，加载模型和向量数据库...")
    start = time.time()

    from embedder import BgeM3Embedder
    from milvus_handler import MilvusHandler
    from retriever import Retriever

    # 加载BGE-M3嵌入模型
    logger.info("加载BGE-M3...")
    _embedder = BgeM3Embedder()
    _embedder.load_model()

    # 连接Milvus
    logger.info("连接Milvus...")
    _milvus = MilvusHandler()
    _milvus.connect()
    _milvus.create_collection()

    # 创建检索器
    _retriever = Retriever(_embedder, _milvus)

    logger.info(f"模型加载完成! 耗时: {time.time()-start:.1f}秒")
    return _retriever


def get_dialogue_manager():
    """获取对话管理器实例（单例）"""
    global _dialogue_mgr
    if _dialogue_mgr is None:
        from dialogue_manager import DialogueManager
        _dialogue_mgr = DialogueManager()
        _dialogue_mgr.create_session("default")
    return _dialogue_mgr


if __name__ == "__main__":
    app = get_flask_app()
    logger.info("启动Web服务: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)
