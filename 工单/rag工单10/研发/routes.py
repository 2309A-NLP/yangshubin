"""
模块功能: Flask API 路由模块
定义全部 HTTP 接口：首页/健康检查/问答/图谱/评估等
工单编号: 人工智能NLP-RAG-金融问答系统部署
"""

import json, logging, time
from pathlib import Path
from flask import Blueprint, request, jsonify, render_template
from app.config import config

logger = logging.getLogger("routes")
api_bp = Blueprint("api", __name__)

# 全局状态（从文件加载）
_state = {"qa_pairs": [], "graph_data": {"nodes": [], "edges": []}, "summary": {}}


def load_state_from_files():
    """从 output 目录加载已保存的数据"""
    out = Path(config.OUTPUT_DIR)
    for name, key in [("qa_results.json", "qa_pairs"),
                       ("knowledge_graph.json", "graph_data"),
                       ("evaluation_summary.json", "summary")]:
        f = out / name
        if f.exists():
            with open(f, "r", encoding="utf-8") as fp:
                _state[key] = json.load(fp)


@api_bp.route("/", methods=["GET"])
def index():
    """首页: Web 聊天界面"""
    return render_template("index.html")


@api_bp.route("/health", methods=["GET"])
def health():
    """健康检查"""
    return jsonify({"status": "ok", "service": "rag-finance-qa", "version": "1.0.0"})


@api_bp.route("/ask", methods=["POST"])
def ask_question():
    """问答接口（非流式）"""
    try:
        question = (request.get_json(force=True).get("question") or "").strip()
        if not question:
            return jsonify({"error": "问题不能为空"}), 400
        from app.rag_engine import get_engine
        result = get_engine().ask(question)
        return jsonify({"question": result["question"], "answer": result["answer"],
                        "sources": result["sources"]})
    except Exception as e:
        logger.error(f"问答异常: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route("/ask/stream", methods=["POST"])
def ask_stream():
    """流式问答（SSE）"""
    try:
        question = (request.get_json(force=True).get("question") or "").strip()
        if not question:
            return jsonify({"error": "问题不能为空"}), 400
        from flask import Response

        def gen():
            from app.rag_engine import get_engine
            from app.llm_client import MiMoClient
            ctx = get_engine().retrieve_context(question)
            prompt = f"你是一个专业的金融问答助手。\n\n参考文档:\n{ctx}\n\n问题: {question}\n\n答案:"
            for chunk in MiMoClient().generate_stream(prompt):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        return Response(gen(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/api/question", methods=["POST"])
def api_question():
    """单问题实时问答（支持 Vector/GraphRAG 切换）"""
    data = request.get_json(force=True)
    question, use_graph = data.get("question", ""), data.get("use_graph", True)
    if not question:
        return jsonify({"error": "请输入问题"}), 400
    t0 = time.time()
    try:
        from app.embedding import embed_query
        from app.graph_retriever import GraphRetriever
        from app.graph_builder import get_graph as load_g
        from app.llm_client import MiMoClient
        qv = embed_query(question)
        if qv is None:
            return jsonify({"error": "向量化失败"}), 500
        gb = load_g()
        gb_ok = gb.load()
        chunks = []
        cp = Path(config.OUTPUT_DIR) / "chunks_data.json"
        if cp.exists():
            with open(cp, "r", encoding="utf-8") as f:
                chunks = json.load(f)
        rt = GraphRetriever(gb if gb_ok else None)
        rt.set_chunks(chunks)
        hits = rt.retrieve(qv.tolist(), question, use_graph=use_graph)
        ctx, sources = "", set()
        for i, ch in enumerate(hits):
            ctx += f"[{i+1}] {ch.get('text',ch.get('content',''))[:800]}\n\n"
            src = ch.get("source_pdf", "")
            if src:
                sources.add(src)
        mode = "知识图谱增强检索" if use_graph else "向量检索"
        prompt = f"你是一位专业的金融年报分析师。\n参考资料:\n{ctx}\n问题: {question}\n仅基于参考资料回答。\n回答:"
        ans = MiMoClient().generate(prompt)
        rt.close()
        return jsonify({"answer": ans, "response_time": round(time.time()-t0,2),
                        "mode": mode, "sources": list(sources)})
    except Exception as e:
        logger.error(f"问答失败: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route("/api/qa_results", methods=["GET"])
def get_qa_results():
    """返回问答结果列表"""
    return jsonify(_state.get("qa_pairs", []))


@api_bp.route("/api/graph_data", methods=["GET"])
def get_graph_data():
    """返回图谱可视化数据"""
    return jsonify(_state.get("graph_data", {"nodes": [], "edges": []}))


@api_bp.route("/api/graph_viz", methods=["GET"])
def graph_viz():
    """返回带颜色的图谱数据"""
    gd = _state.get("graph_data", {"nodes": [], "edges": []})
    colors = {"公司": "#4CAF50", "人物": "#2196F3", "产品": "#FF9800",
              "指标": "#9C27B0", "事件": "#F44336", "时间": "#00BCD4"}
    for n in gd.get("nodes", []):
        n["color"] = colors.get(n.get("type", ""), "#999")
    return jsonify(gd)


@api_bp.route("/api/summary", methods=["GET"])
def get_summary():
    """返回评估对比摘要"""
    return jsonify(_state.get("summary", {}))


@api_bp.route("/stats", methods=["GET"])
def get_stats():
    """系统统计信息"""
    return jsonify({"status": "ok", "config": {
        "milvus_host": config.MILVUS_HOST, "milvus_port": config.MILVUS_PORT,
        "collection": config.MILVUS_COLLECTION, "model_path": config.BGE_MODEL_PATH,
        "data_dir": config.DATA_DIR, "llm_model": config.MIMO_MODEL,
        "llm_api_base": config.MIMO_API_BASE,
    }})


def register_routes(app):
    """注册所有路由到 Flask 应用"""
    app.register_blueprint(api_bp)
    load_state_from_files()
    logger.info("API 路由注册完成")
