"""\napp.py - RAG工单9 Flask Web展示模块\n工单编号: 人工智能NLP-RAG-Graph RAG 优化任务
需求: 交互友好性 — Web界面展示GraphRAG优化前后对比结果
功能: Flask路由(首页/API报告/API摘要)，延迟初始化
"""

import logging, json
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '设计'))
from config import OUTPUT_DIR, WO_ID, LOG_FMT, LOG_DATEFMT

logging.basicConfig(format=LOG_FMT, datefmt=LOG_DATEFMT, level=logging.INFO)
logger = logging.getLogger("app")

app = None


def create_app():
    """创建Flask应用（延迟初始化），避免import时阻塞"""
    from flask import Flask, render_template, jsonify

    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["WO_ID"] = WO_ID

    @app.route("/")
    def index():
        """首页展示GraphRAG优化对比报告"""
        return render_template("index.html", wo_id=WO_ID)

    @app.route("/api/report")
    def api_report():
        """返回评估报告JSON"""
        json_path = OUTPUT_DIR / "evaluation_report.json"
        if not json_path.exists():
            return jsonify({"error": "报告未生成"})
        with open(json_path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))

    @app.route("/api/summary")
    def api_summary():
        """返回优化前后对比摘要"""
        json_path = OUTPUT_DIR / "evaluation_report.json"
        if not json_path.exists():
            return jsonify({"error": "无数据"})
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data.get("summary", {}))

    return app


def get_app():
    """获取Flask应用实例（单例模式）"""
    global app
    if app is None:
        app = create_app()
    return app


if __name__ == "__main__":
    """启动Web服务"""
    flapp = get_app()
    logger.info(f"启动Web: http://127.0.0.1:5009")
    flapp.run(host="127.0.0.1", port=5009, debug=False)
