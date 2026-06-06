"""
模块功能: Docker 容器健康检查脚本
启动一个轻量级 HTTP 服务监听 8080 端口
返回 {"status":"ok"} JSON 供 Docker HEALTHCHECK 指令使用
工单编号: 人工智能NLP-RAG-金融问答系统部署
"""

import sys       # 系统接口
import json      # JSON 处理
from http.server import HTTPServer, BaseHTTPRequestHandler  # HTTP 服务器

# === 配置区域 ===
HEALTHCHECK_HOST = "0.0.0.0"
HEALTHCHECK_PORT = 8080


class HealthCheckHandler(BaseHTTPRequestHandler):
    """简易 HTTP 健康检查处理器，所有路径均返回 {"status":"ok"}"""

    def do_GET(self):
        """处理 GET 请求，返回健康状态 JSON"""
        self._respond()

    def do_HEAD(self):
        """处理 HEAD 请求，仅返回状态码"""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

    def do_POST(self):
        """处理 POST 请求，返回健康状态 JSON"""
        self._respond()

    def _respond(self):
        """统一的 JSON 响应方法，返回 {"status":"ok"}"""
        response = json.dumps({"status": "ok"}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format, *args):
        """关闭默认的请求日志，避免健康检查轮询刷日志"""
        pass


def run_healthcheck():
    """启动健康检查 HTTP 服务（阻塞式），支持 Ctrl+C 优雅关闭"""
    server = HTTPServer((HEALTHCHECK_HOST, HEALTHCHECK_PORT), HealthCheckHandler)
    print(f"健康检查服务启动: http://{HEALTHCHECK_HOST}:{HEALTHCHECK_PORT}/")
    print(f"  端点: GET/HEAD/POST / → {{\"status\":\"ok\"}}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n健康检查服务已关闭。")
        server.server_close()


if __name__ == "__main__":
    run_healthcheck()
