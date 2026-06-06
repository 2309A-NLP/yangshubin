"""
模块功能: Docker 容器入口编排脚本
按顺序执行:
  1. check_env       — 环境检查（Python/依赖/Milvus/模型/API）
  2. wait_for_milvus — 等待 Milvus 服务就绪（带超时重试）
  3. init_pipeline   — 数据初始化流水线（可按需跳过）
  4. start_app       — 启动主 Flask 应用
被 Dockerfile CMD 调用: CMD ["python", "scripts/docker_entrypoint.py"]
工单编号: 人工智能NLP-RAG-金融问答系统部署
"""

import sys       # 系统接口
import os        # 操作系统接口，读取环境变量
import subprocess  # 子进程管理
import time      # 时间相关

# === 配置区域 ===
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
# 环境变量 SKIP_INIT=1 可跳过数据初始化流水线
SKIP_INIT = os.environ.get("SKIP_INIT", "0") == "1"


def run_subprocess(script_name: str, description: str) -> bool:
    """通过 subprocess 执行同级目录下的 Python 脚本

    Args:
        script_name: 脚本文件名（如 check_env.py）
        description: 日志描述文字

    Returns:
        True 表示执行成功，False 表示失败
    """
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    if not os.path.exists(script_path):
        print(f"[入口] ❌ 脚本不存在: {script_path}")
        return False

    print(f"\n{'='*50}")
    print(f"  [{description}] 执行: {script_name}")
    print(f"{'='*50}")

    start_time = time.time()
    # 执行子进程，输出直接打印到终端
    result = subprocess.run([sys.executable, script_path], capture_output=False)
    elapsed = int(time.time() - start_time)

    if result.returncode == 0:
        print(f"\n  ✅ [{description}] 完成 (耗时: {elapsed}s)")
        return True
    else:
        print(f"\n  ❌ [{description}] 失败 (退出码: {result.returncode})")
        return False


def start_main_app():
    """启动主 Flask 应用

    从 app 模块导入并运行，支持优雅退出信号处理。
    使用 APP_HOST/APP_PORT/APP_DEBUG 环境变量配置。
    """
    print(f"\n{'='*50}")
    print("  启动主应用...")
    print(f"{'='*50}")

    try:
        # 从 app 包导入 Flask 应用工厂
        from app import create_app
        app = create_app()

        # 从环境变量读取 Web 配置，带默认值
        host = os.environ.get("APP_HOST", "0.0.0.0")
        port = int(os.environ.get("APP_PORT", "5008"))
        debug = os.environ.get("APP_DEBUG", "0") == "1"

        print(f"  Flask 应用启动: http://{host}:{port}/")
        print(f"  Debug 模式: {'开启' if debug else '关闭'}")
        print(f"  按 Ctrl+C 优雅关闭\n")

        # 启动 Flask 内置开发服务器
        app.run(host=host, port=port, debug=debug, use_reloader=False)

    except ImportError as e:
        print(f"  ❌ 无法导入主应用: {e}")
        print(f"  请检查 app 模块是否正确安装。")
        sys.exit(1)
    except Exception as e:
        print(f"  ❌ 启动主应用失败: {e}")
        sys.exit(1)


def entrypoint():
    """Docker 容器入口主函数 — 启动顺序编排"""
    print(f"\n{'='*50}")
    print("  Docker Entrypoint — RAG 金融问答系统 (工单10)")
    print(f"{'='*50}")

    # === 步骤1: 环境检查 ===
    if not run_subprocess("check_env.py", "环境检查"):
        print("\n  ❌ 环境检查未通过，终止启动。")
        sys.exit(1)

    # === 步骤2: 等待 Milvus 就绪 ===
    if not run_subprocess("wait_for_milvus.py", "等待 Milvus"):
        print("\n  ❌ Milvus 未就绪，终止启动。")
        sys.exit(1)

    # === 步骤3: 数据初始化（可选跳过）===
    if SKIP_INIT:
        print(f"\n  ⏭️  环境变量 SKIP_INIT=1，跳过数据初始化流水线。")
    else:
        if not run_subprocess("init_pipeline.py", "数据初始化"):
            print("\n  ⚠️  数据初始化失败，但将继续启动应用。")
            print("  如需跳过初始化，设置环境变量 SKIP_INIT=1。")

    # === 步骤4: 启动主应用 ===
    start_main_app()


if __name__ == "__main__":
    entrypoint()
