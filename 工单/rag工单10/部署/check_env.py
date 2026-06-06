"""
模块功能: Docker 容器环境检查脚本
检查项:
  1. Python 版本 >= 3.8
  2. 核心依赖包是否可导入
  3. Milvus 连接是否正常
  4. BGE-M3 模型路径是否存在
  5. MiMo API Key 是否配置
可作为 Docker 入口脚本的一部分运行，也可单独执行
工单编号: 人工智能NLP-RAG-金融问答系统部署
"""

import sys          # 系统接口，用于退出码
import importlib    # 动态导入检查依赖
from pathlib import Path  # 路径处理

# === 从 app.config 读取配置 ===
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from app.config import config
    MILVUS_HOST = config.MILVUS_HOST
    MILVUS_PORT = int(config.MILVUS_PORT)
    BGE_MODEL_PATH = Path(config.BGE_MODEL_PATH)
    MIMO_API_KEY = config.MIMO_API_KEY
    MIMO_API_BASE = config.MIMO_API_BASE
except ImportError:
    # 如果 app 模块不可用，使用默认配置
    MILVUS_HOST = "milvus-standalone"
    MILVUS_PORT = 19530
    BGE_MODEL_PATH = Path("/models/bge-m3")
    MIMO_API_KEY = ""
    MIMO_API_BASE = "https://token-plan-cn.xiaomimimo.com/v1"

# 需要检查的核心依赖列表
REQUIRED_PACKAGES = [
    "flask",                 # Web 框架
    "pymilvus",              # Milvus 客户端
    "sentence_transformers", # BGE-M3 向量化
    "fitz",                  # PyMuPDF PDF 解析
    "openai",                # MiMo API 客户端（OpenAI 兼容）
    "networkx",              # 知识图谱
    "numpy",                 # 数值计算
]


def check_python_version() -> bool:
    """检查 Python 版本是否 >= 3.8"""
    version = sys.version_info
    ok = version.major >= 3 and version.minor >= 8
    status = "✓" if ok else "✗"
    print(f"  [{status}] Python 版本: {version.major}.{version.minor}.{version.micro} (需要 >= 3.8)")
    return ok


def check_dependencies() -> bool:
    """尝试导入所有必需的依赖包，验证安装完整性"""
    all_ok = True
    print(f"\n  检查依赖包 ({len(REQUIRED_PACKAGES)} 个):")
    for pkg in REQUIRED_PACKAGES:
        try:
            # 尝试导入包（处理包名和模块名不一致的情况）
            module_name = pkg.replace("-", "_")
            if pkg == "fitz":
                module_name = "fitz"
            importlib.import_module(module_name if pkg != "fitz" else "fitz")
            print(f"    [✓] {pkg}")
        except ImportError:
            print(f"    [✗] {pkg} — 未安装")
            all_ok = False
    return all_ok


def check_milvus() -> bool:
    """尝试连接 Milvus 数据库并检查服务状态"""
    try:
        from pymilvus import connections, utility
        connections.connect(alias="env_check", host=MILVUS_HOST, port=MILVUS_PORT, timeout=5)
        ok = utility.ping_server()
        connections.disconnect("env_check")
        status = "✓" if ok else "✗"
        print(f"  [{status}] Milvus 连接: {MILVUS_HOST}:{MILVUS_PORT}")
        return ok
    except Exception as e:
        print(f"  [✗] Milvus 连接失败: {MILVUS_HOST}:{MILVUS_PORT} — {e}")
        return False


def check_bge_model() -> bool:
    """检查 BGE-M3 模型路径是否存在"""
    ok = BGE_MODEL_PATH.exists() and BGE_MODEL_PATH.is_dir()
    # 如果直接路径不存在，检查是否包含 config.json 作为更多验证
    if ok:
        has_config = (BGE_MODEL_PATH / "config.json").exists()
        extra = f" (路径: {BGE_MODEL_PATH}, config.json: {'✓' if has_config else '✗'})"
    else:
        extra = f" — 路径不存在: {BGE_MODEL_PATH}"
    print(f"  [{'✓' if ok else '✗'}] BGE-M3 模型{extra}")
    return ok


def check_mimo_api() -> bool:
    """检查 MiMo API Key 是否配置"""
    if MIMO_API_KEY and len(MIMO_API_KEY) > 10:
        print(f"  [✓] MiMo API Key 已配置 (前缀: {MIMO_API_KEY[:12]}..., Base: {MIMO_API_BASE})")
        return True
    else:
        print(f"  [✗] MiMo API Key 未配置或格式不正确")
        return False


def run_all_checks() -> dict:
    """运行全部环境检查，返回每项的布尔结果"""
    print("=" * 52)
    print("  环境检查脚本 — check_env.py (工单10)")
    print("=" * 52)

    results = {}
    print("\n[1/5] Python 版本检查")
    results["python"] = check_python_version()

    print("\n[2/5] 依赖包检查")
    results["dependencies"] = check_dependencies()

    print("\n[3/5] Milvus 连接检查")
    results["milvus"] = check_milvus()

    print("\n[4/5] BGE-M3 模型检查")
    results["bge_model"] = check_bge_model()

    print("\n[5/5] MiMo API 检查")
    results["mimo_api"] = check_mimo_api()

    print("\n" + "=" * 52)
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"  汇总: {passed}/{total} 项通过")
    print("=" * 52)

    if all(results.values()):
        print("\n  ✅ 所有环境检查通过，可以正常启动。")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"\n  ⚠️  以下检查未通过: {', '.join(failed)}")
        print("  请根据上述提示修复后重试。")

    return results


if __name__ == "__main__":
    results = run_all_checks()
    sys.exit(0 if all(results.values()) else 1)
