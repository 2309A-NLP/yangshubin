"""
模块功能: 全局配置模块
集中管理所有配置项，包括 MiMo API、Milvus 向量数据库、BGE-M3 模型路径、数据路径等
优先从环境变量读取，支持 Windows / WSL / Docker 跨平台运行
工单编号: 人工智能NLP-RAG-金融问答系统部署
"""

import os          # 操作系统接口，用于读取环境变量和路径判断
from pathlib import Path  # 跨平台路径处理
import platform    # 系统平台检测


class Config:
    """配置类，包含所有应用配置项，从环境变量加载并提供合理默认值"""

    # ======== 跨平台路径自动检测 ========
    # 判断是否在 Docker 容器内部运行
    _is_docker: bool = os.path.exists("/.dockerenv") or os.path.exists("/app")
    # 判断是否在 Windows 系统上运行（包括 Windows 原生 Python）
    _is_windows: bool = platform.system() == "Windows"

    # 项目根目录：Docker 内固定 /app，Windows 用当前目录，WSL/Linux 同理
    PROJECT_ROOT: str = os.environ.get(
        "PROJECT_ROOT",
        "/app" if _is_docker else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    # app 模块目录
    APP_DIR: str = os.path.join(PROJECT_ROOT, "app")
    # Flask 模板目录
    TEMPLATE_DIR: str = os.path.join(APP_DIR, "templates")
    # 输出目录，存储运行日志和中间结果
    OUTPUT_DIR: str = os.path.join(PROJECT_ROOT, "output")

    # ======== MiMo (小米开放平台) LLM 配置 ========
    # MiMo API 密钥（默认使用 Token Plan 订阅密钥）
    MIMO_API_KEY: str = os.environ.get(
        "MIMO_API_KEY",
        "tp-czex5np7bgf6duyuyvqntmw44xcunseatmblffiw9lk9w0jy"
    )
    # MiMo API 基础地址（Token Plan 订阅版，OpenAI 兼容接口）
    MIMO_API_BASE: str = os.environ.get(
        "MIMO_API_BASE",
        "https://token-plan-cn.xiaomimimo.com/v1"
    )
    # 使用的模型名称（小米旗舰推理模型 MiMo-v2.5-pro）
    MIMO_MODEL: str = os.environ.get("MIMO_MODEL", "mimo-v2.5-pro")
    # LLM 生成温度，越低越确定
    LLM_TEMPERATURE: float = float(os.environ.get("LLM_TEMPERATURE", "0.1"))
    # LLM 最大生成 token 数
    LLM_MAX_TOKENS: int = int(os.environ.get("LLM_MAX_TOKENS", "2048"))

    # ======== BGE-M3 文本向量模型配置 ========
    # Docker 内路径固定为 /models/bge-m3（通过 volume 挂载）
    # 本地路径指向 Windows 桌面
    _docker_bge: str = "/models/bge-m3"
    _local_bge: str = r"C:\Users\31326\Desktop\bge-m3" if _is_windows else "/mnt/c/Users/31326/Desktop/bge-m3"
    # 优先读取环境变量，其次是 MODEL_CACHE_DIR 兼容旧版 Docker 配置
    _default_bge: str = os.environ.get("MODEL_CACHE_DIR", "")
    if _default_bge:
        _default_bge = os.path.join(_default_bge, "bge-m3")
    else:
        _default_bge = _docker_bge if _is_docker else _local_bge
    BGE_MODEL_PATH: str = os.environ.get("BGE_MODEL_PATH", _default_bge)
    # 是否使用 GPU 推理（RTX5060 支持 CUDA）
    USE_GPU: bool = os.environ.get("USE_GPU", "true").lower() == "true"
    # 批处理大小（8GB 显存安全值）
    BATCH_SIZE: int = int(os.environ.get("BATCH_SIZE", "2"))
    # BGE-M3 最大序列长度
    MAX_SEQ_LENGTH: int = int(os.environ.get("MAX_SEQ_LENGTH", "1024"))

    # ======== Milvus 向量数据库配置 ========
    # Docker 内通过服务名 milvus-standalone 访问，本地用 localhost
    MILVUS_HOST: str = os.environ.get("MILVUS_HOST", "milvus-standalone" if _is_docker else "localhost")
    # Milvus gRPC 端口
    MILVUS_PORT: str = os.environ.get("MILVUS_PORT", "19530")
    # 连接超时秒数
    MILVUS_TIMEOUT: int = int(os.environ.get("MILVUS_TIMEOUT", "30"))
    # 向量集合名称
    MILVUS_COLLECTION: str = os.environ.get("MILVUS_COLLECTION", "rag_documents")
    # 向量维度（BGE-M3 输出 1024 维）
    VECTOR_DIM: int = int(os.environ.get("VECTOR_DIM", "1024"))

    # ======== PDF 数据路径配置 ========
    _docker_data: str = "/data/pdfs"
    _local_data: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    DATA_DIR: str = os.environ.get("DATA_DIR", _docker_data if _is_docker else _local_data)

    # ======== 文本分块配置 ========
    CHUNK_SIZE: int = int(os.environ.get("CHUNK_SIZE", "300"))
    CHUNK_OVERLAP: int = int(os.environ.get("CHUNK_OVERLAP", "50"))

    # ======== 检索配置 ========
    TOP_K: int = int(os.environ.get("TOP_K", "8"))
    SCORE_THRESHOLD: float = float(os.environ.get("SCORE_THRESHOLD", "0.5"))

    # ======== Flask Web 服务配置 ========
    FLASK_HOST: str = os.environ.get("FLASK_HOST", "0.0.0.0")
    FLASK_PORT: int = int(os.environ.get("FLASK_PORT", "5008"))
    FLASK_DEBUG: bool = os.environ.get("FLASK_DEBUG", "false").lower() == "true"


# 实例化全局配置对象，供其他模块引用
config = Config()

# ===== 模块级常量快捷引用（兼容 from app.config import XXX 的写法） =====
MIMO_API_KEY = config.MIMO_API_KEY
MIMO_API_BASE = config.MIMO_API_BASE
MIMO_MODEL = config.MIMO_MODEL
MILVUS_HOST = config.MILVUS_HOST
MILVUS_PORT = config.MILVUS_PORT
MILVUS_COLLECTION = config.MILVUS_COLLECTION
VECTOR_DIM = config.VECTOR_DIM
BGE_MODEL_PATH = config.BGE_MODEL_PATH
BATCH_SIZE = config.BATCH_SIZE
MAX_SEQ_LENGTH = config.MAX_SEQ_LENGTH
DATA_DIR = config.DATA_DIR
