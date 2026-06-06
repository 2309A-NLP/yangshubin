"""
config.py - RAG工单9 GraphRAG配置模块
需求: GraphRAG优化 — 管理CCF年报路径、MiMo API、BGE-M3模型、评估阈值
功能: 统一配置API密钥、模型路径、Milvus连接、分块参数、评估标准
"""
import os, sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.resolve()
DESKTOP_DIR = Path("/mnt/c/Users/31326/Desktop") if sys.platform != "win32" \
    else Path(os.environ["USERPROFILE"]) / "Desktop"

# ===== CCF竞赛PDF路径 =====
CCF_PDF_DIR = DESKTOP_DIR / "工单" / "RAG 工单" / "附件" / "ccf_competition" / "pdf"
CCF_PDF_DIR_SIMPLE = DESKTOP_DIR / "ccf_competition" / "pdf"
SAMPLE_QUESTIONS_PDF = DESKTOP_DIR / "工单" / "RAG 工单" / "附件" / "sample_questions.pdf"
SAMPLE_QUESTIONS_FALLBACK = PROJECT_DIR / "sample_questions.pdf"
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== 小米MiMo API（OpenAI兼容） =====
MIMO_API_KEY = "tp-czex5np7bgf6duyuyvqntmw44xcunseatmblffiw9lk9w0jy"
MIMO_BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"
MIMO_MODEL = "mimo-v2.5"           # 文本模型(快)，用于QA和评估
MIMO_REASONING_MODEL = "mimo-v2.5-pro"  # 推理模型(慢)，用于实体提取
MIMO_TIMEOUT = 60
MIMO_MAX_TOKENS = 2048

# ===== 备用DeepSeek API =====
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# ===== BGE-M3向量模型 =====
BGE_M3_PATH = str(DESKTOP_DIR / "bge-m3")
BGE_M3_BATCH = 2
BGE_M3_MAX_LEN = 1024
BGE_M3_DEVICE = "cuda"
BGE_M3_FP16 = True

# ===== Milvus向量库 =====
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
MILVUS_COLL = "rag_work_order_9"
MILVUS_DIM = 1024

# ===== 分块 & 检索参数 =====
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_K = 10
GRAPH_EXPAND_K = 5
FINAL_TOP_K = 8

# ===== GraphRAG实体类型 =====
ENTITY_TYPES = ["公司", "人物", "产品", "指标", "事件", "时间"]

# ===== 评估阈值 =====
CONTEXT_PRECISION_THRESHOLD = 0.8
CONTEXT_RECALL_THRESHOLD = 0.9
RESPONSE_TIME_THRESHOLD = 3.0

LOG_FMT = "%(asctime)s | %(levelname)-7s | %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"
WO_ID = "人工智能NLP-RAG-Graph RAG 优化任务"
