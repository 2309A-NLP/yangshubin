# -*- coding: utf-8 -*-
"""
配置文件 —— 集中管理全局参数（工单2优化版）
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
"""

import os
from pathlib import Path

# ============================================================
# 项目路径
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
PDF_PATH = str(PROJECT_ROOT / "招股说明书1.pdf")
OUTPUT_DIR = str(PROJECT_ROOT / "output")

# ============================================================
# BGE-M3 嵌入模型（自动适配多环境）
# ============================================================
_BGE_CANDIDATES = [
    r"C:\Users\31326\Desktop\bge-m3",
    r"/mnt/c/Users/31326/Desktop/bge-m3",
    str(Path.home() / "Desktop" / "bge-m3"),
]
BGE_M3_PATH = next((p for p in _BGE_CANDIDATES if os.path.isdir(p)), _BGE_CANDIDATES[0])
BGE_M3_DIM = 1024

# ============================================================
# BGE-Reranker 重排序模型
# ============================================================
_RERANK_CANDIDATES = [
    r"C:\Users\31326\Desktop\bge-reranker-base",
    r"/mnt/c/Users/31326/Desktop/bge-reranker-base",
    str(Path.home() / "Desktop" / "bge-reranker-base"),
]
BGE_RERANKER_PATH = next((p for p in _RERANK_CANDIDATES if os.path.isdir(p)), None)

# ============================================================
# DeepSeek LLM
# ============================================================
DEEPSEEK_API_KEY = "sk-2ea5dccdfeb04c32a5a1fb3fdcd0e8fa"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"
LLM_TIMEOUT = 60

# ============================================================
# Milvus
# ============================================================
MILVUS_URI = "http://localhost:19530"
MILVUS_COLLECTION = "rag_pdf_qa_v2"  # 新集合，跟工单1区分

# ============================================================
# 切分与检索参数
# ============================================================
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K_RETRIEVAL = 10    # 检索阶段候选数
TOP_K_RERANK = 5        # 重排序后返回数
TOP_K_FINAL = 5         # 最终送给LLM的片段数

# ============================================================
# 性能开关
# ============================================================
USE_RERANKER = False    # 重排序开关（True=更准但+2~3s，False=更快）

# ============================================================
# 设备
# ============================================================
DEVICE = "cuda"
