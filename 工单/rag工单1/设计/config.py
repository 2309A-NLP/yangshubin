# -*- coding: utf-8 -*-
"""
配置文件 —— 集中管理所有全局参数
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
"""

import os
from pathlib import Path

# ============================================================
# 项目路径配置
# ============================================================
# 当前项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent

# PDF文件路径 — 招股说明书1（无水印版）
PDF_PATH = str(PROJECT_ROOT / "招股说明书1-无水印.pdf")

# ============================================================
# BGE-M3 嵌入模型配置
# ============================================================
# BGE-M3 模型本地路径（自动适配 Windows / WSL 环境）
_BGE_M3_CANDIDATES = [
    r"C:\Users\31326\Desktop\bge-m3",          # Windows 原生路径
    r"/mnt/c/Users/31326/Desktop/bge-m3",      # WSL 挂载路径
    str(Path.home() / "Desktop" / "bge-m3"),    # Linux Desktop
    str(Path.home() / "bge-m3"),                # home 目录下
]
BGE_M3_PATH = ""
for _p in _BGE_M3_CANDIDATES:
    if os.path.isdir(_p):
        BGE_M3_PATH = _p
        break
if not BGE_M3_PATH:
    # 保底：直接用第一个
    BGE_M3_PATH = _BGE_M3_CANDIDATES[0]
    print(f"[配置] 未找到 BGE-M3 模型目录，默认使用: {BGE_M3_PATH}")
# BGE-M3 输出的向量维度
BGE_M3_DIM = 1024
# 计算设备：cuda 或 cpu
DEVICE = "cuda"

# ============================================================
# DeepSeek LLM 配置（OpenAI 兼容接口）
# ============================================================
DEEPSEEK_API_KEY = "sk-2ea5dccdfeb04c32a5a1fb3fdcd0e8fa"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"
# LLM 请求超时时间（秒）
LLM_TIMEOUT = 60

# ============================================================
# Milvus 向量数据库配置
# ============================================================
MILVUS_URI = "http://localhost:19530"
# 集合名称：用于存储PDF文档切块+向量
MILVUS_COLLECTION = "rag_pdf_qa"

# ============================================================
# 文本切分参数
# ============================================================
# 每个文本块的最大字符数
CHUNK_SIZE = 512
# 相邻文本块之间的重叠字符数（保持上下文连贯）
CHUNK_OVERLAP = 64

# ============================================================
# 检索参数
# ============================================================
# 每次检索返回的候选文本块数量
TOP_K = 5
