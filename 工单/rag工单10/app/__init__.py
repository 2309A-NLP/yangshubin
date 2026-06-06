"""
模块功能: app 包初始化文件
将 app 目录标记为 Python 包，暴露全部核心模块的引用
工单编号: 人工智能NLP-RAG-金融问答系统部署
"""

# 导入配置模块，整个包共享配置实例
from app.config import config

# 控制 from app import * 的行为，只导出主要模块
__all__ = [
    "config",            # 全局配置
    "app",               # Flask 主入口
    "llm_client",        # MiMo LLM 客户端
    "embedding",         # BGE-M3 向量化
    "document_loader",   # PDF 文档加载
    "text_splitter",     # 文本分块
    "vectorstore",       # Milvus 向量数据库
    "graph_builder",     # 知识图谱
    "rag_engine",        # RAG 问答引擎
    "routes",            # API 路由
]

# 包元信息
__version__ = "1.0.0"
__package_name__ = "rag-ticket-app"
