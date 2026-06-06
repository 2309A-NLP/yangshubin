"""
模块功能: RAG 问答引擎模块
实现 GraphRAG 核心问答流程:
用户问题 → 向量检索 (Milvus) → 知识图谱上下文 → LLM 生成 (MiMo) → 返回答案
整合了向量检索和图谱增强两种检索方式的混合检索策略
工单编号: 人工智能NLP-RAG-金融问答系统部署
"""

import logging              # 日志记录模块
from typing import List, Dict, Optional  # 类型提示
from app.config import config  # 全局配置对象

# 获取当前模块的日志记录器
logger = logging.getLogger("rag_engine")


class RAGEngine:
    """RAG 问答引擎类，整合向量检索、图谱增强和 LLM 生成的完整问答流程"""

    def __init__(self):
        """初始化引擎，各组件延迟加载（首次使用时才初始化）"""
        self._milvus_client = None  # Milvus 客户端（延迟加载）
        self._graph = None          # 知识图谱（延迟加载）
        self._llm_client = None     # MiMo 客户端（延迟加载）

    def _get_milvus(self):
        """延迟获取 Milvus 客户端实例，避免启动时就连接数据库"""
        if self._milvus_client is None:
            from app.vectorstore import MilvusClient
            self._milvus_client = MilvusClient()
            self._milvus_client.connect()
        return self._milvus_client

    def _get_graph(self):
        """延迟获取知识图谱单例实例"""
        if self._graph is None:
            from app.graph_builder import get_graph
            self._graph = get_graph()
        return self._graph

    def _get_llm(self):
        """延迟获取 MiMo LLM 客户端实例"""
        if self._llm_client is None:
            from app.llm_client import MiMoClient
            self._llm_client = MiMoClient()
        return self._llm_client

    def retrieve_context(self, query: str, top_k: int = None) -> str:
        """执行混合检索过程，收集向量检索和图谱检索的上下文

        检索策略包含两步:
        1. 向量检索: 将问题转为向量，在 Milvus 中搜索相似文档块
        2. 图谱检索: 提取问题关键词，在知识图谱中查找相关实体关系

        Args:
            query: 用户输入的问题
            top_k: 向量检索返回的文档块数，默认从配置读取 (8)

        Returns:
            融合后的上下文字符串，包含文档引用和图谱关系
        """
        if top_k is None:
            top_k = config.TOP_K
        # 用于存储所有上下文片段
        context_parts: List[str] = ["以下是参考文档内容:\n"]
        # 第一步: 向量检索 - 从 Milvus 获取语义相似的文档块
        try:
            milvus = self._get_milvus()
            from app.embedding import embed_query
            # 将用户问题编码为查询向量
            query_vec = embed_query(query)
            if query_vec is not None:
                # 在 Milvus 中执行相似度搜索
                results = milvus.search(
                    query_vector=query_vec.tolist(),
                    top_k=top_k,
                )
                # 将搜索结果格式化为上下文文本
                for idx, hit in enumerate(results):
                    context_parts.append(
                        f"[文档{idx + 1}] (来源: {hit['filename']}, 相关度: {hit['score']})\n"
                        f"{hit['text']}\n"
                    )
        except Exception as e:
            logger.warning(f"向量检索失败: {e}")
        # 第二步: 知识图谱上下文增强
        try:
            graph = self._get_graph()
            graph_context: str = graph.get_subgraph_context(query)
            if graph_context:
                context_parts.append("\n" + graph_context + "\n")
        except Exception as e:
            logger.warning(f"图谱检索失败: {e}")
        # 合并所有上下文片段
        full_context: str = "\n".join(context_parts)
        logger.info(f"混合检索完成: 上下文共 {len(full_context)} 字符")
        return full_context

    def generate_answer(self, query: str, context: str) -> str:
        """调用 MiMo LLM 根据检索到的上下文生成答案

        Args:
            query: 用户提出的问题
            context: 检索到的参考文档和图谱上下文

        Returns:
            MiMo 模型生成的答案文本
        """
        # 构建系统级提示词，引导模型基于参考内容回答
        system_prompt: str = (
            "你是一个专业的金融问答助手。请基于以下参考文档内容回答问题。\n"
            "如果参考文档中没有足够信息，请如实说明无法回答。\n"
            "回答时请引用参考文档中的具体内容作为依据。\n"
            "使用中文回答，保持专业、准确、简洁。"
        )
        # 组装完整的提示词（system + 上下文 + 问题）
        prompt: str = f"{system_prompt}\n\n参考文档:\n{context}\n\n问题: {query}\n\n答案:"
        # 调用 MiMo API 生成回答
        llm = self._get_llm()
        answer: str = llm.generate(prompt)
        return answer

    def ask(self, query: str) -> Dict:
        """完整的问答流程: 检索 + 生成

        Args:
            query: 用户输入的问题文本

        Returns:
            包含 question, answer, context, sources 的字典
        """
        logger.info(f"收到用户问题: {query}")
        # 步骤1: 执行混合检索获取上下文
        context: str = self.retrieve_context(query)
        # 步骤2: 基于上下文调用 LLM 生成答案
        answer: str = self.generate_answer(query, context)
        # 步骤3: 整理并返回结构化结果
        result: Dict = {
            "question": query,
            "answer": answer,
            "context": context[:500] + "..." if len(context) > 500 else context,
            "sources": self._extract_sources(context),
        }
        logger.info(f"问答完成: 答案 {len(answer)} 字符")
        return result

    def _extract_sources(self, context: str) -> List[str]:
        """从上下文中提取来源文件名列表

        Args:
            context: 包含文档引用的上下文文本

        Returns:
            去重后的来源文件名列表
        """
        sources: List[str] = []
        for line in context.split("\n"):
            if line.startswith("[文档") and "来源:" in line:
                start = line.find("来源:") + 3
                end = line.find(",", start)
                if end > start:
                    source = line[start:end].strip()
                    if source not in sources:
                        sources.append(source)
        return sources


# 全局 RAG 引擎单例
_engine_instance: RAGEngine = None


def get_engine() -> RAGEngine:
    """获取 RAG 引擎全局单例

    Returns:
        RAGEngine 实例（线程安全、延迟初始化）
    """
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = RAGEngine()
    return _engine_instance
