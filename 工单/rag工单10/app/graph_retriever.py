"""
模块功能: 混合检索器（向量检索 + 图谱扩展）
集成 Milvus 向量搜索和 NetworkX 知识图谱扩展
使用 MiMo API 提取查询实体，支持纯向量和 GraphRAG 两种模式
工单编号: 人工智能NLP-RAG-金融问答系统部署
"""

import logging, json
from collections import defaultdict
from app.config import config

logger = logging.getLogger("graph_retriever")
_query_cache: dict = {}


def extract_query_entities(query: str) -> list:
    """使用 MiMo API 从查询中提取金融实体"""
    if query in _query_cache:
        return _query_cache[query]
    prompt = f"""从以下金融问题中提取核心实体名(公司/人名/产品/指标)。
返回JSON: {{"entities": ["实体1", "实体2"]}}
问题: {query}"""
    try:
        from app.llm_client import MiMoClient
        result = MiMoClient().generate(prompt)
        if '{' in result:
            result = result[result.index('{'):result.rindex('}')+1]
            entities = json.loads(result).get("entities", [])
        else:
            entities = []
    except Exception as e:
        logger.warning(f"实体提取失败: {e}")
        entities = []
    _query_cache[query] = entities
    return entities


class GraphRetriever:
    """向量+图谱混合检索器"""

    def __init__(self, graph_builder=None):
        self.graph_builder = graph_builder
        self.chunk_store = []
        self._milvus_mgr = None

    def set_chunks(self, chunks: list):
        """设置 chunk 存储"""
        self.chunk_store = chunks or []

    def _get_milvus(self):
        """获取 Milvus 客户端"""
        if self._milvus_mgr is None:
            from app.vectorstore import MilvusClient
            self._milvus_mgr = MilvusClient()
            self._milvus_mgr.connect()
        return self._milvus_mgr

    def retrieve_vector_only(self, qv: list, top_k: int = None) -> list:
        """纯向量检索"""
        mgr = self._get_milvus()
        return [{"text": r.get("text",""), "score": r.get("score",0),
                 "source_pdf": r.get("filename",""), "page_num": 0}
                for r in mgr.search(qv, top_k=top_k or config.TOP_K)]

    def _get_chunks_by_entities(self, entities: list, expand_k: int = 5) -> list:
        """通过实体获取图谱扩展的 chunks"""
        if not self.graph_builder or not self.graph_builder._built:
            return []
        indices = set()
        for ent in entities:
            if ent in self.graph_builder.entity_to_chunks:
                indices.update(self.graph_builder.entity_to_chunks[ent])
            for rel in self.graph_builder.get_related_entities(ent, 2)[1:]:
                if rel in self.graph_builder.entity_to_chunks:
                    indices.update(self.graph_builder.entity_to_chunks[rel])
        indices = list(indices)[:expand_k*3]
        smap = {c["index"]: c for c in self.chunk_store}
        return [smap[i] for i in indices if i in smap]

    def retrieve(self, qv: list, qtext: str, use_graph: bool = True,
                 top_k: int = None, final_k: int = None) -> list:
        """混合检索：向量检索 + 图谱扩展 + 融合排序"""
        top_k = top_k or config.TOP_K
        final_k = final_k or config.TOP_K
        retrieved, seen = [], set()
        # 1. 向量检索
        for hit in self.retrieve_vector_only(qv, top_k):
            if hit["text"] not in seen:
                retrieved.append({**hit, "source": "vector"})
                seen.add(hit["text"])
        # 2. 图谱扩展
        if use_graph and self.graph_builder and self.graph_builder._built:
            entities = extract_query_entities(qtext)
            if entities:
                logger.info(f"图谱扩展: {entities}")
                for ch in self._get_chunks_by_entities(entities):
                    txt = ch.get("content", ch.get("text", ""))
                    if txt not in seen:
                        retrieved.append({"text": txt, "score": 0.5,
                            "source_pdf": ch.get("source_pdf",""),
                            "page_num": ch.get("page_num",0), "source": "graph"})
                        seen.add(txt)
        # 3. 排序返回
        retrieved.sort(key=lambda x: x.get("score",0), reverse=True)
        return retrieved[:final_k]

    def close(self):
        """释放资源"""
        if self._milvus_mgr:
            try:
                self._milvus_mgr.close()
            except Exception:
                pass
