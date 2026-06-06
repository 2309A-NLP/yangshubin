"""\ngraph_retriever.py - RAG工单9 混合检索（向量+图谱）模块\n工单编号: 人工智能NLP-RAG-Graph RAG 优化任务
需求: 检索层面优化 — 向量检索+图谱实体扩展+公司感知重排，支持优化前后对比
功能: MiMo实体提取→图谱BFS扩展→向量检索→RRF融合→返回TOP_K结果
"""
import logging, re
from collections import OrderedDict
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '设计'))
from config import TOP_K, GRAPH_EXPAND_K, FINAL_TOP_K, MIMO_API_KEY, MIMO_BASE_URL, MIMO_MODEL, LOG_FMT, LOG_DATEFMT
from milvus_handler import MilvusManager

logging.basicConfig(format=LOG_FMT, datefmt=LOG_DATEFMT, level=logging.INFO)
logger = logging.getLogger("graph_retriever")
_entity_cache = {}


def extract_query_entities(query_text):
    """使用MiMo从查询中提取金融实体"""
    key = f"entities|{query_text[:80]}"
    if key in _entity_cache:
        return _entity_cache[key]
    prompt = f"从以下金融问题中提取公司/人物名称，只返回名称列表JSON。\n问题: {query_text}\nJSON: {{\"entities\":[\"平安银行\",\"招商银行\"]}}"
    try:
        from openai import OpenAI
        c = OpenAI(api_key=MIMO_API_KEY, base_url=MIMO_BASE_URL)
        r = c.chat.completions.create(model=MIMO_MODEL, timeout=15,
                                       messages=[{"role": "user", "content": prompt}],
                                       temperature=0, max_tokens=200)
        text = (r.choices[0].message.content or "").strip()
        if not text:
            extra = r.choices[0].message.model_extra or {}
            text = extra.get("reasoning_content", "").strip()
        if '{' in text:
            import json as j
            data = j.loads(text[text.index('{'):text.rindex('}')+1])
            entities = data.get("entities", [])
            _entity_cache[key] = entities
            return entities
    except Exception as e:
        logger.warning(f"实体提取失败: {e}")
    _entity_cache[key] = []
    return []


def fuzzy_match_entity(query_entity, graph_nodes, threshold=0.6):
    """模糊匹配实体名称"""
    matches = []
    for node in graph_nodes:
        if query_entity in node or node in query_entity:
            matches.append(node)
            continue
        common = set(query_entity) & set(node)
        if len(common) / max(len(set(query_entity)), 1) > threshold:
            matches.append(node)
    return matches


class GraphRetriever:
    """GraphRAG混合检索器：向量检索 + 图谱实体扩展 + 公司感知重排 + 融合排序"""
    def __init__(self, graph_builder=None):
        self.milvus = MilvusManager()
        self.graph_builder = graph_builder
        self.chunk_store = {}

    def set_chunks(self, chunks):
        self.chunk_store = {c["index"]: c for c in chunks} if chunks else {}

    def retrieve(self, query_vector, query_text, use_graph=True):
        """混合检索：向量召回→公司感知重排→图谱扩展→融合排序"""
        hits = self.milvus.search(query_vector, top_k=TOP_K)
        result_map = OrderedDict()
        for h in hits:
            result_map[h.get("chunk_index", id(h))] = h
        # 公司感知重排：检测问题中公司名，提升相关chunk分数
        companies = re.findall(r'(平安|招商|中信|国泰|君安|邮储|太平洋|人寿)[\u4e00-\u9fff]{0,4}(?:银行|证券|保险|集团)', query_text)
        target = companies[0] if companies else None
        if target:
            filtered = OrderedDict()
            for idx, h in result_map.items():
                if target in h.get("source_pdf", ""):
                    h["score"] *= 1.5
                    filtered[idx] = h
            others = [(idx, h) for idx, h in result_map.items() if idx not in filtered]
            for idx, h in others[:3]:
                h["score"] *= 0.5
                filtered[idx] = h
            result_map = filtered
        # 图谱扩展
        if use_graph and self.graph_builder and self.graph_builder._built:
            query_entities = extract_query_entities(query_text)
            all_nodes = list(self.graph_builder.graph.nodes())
            expanded = set()
            for qe in query_entities:
                for m in fuzzy_match_entity(qe, all_nodes):
                    expanded.update(self.graph_builder.get_related_entities(m, max_hops=2))
            extra_indices = set()
            for entity in expanded:
                extra_indices.update(self.graph_builder.entity_to_chunks.get(entity, set()))
            existing = {h.get("chunk_index", -1) for h in hits if h.get("chunk_index") is not None}
            for ci in extra_indices:
                if ci not in existing and ci not in result_map and ci in self.chunk_store:
                    ch = self.chunk_store[ci]
                    result_map[ci] = {"score": 0.7, "content": ch["content"],
                                      "source_pdf": ch.get("source_pdf", ""),
                                      "page_num": ch.get("page_num", 0), "chunk_index": ci}
        merged = sorted(result_map.values(), key=lambda x: x.get("score", 0), reverse=True)
        logger.info(f"混合检索: {len(hits)}条 → {FINAL_TOP_K}条")
        return merged[:FINAL_TOP_K]

    def retrieve_vector_only(self, query_vector):
        """纯向量检索（优化前对比基线）"""
        return self.milvus.search(query_vector, top_k=FINAL_TOP_K)

    def close(self):
        self.milvus.close()


if __name__ == "__main__":
    from embedder import BgeM3Embedder
    e = BgeM3Embedder()
    r = e.encode_query(["平安银行盈利情况"])
    v = r["dense_vecs"][0].tolist()
    ret = GraphRetriever()
    print(f"检索: {len(ret.retrieve(v, '平安银行盈利情况', False))}条")
    ret.close()
