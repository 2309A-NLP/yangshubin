"""
模块功能: 知识图谱构建模块（MiMo 实体提取 + NetworkX 图）
使用 MiMo API 从文档中提取金融实体关系，构建 NetworkX 图
支持 entity_to_chunks 映射和子图扩展，用于 GraphRAG 混合检索
工单编号: 人工智能NLP-RAG-金融问答系统部署
"""

import json, logging
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import networkx as nx
from app.config import config

logger = logging.getLogger("graph_builder")
_extraction_cache: Dict[str, dict] = {}


def extract_entities_relations(text: str, chunk_index: int = 0) -> dict:
    """使用 MiMo API 从文本块中提取金融实体和关系"""
    cache_key = text[:100]
    if cache_key in _extraction_cache:
        return _extraction_cache[cache_key]
    prompt = f"""从以下金融年报文本中提取实体和关系。
实体类型: 公司, 人物, 产品, 指标, 事件, 时间
关系类型: 属于, 担任, 发布, 增长, 下降, 投资, 合作, 控股
返回JSON: {{"entities":[{{"name":"实体名","type":"公司|人物|产品|指标|事件|时间"}}],
  "relations":[{{"source":"实体名","relation":"关系类型","target":"实体名"}}]}}
文本:{text[:1500]}"""
    try:
        from app.llm_client import MiMoClient
        result = MiMoClient().generate(prompt)
        if '{' in result:
            result = result[result.index('{'):result.rindex('}') + 1]
            data = json.loads(result)
        else:
            data = {"entities": [], "relations": []}
    except Exception as e:
        logger.warning(f"实体提取失败: {e}")
        data = {"entities": [], "relations": []}
    for e in data.get("entities", []):
        e["chunk_index"] = chunk_index
    for r in data.get("relations", []):
        r["chunk_index"] = chunk_index
    _extraction_cache[cache_key] = data
    return data


class KnowledgeGraph:
    """基于 NetworkX 的金融知识图谱构建器"""

    def __init__(self):
        self.graph: nx.Graph = nx.Graph()
        self.entity_to_chunks: Dict[str, set] = defaultdict(set)
        self._built: bool = False

    def build_from_chunks(self, chunks: list, max_chunks: int = 15) -> dict:
        """从文本块列表中提取实体关系并构建知识图谱"""
        self.graph = nx.Graph()
        logger.info(f"构建图谱(处理{min(len(chunks),max_chunks)}/共{len(chunks)}块)...")
        for i, chunk in enumerate(chunks[:max_chunks]):
            text = chunk.get("content", chunk.get("text", ""))
            data = extract_entities_relations(text, chunk["index"])
            for ent in data.get("entities", []):
                name = ent["name"].strip()
                if not name:
                    continue
                self.graph.add_node(name, type=ent.get("type","未知"),
                                    source_pdf=chunk.get("source_pdf",""))
                self.entity_to_chunks[name].add(chunk["index"])
            for rel in data.get("relations", []):
                src, tgt = rel.get("source","").strip(), rel.get("target","").strip()
                if src and tgt:
                    self.graph.add_edge(src, tgt, relation=rel.get("relation","相关"))
        self._built = True
        return {"nodes": self.graph.number_of_nodes(), "edges": self.graph.number_of_edges()}

    def build_from_documents(self, data_dir: str) -> Dict[str, int]:
        """从 PDF 目录加载文档并构建知识图谱"""
        from app.document_loader import load_documents
        from app.text_splitter import split_text
        docs = load_documents(data_dir)
        chunks = []
        for doc in docs:
            text = doc.get("text", "")
            if text:
                for idx, ch in enumerate(split_text(text)):
                    chunks.append({"content": ch, "index": len(chunks)+idx,
                                   "source_pdf": doc.get("filename","unknown")})
        return self.build_from_chunks(chunks, max_chunks=30)

    def get_related_entities(self, entity: str, max_hops: int = 2) -> list:
        """BFS 获取实体关联列表"""
        if not self._built or entity not in self.graph:
            return []
        visited, queue = {entity}, [(entity, 0)]
        while queue:
            cur, depth = queue.pop(0)
            if depth >= max_hops:
                continue
            for n in self.graph.neighbors(cur):
                if n not in visited:
                    visited.add(n)
                    queue.append((n, depth+1))
        return list(visited)

    def get_subgraph_context(self, query: str, max_nodes: int = 20) -> str:
        """获取查询相关的子图文本"""
        matched = [n for n in self.graph.nodes() if query in str(n)]
        if not matched:
            return ""
        lines = ["【知识图谱上下文】"]
        for node in matched[:max_nodes]:
            for nb in self.graph.neighbors(node):
                rel = self.graph.get_edge_data(node, nb).get("relation", "相关")
                lines.append(f"  - {node} --[{rel}]--> {nb}")
        return "\n".join(lines)

    def get_graph_data(self) -> dict:
        """返回前端可视化的图谱数据"""
        if not self._built:
            return {"nodes": [], "edges": [], "entity_to_chunks": {}}
        nodes = [{"name": n, "type": d.get("type","未知"),
                  "source_pdf": d.get("source_pdf","")}
                 for n, d in self.graph.nodes(data=True)]
        edges = [{"source": u, "target": v, "relation": d.get("relation","相关")}
                 for u, v, d in self.graph.edges(data=True)]
        return {"nodes": nodes, "edges": edges,
                "entity_to_chunks": {k: list(v) for k,v in self.entity_to_chunks.items()}}

    def save(self, output_dir: str = None):
        """保存图谱数据到 JSON 文件"""
        if not self._built:
            return
        fpath = (Path(output_dir or config.OUTPUT_DIR) / "knowledge_graph.json")
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(self.get_graph_data(), f, ensure_ascii=False, indent=2)

    def load(self, file_path: str = None):
        """从 JSON 文件加载图谱数据"""
        from pathlib import Path
        fpath = file_path or str(Path(config.OUTPUT_DIR) / "knowledge_graph.json")
        if not Path(fpath).exists():
            return False
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                gd = json.load(f)
            self.graph = nx.Graph()
            for n in gd.get("nodes", []):
                self.graph.add_node(n["name"], type=n.get("type",""),
                                    source_pdf=n.get("source_pdf",""))
            for e in gd.get("edges", []):
                self.graph.add_edge(e["source"], e["target"],
                                    relation=e.get("relation","相关"))
            self.entity_to_chunks = defaultdict(
                set, {k: set(v) for k,v in gd.get("entity_to_chunks",{}).items()})
            self._built = True
            return True
        except Exception as e:
            logger.error(f"图谱加载失败: {e}")
            return False


_graph_instance: KnowledgeGraph = None


def get_graph() -> KnowledgeGraph:
    """获取全局知识图谱单例"""
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = KnowledgeGraph()
    return _graph_instance


def build_knowledge_graph(data_dir: str) -> Dict[str, int]:
    """便捷函数: 从数据目录构建知识图谱"""
    return get_graph().build_from_documents(data_dir)
