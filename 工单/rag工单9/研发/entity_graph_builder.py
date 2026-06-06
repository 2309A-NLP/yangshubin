"""
entity_graph_builder.py - RAG工单9 实体提取与知识图谱构建模块
工单编号: 人工智能NLP-RAG-Graph RAG 优化任务
需求: prompt构建层面优化 — 使用MiMo从CCF年报提取金融实体和关系，构建NetworkX图谱
功能: MiMo API实体关系提取 → NetworkX知识图谱 → BFS关联扩展 → 图谱保存
"""
import logging, json, time, re
from collections import defaultdict
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '设计'))
from config import MIMO_API_KEY, MIMO_BASE_URL, OUTPUT_DIR, LOG_FMT, LOG_DATEFMT

# 实体提取用mimo-v2.5（文本模型，直接返回结果）
from config import MIMO_MODEL as ENTITY_MODEL

logging.basicConfig(format=LOG_FMT, datefmt=LOG_DATEFMT, level=logging.INFO)
logger = logging.getLogger("entity_graph_builder")
_extraction_cache = {}


def _parse_json_robust(text):
    """从LLM输出中健壮地提取JSON对象，处理多种格式错误"""
    if not text:
        return None
    # 策略1: 直接解析
    try:
        return json.loads(text)
    except:
        pass
    # 策略2: 从后往前找最后一个完整JSON对象
    if '{' in text and '}' in text:
        # 先试最后一个{到最后一个}
        last_open = text.rfind('{')
        last_close = text.rfind('}')
        if last_open < last_close:
            sub = text[last_open:last_close + 1]
            try:
                return json.loads(sub)
            except:
                pass
        # 再试第一个{到最后一个}
        sub = text[text.index('{'):text.rindex('}') + 1]
        try:
            return json.loads(sub)
        except:
            pass
    # 策略3: 提取所有可能的JSON对象，合并entities和relations
    all_ents, all_rels = [], []
    for match in re.finditer(r'\{[^{}]*"entities"[^{}]*\[[^\]]*\][^{}]*"relations"[^{}]*\[[^\]]*\][^{}]*\}', text, re.DOTALL):
        try:
            obj = json.loads(match.group())
            all_ents.extend(obj.get("entities", []))
            all_rels.extend(obj.get("relations", []))
        except:
            pass
    if all_ents or all_rels:
        return {"entities": all_ents, "relations": all_rels}
    # 策略4: 分别提取entities和relations数组
    ent_match = re.search(r'"entities"\s*:\s*(\[.*?\])', text, re.DOTALL)
    rel_match = re.search(r'"relations"\s*:\s*(\[.*?\])', text, re.DOTALL)
    if ent_match:
        try:
            all_ents = json.loads(ent_match.group(1))
        except:
            pass
    if rel_match:
        try:
            all_rels = json.loads(rel_match.group(1))
        except:
            pass
    if all_ents or all_rels:
        return {"entities": all_ents, "relations": all_rels}
    return None


def extract_entities_relations(text, chunk_index=0):
    """使用MiMo从文本块中提取实体和关系（带缓存）"""
    cache_key = text[:100]
    if cache_key in _extraction_cache:
        return _extraction_cache[cache_key]
    # 精简prompt，降低模型自由发挥空间
    prompt = (
        "从以下金融年报文本中提取实体和关系，严格只返回一个JSON对象，不要任何其他文字。\n"
        "实体类型限: 公司, 人物, 产品, 指标, 事件, 时间\n"
        "关系类型限: 属于, 担任, 发布, 增长, 下降, 投资, 合作, 控股\n"
        "输出格式:\n"
        '{"entities":[{"name":"实体名","type":"类型"}],'
        '"relations":[{"source":"实体名","relation":"关系","target":"实体名"}]}\n'
        f"文本:\n{text[:1500]}"
    )
    try:
        from openai import OpenAI
        client = OpenAI(api_key=MIMO_API_KEY, base_url=MIMO_BASE_URL)
        resp = client.chat.completions.create(
            model=ENTITY_MODEL, timeout=45,
            messages=[
                {"role": "system", "content": "你是实体提取器。只返回JSON，不要任何解释或推理过程。"},
                {"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=1000)
        content = (resp.choices[0].message.content or "").strip()
        extra = resp.choices[0].message.model_extra or {}
        reasoning = extra.get("reasoning_content", "").strip()
        # 优先用content中的JSON，其次用reasoning末尾的JSON
        data = _parse_json_robust(content)
        if not data:
            data = _parse_json_robust(reasoning)
        if data and ("entities" in data or "relations" in data):
            # 过滤无效实体
            valid_ents = [e for e in data.get("entities", [])
                          if isinstance(e, dict) and e.get("name", "").strip()]
            valid_rels = [r for r in data.get("relations", [])
                          if isinstance(r, dict) and r.get("source", "").strip() and r.get("target", "").strip()]
            data = {"entities": valid_ents, "relations": valid_rels}
        else:
            logger.warning(f"解析失败(chunk{chunk_index}), 原文: {result_text[:200]}")
            data = {"entities": [], "relations": []}
    except Exception as e:
        logger.warning(f"提取失败(chunk{chunk_index}): {e}")
        data = {"entities": [], "relations": []}
    for e in data.get("entities", []):
        e["chunk_index"] = chunk_index
    for r in data.get("relations", []):
        r["chunk_index"] = chunk_index
    _extraction_cache[cache_key] = data
    return data


class GraphBuilder:
    """基于NetworkX的金融知识图谱构建器，支持MiMo实体提取"""
    def __init__(self):
        self.graph = None
        self.entity_to_chunks = defaultdict(set)
        self._built = False

    def build_from_chunks(self, chunks, max_chunks=30):
        """从文本块中提取实体关系并构建知识图谱"""
        import networkx as nx
        self.graph = nx.Graph()
        n = min(len(chunks), max_chunks)
        logger.info(f"构建图谱(处理{n}块)...")
        success, fail = 0, 0
        for i, chunk in enumerate(chunks[:max_chunks]):
            text = chunk["content"]
            data = extract_entities_relations(text, chunk["index"])
            ents = data.get("entities", [])
            rels = data.get("relations", [])
            if ents:
                success += 1
            else:
                fail += 1
            for ent in ents:
                name = ent["name"].strip()
                if not name:
                    continue
                self.graph.add_node(name, type=ent.get("type", "未知"), source_pdf=chunk.get("source_pdf", ""))
                self.entity_to_chunks[name].add(chunk["index"])
            for rel in rels:
                src, tgt = rel.get("source", "").strip(), rel.get("target", "").strip()
                if src and tgt:
                    self.graph.add_edge(src, tgt, relation=rel.get("relation", "相关"))
        self._built = True
        logger.info(f"图谱完成! {self.graph.number_of_nodes()}节点, {self.graph.number_of_edges()}条边 (成功{success}/失败{fail})")

    def save(self):
        """保存图结构和实体映射到文件"""
        if not self._built:
            return
        nodes = [{"name": n, "type": d.get("type", "未知"), "source_pdf": d.get("source_pdf", "")}
                 for n, d in self.graph.nodes(data=True)]
        edges = [{"source": u, "target": v, "relation": d.get("relation", "相关")}
                 for u, v, d in self.graph.edges(data=True)]
        data = {"nodes": nodes, "edges": edges,
                "entity_to_chunks": {k: list(v) for k, v in self.entity_to_chunks.items()}}
        with open(str(OUTPUT_DIR / "knowledge_graph.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"图谱已保存: {len(nodes)}节点, {len(edges)}边")

    def get_related_entities(self, entity_name, max_hops=2):
        """BFS获取实体关联节点"""
        if not self._built or entity_name not in self.graph:
            return []
        visited, queue = {entity_name}, [(entity_name, 0)]
        while queue:
            cur, depth = queue.pop(0)
            if depth >= max_hops:
                continue
            for nb in self.graph.neighbors(cur):
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, depth + 1))
        return list(visited)


if __name__ == "__main__":
    test = "平安银行2019年实现营业收入1379亿元，董事长谢永林出席发布会。"
    print(json.dumps(extract_entities_relations(test), ensure_ascii=False, indent=2))
