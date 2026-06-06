"""
模块功能: 完整流水线协调器
一键执行 RAG 评估流程: PDF解析→分块→向量化→入库→图谱构建
→纯向量问答→GraphRAG问答→评估对比→报告生成
工单编号: 人工智能NLP-RAG-金融问答系统部署
"""

import json, sys, time, logging
from pathlib import Path
from app.config import config
from app.llm_client import MiMoClient
from app.embedding import embed_query
from app.document_loader import load_documents, load_pdf
from app.text_splitter import split_text
from app.vectorstore import store_embeddings
from app.graph_builder import get_graph
from app.graph_retriever import GraphRetriever
from app.evaluator import RagasEvaluator

logger = logging.getLogger("pipeline")
_step = [1]


def step(msg, fn, *args, **kw):
    """带计时和日志的步骤执行器"""
    logger.info(f"📌 步{_step[0]}: {msg}")
    _step[0] += 1
    t0 = time.time()
    r = fn(*args, **kw)
    logger.info(f"  ✅ {time.time()-t0:.1f}s")
    return r


def _milvus_has_data() -> bool:
    """检查 Milvus 是否有数据"""
    try:
        from pymilvus import utility, Collection
        from app.vectorstore import MilvusClient
        c = MilvusClient()
        if not c.connect():
            return False
        if utility.has_collection(config.MILVUS_COLLECTION):
            col = Collection(config.MILVUS_COLLECTION)
            col.load()
            cnt = col.num_entities
            col.release()
            c.close()
            return cnt > 0
        c.close()
        return False
    except Exception:
        return False


def pipeline_pcv():
    """①~④: 解析→分块→向量化→入库"""
    docs = load_documents(config.DATA_DIR)
    if not docs:
        logger.error("未找到 PDF"); sys.exit(1)
    all_chunks = []
    idx = 0
    for doc in docs:
        for ch in split_text(doc.get("text","")):
            all_chunks.append({"content":ch,"index":idx,
                "source_pdf":doc.get("filename","unknown"),"char_count":len(ch)})
            idx += 1
    with open(Path(config.OUTPUT_DIR)/"chunks_data.json","w",encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    cnt = store_embeddings(config.DATA_DIR)
    logger.info(f"入库: {cnt}条")
    return all_chunks


def pipeline_graph(chunks):
    """⑤ 构建知识图谱"""
    b = get_graph()
    b.build_from_chunks(chunks, max_chunks=30)
    b.save()
    return b


def run_questions(questions, gb, use_graph, chunks=None):
    """⑥⑦ 执行问答"""
    rt = GraphRetriever(gb if gb and gb._built else None)
    if chunks:
        rt.set_chunks(chunks)
    llm = MiMoClient()
    mode = "GraphRAG" if use_graph else "Vector"
    logger.info(f"🔍 {mode}({len(questions)}题)...")
    results = []
    for i, q in enumerate(questions):
        t0 = time.time()
        qv = embed_query(q["question"])
        if qv is None:
            continue
        hits = rt.retrieve(qv.tolist(), q["question"], use_graph=use_graph) if use_graph \
            else rt.retrieve_vector_only(qv.tolist())
        ctx, srcs = "", set()
        for i2, ch in enumerate(hits):
            ctx += f"[{i2+1}] {ch.get('text',ch.get('content',''))[:800]}\n\n"
            s = ch.get("source_pdf","")
            if s: srcs.add(s)
        ans = llm.generate(f"你是一位专业的金融年报分析师。\n参考资料:\n{ctx}\n问题: {q['question']}\n仅基于参考资料回答。\n回答:")
        rt2 = round(time.time()-t0, 2)
        results.append({"question":q["question"],"reference_answer":q.get("reference_answer",""),
            "retrieved_chunks":hits[:3],"response_time":rt2,
            "qa_result":{"answer":ans,"response_time":rt2,"model":config.MIMO_MODEL,"mode":mode,"sources":list(srcs)}})
        logger.info(f"  [{i+1}/{len(questions)}] ✅ {q['question'][:30]}...")
    rt.close()
    return results


def print_summary(s):
    """打印评估摘要"""
    if not s:
        return
    b, a, imp = s["before"], s["after"], s["improvement"]
    print(f"\n{'='*55}\n  工单10 | {s['total_questions']}题 | LLM: {config.MIMO_MODEL}\n{'='*55}")
    print(f"  指标         | Vector      | GraphRAG    | 变化\n  {'-'*52}")
    for k in ["相关性","完整性","准确性","流畅性"]:
        print(f"  {k:12s} | {b.get(k,0):.2f}          | {a.get(k,0):.2f}          | {imp.get(k+'_delta',0):+.2f}")
    print(f"  响应时间     | {b.get('avg_response_time',0):.2f}s       | {a.get('avg_response_time',0):.2f}s       | {imp.get('time_delta',0):+.2f}s\n{'='*55}")


def run_pipeline(questions, test_mode=False):
    """执行完整评估流水线"""
    if test_mode:
        questions = questions[:2]
    logger.info(f"加载 {len(questions)} 个测试问题")
    if not _milvus_has_data():
        chunks = step("①~④ 解析→分块→向量化→入库", pipeline_pcv)
        gb = step("⑤ 构建知识图谱", pipeline_graph, chunks)
    else:
        logger.info("✅ Milvus 已有数据")
        chunks = []
        cp = Path(config.OUTPUT_DIR)/"chunks_data.json"
        if cp.exists():
            with open(cp,"r",encoding="utf-8") as f:
                chunks = json.load(f)
        gb = get_graph()
        gb.load()
    logger.info(f"\n{'='*40}\n🔵 纯向量\n{'='*40}")
    before = step("⑥ 纯向量问答", run_questions, questions, gb, False, chunks)
    logger.info(f"\n{'='*40}\n🟢 GraphRAG\n{'='*40}")
    after = step("⑦ GraphRAG问答", run_questions, questions, gb, True, chunks)
    all_r = before + after
    with open(Path(config.OUTPUT_DIR)/"qa_results.json","w",encoding="utf-8") as f:
        json.dump(all_r, f, ensure_ascii=False, indent=2)
    ev = RagasEvaluator()
    step("⑧ 评估Vector", ev.evaluate_batch, before, "before")
    step("⑨ 评估GraphRAG", ev.evaluate_batch, after, "after")
    summary = step("⑩ 对比报告", ev.compare_and_save)
    print_summary(summary)
    return summary, gb, all_r
