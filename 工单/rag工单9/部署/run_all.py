"""\nrun_all.py - RAG工单9 全流程统一入口\n工单编号: 人工智能NLP-RAG-Graph RAG 优化任务（调用全部模块 + 小米MiMo API）
需求: GraphRAG优化 — 解析→向量化→图谱→优化前后对比→MiMo评估
功能: 1-5数据流水线+图谱构建 6-7优化前(纯向量)/优化后(GraphRAG)检索问答 8-10评估对比+Web
"""
import logging, time, json, argparse, os, sys

import sys, os
_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_root, '..', '设计'))
sys.path.insert(0, os.path.join(_root, '..', '研发'))
sys.path.insert(0, os.path.join(_root, '..', '测试'))
sys.path.insert(0, os.path.join(_root, '..', '部署'))
from config import OUTPUT_DIR, WO_ID, LOG_FMT, LOG_DATEFMT

logging.basicConfig(format=LOG_FMT, datefmt=LOG_DATEFMT, level=logging.INFO)
logger = logging.getLogger("run_all")
logger.info(f"{'='*50}\nRAG工单9 GraphRAG优化 | {WO_ID}\n{'='*50}")


def banner(t):
    logger.info(f"\n{'='*30}\n{t}\n{'='*30}")


def step_parse():
    """① 解析CCF年报 + sample_questions"""
    banner("📄 [1] PDF解析(CCF年报+测试题)")
    from pdf_parser import parse_ccf_pdfs, parse_sample_questions
    pdf = parse_ccf_pdfs()
    qs = parse_sample_questions()
    logger.info(f"✅ PDF: {len(pdf['pages'])}页 | 测试题: {len(qs)}题\n")
    return pdf, qs


def step_chunk(pdf):
    """② 文本分块"""
    banner("🔪 [2] 文本分块")
    from text_chunker import build_chunks, save_chunks
    chunks = build_chunks(pdf)
    save_chunks(chunks)
    json.dump(chunks, open(os.path.join(OUTPUT_DIR, "chunks_data.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    logger.info(f"✅ 分块: {len(chunks)}块\n")
    return chunks


def step_embed(chunks):
    """③ BGE-M3向量化"""
    banner("🧠 [3] BGE-M3向量化")
    from embedder import create_embeddings
    emb = create_embeddings(chunks)
    logger.info(f"✅ 向量: {len(emb['dense_vectors'])}个\n")
    return emb


def step_milvus(emb):
    """④ Milvus入库"""
    banner("🗄️  [4] Milvus入库")
    from milvus_handler import create_index_and_insert
    cnt = create_index_and_insert(emb["dense_vectors"], emb["chunk_texts"], emb["chunk_metas"])
    logger.info(f"✅ Milvus: {cnt}条\n")


def step_graph(chunks):
    """⑤ MiMo实体提取→知识图谱"""
    banner("🔗 [5] 实体提取+知识图谱(MiMo)")
    from entity_graph_builder import GraphBuilder
    builder = GraphBuilder()
    builder.build_from_chunks(chunks, max_chunks=30)
    builder.save()
    logger.info(f"✅ 图谱: {builder.graph.number_of_nodes()}节点\n")
    return builder


def run_qa(questions, graph_builder, use_graph, chunks):
    """⑥-⑦ 检索+MiMo问答"""
    from embedder import BgeM3Embedder
    from graph_retriever import GraphRetriever
    from qa_generator import QAGenerator
    e, ret, qa = BgeM3Embedder(), GraphRetriever(graph_builder), QAGenerator()
    e.load()
    ret.set_chunks(chunks)
    if not ret.chunk_store:
        ret.set_chunks([])
    results, mode_name = [], "GraphRAG" if use_graph else "VectorOnly"
    logger.info(f"🔍 {mode_name}模式 ({len(questions)}题)...")
    for q in questions:
        qv = e.encode_query(q["question"])["dense_vecs"][0].tolist()
        t0 = time.time()
        ctx = ret.retrieve(qv, q["question"], use_graph=use_graph) if use_graph else ret.retrieve_vector_only(qv)
        ans = qa.generate(q["question"], ctx, use_graph=use_graph)
        results.append({"question": q["question"], "reference_answer": q.get("reference_answer", ""),
                        "retrieved_chunks": ctx, "response_time": round(time.time()-t0+ans["response_time"], 2),
                        "qa_result": ans})
        print(f"  [{mode_name}] {q['question'][:40]}... ⏱{ans['response_time']:.1f}s")
    ret.close()
    return results


def step_eval(before, after):
    """⑧-⑩ MiMo评估+对比+HTML报告"""
    banner("📊 [8-10] MiMo评估对比")
    from evaluator import RagasEvaluator
    ev = RagasEvaluator()
    ev.evaluate_batch(before, "before")
    ev.evaluate_batch(after, "after")
    summary = ev.compare_and_save()
    if summary:
        b, a = summary["before"], summary["after"]
        print(f"\n{'='*55}")
        print(f"  指标         | 优化前(Vector) | 优化后(GraphRAG) | 变化")
        print(f"  {'-'*52}")
        print(f"  上下文精度   | {b['avg_precision']:.4f}          | {a['avg_precision']:.4f}           | {summary['improvement']['precision_delta']:+.4f}")
        print(f"  上下文召回   | {b['avg_recall']:.4f}          | {a['avg_recall']:.4f}           | {summary['improvement']['recall_delta']:+.4f}")
        print(f"  响应时间     | {b['avg_response_time']}s        | {a['avg_response_time']}s         | {summary['improvement']['time_delta']:+.2f}s")
        print(f"{'='*55}")
        print(f"  阈值: 精度≥0.8 召回≥0.9")
        print(f"  优化前达标: {'✅' if summary.get('before_meets_threshold') else '❌'}")
        print(f"  优化后达标: {'✅' if summary.get('after_meets_threshold') else '❌'}")
        print(f"  报告: {OUTPUT_DIR / 'evaluation_report.html'}\n")
    return summary


def full_pipeline():
    """完整10步流水线"""
    t0 = time.time()
    pdf, questions = step_parse()
    chunks = step_chunk(pdf)
    emb = step_embed(chunks)
    step_milvus(emb)
    del emb
    builder = step_graph(chunks)
    logger.info(f"\n{'='*30}\n优化前: 纯向量检索\n{'='*30}")
    before = run_qa(questions, builder, False, chunks)
    logger.info(f"\n{'='*30}\n优化后: GraphRAG混合检索\n{'='*30}")
    after = run_qa(questions, builder, True, chunks)
    step_eval(before, after)
    logger.info(f"\n🏁 全流程完成! 总耗时: {time.time()-t0:.1f}秒")
    from app import get_app
    get_app().run(host="0.0.0.0", port=5009, debug=False)


def quick_test():
    """快速测试（缓存数据+MiMo）"""
    import networkx as nx
    from collections import defaultdict
    from entity_graph_builder import GraphBuilder
    p = OUTPUT_DIR / "chunks_data.json"
    if not p.exists():
        logger.error(f"缓存不存在: {p}")
        return
    chunks = json.load(open(p, encoding="utf-8"))
    logger.info(f"📦 缓存: {len(chunks)}块")
    graph_builder = GraphBuilder()
    gp = OUTPUT_DIR / "knowledge_graph.json"
    if gp.exists():
        graph_builder.graph = nx.Graph()
        gd = json.load(open(gp, encoding="utf-8"))
        for n in gd.get("nodes", []):
            graph_builder.graph.add_node(n["name"], type=n.get("type", ""), source_pdf=n.get("source_pdf", ""))
        for e in gd.get("edges", []):
            graph_builder.graph.add_edge(e["source"], e["target"], relation=e.get("relation", "相关"))
        graph_builder.entity_to_chunks = defaultdict(set, {k: set(v) for k, v in gd.get("entity_to_chunks", {}).items()})
        graph_builder._built = True
        logger.info(f"图谱: {graph_builder.graph.number_of_nodes()}节点")
    from pdf_parser import parse_sample_questions
    questions = parse_sample_questions()
    before = run_qa(questions, graph_builder, False, chunks)
    after = run_qa(questions, graph_builder, True, chunks)
    step_eval(before, after)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG工单9 GraphRAG优化 (MiMo API)")
    parser.add_argument("--quick", action="store_true", help="快速测试(缓存数据)")
    parser.add_argument("--web-only", action="store_true", help="仅启动Web")
    args = parser.parse_args()
    if args.web_only:
        from app import get_app
        get_app().run(host="0.0.0.0", port=5009, debug=False)
    elif args.quick:
        quick_test()
    else:
        full_pipeline()
