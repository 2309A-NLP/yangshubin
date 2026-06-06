"""\nrun.py - RAG工单9 主流水线\n工单编号: 人工智能NLP-RAG-Graph RAG 优化任务（调用所有子模块+MiMo API）
需求: GraphRAG优化 — 协调全流程：解析→分块→向量化→Milvus→图谱→优化前后对比→评估
功能: 1-4数据流水线 5图谱构建 6-7优化前后检索 8-10评估对比 Web展示
"""

import logging, sys, time, argparse
import sys, os
_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_root, '..', '设计'))
sys.path.insert(0, os.path.join(_root, '..', '研发'))
sys.path.insert(0, os.path.join(_root, '..', '测试'))
sys.path.insert(0, os.path.join(_root, '..', '部署'))
from config import LOG_FMT, LOG_DATEFMT, WO_ID, OUTPUT_DIR

logging.basicConfig(format=LOG_FMT, datefmt=LOG_DATEFMT, level=logging.INFO)
logger = logging.getLogger("run")


def step(msg, fn, *args, **kw):
    """通用步骤执行器，带日志和计时"""
    logger.info(f"步: {msg}")
    t0 = time.time()
    r = fn(*args, **kw)
    logger.info(f"  ✓ {time.time() - t0:.1f}s")
    return r


def print_summary(summary):
    """打印优化前后对比摘要到控制台"""
    if not summary:
        return
    b, a = summary["before"], summary["after"]
    imp = summary["improvement"]
    print(f"\n{'='*55}")
    print(f"  {WO_ID}")
    print(f"  {summary.get('total_questions', 0)}题对比")
    print(f"{'='*55}")
    print(f"  指标         | 优化前(Vector) | 优化后(GraphRAG) | 变化")
    print(f"  {'-'*52}")
    print(f"  上下文精度   | {b.get('avg_precision', 0):.4f}          | {a.get('avg_precision', 0):.4f}           | {imp.get('precision_delta', 0):+.4f}")
    print(f"  上下文召回   | {b.get('avg_recall', 0):.4f}          | {a.get('avg_recall', 0):.4f}           | {imp.get('recall_delta', 0):+.4f}")
    print(f"  响应时间     | {b.get('avg_response_time', 0)}s        | {a.get('avg_response_time', 0)}s         | {imp.get('time_delta', 0):+.2f}s")
    print(f"{'='*55}")
    print(f"  阈值: 精度≥0.8 召回≥0.9")
    print(f"  优化前达标: {'✅' if summary.get('before_meets_threshold') else '❌'}")
    print(f"  优化后达标: {'✅' if summary.get('after_meets_threshold') else '❌'}")
    print(f"\n  报告: {OUTPUT_DIR / 'evaluation_report.html'}")
    print(f"  Web:  http://127.0.0.1:5009\n")


def _check_milvus():
    """检查Milvus集合是否已有数据"""
    try:
        from pymilvus import utility, Collection
        from milvus_handler import MilvusManager
        mgr = MilvusManager()
        mgr.connect()
        ok = utility.has_collection("rag_work_order_9") and Collection("rag_work_order_9").num_entities > 0
        Collection("rag_work_order_9").release()
        mgr.close()
        return ok
    except:
        return False


def pipeline_parse_chunk_vectorize():
    """流水线①~④: 解析PDF→分块→向量化→Milvus入库"""
    from pdf_parser import parse_ccf_pdfs
    from text_chunker import build_chunks, save_chunks
    from embedder import create_embeddings
    from milvus_handler import create_index_and_insert

    pdf = parse_ccf_pdfs()
    if not pdf["pages"]:
        raise RuntimeError("PDF解析失败")
    chunks = build_chunks(pdf)
    save_chunks(chunks)
    data = create_embeddings(chunks)
    create_index_and_insert(data["dense_vectors"], data["chunk_texts"], data["chunk_metas"])
    return chunks, data


def pipeline_build_graph(chunks):
    """流水线⑤: 构建知识图谱，返回builder和chunks"""
    from entity_graph_builder import GraphBuilder
    builder = GraphBuilder()
    builder.build_from_chunks(chunks, max_chunks=30)
    builder.save()
    return builder, chunks


def run_test_questions(questions, graph_builder, use_graph, chunks=None):
    """对测试问题执行检索+问答"""
    from embedder import BgeM3Embedder
    from graph_retriever import GraphRetriever
    from qa_generator import QAGenerator
    embedder = BgeM3Embedder(); retriever = GraphRetriever(graph_builder); qa = QAGenerator()
    if chunks:
        retriever.set_chunks(chunks)
    # 如果chunk_store为空且有graph_builder，从graph数据补一下
    if not retriever.chunk_store:
        retriever.set_chunks([])
    results = []; mode_name = "GraphRAG" if use_graph else "Vector"
    logger.info(f"运行{mode_name}模式({len(questions)}题)...")
    for q in questions:
        qvec = embedder.encode_query(q["question"])["dense_vecs"][0].tolist()
        t0 = time.time()
        chunks = retriever.retrieve(qvec, q["question"], use_graph=use_graph) if use_graph else retriever.retrieve_vector_only(qvec)
        qa_result = qa.generate(q["question"], chunks, use_graph=use_graph)
        results.append({"question":q["question"],"reference_answer":q.get("reference_answer",""),
                        "retrieved_chunks":chunks,"response_time":round(time.time()-t0+qa_result["response_time"],2),
                        "qa_result":qa_result})
    retriever.close(); return results


def main():
    """主流程：执行完整GraphRAG优化对比"""
    parser = argparse.ArgumentParser(description="RAG工单9 GraphRAG优化对比")
    parser.add_argument("--web-only", action="store_true", help="仅启动Web")
    parser.add_argument("--test", action="store_true", help="测试模式(仅2题)")
    parser.add_argument("--rebuild", action="store_true", help="强制重建所有数据")
    args = parser.parse_args()

    if args.web_only:
        from app import get_app
        return get_app().run(host="127.0.0.1", port=5009, debug=False)

    t0 = time.time()

    # 提取测试问题
    from pdf_parser import parse_sample_questions
    questions = step("提取测试问题", parse_sample_questions)
    if args.test:
        questions = questions[:2]

    # 检查/重建数据
    if args.rebuild or not _check_milvus():
        if args.rebuild:
            logger.info("--rebuild: 强制重建")
        chunks, _ = step("①~④解析+分块+向量化+入库", pipeline_parse_chunk_vectorize)
        graph_builder, chunks = step("⑤构建知识图谱", pipeline_build_graph, chunks)
        # 存档chunks供后续图谱扩展使用
        import json as _j
        with open(str(OUTPUT_DIR / "chunks_data.json"), "w", encoding="utf-8") as _f:
            _j.dump(chunks, _f, ensure_ascii=False, indent=2)
    else:
        logger.info("Milvus已有数据, 跳过①~⑤")
        # 加载图谱和chunks
        import json, networkx as nx
        from collections import defaultdict
        from entity_graph_builder import GraphBuilder
        graph_builder = None; chunks = []
        graph_path = OUTPUT_DIR / "knowledge_graph.json"
        chunks_path = OUTPUT_DIR / "chunks_data.json"
        if graph_path.exists() and chunks_path.exists():
            with open(chunks_path, "r", encoding="utf-8") as _f: chunks = json.load(_f)
            graph_builder = GraphBuilder()
            graph_builder.graph = nx.Graph()
            with open(graph_path, "r", encoding="utf-8") as _f: gdata = json.load(_f)
            for n in gdata.get("nodes", []): graph_builder.graph.add_node(n["name"], type=n.get("type",""), source_pdf=n.get("source_pdf",""))
            for e in gdata.get("edges", []): graph_builder.graph.add_edge(e["source"], e["target"], relation=e.get("relation","相关"))
            graph_builder.entity_to_chunks = defaultdict(set, {k:set(v) for k,v in gdata.get("entity_to_chunks",{}).items()})
            graph_builder._built = True
            logger.info(f"图谱已加载: {graph_builder.graph.number_of_nodes()}节点, {len(chunks)}个chunk")

    # 优化前：纯向量检索
    logger.info(f"\n{'=' * 40}\n优化前: 纯向量检索\n{'=' * 40}")
    before_results = step("⑥优化前检索+问答", run_test_questions, questions, graph_builder, False, chunks)

    # 优化后：GraphRAG混合检索
    logger.info(f"\n{'=' * 40}\n优化后: GraphRAG混合检索\n{'=' * 40}")
    after_results = step("⑦优化后检索+问答", run_test_questions, questions, graph_builder, True, chunks)

    # RAGAS风格评估
    from evaluator import RagasEvaluator
    evaluator = RagasEvaluator()
    step("⑧评估优化前", evaluator.evaluate_batch, before_results, "before")
    step("⑨评估优化后", evaluator.evaluate_batch, after_results, "after")
    summary = step("⑩对比分析", evaluator.compare_and_save)

    print_summary(summary)
    logger.info(f"总耗时: {time.time() - t0:.1f}秒")

    # 启动Web
    from app import get_app
    get_app().run(host="127.0.0.1", port=5009, debug=False)


if __name__ == "__main__":
    main()
