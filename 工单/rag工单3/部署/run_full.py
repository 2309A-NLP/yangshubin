"""
完整流水线 - 一次性跑完 MinerU 解析 → 向量化 → Milvus → QA
"""
import sys, os, time, json
# ===== 路径桥接：将所有子目录加入 sys.path =====
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ["设计", "研发", "测试", "优化", "部署"]:
    _p = os.path.join(_ROOT, _sub)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
sys.stdout.reconfigure(line_buffering=True)

from config import log, TEST_QUESTIONS

print("=" * 50)
print("完整流水线启动 (MinerU + FP16)")
print("=" * 50)

# Step 1: MinerU 解析
print("\n[1/6] MinerU PDF 解析...")
from pdf_parser import parse_pdf, save_parsed_output
md_content, raw_tables = parse_pdf()
save_parsed_output(md_content, raw_tables)
print(f"  -> {len(md_content)} 字符, {len(raw_tables)} 个表格")

# Step 2: 表格解析
print("\n[2/6] 表格解析...")
from table_parser import tables_to_text_blocks
table_blocks = tables_to_text_blocks(raw_tables)
print(f"  -> {len(table_blocks)} 个文本块")

# Step 3: 文本切分
print("\n[3/6] 文本切分...")
from pdf_parser import extract_content_sections
from text_splitter import split_by_sections, merge_text_and_table_chunks, save_chunks_json, save_chunks_text
sections = extract_content_sections(md_content)
text_chunks = split_by_sections(sections)
merged_chunks = merge_text_and_table_chunks(text_chunks, table_blocks)
save_chunks_json(merged_chunks)
save_chunks_text(merged_chunks)
print(f"  -> {len(merged_chunks)} 个片段 (文本{len(text_chunks)}+表格{len(table_blocks)})")

# Step 4: 向量化
print("\n[4/6] BGE-M3 向量化 (FP16)...")
from embedding import BGEM3Embedding
embedder = BGEM3Embedding()
texts = [c["text"] for c in merged_chunks]
start = time.time()
embeddings = embedder.encode_dense(texts)
print(f"  -> {embeddings.shape} (耗时 {time.time()-start:.1f}s)")

# Step 5: Milvus
print("\n[5/6] Milvus 存储...")
from vector_store import MilvusVectorStore
store = MilvusVectorStore()
store.create_collection(drop_if_exists=True)
store.insert(merged_chunks, embeddings)
stats = store.get_collection_stats()
print(f"  -> {stats}")

# Step 6: QA
print("\n[6/6] 检索 + QA (15个问题)...")
from retriever import HybridRetriever
from llm_qa import DeepSeekQA
retriever = HybridRetriever(embedder, store)
qa = DeepSeekQA()

results = []
for idx, q in enumerate(TEST_QUESTIONS):
    ctx = retriever.retrieve(q["question"], top_k=5)
    result = qa.generate_answer(q["question"], ctx)
    result["id"] = q["id"]
    result["question"] = q["question"]
    results.append(result)

    conf = result.get("confidence", 0)
    label = "OK" if conf > 0.5 else "??"
    print(f"  [{label}] Q{q['id']}: {q['question'][:35]}... conf:{conf:.2f} {result.get('latency',0):.1f}s")

qa.save_results(results)
print(f"\n全部完成！{len(results)} 个问题 -> output/qa_results.json")
