"""
从缓存恢复流水线：加载 chunks.json → 重新 embedding → 存储到 Milvus → QA
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
"""
import sys, os, json, time
# ===== 路径桥接：将所有子目录加入 sys.path =====
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ["设计", "研发", "测试", "优化", "部署"]:
    _p = os.path.join(_ROOT, _sub)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
sys.stdout.reconfigure(line_buffering=True)
from config import OUTPUT_DIR, log, ensure_dirs

ensure_dirs()

# ===== Step 1: 加载缓存 =====
log("加载缓存 chunks.json ...", "RESUME")
chunks_path = os.path.join(OUTPUT_DIR, "chunks.json")
with open(chunks_path, "r", encoding="utf-8") as f:
    chunks = json.load(f)
log(f"加载 {len(chunks)} 个片段", "RESUME")

# ===== Step 2: BGE-M3 向量化 =====
log("=" * 60, "PIPELINE")
log("Step 4/6: BGE-M3 向量化", "PIPELINE")
from embedding import BGEM3Embedding
embedder = BGEM3Embedding()
texts = [c["text"] for c in chunks]
embeddings = embedder.encode_dense(texts)
log(f"向量化完成，形状: {embeddings.shape}", "PIPELINE")

# ===== Step 3: 存储到 Milvus =====
log("=" * 60, "PIPELINE")
log("Step 5/6: 向量存储 (Milvus)", "PIPELINE")
from vector_store import MilvusVectorStore
store = MilvusVectorStore()
store.create_collection(drop_if_exists=True)
store.insert(chunks, embeddings)
stats = store.get_collection_stats()
log(f"Milvus 存储完成: {stats}", "PIPELINE")

# ===== Step 4: 检索 + 问答 =====
log("=" * 60, "PIPELINE")
log("Step 6/6: 检索 + 问答", "PIPELINE")
from config import TEST_QUESTIONS
from retriever import HybridRetriever
from llm_qa import DeepSeekQA

retriever = HybridRetriever(embedder, store)
qa = DeepSeekQA()

def get_context(query):
    results = retriever.retrieve(query, top_k=5)
    return [r for r in results]

results = qa.batch_qa(TEST_QUESTIONS, get_context)
qa.save_results(results)

log(f"共处理 {len(results)} 个问题", "QA")
for r in results:
    conf = r.get("confidence", 0)
    lat = r.get("latency", 0)
    label = "✅" if conf > 0.5 else "⚠️"
    log(f"{label} Q{r.get('id','')}: {r.get('question','')[:40]}... | 置信度:{conf:.2f} | 延迟:{lat:.2f}s", "QA")
    print(f"   答案: {r.get('answer','')[:150]}")
    print()

log("流水线恢复完成 ✅", "RESUME")
