"""
快速重建：清空旧Milvus集合，重新索引两份PDF的文本（不重新描述图片）
用已有的image_descriptions.json和chunks.json
"""
import sys, os, json, time
# ===== 路径桥接 =====
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ["设计", "研发", "测试", "优化", "部署"]:
    _p = os.path.join(_ROOT, _sub)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
sys.path.insert(0, r"C:\Users\31326\Desktop\rag工单4")

from config import PDF_FILES, OUTPUT_DIR, WORK_ORDER_ID
from pdf_parser import extract_pdf
from text_chunker import build_chunks_with_images, save_chunks
from embedder import create_embeddings_for_chunks
from milvus_handler import MilvusHandler

print("=" * 50)
print(f"RAG工单4 快速重建")
print(f"工单: {WORK_ORDER_ID}")
print("=" * 50)

# 先清理旧集合
handler = MilvusHandler()
handler.connect()
handler.drop_collection()
print("[OK] 已删除旧集合")

# 处理两份PDF的全部文本（跳过图片描述，用已有的描述）
all_chunks = []
existing_descriptions = []

# 加载已有的图片描述
desc_path = os.path.join(OUTPUT_DIR, "image_descriptions.json")
if os.path.exists(desc_path):
    with open(desc_path, "r", encoding="utf-8") as f:
        existing_descriptions = json.load(f)
    print(f"[OK] 加载已有图片描述: {len(existing_descriptions)}条")

for pdf_name, pdf_path in PDF_FILES.items():
    print(f"\n--- 处理PDF: {pdf_name} ---")
    
    # 解析PDF（只提取文本，不重新渲染图片）
    result = extract_pdf(pdf_path)
    if result is None:
        print(f"[FAIL] 解析失败: {pdf_name}")
        continue
    
    pages = result["pages"]
    total_text = result["total_text"]
    
    # 过滤出属于当前PDF的图片描述
    pdf_descriptions = [d for d in existing_descriptions if d.get("source_pdf", "") == pdf_name]
    print(f"  文本页数: {len(pages)}, 关联图片描述: {len(pdf_descriptions)}条")
    
    # 构建文本结果
    text_result = {"pages": pages, "total_text": total_text}
    
    # 分块
    chunks = build_chunks_with_images(text_result, pdf_descriptions)
    all_chunks.extend(chunks)
    print(f"  分块: {len(chunks)}块")

print(f"\n[OK] 总块数: {len(all_chunks)}")

# 向量化
print("\n--- BGE-M3 向量化 ---")
embeddings = create_embeddings_for_chunks(all_chunks)
print(f"[OK] 向量完成: {len(embeddings['dense_vectors'])}条")

# 入库
print("\n--- 存入Milvus ---")
handler.create_collection(drop_if_exists=True)
ids = handler.insert_data(
    vectors=embeddings["dense_vectors"],
    chunk_texts=embeddings["chunk_texts"],
    chunk_metas=embeddings["chunk_metas"],
)
print(f"[OK] 入库完成: {len(ids)}条")

# 启动Web
print("\n--- 启动Web服务 ---")
print(f"  http://localhost:5000")
print(f"  工单: {WORK_ORDER_ID}")

from app import get_flask_app
app = get_flask_app()
app.run(host="127.0.0.1", port=5000, debug=False)
