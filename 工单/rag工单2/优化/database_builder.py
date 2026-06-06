# -*- coding: utf-8 -*-
"""
知识库构建 —— PDF清洗 → 语义切分 → BGE-M3嵌入 → Milvus入库
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
"""

import os
import time
import json
from pathlib import Path
from datetime import datetime

from config import PDF_PATH, OUTPUT_DIR
from pdf_cleaner import clean_pdf_text
from text_splitter import semantic_chunking
from embedding_model import EmbeddingModel
from vector_store import VectorStore


def _log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  [{ts}] {msg}")


def _export_data(text: str, chunks: list):
    """导出处理后的数据到output目录"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 完整文本
    with open(os.path.join(OUTPUT_DIR, "01_清洗后全文.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    _log(f"导出全文: {len(text)}字符")

    # 文本块明细
    with open(os.path.join(OUTPUT_DIR, "02_文本块明细.txt"), "w", encoding="utf-8") as f:
        for i, c in enumerate(chunks, 1):
            f.write(f"{'='*50}\n【第{i}块】ID:{c['chunk_id']}\n{c['text']}\n\n")

    # 文本块JSON
    with open(os.path.join(OUTPUT_DIR, "03_文本块总览.json"), "w", encoding="utf-8") as f:
        summary = [{"序号": i+1, "chunk_id": c["chunk_id"],
                     "长度": len(c["text"]),
                     "预览": c["text"][:80]}
                    for i, c in enumerate(chunks)]
        json.dump(summary, f, ensure_ascii=False, indent=2)

    _log(f"导出文本块: {len(chunks)}条")
    _log(f"导出目录: {OUTPUT_DIR}")


def build():
    """构建知识库"""
    _log("="*50)
    _log("开始构建知识库（优化版）")

    t0 = time.time()

    # Step 1: PDF清洗
    _log("[1/4] PDF清洗（去页眉页脚+提取文本）...")
    t1 = time.time()
    text = clean_pdf_text(PDF_PATH)
    _log(f"  完成: {len(text)}字符 ({(time.time()-t1):.1f}s)")

    # Step 2: 语义切分
    _log("[2/4] 语义切分...")
    t1 = time.time()
    chunks = semantic_chunking(text)
    _log(f"  完成: {len(chunks)}块 ({(time.time()-t1):.1f}s)")

    # Step 3: BGE-M3嵌入
    _log("[3/4] BGE-M3生成向量...")
    t1 = time.time()
    emb = EmbeddingModel()
    vectors = emb.encode([c["text"] for c in chunks])
    _log(f"  完成: 形状{vectors.shape} ({(time.time()-t1):.1f}s)")

    # Step 4: Milvus入库
    _log("[4/4] Milvus入库...")
    t1 = time.time()
    store = VectorStore()
    store.create_collection(drop_if_exists=True)
    store.insert(chunks, vectors.tolist())
    store.close()
    _log(f"  完成: 入库{len(chunks)}条 ({(time.time()-t1):.1f}s)")

    # 导出数据
    _export_data(text, chunks)

    # 预热：构建BM25缓存，下次提问秒加载
    _log("[预热] 构建BM25检索缓存...")
    from hybrid_retriever import HybridRetriever
    hr = HybridRetriever()
    hr.warm_up()
    _log("[预热] BM25缓存就绪")

    _log(f"✅ 构建完成！总耗时: {(time.time()-t0):.1f}s")
    print(f"\n  💡 output/ 文件夹可查看处理好的数据")
    return {"chunks": len(chunks), "elapsed": round(time.time()-t0, 1)}


if __name__ == "__main__":
    build()
