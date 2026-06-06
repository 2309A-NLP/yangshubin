# -*- coding: utf-8 -*-
"""
数据查看工具 —— 查看PDF解析、文本切分、Milvus入库后的数据
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
"""

import sys
import os
from pprint import pprint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PDF_PATH, MILVUS_URI, MILVUS_COLLECTION, CHUNK_SIZE, CHUNK_OVERLAP
from pdf_parser import extract_text_from_pdf, extract_text_with_metadata
from text_splitter import split_text_simple
from pymilvus import MilvusClient


def show_pdf_info():
    """查看PDF基本信息（总字符数、页数）"""
    print("\n" + "=" * 60)
    print("  📄 PDF 基本信息")
    print("=" * 60)
    print(f"  PDF路径: {PDF_PATH}")

    full_text = extract_text_from_pdf(PDF_PATH)
    pages = extract_text_with_metadata(PDF_PATH)

    print(f"  总字符数: {len(full_text)}")
    print(f"  总页数: {len(pages)}")
    print(f"  文件大小: {os.path.getsize(PDF_PATH) / 1024:.1f} KB")


def show_pdf_preview(length: int = 500):
    """预览PDF前N个字符的内容"""
    print("\n" + "=" * 60)
    print(f"  📖 PDF 内容预览（前{length}字）")
    print("=" * 60)
    text = extract_text_from_pdf(PDF_PATH)
    print(text[:length])
    print(f"\n  ...（共 {len(text)} 字符，仅显示前 {length} 字）")


def show_chunks_info():
    """查看切分后的文本块信息"""
    print("\n" + "=" * 60)
    print("  ✂️  文本切分结果")
    print("=" * 60)
    print(f"  切分参数: chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")

    full_text = extract_text_from_pdf(PDF_PATH)
    chunks = split_text_simple(full_text)

    print(f"  切分后文本块数量: {len(chunks)}")
    print(f"  平均每块字符数: {sum(len(c['text']) for c in chunks) // len(chunks)}")
    print(f"  最短块: {min(len(c['text']) for c in chunks)} 字符")
    print(f"  最长块: {max(len(c['text']) for c in chunks)} 字符")

    return chunks


def show_chunks_table(start: int = 0, count: int = 5):
    """以表格形式展示文本块内容"""
    full_text = extract_text_from_pdf(PDF_PATH)
    chunks = split_text_simple(full_text)

    print("\n" + "=" * 60)
    print(f"  📋 文本块列表（第{start+1}-{min(start+count, len(chunks))}条，共{len(chunks)}条）")
    print("=" * 60)

    for i, chunk in enumerate(chunks[start:start+count], start=start+1):
        print(f"\n  【块 {i}】")
        print(f"  ID: {chunk['chunk_id']}")
        print(f"  位置: [{chunk['start_pos']} ~ {chunk['end_pos']}] （偏移量: {chunk['start_pos']}）")
        print(f"  内容预览: {chunk['text'][:150]}...")
        print(f"  块长度: {len(chunk['text'])} 字符")
        print("  " + "-" * 50)


def show_milvus_data(limit: int = 5):
    """查看Milvus中存储的数据"""
    print("\n" + "=" * 60)
    print("  🗄️  Milvus 向量数据库中的数据")
    print("=" * 60)

    client = MilvusClient(uri=MILVUS_URI)

    if not client.has_collection(MILVUS_COLLECTION):
        print("  ❌ 集合不存在，请先运行 python database_builder.py 构建知识库")
        return

    # 查询统计
    stats = client.get_collection_stats(MILVUS_COLLECTION)
    print(f"  集合名称: {MILVUS_COLLECTION}")
    print(f"  总数据条数: {stats.get('row_count', 0)}")

    # 查询数据
    results = client.query(
        collection_name=MILVUS_COLLECTION,
        limit=limit,
        output_fields=["id", "text", "metadata"]
    )

    print(f"\n  前 {limit} 条数据预览：")
    for i, item in enumerate(results, 1):
        print(f"\n  【记录 {i}】")
        print(f"  ID: {item.get('id')}")
        print(f"  Metadata: {item.get('metadata', '')}")
        print(f"  文本: {item.get('text', '')[:200]}...")

    # 搜索一条验证
    print(f"\n  📊 数据验证：查询一条记录看能否正常检索...")
    from embedding_model import EmbeddingModel
    emb = EmbeddingModel()
    vec = emb.encode_query("测试检索")[0].tolist()
    search_result = client.search(
        collection_name=MILVUS_COLLECTION,
        data=[vec],
        limit=1,
        output_fields=["id", "text"],
        metric_type="IP"
    )
    if search_result and search_result[0]:
        hit = search_result[0][0]
        print(f"  ✅ 检索正常！命中ID={hit.get('id')}, 相似度={hit.get('distance'):.4f}")

    client.close()


def show_all():
    """一次性展示所有数据信息"""
    print("╔" + "═" * 58 + "╗")
    print("║          RAG 数据查看工具                                  ║")
    print("╚" + "═" * 58 + "╝")

    show_pdf_info()
    show_pdf_preview(300)
    show_chunks_table(0, 3)
    show_milvus_data(3)


if __name__ == "__main__":
    print("""
用法：
  python view_data.py info      — 查看PDF基本信息
  python view_data.py preview   — 预览PDF内容（前500字）
  python view_data.py chunks    — 查看文本切分结果
  python view_data.py milvus    — 查看Milvus中的存储数据
  python view_data.py all       — 一次性查看全部信息
    """)

    import sys as _sys
    cmd = _sys.argv[1].lower() if len(_sys.argv) > 1 else "all"

    commands = {
        "info": show_pdf_info,
        "preview": show_pdf_preview,
        "chunks": show_chunks_table,
        "milvus": show_milvus_data,
        "all": show_all,
    }

    if cmd in commands:
        commands[cmd]()
    else:
        print(f"未知命令: {cmd}")
        print("可用命令: info, preview, chunks, milvus, all")
