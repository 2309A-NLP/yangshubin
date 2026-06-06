"""\ntext_chunker.py - RAG工单9 文本分块模块\n工单编号: 人工智能NLP-RAG-Graph RAG 优化任务
需求: GraphRAG数据预处理 — 将CCF年报文本切块，供向量化和图谱构建
功能: 固定大小切块(带重叠窗口)，保留元数据(source_pdf/page_num)
"""

import logging, json, os, re
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '设计'))
from config import CHUNK_SIZE, CHUNK_OVERLAP, OUTPUT_DIR, LOG_FMT, LOG_DATEFMT

logging.basicConfig(format=LOG_FMT, datefmt=LOG_DATEFMT, level=logging.INFO)
logger = logging.getLogger("text_chunker")


def split_text(text, chunk_size=None, overlap=None):
    """
    将长文本按指定大小切块（带重叠窗口）
    参数:
        text: 输入文本字符串
        chunk_size: 每块最大字符数
        overlap: 块间重叠字符数
    返回:
        list: [{"content": str, "index": int}, ...]
    """
    chunk_size = chunk_size or CHUNK_SIZE
    overlap = overlap or CHUNK_OVERLAP
    # 按换行分割为段落列表
    paragraphs = [p.strip() for p in re.split(r'\n+', text) if p.strip()]
    chunks = []
    current_chunk = ""
    # 逐段合并到当前块中，超过chunk_size则切分
    for para in paragraphs:
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += para + "\n"
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            # 新块带上上一块末尾内容作为重叠
            if overlap > 0 and chunks:
                last = chunks[-1]
                overlap_text = last[-overlap:] if len(last) > overlap else last
                current_chunk = overlap_text + "\n" + para + "\n"
            else:
                current_chunk = para + "\n"
    # 保存最后一块
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return [{"content": c, "index": i} for i, c in enumerate(chunks)]


def build_chunks(pdf_result):
    """
    将CCF年报PDF解析结果切块
    参数:
        pdf_result: pdf_parser返回的解析结果
    返回:
        list: 带元数据的文本块列表
    """
    logger.info("构建文档块...")
    all_chunks = []
    for page in pdf_result["pages"]:
        page_chunks = split_text(page["text"])
        for c in page_chunks:
            all_chunks.append({
                "content": c["content"],
                "index": len(all_chunks),
                "source_pdf": page.get("source_pdf", "unknown.pdf"),
                "page_num": page["page_num"],
            })
    logger.info(f"文档块构建完成: {len(all_chunks)} 块")
    return all_chunks


def save_chunks(chunks):
    """保存分块结果到output目录"""
    path = os.path.join(OUTPUT_DIR, "chunks.json")
    data = [{"index": c["index"], "preview": c["content"][:80],
             "source": c["source_pdf"], "page": c["page_num"]} for c in chunks]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"分块结果已保存: {path}")


if __name__ == "__main__":
    """单独测试分块功能"""
    test = "第一段。第二段。\n新段落。"
    result = split_text(test, chunk_size=20, overlap=5)
    for c in result:
        print(f"块{c['index']}: {c['content'][:30]}")
