# -*- coding: utf-8 -*-
"""
文本切分模块 —— 将PDF全文按固定大小切分为语义块
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
"""

import uuid
from typing import List, Dict

from config import CHUNK_SIZE, CHUNK_OVERLAP


def split_text_simple(text: str, chunk_size: int = None,
                      chunk_overlap: int = None) -> List[Dict]:
    """
    按固定字符数切分文本，带重叠窗口

    Args:
        text: 待切分的完整文本字符串
        chunk_size: 每个文本块的最大字符数（默认取配置值）
        chunk_overlap: 相邻块之间的重叠字符数（默认取配置值）

    Returns:
        切分结果列表，每项包含：
        - chunk_id: 唯一标识
        - text: 文本块内容
        - start_pos: 在原文本中的起始位置
        - end_pos: 在原文本中的结束位置
    """
    if chunk_size is None:
        chunk_size = CHUNK_SIZE
    if chunk_overlap is None:
        chunk_overlap = CHUNK_OVERLAP

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk_text = text[start:end]

        # 跳过空文本块
        if chunk_text.strip():
            chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "text": chunk_text,
                "start_pos": start,
                "end_pos": end
            })

        # 移动窗口：如果已到达末尾则结束
        if end >= text_len:
            break
        start = end - chunk_overlap

    return chunks


def split_by_paragraphs(text: str, chunk_size: int = None,
                        chunk_overlap: int = None) -> List[Dict]:
    """
    按段落（双换行）初步分割，再合并到接近chunk_size

    Args:
        text: 待切分的完整文本
        chunk_size: 目标块大小
        chunk_overlap: 块间重叠

    Returns:
        按段落语义合并后的文本块列表
    """
    if chunk_size is None:
        chunk_size = CHUNK_SIZE
    if chunk_overlap is None:
        chunk_overlap = CHUNK_OVERLAP

    # 先按双换行分割为段落
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current_chunk = ""
    current_start = 0
    char_offset = 0

    for para in paragraphs:
        # 如果当前块+新段落不超过上限，则合并
        if len(current_chunk) + len(para) < chunk_size:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
                current_start = char_offset
        else:
            # 保存当前块，开始新的块
            if current_chunk:
                chunks.append({
                    "chunk_id": str(uuid.uuid4()),
                    "text": current_chunk,
                    "start_pos": current_start,
                    "end_pos": char_offset
                })
            current_chunk = para
            current_start = char_offset

        char_offset += len(para) + 2  # +2 因为\n\n

    # 处理最后一个块
    if current_chunk:
        chunks.append({
            "chunk_id": str(uuid.uuid4()),
            "text": current_chunk,
            "start_pos": current_start,
            "end_pos": len(text)
        })

    return chunks


if __name__ == "__main__":
    # 自测模块
    from config import PDF_PATH
    from pdf_parser import extract_text_from_pdf

    text = extract_text_from_pdf(PDF_PATH)
    chunks = split_text_simple(text)
    print(f"原始文本长度：{len(text)}")
    print(f"切分后文本块数量：{len(chunks)}")

    # 显示前3个块的预览
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- 块 {i + 1} (id={chunk['chunk_id'][:8]}...) ---")
        print(f"位置 [{chunk['start_pos']}:{chunk['end_pos']}]")
        print(chunk['text'][:150])
