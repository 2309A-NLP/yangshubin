# -*- coding: utf-8 -*-
"""
文本切分模块（优化版）—— 语义切分 + 重叠窗口
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
"""

import re
from typing import List, Dict
from config import CHUNK_SIZE, CHUNK_OVERLAP


def _split_by_paragraphs(text: str) -> List[str]:
    """按双换行或章节标题切分为语义段落"""
    # 先按双换行切
    paragraphs = re.split(r"\n\s*\n", text)
    # 再按章节标题切（如 "1-1-XX" 或 "第X节"）
    result = []
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        # 如果段落太长（超过chunk_size），继续细分
        if len(p) > CHUNK_SIZE * 1.5:
            sub = re.split(r"(?<=\。|\！|\？)(?=\s*[^\s])", p)
            result.extend(s.strip() for s in sub if s.strip())
        else:
            result.append(p)
    return result


def semantic_chunking(text: str, chunk_size: int = None, overlap: int = None) -> List[Dict]:
    """
    语义切分：先按段落合并，尽量保持语义完整

    Args:
        text: 清洗后的文本
        chunk_size: 每块目标字符数
        overlap: 块间重叠字符数

    Returns:
        [{"chunk_id": str, "text": str, "start_pos": int}, ...]
    """
    import uuid
    if chunk_size is None:
        chunk_size = CHUNK_SIZE
    if overlap is None:
        overlap = CHUNK_OVERLAP

    paragraphs = _split_by_paragraphs(text)
    chunks = []
    current_chunk, current_len = [], 0
    char_offset = 0

    for para in paragraphs:
        if current_len + len(para) + 2 <= chunk_size:
            current_chunk.append(para)
            current_len += len(para) + 2
        else:
            if current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                chunks.append({
                    "chunk_id": str(uuid.uuid4()),
                    "text": chunk_text,
                    "start_pos": char_offset
                })
            current_chunk = [para]
            current_len = len(para)
        char_offset += len(para) + 2

    if current_chunk:
        chunks.append({
            "chunk_id": str(uuid.uuid4()),
            "text": "\n\n".join(current_chunk),
            "start_pos": char_offset
        })

    # 如果没有重叠要求，或者块数太少，直接返回
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    # 添加重叠块
    overlapped = []
    for i, c in enumerate(chunks):
        overlapped.append(c)
        if i < len(chunks) - 1:
            # 从当前块末尾取 overlap 字符，拼到下一块开头
            tail = c["text"][-overlap:] if len(c["text"]) > overlap else c["text"]
            next_text = tail + "\n\n" + chunks[i + 1]["text"]
            overlapped.append({
                "chunk_id": str(uuid.uuid4()),
                "text": next_text[:chunk_size],
                "start_pos": c["start_pos"] + max(0, len(c["text"]) - overlap)
            })

    return overlapped


if __name__ == "__main__":
    from pdf_cleaner import clean_pdf_text
    from config import PDF_PATH

    text = clean_pdf_text(PDF_PATH)
    chunks = semantic_chunking(text)
    print(f"原始长度: {len(text)}")
    print(f"语义切分后: {len(chunks)} 块")
    for i, c in enumerate(chunks[:5]):
        print(f"\n块{i+1}: ({len(c['text'])}字符) {c['text'][:80]}...")
