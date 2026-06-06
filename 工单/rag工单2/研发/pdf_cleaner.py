# -*- coding: utf-8 -*-
"""
PDF预处理模块 —— 去页眉页脚、清洗文本、提取表格
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
"""

import fitz
import re
from collections import Counter
from typing import List, Dict


def _detect_header_footer(pdf_path: str, sample_pages: int = 10) -> dict:
    """
    通过采样前N页，自动检测页眉页脚文本

    Returns: {"header_texts": [...], "footer_texts": [...], "header_y_max": float, "footer_y_min": float}
    """
    doc = fitz.open(pdf_path)
    top_texts, bottom_texts, page_h = [], [], []

    for i in range(min(sample_pages, len(doc))):
        page = doc[i]
        rect = page.rect
        page_h.append(rect.height)
        blocks = page.get_text("blocks")

        for b in blocks:
            x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4].strip()
            if not text or len(text) < 8:
                continue
            # 顶部10%区域 → 候选页眉
            if y0 < rect.height * 0.10:
                top_texts.append(text[:60])
            # 底部10%区域 → 候选页脚
            if y1 > rect.height * 0.90:
                bottom_texts.append(text[:60])

    doc.close()
    avg_h = sum(page_h) / len(page_h) if page_h else 842

    # 取出现次数 > 一半采样页的文本 = 页眉/页脚
    header_texts = {t for t, c in Counter(top_texts).items() if c > sample_pages // 2}
    footer_texts = {t for t, c in Counter(bottom_texts).items() if c > sample_pages // 2}

    return {
        "header_texts": header_texts,
        "footer_texts": footer_texts,
        "header_y_max": avg_h * 0.10,
        "footer_y_min": avg_h * 0.90,
    }


def clean_pdf_text(pdf_path: str) -> str:
    """
    提取PDF文本，自动去除页眉页脚，返回干净文本

    Args:
        pdf_path: PDF文件路径

    Returns:
        清洗后的纯文本（按页用换行拼接）
    """
    doc = fitz.open(pdf_path)
    rules = _detect_header_footer(pdf_path)
    pages_text = []

    for page_num, page in enumerate(doc, 1):
        rect = page.rect
        blocks = page.get_text("blocks")
        clean_blocks = []

        for b in blocks:
            x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4].strip()
            if not text:
                continue

            # 跳过页眉区域中的匹配文本
            if y0 < rules["header_y_max"] and any(
                h in text for h in rules["header_texts"]
            ):
                continue

            # 跳过页脚区域中的匹配文本
            if y1 > rules["footer_y_min"] and any(
                f in text for f in rules["footer_texts"]
            ):
                continue

            # 跳过页码行（纯数字或 "1-1-XX" 格式）
            if re.match(r"^\d{1,3【报错】}-\d{1,3【报错】}-\d{1,3【报错】}$", text.strip()):
                continue
            if re.match(r"^\d{1,4}$", text.strip()) and len(text) < 6:
                continue

            clean_blocks.append(text)

        page_text = "\n".join(clean_blocks)
        if page_text.strip():
            pages_text.append(page_text)

    doc.close()
    return "\n\n".join(pages_text)


def extract_tables(pdf_path: str) -> List[Dict]:
    """
    提取PDF中的表格数据（通过检测页面上对齐的文本块）

    Returns: [{"page": 1, "table_text": "..."}, ...]
    """
    doc = fitz.open(pdf_path)
    tables = []

    for page_num, page in enumerate(doc, 1):
        blocks = page.get_text("blocks")
        rows = []

        for b in blocks:
            text = b[4].strip()
            # 含多个空格或制表符 → 可能是表格行
            if text and ("  " in text or "\t" in text):
                rows.append(text)

        if rows:
            tables.append({"page": page_num, "table_text": "\n".join(rows)})

    doc.close()
    return tables


if __name__ == "__main__":
    from config import PDF_PATH
    text = clean_pdf_text(PDF_PATH)
    print(f"清洗后长度: {len(text)} 字符")
    print(f"前300字:\n{text[:300]}")
