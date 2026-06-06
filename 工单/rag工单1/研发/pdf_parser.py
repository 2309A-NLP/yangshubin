# -*- coding: utf-8 -*-
"""
PDF解析模块 —— 使用 PyMuPDF 提取招股说明书中的文本内容
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
"""

import fitz  # PyMuPDF
from typing import List, Dict


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    从PDF文件中提取全部文本内容

    Args:
        pdf_path: PDF文件的绝对路径

    Returns:
        拼接后的完整文本字符串，每页之间用换行分隔
    """
    doc = fitz.open(pdf_path)
    pages_text = []

    for page_num, page in enumerate(doc, start=1):
        # 提取当前页的纯文本
        page_text = page.get_text()
        if page_text.strip():
            pages_text.append(page_text)

    doc.close()
    return "\n".join(pages_text)


def extract_text_with_metadata(pdf_path: str) -> List[Dict]:
    """
    逐页提取文本，每页附带页码元信息

    Args:
        pdf_path: PDF文件的绝对路径

    Returns:
        每页一个字典：[{"page_num": 1, "text": "..."}, ...]
    """
    doc = fitz.open(pdf_path)
    pages = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if text.strip():
            pages.append({
                "page_num": page_num,
                "text": text.strip()
            })

    doc.close()
    return pages


def extract_tables_from_pdf(pdf_path: str) -> List[Dict]:
    """
    提取PDF中可能存在的表格数据（通过查找含制表符或连续空格的行）

    Args:
        pdf_path: PDF文件的绝对路径

    Returns:
        每个表格一页，包含页码和原文块
    """
    doc = fitz.open(pdf_path)
    tables = []

    for page_num, page in enumerate(doc, start=1):
        # 获取页面上的文本块（按行分割）
        blocks = page.get_text("blocks")
        table_lines = []

        for block in blocks:
            # 文本块坐标：(x0, y0, x1, y1, text, block_no, block_type)
            text = block[4].strip()
            # 如果文本包含多个空格或短横，可能是表格数据
            if text and ("  " in text or "\t" in text):
                table_lines.append(text)

        if table_lines:
            tables.append({
                "page_num": page_num,
                "table_text": "\n".join(table_lines)
            })

    doc.close()
    return tables


if __name__ == "__main__":
    # 本模块自测：解析PDF并打印基本信息
    from config import PDF_PATH

    full_text = extract_text_from_pdf(PDF_PATH)
    pages = extract_text_with_metadata(PDF_PATH)
    tables = extract_tables_from_pdf(PDF_PATH)

    print(f"PDF总字符数：{len(full_text)}")
    print(f"总页数：{len(pages)}")
    print(f"检测到表格数据页数：{len(tables)}")
    print(f"前200字预览：\n{full_text[:200]}")
