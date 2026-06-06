"""
text_chunker.py - RAG工单5 文本分块模块
工单编号: 人工智能NLP-RAG-Query理解优化任务
完成需求: 将PDF长文本切分为固定大小的块，带重叠窗口
功能说明: 支持按段落切分、块间重叠、为每个块附加来源元数据
"""

import logging  # 日志记录
import json     # JSON序列化保存分块结果
import os       # 文件路径操作
import re       # 正则表达式，用于按换行分割段落

# 导入配置项
from config import (
    CHUNK_SIZE, CHUNK_OVERLAP,  # 分块大小和重叠窗口
    OUTPUT_DIR, LOG_FORMAT, LOG_DATE_FORMAT  # 输出路径和日志格式
)

# 设置日志记录器
logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT, level=logging.INFO)
logger = logging.getLogger("text_chunker")


def split_text(text, chunk_size=None, overlap=None):
    """
    将长文本按指定大小切分为块，带重叠窗口
    参数:
        text: 输入的长文本字符串
        chunk_size: 每块最大字符数（默认用配置值300）
        overlap: 块间重叠字符数（默认用配置值50）
    返回:
        list: [{"content": str, "index": int}, ...]
    """
    # 使用配置参数的默认值
    chunk_size = chunk_size or CHUNK_SIZE
    overlap = overlap or CHUNK_OVERLAP

    # 按换行符分割成段落，去除空段落
    paragraphs = re.split(r'\n+', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    current_chunk = ""  # 当前正在构建的块

    # 逐段向当前块中添加内容
    for para in paragraphs:
        if len(current_chunk) + len(para) < chunk_size:
            # 当前块还能容纳这段文本
            current_chunk += para + "\n"
        else:
            # 当前块已满，先保存再开始新块
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            # 新块开头带上上一块末尾的重叠内容
            if overlap > 0 and chunks:
                last_chunk = chunks[-1]
                # 取上一块末尾的overlap个字符
                overlap_text = (
                    last_chunk[-overlap:]
                    if len(last_chunk) > overlap
                    else last_chunk
                )
                current_chunk = overlap_text + "\n" + para + "\n"
            else:
                current_chunk = para + "\n"

    # 保存最后一块
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    logger.info(f"分块完成: {len(chunks)} 块 (大小={chunk_size}, 重叠={overlap})")

    # 为每个块添加索引编号
    return [{"content": c, "index": i} for i, c in enumerate(chunks)]


def build_chunks(pdf_result):
    """
    将PDF解析结果中的所有文本切块，附加来源元数据
    参数:
        pdf_result: pdf_parser返回的解析结果（含pages列表）
    返回:
        list: [{"content", "index", "source_pdf", "page_num"}, ...]
    """
    logger.info("开始构建文档块")
    all_chunks = []

    # 逐页处理文本，每页独立分块
    for page in pdf_result["pages"]:
        page_text = page["text"]
        source_pdf = page.get("source_pdf", "招股说明书2.pdf")
        page_num = page["page_num"]

        # 对当前页文本进行分块
        page_chunks = split_text(page_text)

        # 为每个块添加来源元数据
        for chunk in page_chunks:
            all_chunks.append({
                "content": chunk["content"],     # 块文本内容
                "index": len(all_chunks),        # 全局索引
                "source_pdf": source_pdf,         # 来源PDF文件名
                "page_num": page_num,             # 来源页码
            })

    logger.info(f"文档块构建完成! 共 {len(all_chunks)} 块")
    return all_chunks


def save_chunks(chunks):
    """
    将分块结果保存到output目录
    参数:
        chunks: 分块数据列表
    返回:
        str: 保存的文件路径
    """
    output_path = os.path.join(OUTPUT_DIR, "chunks.json")
    # 只保存预览信息，不保存完整文本（省空间）
    save_data = []
    for c in chunks:
        save_data.append({
            "index": c["index"],
            "preview": c["content"][:80] + "..." if len(c["content"]) > 80
                       else c["content"],
            "length": len(c["content"]),
            "source_pdf": c["source_pdf"],
            "page_num": c["page_num"],
        })
    # 写入JSON文件
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    logger.info(f"分块结果已保存: {output_path}")
    return output_path


if __name__ == "__main__":
    """单独测试分块功能"""
    test = "第一段。第二段。第三段。\n新段落。"
    chunks = split_text(test, chunk_size=20, overlap=5)
    for c in chunks:
        print(f"块 {c['index']}: {c['content'][:40]}")
