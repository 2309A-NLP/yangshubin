"""
text_chunker.py - RAG工单4 文本分块模块
工单编号: 人工智能NLP-RAG-图像内容解析及检索优化
功能: 将PDF文本和图像描述文本切分为固定大小的文本块，
      带重叠窗口，便于后续向量化检索
"""

import logging
import json
import os
import re

# 导入配置
from config import CHUNK_SIZE, CHUNK_OVERLAP, OUTPUT_DIR, PDF_FILES, LOG_FORMAT, LOG_DATE_FORMAT

# 默认PDF名称
PDF_NAMES = list(PDF_FILES.keys())
PDF_NAME = PDF_NAMES[0]

# 设置日志
logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT, level=logging.INFO)
logger = logging.getLogger("text_chunker")


def split_text_into_chunks(text, chunk_size=None, overlap=None):
    """
    将长文本按指定大小切分为块（带重叠窗口）
    参数:
        text: 输入文本
        chunk_size: 每块最大字符数
        overlap: 块间重叠字符数
    返回:
        list: [{"content": str, "chunk_index": int}, ...]
    """
    # 使用配置中的默认值
    chunk_size = chunk_size or CHUNK_SIZE
    overlap = overlap or CHUNK_OVERLAP
    
    # 按句号、问号、感叹号、换行符分割段落
    # 先按换行分割，再合并成块
    paragraphs = re.split(r'\n+', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        # 如果当前段落加上去不超过块大小，直接加
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += para + "\n"
        else:
            # 当前块已满，保存
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            # 新块从当前段落开始（带重叠）
            if overlap > 0 and chunks:
                # 从上一个块的末尾取重叠部分
                last_chunk = chunks[-1]
                overlap_text = last_chunk[-overlap:] if len(last_chunk) > overlap else last_chunk
                current_chunk = overlap_text + "\n" + para + "\n"
            else:
                current_chunk = para + "\n"
    
    # 保存最后一块
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    logger.info(f"文本分块完成: {len(chunks)} 块 (块大小={chunk_size}, 重叠={overlap})")
    
    # 为每个块添加索引
    result = []
    for i, chunk in enumerate(chunks):
        result.append({
            "content": chunk,
            "chunk_index": i,
        })
    
    return result


def build_chunks_with_images(text_result, image_descriptions):
    """
    将文本块和图片描述合并，构建完整的文档块列表
    参数:
        text_result: pdf_parser返回的文本提取结果
        image_descriptions: image_describer返回的图片描述列表
    返回:
        list: [{
            "content": str,          # 块内容
            "chunk_index": int,       # 块索引
            "source_pdf": str,        # 来源PDF
            "page_num": int,          # 来源页码
            "has_image": bool,        # 是否包含图片描述
            "image_file": str,        # 关联的图片文件名（如有）
        }, ...]
    """
    logger.info("开始构建文档块（文本+图片描述）")

    all_chunks = []

    # ---- 处理文本页 ----
    if text_result and "pages" in text_result:
        for page in text_result["pages"]:
            page_num = page["page_num"]
            page_text = page["text"]

            # 页面来源PDF（如果有）
            source_pdf = page.get("source_pdf", "招股说明书2.pdf")

            # 将页面文本切块
            page_chunks = split_text_into_chunks(page_text)

            for chunk in page_chunks:
                all_chunks.append({
                    "content": chunk["content"],
                    "chunk_index": len(all_chunks),
                    "source_pdf": source_pdf,
                    "page_num": page_num,
                    "has_image": False,
                    "image_file": "",
                })
    
    # ---- 插入图片描述块 ----
    if image_descriptions:
        for img_desc in image_descriptions:
            # 构建包含图片描述的文本块
            img_text = (
                f"[图片内容] (来源: 第{img_desc['page_num']}页, "
                f"图片文件: {img_desc['filename']})\n"
                f"图片描述: {img_desc['description']}"
            )
            
            # 获取图片来源PDF
            img_source_pdf = img_desc.get("source_pdf", PDF_NAME)
            
            all_chunks.append({
                "content": img_text,
                "chunk_index": len(all_chunks),
                "source_pdf": img_source_pdf,
                "page_num": img_desc["page_num"],
                "has_image": True,
                "image_file": img_desc["filename"],
            })
    
    logger.info(f"文档块构建完成! 共 {len(all_chunks)} 块 "
                f"(文本块 + 图片描述块)")
    
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
    
    # 准备保存的数据（限制content长度，但保留完整数据）
    save_data = []
    for chunk in chunks:
        save_data.append({
            "chunk_index": chunk["chunk_index"],
            "content_preview": chunk["content"][:100] + "..." if len(chunk["content"]) > 100 else chunk["content"],
            "content_length": len(chunk["content"]),
            "source_pdf": chunk["source_pdf"],
            "page_num": chunk["page_num"],
            "has_image": chunk["has_image"],
            "image_file": chunk["image_file"],
        })
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"分块结果已保存: {output_path}")
    return output_path


if __name__ == "__main__":
    """单独测试分块功能"""
    test_text = "这是第一段。这是第二段。这是第三段。\n这是新段落。"
    chunks = split_text_into_chunks(test_text, chunk_size=20, overlap=5)
    for c in chunks:
        print(f"块 {c['chunk_index']}: {c['content'][:50]}...")
