"""
pdf_parser.py - RAG工单4 PDF解析模块
工单编号: 人工智能NLP-RAG-图像内容解析及检索优化
功能: 使用PyMuPDF提取PDF中的文本内容和图片，
      按页处理，返回结构化数据
"""

import fitz  # PyMuPDF
import os
import logging
from pathlib import Path

from config import (
    EXTRACTED_IMAGES_DIR,
    IMAGE_MIN_WIDTH, IMAGE_MIN_HEIGHT, IMAGE_MIN_SIZE,
    PDF_FILES,
    LOG_FORMAT, LOG_DATE_FORMAT
)

logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT, level=logging.INFO)
logger = logging.getLogger("pdf_parser")


def extract_pdf(pdf_path=None):
    """
    提取PDF中的文本和图片
    参数:
        pdf_path: PDF文件路径，默认为招股说明书2.pdf
    返回:
        dict: {
            "pages": [...],
            "images": [...],
            "total_pages": int,
            "total_text": str,
            "pdf_name": str,
        }
    """
    # 使用默认路径或传入路径
    if pdf_path is None:
        pdf_path = list(PDF_FILES.values())[0]

    # 确定PDF名称
    pdf_name = os.path.basename(pdf_path)
    for name, path in PDF_FILES.items():
        if path == pdf_path or os.path.basename(path) == os.path.basename(pdf_path):
            pdf_name = name
            break

    # 检查PDF是否存在
    if not os.path.exists(pdf_path):
        logger.error(f"PDF文件不存在: {pdf_path}")
        return None

    logger.info(f"开始解析PDF: {pdf_name} ({pdf_path})")

    # 打开PDF文档
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    logger.info(f"PDF总页数: {total_pages}")

    pages_data = []
    images_data = []
    all_text = []
    img_global_index = 0

    # 找出包含图表的关键页码（组织结构图/流程图所在页）用于渲染
    import re as _re
    key_pages_for_render = set()
    for pn in range(total_pages):
        txt = doc[pn].get_text()
        if _re.search(r'组织结构|组织架构|销售部|增长率|IC.?市场|应用结构', txt):
            key_pages_for_render.add(pn)

    # 逐页遍历
    for page_num in range(total_pages):
        page = doc[page_num]

        # 提取文本
        page_text = page.get_text()
        all_text.append(page_text)

        # 提取图片
        img_list = page.get_images(full=True)
        page_image_count = 0

        for img_idx, img_info in enumerate(img_list):
            xref = img_info[0]
            img_width = img_info[2]
            img_height = img_info[3]

            # 过滤小图片
            if img_width < IMAGE_MIN_WIDTH or img_height < IMAGE_MIN_HEIGHT:
                continue

            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            img_ext = base_image["ext"]

            if len(img_bytes) < IMAGE_MIN_SIZE:
                continue

            img_filename = f"page{page_num+1:04d}_img{img_global_index:04d}.{img_ext}"
            img_path = os.path.join(EXTRACTED_IMAGES_DIR, img_filename)

            with open(img_path, "wb") as f:
                f.write(img_bytes)

            images_data.append({
                "page_num": page_num + 1,
                "img_index": img_global_index,
                "filename": img_filename,
                "path": img_path,
                "width": img_width,
                "height": img_height,
                "size": len(img_bytes),
                "source_pdf": pdf_name,
            })

            img_global_index += 1
            page_image_count += 1

        # 页面渲染：关键图表页整体转为图片（捕捉矢量图/组织结构图）
        if page_num in key_pages_for_render:
            pix = page.get_pixmap(dpi=150)
            render_filename = f"render_page{page_num+1:04d}.png"
            render_path = os.path.join(EXTRACTED_IMAGES_DIR, render_filename)
            pix.save(render_path)

            render_size = os.path.getsize(render_path)
            if render_size > IMAGE_MIN_SIZE:
                images_data.append({
                    "page_num": page_num + 1,
                    "img_index": img_global_index,
                    "filename": render_filename,
                    "path": render_path,
                    "width": pix.width,
                    "height": pix.height,
                    "size": render_size,
                    "source_pdf": pdf_name,
                })
                img_global_index += 1
                page_image_count += 1
                logger.info(f"  渲染图表页 {page_num+1} ({pix.width}x{pix.height})")

        # 记录页面信息
        pages_data.append({
            "page_num": page_num + 1,
            "text": page_text,
            "image_count": page_image_count,
            "source_pdf": pdf_name,
        })

        if (page_num + 1) % 50 == 0:
            logger.info(f"已处理 {page_num+1}/{total_pages} 页，当前共提取 {len(images_data)} 张图片")

    doc.close()

    result = {
        "pages": pages_data,
        "images": images_data,
        "total_pages": total_pages,
        "total_images": len(images_data),
        "total_text": "\n".join(all_text),
        "pdf_name": pdf_name,
    }

    logger.info(f"PDF解析完成! 共 {total_pages} 页, 提取 {len(images_data)} 张有效图片")
    return result


if __name__ == "__main__":
    result = extract_pdf()
    if result:
        print(f"\n=== 解析结果摘要 ===")
        print(f"总页数: {result['total_pages']}")
        print(f"有效图片: {result['total_images']} 张")
        print(f"总文本长度: {len(result['total_text'])} 字符")
        print(f"前500字符预览:")
        print(result['total_text'][:500])
