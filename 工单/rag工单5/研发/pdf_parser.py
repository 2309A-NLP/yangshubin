"""
pdf_parser.py - RAG工单5 PDF解析模块
工单编号: 人工智能NLP-RAG-Query理解优化任务
完成需求: 解析双PDF（力源信息招股说明书+兴图新科招股说明书）
功能说明: 使用PyMuPDF提取文本+图片，按页处理，返回结构化数据
"""

import fitz        # PyMuPDF，用于PDF解析
import os          # 文件路径操作
import logging     # 日志记录
from pathlib import Path  # 跨平台路径

# 导入配置项
from config import (
    PDF_PATHS, PDF_NAMES,            # PDF文件路径和名称
    IMAGE_MIN_WIDTH, IMAGE_MIN_HEIGHT, IMAGE_MIN_SIZE,  # 图片过滤参数
    LOG_FORMAT, LOG_DATE_FORMAT, PROJECT_DIR  # 日志和项目目录
)

# 设置日志记录器
logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT, level=logging.INFO)
logger = logging.getLogger("pdf_parser")

# 图片输出目录
EXTRACTED_IMAGES_DIR = PROJECT_DIR / "extracted_images"
EXTRACTED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def extract_pdf(pdf_path, pdf_name):
    """
    解析单个PDF文件，提取文本和图片
    参数:
        pdf_path: PDF文件绝对路径
        pdf_name: PDF文件名（用于标识来源）
    返回:
        dict: 包含pages(每页文本), images(图片信息),
              total_pages, total_images, total_text
    """
    # 检查PDF文件是否存在
    if not os.path.exists(pdf_path):
        logger.error(f"PDF文件不存在: {pdf_path}")
        return None

    # 打开PDF文件
    logger.info(f"解析PDF: {pdf_name}")
    doc = fitz.open(pdf_path)
    total_pages = len(doc)  # 获取总页数
    logger.info(f"总页数: {total_pages}")

    pages_data = []      # 存储每页的文本数据
    images_data = []     # 存储提取的图片数据
    all_text = []        # 存储全部文本（用换行符拼接）
    img_global_index = 0 # 全局图片索引计数器

    # 逐页遍历PDF文档
    for page_num in range(total_pages):
        page = doc[page_num]  # 获取当前页对象

        # 提取当前页的纯文本内容
        page_text = page.get_text()
        all_text.append(page_text)

        # 提取当前页中嵌入的图片
        img_list = page.get_images(full=True)
        page_image_count = 0  # 当前页图片计数

        # 遍历当前页的所有图片资源
        for img_info in img_list:
            xref = img_info[0]       # 图片在PDF中的引用编号
            img_width = img_info[2]  # 图片宽度（像素）
            img_height = img_info[3] # 图片高度（像素）

            # 过滤过小的图片（图标、装饰元素等）
            if img_width < IMAGE_MIN_WIDTH or img_height < IMAGE_MIN_HEIGHT:
                continue

            # 提取图片的二进制数据
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]  # 图片二进制内容
            img_ext = base_image["ext"]      # 图片扩展名（jpg/png等）

            # 过滤过小的文件（避免提取二维码、小图标）
            if len(img_bytes) < IMAGE_MIN_SIZE:
                continue

            # 生成图片文件名：来源PDF_页码_序号.扩展名
            img_filename = (
                f"{pdf_name.replace('.pdf','')}"
                f"_p{page_num+1:04d}_i{img_global_index:04d}.{img_ext}"
            )
            # 保存图片到磁盘
            img_path = os.path.join(EXTRACTED_IMAGES_DIR, img_filename)
            with open(img_path, "wb") as f:
                f.write(img_bytes)

            # 记录图片的元数据信息
            images_data.append({
                "page_num": page_num + 1,   # 所在页码（1-indexed）
                "img_index": img_global_index, # 全局序号
                "filename": img_filename,   # 文件名
                "path": img_path,           # 绝对路径
                "width": img_width,         # 宽度
                "height": img_height,       # 高度
                "size": len(img_bytes),     # 文件大小
                "source_pdf": pdf_name,     # 来源PDF
            })
            img_global_index += 1
            page_image_count += 1

        # 记录当前页的基本信息
        pages_data.append({
            "page_num": page_num + 1,
            "text": page_text,
            "image_count": page_image_count,
            "source_pdf": pdf_name,
        })

        # 每50页输出一次进度，方便监控
        if (page_num + 1) % 50 == 0:
            logger.info(f"已处理 {page_num+1}/{total_pages} 页, "
                        f"共提取 {len(images_data)} 张图片")

    # 关闭PDF文档释放资源
    doc.close()

    # 返回结构化的解析结果
    return {
        "pages": pages_data,          # 每页文本+元数据
        "images": images_data,        # 所有图片信息
        "total_pages": total_pages,   # 总页数
        "total_images": len(images_data),  # 总图片数
        "total_text": "\n".join(all_text), # 全部文本
        "pdf_name": pdf_name,         # PDF文件名
    }


def extract_all_pdfs():
    """
    解析所有配置的PDF文件（遍历PDF_PATHS列表）
    返回:
        dict: 合并后的完整数据（pages/images/total_text等）
    """
    all_pages = []
    all_images = []
    all_text = []
    total_pages = 0
    total_images = 0

    # 逐个解析每个PDF并合并结果
    for pdf_path, pdf_name in zip(PDF_PATHS, PDF_NAMES):
        result = extract_pdf(pdf_path, pdf_name)
        if result:
            all_pages.extend(result["pages"])
            all_images.extend(result["images"])
            all_text.append(result["total_text"])
            total_pages += result["total_pages"]
            total_images += result["total_images"]

    logger.info(f"双PDF解析完成! 共 {total_pages} 页, {total_images} 张图片")
    return {
        "pages": all_pages,
        "images": all_images,
        "total_pages": total_pages,
        "total_images": total_images,
        "total_text": "\n".join(all_text),
    }


if __name__ == "__main__":
    """单独测试PDF解析功能"""
    result = extract_all_pdfs()
    if result:
        print(f"总页数: {result['total_pages']}")
        print(f"总图片: {result['total_images']} 张")
        print(f"总文本: {len(result['total_text'])} 字符")
