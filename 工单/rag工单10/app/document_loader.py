"""
模块功能: PDF 文档加载器模块
使用 PyMuPDF (fitz) 解析 PDF 文件，提取其中的文本内容
支持批量加载目录下所有 PDF，返回结构化的文档列表供后续处理
工单编号: 人工智能NLP-RAG-金融问答系统部署
"""

import os          # 操作系统接口，用于文件路径判断
import logging     # 日志记录模块
from pathlib import Path  # 跨平台路径处理工具

import fitz        # PyMuPDF 库，专业的 PDF 解析工具

# 获取当前模块的日志记录器
logger = logging.getLogger("document_loader")

# 支持的文档文件扩展名集合
SUPPORTED_EXTENSIONS: tuple = (".pdf",)


def load_pdf(file_path: str) -> str:
    """加载单个 PDF 文件，提取全部页面的文本内容

    Args:
        file_path: PDF 文件的完整路径（支持 Windows 和 Linux 路径）

    Returns:
        提取到的全部文本内容（按页码顺序拼接），失败时返回空字符串
    """
    # 第一步：检查文件是否存在
    if not os.path.isfile(file_path):
        logger.warning(f"文件不存在: {file_path}")
        return ""
    try:
        # 第二步：使用 PyMuPDF 打开 PDF 文档
        doc: fitz.Document = fitz.open(file_path)
        # 用于存储所有页面的文本列表
        all_text: list = []
        # 第三步：遍历每一页提取文本
        for page_num in range(len(doc)):
            page: fitz.Page = doc[page_num]
            # 提取当前页面的纯文本内容
            page_text: str = page.get_text()
            # 过滤空白页面
            if page_text.strip():
                all_text.append(page_text)
        # 第四步：关闭文档释放系统资源，先获取页数再关闭
        page_count = len(doc)
        doc.close()
        # 合并所有页面文本，用换行符分隔
        full_text: str = "\n".join(all_text)
        logger.info(f"PDF 加载成功: {file_path}, {page_count} 页, {len(full_text)} 字符")
        return full_text
    except Exception as e:
        # 捕获解析过程中的各种异常
        logger.error(f"PDF 解析失败: {file_path}, 错误: {e}")
        return ""


def load_documents(data_dir: str) -> list:
    """加载指定目录下的所有 PDF 文档文件

    递归遍历目录，找到所有 PDF 文件并提取文本内容。

    Args:
        data_dir: PDF 文件所在的目录路径

    Returns:
        文档字典列表，每个元素包含 filename, filepath, text, char_count 字段
    """
    # 将字符串路径转为 Path 对象，方便路径操作
    data_path: Path = Path(data_dir)
    # 检查目录是否存在
    if not data_path.exists():
        logger.warning(f"数据目录不存在: {data_dir}")
        return []
    # 存储所有文档的列表
    documents: list = []
    # 递归遍历目录下所有文件
    for file_path in data_path.rglob("*"):
        # 根据文件扩展名判断是否为 PDF
        if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            filename: str = file_path.name
            logger.info(f"开始处理文档: {filename}")
            # 调用 single PDF 加载函数提取文本
            text: str = load_pdf(str(file_path))
            # 只保留有有效文本内容的文档
            if text.strip():
                documents.append({
                    "filename": filename,      # 文档文件名
                    "filepath": str(file_path), # 文档完整路径
                    "text": text,               # 文档文本内容
                    "char_count": len(text),    # 文档字符总数
                })
    # 输出加载统计信息
    logger.info(f"文档加载完成: 共 {len(documents)} 个有效文档")
    return documents


def get_document_stats(documents: list) -> dict:
    """统计已加载文档的基本信息

    Args:
        documents: load_documents 函数返回的文档列表

    Returns:
        包含文档数量、总字符数、文件名列表的统计字典
    """
    # 计算文档数量
    doc_count: int = len(documents)
    # 计算所有文档的总字符数
    total_chars: int = sum(d.get("char_count", 0) for d in documents)
    # 提取所有文档的文件名
    filenames: list = [d.get("filename", "") for d in documents]
    # 返回统计结果字典
    return {
        "doc_count": doc_count,
        "total_chars": total_chars,
        "filenames": filenames,
    }
