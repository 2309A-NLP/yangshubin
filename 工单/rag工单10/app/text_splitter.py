"""
模块功能: 文本分块模块
将长文本按指定大小切分成有重叠的文本块（Chunk）
优先按段落切分保证语义完整，段落过长时按字符强制切分
工单编号: 人工智能NLP-RAG-金融问答系统部署
"""

import re              # 正则表达式库，用于段落分割
import logging         # 日志记录模块
from typing import List, Dict  # 类型提示

from app.config import config  # 全局配置，读取分块参数

# 获取当前模块的日志记录器
logger = logging.getLogger("text_splitter")


def split_by_paragraphs(text: str) -> List[str]:
    """按段落拆分文本，识别空行作为段落分隔符

    Args:
        text: 原始文本字符串

    Returns:
        段落列表，已过滤空白段落
    """
    # 按两个及以上换行符（即段落间的空行）分割文本
    paragraphs: List[str] = re.split(r"\n\s*\n", text.strip())
    # 过滤掉纯空白或空的段落
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    return paragraphs


def split_text(text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
    """将文本切分为有重叠的文本块

    切分策略:
    1. 优先按段落切分，保持语义单元完整
    2. 超长段落（超过 chunk_size）按字符强制切分
    3. 相邻块之间保留重叠部分维持上下文连贯

    Args:
        text: 待切分的原始文本
        chunk_size: 每个块的最大字符数，默认从配置读取 (300)
        chunk_overlap: 块间重叠字符数，默认从配置读取 (50)

    Returns:
        切分后的文本块列表
    """
    # 使用配置中的默认值
    if chunk_size is None:
        chunk_size = config.CHUNK_SIZE
    if chunk_overlap is None:
        chunk_overlap = config.CHUNK_OVERLAP
    # 空文本直接返回空列表
    if not text or not text.strip():
        return []
    # 第一步：按段落拆分原始文本
    paragraphs: List[str] = split_by_paragraphs(text)
    chunks: List[str] = []
    # 当前正在构建的块内容
    current_chunk: str = ""
    # 第二步：遍历每个段落，决定如何合并或切分
    for para in paragraphs:
        # 情况1：段落本身超长，需要强制字符级切分
        if len(para) > chunk_size:
            # 先保存当前累积的块
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            # 对超长段落按 chunk_size 滑动窗口切分
            start: int = 0
            while start < len(para):
                end: int = start + chunk_size
                chunk: str = para[start:end]
                chunks.append(chunk)
                # 移动起始位置时保留重叠部分
                start = end - chunk_overlap
        else:
            # 情况2：段落正常，尝试追加到当前块
            if current_chunk and (len(current_chunk) + len(para) + 1 > chunk_size):
                # 追加后会超限，保存当前块并开始新块
                chunks.append(current_chunk)
                current_chunk = ""
            # 将段落添加到当前块（添加换行符分隔）
            if current_chunk:
                current_chunk += "\n" + para
            else:
                current_chunk = para
    # 第三步：处理最后一个累积的块
    if current_chunk:
        chunks.append(current_chunk)
    # 记录分块统计信息
    logger.info(f"文本分块完成: 输入 {len(text)} 字符 -> 输出 {len(chunks)} 个块")
    return chunks


def split_documents(data_dir: str) -> List[Dict]:
    """加载文档并进行分块处理（一站式函数）

    从指定目录加载所有 PDF，对每篇文档的文本进行分块，
    返回包含文件名、块序号和块内容的结构化列表。

    Args:
        data_dir: PDF 文件所在目录路径

    Returns:
        分块结果列表，每个元素包含 filename, chunk_index, text, char_count
    """
    from app.document_loader import load_documents
    # 加载指定目录下的所有 PDF 文档
    documents: list = load_documents(data_dir)
    all_chunks: List[Dict] = []
    # 对每篇文档进行分块处理
    for doc in documents:
        filename: str = doc.get("filename", "unknown")
        text: str = doc.get("text", "")
        # 跳过空文档
        if not text:
            continue
        # 对文档文本执行分块
        chunks: List[str] = split_text(text)
        # 为每个块创建结构化记录
        for idx, chunk in enumerate(chunks):
            all_chunks.append({
                "filename": filename,     # 来源文档文件名
                "chunk_index": idx,       # 块在文档中的序号
                "text": chunk,            # 块文本内容
                "char_count": len(chunk), # 块字符数
            })
    logger.info(f"文档分块完成: 共 {len(all_chunks)} 个块")
    return all_chunks
