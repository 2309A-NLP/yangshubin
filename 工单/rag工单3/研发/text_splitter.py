"""
文本切分模块 - 将文本和表格内容切分为可控片段
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
"""
import os  # 导入操作系统接口模块
import json  # 导入 JSON 处理模块
import re  # 导入正则表达式模块
from copy import deepcopy  # 导入深拷贝函数
from config import CHUNK_SIZE, CHUNK_OVERLAP, OUTPUT_DIR, log  # 从配置模块导入切分参数和日志函数


def split_text(text: str, chunk_size: int = CHUNK_SIZE,  # 将文本按指定大小切分为重叠片段
               overlap: int = CHUNK_OVERLAP) -> list:  # 重叠大小参数
    """
    将文本按指定大小切分为重叠片段

    Returns:
        [{chunk_index: int, text: str, char_count: int}, ...]
    """
    if not text or len(text) <= chunk_size:  # 如果文本为空或长度不超过块大小
        return [{"chunk_index": 0, "text": text, "char_count": len(text or "")}]  # 直接返回包含整段文本的单个块

    chunks = []  # 初始化块列表
    start = 0  # 起始位置初始化为 0
    index = 0  # 块索引初始化为 0

    while start < len(text):  # 循环切分直到文本末尾
        end = start + chunk_size  # 计算当前块的结束位置
        chunk_text = text[start:end]  # 截取当前块的文本内容

        chunks.append({  # 将当前块加入列表
            "chunk_index": index,  # 块索引
            "text": chunk_text,  # 块文本内容
            "char_count": len(chunk_text),  # 块字符数
        })  # 块添加完成
        index += 1  # 索引递增

        # 移动起始位置（带重叠）
        step = chunk_size - overlap  # 计算滑动步长（块大小减去重叠量）
        if step <= 0:  # 防止步长小于等于 0 导致死循环
            step = chunk_size // 2  # 使用块大小的一半作为步长
        start += step  # 移动起始位置

    log(f"文本切分为 {len(chunks)} 个片段 (chunk_size={chunk_size}, overlap={overlap})", "SPLIT")  # 日志记录切分结果
    return chunks  # 返回切分结果列表


def split_by_sections(sections: list, chunk_size: int = CHUNK_SIZE) -> list:  # 按章节结构切分，保持章节完整性
    """
    按章节结构切分，保持章节完整性

    Args:
        sections: [{level, title, content}, ...] 来自 pdf_parser

    Returns:
        [{chunk_index, text, section_title, section_level, metadata}, ...]
    """
    chunks = []  # 初始化块列表
    index = 0  # 块索引初始化为 0

    for sec in sections:  # 遍历章节列表
        text = "\n".join(sec["content"]).strip()  # 将章节内容拼接为字符串并去除首尾空白
        if not text:  # 如果章节内容为空
            continue  # 跳过该章节

        title = sec["title"] or "无标题"  # 获取章节标题
        level = sec.get("level", 0)  # 获取章节层级

        # 如果文本较短，直接作为一个 chunk
        if len(text) <= chunk_size:  # 如果章节文本长度不超过块大小
            chunks.append({  # 将当前块加入列表
                "chunk_index": index,  # 块索引
                "text": f"{'#' * level} {title}\n{text}",
                "section_title": title,
                "section_level": level,
                "char_count": len(text),
            })  # 块添加完成
            index += 1  # 索引递增
        else:  # 长文本需要按段落进一步切分
            # 长文本按段落切分
            paragraphs = text.split("\n\n")  # 按空行分割段落
            current_text = f"{'#' * level} {title}\n"  # 初始化当前块文本
            for para in paragraphs:  # 遍历段落
                para = para.strip()  # 去除段落首尾空白
                if not para:  # 跳过空段落
                    continue  # 跳过该章节
                if len(current_text) + len(para) > chunk_size and len(current_text) > 200:  # 如果添加段落会超长且当前块已有足够内容
                    chunks.append({  # 将当前块加入列表
                        "chunk_index": index,  # 块索引
                        "text": current_text,
                        "section_title": title,
                        "section_level": level,
                        "char_count": len(current_text),
                    })  # 块添加完成
                    index += 1  # 索引递增
                    current_text = f"{'#' * level} {title} ({index})\n{para}\n"
                else:
                    current_text += para + "\n\n"  # 将段落追加到当前块

            if len(current_text.strip()) > 50:  # 如果剩余文本长度超过 50 字符
                chunks.append({  # 将当前块加入列表
                    "chunk_index": index,  # 块索引
                    "text": current_text,
                    "section_title": title,
                    "section_level": level,
                    "char_count": len(current_text),
                })  # 块添加完成
                index += 1  # 索引递增

    log(f"按章节切分为 {len(chunks)} 个片段", "SPLIT")
    return chunks  # 返回切分结果列表


def merge_text_and_table_chunks(text_chunks: list, table_blocks: list) -> list:  # 合并文本和表格片段，表格优先保留完整结构
    """
    合并文本和表格片段，表格优先保留完整结构

    Returns:
        [{chunk_index, text, type: text|table, source_pdf, metadata}, ...]
    """
    merged = []  # 初始化合并列表

    # 先加入所有文本片段
    for tc in text_chunks:  # 遍历文本块列表
        text = tc["text"]
        # 从文本内容中提取 source_pdf（来自 <!-- source: filename.pdf --> 注释）
        source_pdf = ""
        m = re.search(r'<!-- source:\s*([^>]+)\s*-->', text)
        if m:
            source_pdf = m.group(1).strip()

        merged.append({  # 添加元素
            "chunk_index": len(merged),  # 块索引（基于当前合并列表长度）
            "text": text,  # 文本内容
            "type": "text",  # 标记为文本类型
            "section_title": tc.get("section_title", ""),  # 章节标题
            "section_level": tc.get("section_level", 0),  # 章节层级
            "source_pdf": source_pdf,  # 来源 PDF 文件名
            "metadata": {"source_pdf": source_pdf} if source_pdf else {},  # 元数据
        })  # 块添加完成

    # 再加入表格片段（用单独的索引标识）
    for tb in table_blocks:  # 遍历表格块列表
        meta = tb.get("metadata", {}).copy()  # 复制元数据
        source_pdf = tb.get("metadata", {}).get("source_pdf", "")
        merged.append({  # 添加元素
            "chunk_index": len(merged),  # 块索引（基于当前合并列表长度）
            "text": tb["text"],  # 表格文本内容
            "type": "table",  # 标记为表格类型
            "table_index": tb.get("table_index", 0),  # 表格索引
            "source_pdf": source_pdf,  # 来源 PDF 文件名
            "metadata": meta,  # 表格元数据
        })  # 块添加完成

    log(f"合并后共 {len(merged)} 个片段 (文本{len(text_chunks)}+表格{len(table_blocks)})", "SPLIT")
    return merged


def save_chunks_json(chunks: list, filename: str = "chunks.json"):  # 保存切分结果到 JSON 文件
    """保存切分结果到 JSON"""
    out_path = os.path.join(OUTPUT_DIR, filename)  # 构造输出文件路径
    with open(out_path, "w", encoding="utf-8") as f:  # 以写模式打开文件
        json.dump(chunks, f, ensure_ascii=False, indent=2)  # 序列化为 JSON 并写入
    log(f"切分结果已保存: {out_path} ({len(chunks)} 个片段)", "SPLIT")  # 日志记录


def save_chunks_text(chunks: list, filename: str = "chunks.txt"):  # 保存切分结果为可读文本
    """保存切分结果为可读文本"""
    out_path = os.path.join(OUTPUT_DIR, filename)  # 构造输出文件路径
    with open(out_path, "w", encoding="utf-8") as f:  # 以写模式打开文件
        for c in chunks:  # 遍历所有块
            f.write(f"{'='*60}\n")
            f.write(f"片段 #{c['chunk_index']} | 类型: {c.get('type', 'text')}\n")
            if c.get("section_title"):  # 如果有章节标题
                f.write(f"章节: {c['section_title']}\n")
            if c.get("source_pdf"):  # 如果有来源 PDF
                f.write(f"来源: {c['source_pdf']}\n")
            f.write(f"{'='*60}\n")
            f.write(c["text"])
            f.write("\n\n")
    log(f"切分可读文本已保存: {out_path}", "SPLIT")


if __name__ == "__main__":  # 主程序入口（测试用）
    # 测试
    test = "这是一段测试文本。" * 200  # 生成测试用长文本
    chunks = split_text(test)  # 测试切分函数
    print(f"切分成 {len(chunks)} 个片段")  # 打印切分结果
    save_chunks_json(chunks)  # 保存测试结果到 JSON 文件
