# -*- coding: utf-8 -*-
"""
数据初始化模块 —— 将PDF解析、切分、嵌入、入库一站式完成
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统

处理完成后，数据会导出到 output/ 文件夹供查看
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

from config import PDF_PATH, TOP_K
from pdf_parser import extract_text_from_pdf, extract_text_with_metadata
from text_splitter import split_text_simple
from embedding_model import EmbeddingModel
from vector_store import VectorStore

# ============================================================
# 日志工具
# ============================================================
_LOG_FILE = None


def log(msg: str, step: str = "INFO"):
    """带时间戳的日志输出，同时打印到终端和日志文件"""
    global _LOG_FILE
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] [{step}] {msg}"
    print(line)

    if _LOG_FILE is not None:
        _LOG_FILE.write(line + "\n")
        _LOG_FILE.flush()


def log_start(step: str, desc: str):
    """开始一个步骤"""
    log("─" * 50, step)
    log(f"▶ 开始: {desc}", step)


def log_end(step: str, desc: str):
    """完成一个步骤"""
    log(f"✓ 完成: {desc}", step)


# ============================================================
# 数据导出
# ============================================================
def export_data(text: str, chunks: list, output_dir: str = None):
    """
    将处理好的数据导出到文件夹，方便查看

    Args:
        text: PDF全文文本
        chunks: 切分后的文本块列表
        output_dir: 输出目录（默认 output/）
    """
    if output_dir is None:
        output_dir = str(Path(__file__).resolve().parent / "output")

    os.makedirs(output_dir, exist_ok=True)
    log("=" * 50, "导出")

    # 1. 导出完整PDF文本
    txt_path = os.path.join(output_dir, "01_全文提取.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"PDF全文提取 — 共 {len(text)} 字符\n")
        f.write("=" * 60 + "\n\n")
        f.write(text)
    log(f"  导出全文: {txt_path} ({len(text)} 字符)", "导出")

    # 2. 导出文本块总览（JSON）
    chunks_json = []
    for i, c in enumerate(chunks, 1):
        chunks_json.append({
            "序号": i,
            "chunk_id": c["chunk_id"],
            "起始位置": c["start_pos"],
            "结束位置": c["end_pos"],
            "长度": len(c["text"]),
            "内容预览": c["text"][:100] + ("..." if len(c["text"]) > 100 else "")
        })

    json_path = os.path.join(output_dir, "02_文本块总览.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(chunks_json, f, ensure_ascii=False, indent=2)
    log(f"  导出总览: {json_path} ({len(chunks)} 条)", "导出")

    # 3. 导出文本块明细（可读格式，每块用分隔线隔开）
    detail_path = os.path.join(output_dir, "03_文本块明细.txt")
    with open(detail_path, "w", encoding="utf-8") as f:
        f.write(f"文本块明细 — 共 {len(chunks)} 块\n")
        f.write("=" * 60 + "\n\n")
        for i, c in enumerate(chunks, 1):
            f.write(f"{'=' * 60}\n")
            f.write(f"【第 {i} 块】ID: {c['chunk_id']}\n")
            f.write(f"位置: [{c['start_pos']} ~ {c['end_pos']}]  长度: {len(c['text'])} 字符\n")
            f.write("-" * 60 + "\n")
            f.write(c["text"])
            f.write("\n\n")
    log(f"  导出明细: {detail_path}", "导出")

    # 4. 导出统计信息
    stats_path = os.path.join(output_dir, "00_处理统计.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("PDF 处理统计\n")
        f.write("=" * 40 + "\n")
        f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"PDF总字符数: {len(text)}\n")
        f.write(f"文本块数量: {len(chunks)}\n")
        f.write(f"每块大小: 512 字符\n")
        f.write(f"块间重叠: 64 字符\n")
        avg_len = sum(len(c["text"]) for c in chunks) // len(chunks)
        f.write(f"平均块长度: {avg_len} 字符\n")
        f.write(f"最长块: {max(len(c['text']) for c in chunks)} 字符\n")
        f.write(f"最短块: {min(len(c['text']) for c in chunks)} 字符\n")
    log(f"  导出统计: {stats_path}", "导出")

    log(f"✅ 全部数据已导出到: {output_dir}", "导出")
    print(f"\n  💡 去这里查看处理好的数据: {output_dir}\\")
    print(f"     00_处理统计.txt   — 统计信息")
    print(f"     01_全文提取.txt   — PDF完整文字")
    print(f"     02_文本块总览.json — 所有文本块列表")
    print(f"     03_文本块明细.txt  — 逐块查看内容")


# ============================================================
# 知识库构建
# ============================================================
def build_knowledge_base(pdf_path: str = PDF_PATH,
                         drop_old: bool = True) -> Dict:
    """
    构建知识库：解析PDF → 切分 → 嵌入 → 存入Milvus → 导出数据

    Args:
        pdf_path: PDF文件路径
        drop_old: 是否覆盖已存在的Milvus集合

    Returns:
        处理统计信息
    """
    global _LOG_FILE

    # 创建日志文件
    log_dir = Path(__file__).resolve().parent / "output"
    os.makedirs(str(log_dir), exist_ok=True)
    _LOG_FILE = open(str(log_dir / "构建日志.txt"), "w", encoding="utf-8")

    log("=" * 60, "启动")
    log("  RAG 知识库构建流程", "启动")
    log("=" * 60, "启动")

    overall_start = time.time()

    # ============================================================
    # Step 1: 解析PDF
    # ============================================================
    log_start("Step1", "解析PDF文档")
    step_start = time.time()

    pdf_name = os.path.basename(pdf_path)
    log(f"  PDF文件: {pdf_name}", "Step1")
    log(f"  文件大小: {os.path.getsize(pdf_path) / 1024:.1f} KB", "Step1")

    text = extract_text_from_pdf(pdf_path)
    pages = extract_text_with_metadata(pdf_path)

    log(f"  提取文本: {len(text)} 字符", "Step1")
    log(f"  PDF页数: {len(pages)} 页", "Step1")
    log(f"  耗时: {time.time() - step_start:.2f}s", "Step1")
    log_end("Step1", "PDF解析完成")

    # ============================================================
    # Step 2: 文本切分
    # ============================================================
    log_start("Step2", "文本切分")
    step_start = time.time()

    chunks = split_text_simple(text)

    log(f"  切分参数: chunk_size=512, overlap=64", "Step2")
    log(f"  切分块数: {len(chunks)}", "Step2")
    avg_len = sum(len(c["text"]) for c in chunks) // len(chunks)
    log(f"  平均长度: {avg_len} 字符/块", "Step2")
    log(f"  耗时: {time.time() - step_start:.2f}s", "Step2")
    log_end("Step2", "文本切分完成")

    # ============================================================
    # Step 3: 生成嵌入向量
    # ============================================================
    log_start("Step3", "生成BGE-M3嵌入向量")
    step_start = time.time()

    emb_model = EmbeddingModel()
    log(f"  模型路径: {emb_model.model_path}", "Step3")
    log(f"  计算设备: {emb_model.device}", "Step3")

    texts_to_encode = [c["text"] for c in chunks]
    log(f"  待编码文本: {len(texts_to_encode)} 条", "Step3")

    embeddings = emb_model.encode(texts_to_encode)

    log(f"  向量维度: {embeddings.shape[1]}", "Step3")
    log(f"  向量形状: {embeddings.shape}", "Step3")
    log(f"  耗时: {time.time() - step_start:.2f}s", "Step3")
    log_end("Step3", "嵌入向量生成完成")

    # ============================================================
    # Step 4: 存入Milvus
    # ============================================================
    log_start("Step4", "存入Milvus向量数据库")
    step_start = time.time()

    store = VectorStore()
    log(f"  Milvus地址: localhost:19530", "Step4")
    log(f"  集合名称: {store.collection_name}", "Step4")

    store.create_collection(drop_if_exists=drop_old)
    log(f"  集合已就绪", "Step4")

    inserted = store.insert_embeddings(chunks, embeddings.tolist())
    log(f"  成功入库: {inserted} 条", "Step4")
    log(f"  耗时: {time.time() - step_start:.2f}s", "Step4")
    store.close()
    log_end("Step4", "Milvus入库完成")

    # ============================================================
    # 导出数据到文件夹
    # ============================================================
    export_data(text, chunks)

    # ============================================================
    # 完成
    # ============================================================
    total_elapsed = time.time() - overall_start
    result = {
        "status": "success",
        "total_chars": len(text),
        "total_pages": len(pages),
        "total_chunks": len(chunks),
        "inserted": inserted,
        "elapsed_seconds": round(total_elapsed, 2)
    }

    log("=" * 60, "完成")
    log(f"✅ 知识库构建完成！总耗时: {total_elapsed:.2f}秒", "完成")
    log(f"   PDF: {len(pages)} 页 / {len(text)} 字符", "完成")
    log(f"   文本块: {len(chunks)} 块 → 入库: {inserted} 条", "完成")
    log(f"   导出目录: {log_dir}", "完成")
    log("=" * 60, "完成")

    if _LOG_FILE is not None:
        _LOG_FILE.close()
        _LOG_FILE = None

    return result


if __name__ == "__main__":
    build_knowledge_base()
