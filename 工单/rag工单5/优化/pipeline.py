"""
pipeline.py - RAG工单5 流水线步骤模块
工单编号: 人工智能NLP-RAG-Query理解优化任务
完成需求: 串联完整流水线（解析PDF→分块→向量化→入库→启动Web）
功能说明: 每个步骤独立函数，供run.py和main.py调用
"""

import logging  # 日志
import time     # 计时

# 导入配置
from config import PDF_NAMES, OUTPUT_DIR, WORK_ORDER_ID, LOG_FORMAT, LOG_DATE_FORMAT

# 设置日志
logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT, level=logging.INFO)
logger = logging.getLogger("pipeline")


def step1_parse_pdfs():
    """
    步骤1: 解析双PDF（力源信息+兴图新科）
    使用pdf_parser模块提取文本和图片
    返回: 合并后的PDF解析结果
    """
    logger.info("[步骤1/5] 解析双PDF...")
    start = time.time()

    from pdf_parser import extract_all_pdfs
    result = extract_all_pdfs()

    if result is None or not result.get("pages"):
        logger.error("PDF解析失败")
        return None

    elapsed = time.time() - start
    logger.info(f"[步骤1/5] 完成! {result['total_pages']}页, "
                f"{result['total_images']}张图片, {elapsed:.1f}秒")
    return result


def step2_chunk_text(pdf_result):
    """
    步骤2: 文本分块（按段落切分+重叠窗口）
    参数: pdf_result: 步骤1的解析结果
    返回: 文本块列表
    """
    logger.info("[步骤2/5] 文本分块...")
    start = time.time()

    from text_chunker import build_chunks, save_chunks
    chunks = build_chunks(pdf_result)
    save_chunks(chunks)

    elapsed = time.time() - start
    logger.info(f"[步骤2/5] 完成! {len(chunks)} 个文本块, {elapsed:.2f}秒")
    return chunks


def step2_5_describe_images(chunks):
    """
    步骤2.5: 生成图片描述并合并到文本块中
    使用MiMo多模态API描述PDF中的图片
    参数: chunks: 文本块列表（会被扩展）
    返回: 合并后的chunks列表（包含图片描述chunks）
    """
    logger.info("[步骤2.5/5] 生成图片描述...")
    start = time.time()

    from image_describer import describe_all_images
    image_chunks = describe_all_images()

    if image_chunks:
        # 图片描述chunks排在文本chunks后面
        base_index = len(chunks)
        for i, ic in enumerate(image_chunks):
            ic["index"] = base_index + i
        chunks.extend(image_chunks)
        logger.info(f"[步骤2.5/5] 合并 {len(image_chunks)} 个图片描述")
    else:
        logger.warning("[步骤2.5/5] 未生成图片描述，使用纯文本chunks")

    elapsed = time.time() - start
    logger.info(f"[步骤2.5/5] 完成! 总chunks: {len(chunks)}, 耗时: {elapsed:.1f}秒")
    return chunks


def step3_create_embeddings(chunks):
    """
    步骤3: BGE-M3向量化（将文本块转为稠密向量）
    参数: chunks: 文本块列表
    返回: 嵌入结果（含dense_vectors, chunk_texts, chunk_metas）
    """
    logger.info("[步骤3/5] BGE-M3向量化...")
    start = time.time()

    from embedder import create_embeddings
    embeddings = create_embeddings(chunks)

    elapsed = time.time() - start
    vec_count = len(embeddings["dense_vectors"])
    vec_dim = len(embeddings["dense_vectors"][0]) if vec_count > 0 else 0
    logger.info(f"[步骤3/5] 完成! {vec_count} 个向量, 维度{vec_dim}, {elapsed:.1f}秒")
    return embeddings


def step4_store_to_milvus(embeddings):
    """
    步骤4: 将向量存入Milvus数据库
    参数: embeddings: 步骤3的嵌入结果
    返回: MilvusHandler实例
    """
    logger.info("[步骤4/5] 存入Milvus...")
    start = time.time()

    from milvus_handler import MilvusHandler
    handler = MilvusHandler()
    handler.connect()
    handler.create_collection(drop_if_exists=True)
    ids = handler.insert(
        vectors=embeddings["dense_vectors"],
        texts=embeddings["chunk_texts"],
        metas=embeddings["chunk_metas"],
    )

    elapsed = time.time() - start
    logger.info(f"[步骤4/5] 完成! {len(ids)} 条向量, {elapsed:.1f}秒")
    return handler


def step5_start_web(host="127.0.0.1", port=5000):
    """
    步骤5: 启动多轮对话Web服务
    参数:
        host: 监听地址
        port: 监听端口
    """
    logger.info("[步骤5/5] 启动Web服务...")

    from app import get_flask_app
    app = get_flask_app()

    print(f"\n{'='*50}")
    print(f"  🌐 多轮对话服务: http://{host}:{port}")
    print(f"  🆔 工单编号: {WORK_ORDER_ID}")
    print(f"  💡 支持指代消解、省略补全、多轮对话")
    print(f"  🖼️  图片描述已集成（组织结构图可检索）")
    print(f"  {'='*50}\n")

    app.run(host=host, port=port, debug=False)


def run_pipeline():
    """完整流水线：步骤1→2→2.5→3→4→5"""
    pipeline_start = time.time()

    pdf_result = step1_parse_pdfs()
    if pdf_result is None:
        return

    chunks = step2_chunk_text(pdf_result)
    chunks = step2_5_describe_images(chunks)
    embeddings = step3_create_embeddings(chunks)
    step4_store_to_milvus(embeddings)

    total = time.time() - pipeline_start
    logger.info(f"流水线全部完成! 总耗时: {total:.1f}秒")
    step5_start_web()
