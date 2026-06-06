"""
run.py - RAG工单4 主运行入口
工单编号: 人工智能NLP-RAG-图像内容解析及检索优化
功能: 串联所有模块的完整流程：
      1. 解析PDF → 2. 提取图片并描述 → 3. 文本分块
      → 4. 向量化 → 5. 存入Milvus → 6. 启动Web服务
      支持处理多个PDF（力源信息 + 兴图新科）
"""

import logging
import json
import os
import sys
import time
import argparse
# ===== 路径桥接 =====
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ["设计", "研发", "测试", "优化", "部署"]:
    _p = os.path.join(_ROOT, _sub)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import (
    PDF_FILES, PDF_PATH, PDF_NAME, OUTPUT_DIR, WORK_ORDER_ID,
    TEST_QUESTIONS, LOG_FORMAT, LOG_DATE_FORMAT
)

logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT, level=logging.INFO)
logger = logging.getLogger("run")

logger.info("=" * 60)
logger.info(f"RAG工单4 启动")
logger.info(f"工单编号: {WORK_ORDER_ID}")
logger.info(f"可处理PDF: {list(PDF_FILES.keys())}")
logger.info("=" * 60)

# 全局变量（跨步骤共享）
_global_embedder = None
_global_milvus = None
_global_pipeline = None


def step1_parse_pdf(pdf_path, pdf_name):
    """步骤1: 解析单个PDF，提取文本和图片"""
    logger.info(f"[步骤1/6] 开始解析PDF: {pdf_name}...")
    start = time.time()

    from pdf_parser import extract_pdf
    result = extract_pdf(pdf_path)

    if result is None:
        logger.error(f"PDF解析失败: {pdf_path}")
        return None, None, None

    elapsed = time.time() - start
    logger.info(f"[步骤1/6] 完成! {pdf_name}: {result['total_pages']} 页, "
                f"{result['total_images']} 张图片, 耗时 {elapsed:.2f}秒")

    return result["pages"], result["images"], result["total_text"]


def step2_describe_images(images_data):
    """步骤2: 使用多模态模型描述图片内容"""
    logger.info("[步骤2/6] 开始描述图片内容...")
    start = time.time()

    if not images_data:
        logger.warning("没有图片需要描述，跳过步骤2")
        return []

    from image_describer import describe_all_images
    descriptions = describe_all_images(images_data, max_images=50)

    elapsed = time.time() - start
    logger.info(f"[步骤2/6] 完成! 描述了 {len(descriptions)} 张图片, 耗时 {elapsed:.2f}秒")
    return descriptions


def step3_chunk_text(pages_data, image_descriptions):
    """步骤3: 文本分块 + 图片描述合并"""
    logger.info("[步骤3/6] 开始文本分块...")
    start = time.time()

    from text_chunker import build_chunks_with_images, save_chunks

    text_result = {
        "pages": pages_data,
        "total_text": "\n".join([p["text"] for p in pages_data]),
    }

    chunks = build_chunks_with_images(text_result, image_descriptions)
    save_chunks(chunks)

    elapsed = time.time() - start
    logger.info(f"[步骤3/6] 完成! 共 {len(chunks)} 个文本块, 耗时 {elapsed:.2f}秒")
    return chunks


def step4_create_embeddings(chunks):
    """步骤4: BGE-M3向量化"""
    logger.info("[步骤4/6] 开始生成向量嵌入...")
    start = time.time()

    from embedder import create_embeddings_for_chunks
    embeddings = create_embeddings_for_chunks(chunks)

    elapsed = time.time() - start
    logger.info(f"[步骤4/6] 完成! {len(embeddings['dense_vectors'])} 个向量, "
                f"耗时 {elapsed:.2f}秒")
    return embeddings


def step5_store_to_milvus(embeddings):
    """步骤5: 向量存入Milvus"""
    logger.info("[步骤5/6] 开始存入Milvus向量数据库...")
    start = time.time()

    from milvus_handler import MilvusHandler

    handler = MilvusHandler()
    handler.connect()
    handler.create_collection(drop_if_exists=True)

    ids = handler.insert_data(
        vectors=embeddings["dense_vectors"],
        chunk_texts=embeddings["chunk_texts"],
        chunk_metas=embeddings["chunk_metas"],
    )

    elapsed = time.time() - start
    logger.info(f"[步骤5/6] 完成! 存入 {len(ids)} 条向量, 耗时 {elapsed:.2f}秒")
    return handler


def step6_start_web(host="127.0.0.1", port=5000):
    """步骤6: 启动Flask Web服务"""
    logger.info("[步骤6/6] 启动Web服务...")

    from app import get_flask_app
    app = get_flask_app()

    logger.info(f"Web服务已启动: http://{host}:{port}")
    print(f"\n{'='*50}")
    print(f"  Web服务已启动: http://{host}:{port}")
    print(f"  工单编号: {WORK_ORDER_ID}")
    print(f"{'='*50}\n")

    app.run(host=host, port=port, debug=False)


def run_pipeline(skip_images=False, max_images=20):
    """完整流水线"""
    pipeline_start = time.time()

    # ---- 处理两份PDF ----
    all_chunks = []
    all_embeddings_vectors = []
    all_embeddings_texts = []
    all_embeddings_metas = []

    for pdf_name, pdf_path in PDF_FILES.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"处理PDF: {pdf_name}")
        logger.info(f"{'='*50}")

        # 步骤1: 解析PDF
        pages_data, images_data, _ = step1_parse_pdf(pdf_path, pdf_name)
        if pages_data is None:
            continue

        # 步骤2: 描述图片
        image_descriptions = []
        if not skip_images and images_data:
            if len(images_data) > max_images:
                sorted_imgs = sorted(images_data, key=lambda x: x["width"] * x["height"], reverse=True)
                images_data = sorted_imgs[:max_images]
            image_descriptions = step2_describe_images(images_data)
        else:
            logger.info("跳过图片描述步骤")

        # 步骤3: 分块
        chunks = step3_chunk_text(pages_data, image_descriptions)

        # 步骤4: 向量化
        embeddings = step4_create_embeddings(chunks)

        all_chunks.extend(chunks)
        all_embeddings_vectors.extend(embeddings["dense_vectors"])
        all_embeddings_texts.extend(embeddings["chunk_texts"])
        all_embeddings_metas.extend(embeddings["chunk_metas"])

    # 合并所有向量
    merged_embeddings = {
        "dense_vectors": all_embeddings_vectors,
        "chunk_texts": all_embeddings_texts,
        "chunk_metas": all_embeddings_metas,
    }

    # 步骤5: 入库
    handler = step5_store_to_milvus(merged_embeddings)

    # 保存全局引用
    global _global_embedder, _global_milvus, _global_pipeline
    from embedder import BgeM3Embedder
    from milvus_handler import MilvusHandler
    from retriever import RetrievalPipeline

    # 这些已经在create_embeddings_for_chunks和Milvus中加载过了
    # 重新获取pipeline引用
    _global_embedder = BgeM3Embedder()
    _global_embedder.load_model()
    _global_milvus = handler
    _global_pipeline = RetrievalPipeline(_global_embedder, _global_milvus)

    total_elapsed = time.time() - pipeline_start
    logger.info(f"流水线全部完成! 总耗时: {total_elapsed:.2f}秒")

    step6_start_web()


def quick_test():
    """快速启动Web（假设数据已在Milvus中）"""
    logger.info("快速模式：直接启动Web服务（不重新解析PDF）")
    step6_start_web()


def run_qa_test():
    """运行全部16个问题的问答测试"""
    from embedder import BgeM3Embedder
    from milvus_handler import MilvusHandler
    from retriever import RetrievalPipeline
    from qa_generator import generate_answer, batch_qa

    logger.info("命令行问答测试模式 - 全部16个问题")

    # 加载模型
    embedder = BgeM3Embedder()
    embedder.load_model()

    # 连接Milvus
    milvus = MilvusHandler()
    milvus.connect()
    milvus.create_collection()

    # 创建检索流水线
    pipeline = RetrievalPipeline(embedder, milvus)

    # 运行全部16个问题
    results = batch_qa(TEST_QUESTIONS, pipeline)

    # 打印摘要
    correct = 0
    for r in results:
        status = "CORRECT" if r["confidence"] == "high" else "LOW"
        if r["confidence"] == "high":
            correct += 1
        print(f"[{status}] Q{r['id']}: {r['answer'][:80]}...")
        print(f"        置信度={r['confidence']}, 耗时={r['response_time']}秒")

    total = len(results)
    accuracy = correct / total * 100
    print(f"\n{'='*50}")
    print(f"测试完成! {correct}/{total} 条高置信度 = {accuracy:.1f}%")
    print(f"{'='*50}")

    # 保存评估结果
    eval_path = os.path.join(OUTPUT_DIR, "batch_eval_results.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"评估结果已保存: {eval_path}")


def run_single_pdf_test():
    """测试单个PDF的快速验证（不启动Web）"""
    from embedder import BgeM3Embedder
    from milvus_handler import MilvusHandler
    from retriever import RetrievalPipeline
    from qa_generator import generate_answer

    logger.info("单PDF快速测试模式")

    # 处理第一个PDF（力源信息）
    pdf_name = "招股说明书2.pdf"
    pdf_path = PDF_FILES[pdf_name]

    pages_data, images_data, _ = step1_parse_pdf(pdf_path, pdf_name)
    if pages_data is None:
        return

    from image_describer import describe_all_images
    image_descriptions = describe_all_images(images_data, max_images=20)

    chunks = step3_chunk_text(pages_data, image_descriptions)
    embeddings = step4_create_embeddings(chunks)

    handler = step5_store_to_milvus(embeddings)

    # 测试力源信息的问题
    embedder = BgeM3Embedder()
    embedder.load_model()
    pipeline = RetrievalPipeline(embedder, handler)

    liyuan_questions = [q for q in TEST_QUESTIONS if q["id"] <= 6]
    from qa_generator import batch_qa
    results = batch_qa(liyuan_questions, pipeline)

    for r in results:
        print(f"\nQ{r['id']}: {r['question']}")
        print(f"A: {r['answer']}")
        print(f"   置信度={r['confidence']}, 耗时={r['response_time']}秒")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG工单4 - PDF图像内容解析及检索优化")
    parser.add_argument("--skip-images", action="store_true", help="跳过图片描述步骤")
    parser.add_argument("--web-only", action="store_true", help="仅启动Web服务（不重新解析PDF）")
    parser.add_argument("--test", action="store_true", help="命令行问答测试（全部16个问题）")
    parser.add_argument("--test-single", action="store_true", help="单PDF快速测试")
    parser.add_argument("--max-images", type=int, default=20, help="最多描述的图片数")

    args = parser.parse_args()

    if args.web_only:
        quick_test()
    elif args.test:
        run_qa_test()
    elif args.test_single:
        run_single_pdf_test()
    else:
        run_pipeline(skip_images=args.skip_images, max_images=args.max_images)
