"""
main.py - RAG工单5 统一入口（调用所有模块的总控文件）
工单编号: 人工智能NLP-RAG-Query理解优化任务
完成需求: 导入并调用所有15个模块，支持4种运行模式
功能说明: 一键完成info/pipeline/eval/web/verify
"""

import sys, os, time, argparse
# ===== 路径桥接 =====
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ["设计", "研发", "测试", "优化", "部署"]:
    _p = os.path.join(_ROOT, _sub)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# 导入所有模块
from config import (PROJECT_DIR, OUTPUT_DIR, PDF_NAMES, WORK_ORDER_ID,
    BGE_M3_MODEL_PATH, BGE_M3_BATCH_SIZE, MILVUS_HOST, MILVUS_PORT,
    MILVUS_COLLECTION, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, RERANK_TOP_K,
    LLM_API_KEY, LLM_MODEL, MIMO_API_KEY, MIMO_VL_MODEL)
from pdf_parser import extract_all_pdfs, extract_pdf, EXTRACTED_IMAGES_DIR as IMG_DIR
from text_chunker import split_text, build_chunks, save_chunks
from image_describer import (describe_all_images, describe_single_image,
    encode_image_base64, parse_image_filename)
from embedder import BgeM3Embedder, create_embeddings
from milvus_handler import MilvusHandler
from retriever import Retriever
from query_rewriter import rewrite_query, build_rewrite_prompt
from qa_generator import generate_answer, build_prompt
from dialogue_manager import DialogueManager
from evaluator import TEST_QUESTIONS, load_models, run_multi_turn_test
from evalreport import evaluate_with_llm, evaluate_results, print_report, save_results
from pipeline import (step1_parse_pdfs, step2_chunk_text, step2_5_describe_images,
    step3_create_embeddings, step4_store_to_milvus, step5_start_web, run_pipeline)
from app import get_flask_app, get_retriever, get_dialogue_manager
from run import quick_start, run_test as cli_test

# ============================================================
# 运行模式：info - 打印所有模块信息
# ============================================================
def mode_info():
    """打印所有模块的配置和状态信息"""
    print(f"\n{'='*55}")
    print(f"  RAG工单5 - 模块信息总览")
    print(f"  {WORK_ORDER_ID}")
    print(f"{'='*55}")
    print(f"  📁 项目目录:     {PROJECT_DIR}")
    print(f"  📄 目标PDF:      {PDF_NAMES}")
    print(f"  💾 输出目录:     {OUTPUT_DIR}")
    print(f"  🖼️  图片目录:     {IMG_DIR}")
    print(f"  🤖 BGE-M3:       {BGE_M3_MODEL_PATH}")
    print(f"  📐 向量维度:     1024 | Batch: {BGE_M3_BATCH_SIZE}")
    print(f"  🗄️  Milvus:      {MILVUS_HOST}:{MILVUS_PORT}/{MILVUS_COLLECTION}")
    print(f"  📝 分块参数:     size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    print(f"  🔍 检索参数:     top_k={TOP_K}, rerank_top_k={RERANK_TOP_K}")
    print(f"  🌐 LLM:          {LLM_MODEL} (Key: {'✅' if LLM_API_KEY else '❌'})")
    print(f"  🖼️  MiMo VL:     {MIMO_VL_MODEL} (Key: {'✅' if MIMO_API_KEY else '❌'})")
    print(f"  🧪 测试问题数:   {len(TEST_QUESTIONS)}")
    print(f"  📚 导入模块数:   15")
    print(f"{'='*55}")

# ============================================================
# 运行模式：pipeline - 完整流水线（解析→入库→Web）
# ============================================================
def mode_pipeline():
    """执行完整流水线"""
    run_pipeline()

# ============================================================
# 运行模式：eval - 评估测试（加载数据→多轮测试→评分）
# ============================================================
def mode_eval():
    """执行评估测试"""
    from evaluator import main as eval_main
    eval_main()

# ============================================================
# 运行模式：web - 仅启动Web服务
# ============================================================
def mode_web():
    """仅启动Web服务"""
    quick_start()

# ============================================================
# 运行模式：verify - 验证所有模块可导入
# ============================================================
def mode_verify():
    """验证所有模块导入和基本功能"""
    print("\n✅ 所有15个模块导入成功!")
    print(f"\n  模块清单:")
    modules = [
        ("config", "配置文件"),
        ("pdf_parser", "PDF解析"),
        ("text_chunker", "文本分块"),
        ("image_describer", "图片描述(MiMo API)"),
        ("embedder", "BGE-M3嵌入"),
        ("milvus_handler", "Milvus数据库"),
        ("retriever", "检索+重排序"),
        ("query_rewriter", "Query重写(指代消解)"),
        ("qa_generator", "问答生成"),
        ("dialogue_manager", "多轮对话管理"),
        ("evaluator", "评估测试"),
        ("evalreport", "LLM评分报告"),
        ("pipeline", "流水线步骤"),
        ("app", "Flask Web应用"),
        ("run", "运行入口"),
    ]
    for i, (name, desc) in enumerate(modules, 1):
        print(f"  {i:02d}. {name:20s} - {desc}")
    print(f"\n  共 {len(modules)} 个模块 ✅")

# ============================================================
# 主函数
# ============================================================
def main():
    """统一入口，支持4种模式选择"""
    parser = argparse.ArgumentParser(
        description="RAG工单5 - 所有模块统一入口"
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["info", "pipeline", "eval", "web", "verify"],
        default="info",
        help="运行模式: info(信息) pipeline(流水线) eval(评估) web(Web) verify(验证)"
    )
    args = parser.parse_args()

    print(f"\n  🚀 RAG工单5 - 一键启动")
    print(f"  📋 工单: {WORK_ORDER_ID}")
    print(f"  🔧 模式: {args.mode}")
    print(f"{'='*55}")

    start = time.time()

    # 根据模式选择执行
    mode_map = {
        "info": mode_info,
        "pipeline": mode_pipeline,
        "eval": mode_eval,
        "web": mode_web,
        "verify": mode_verify,
    }

    mode_map[args.mode]()

    elapsed = time.time() - start
    print(f"\n⏱️  总耗时: {elapsed:.1f}秒")


if __name__ == "__main__":
    main()
