"""
run.py - RAG工单5 主运行入口
工单编号: 人工智能NLP-RAG-Query理解优化任务
完成需求: 命令行入口，支持完整流水线/快速Web/命令行测试三种模式
功能说明: argparse解析参数→调用pipeline或快速启动或测试
"""

import logging
import os
import sys
import argparse
# ===== 路径桥接 =====
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ["设计", "研发", "测试", "优化", "部署"]:
    _p = os.path.join(_ROOT, _sub)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# 导入配置
from config import PDF_NAMES, WORK_ORDER_ID, LOG_FORMAT, LOG_DATE_FORMAT

# 设置日志
logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT, level=logging.INFO)
logger = logging.getLogger("run")

logger.info("=" * 55)
logger.info(f"RAG工单5 启动")
logger.info(f"工单编号: {WORK_ORDER_ID}")
logger.info(f"目标PDF: {PDF_NAMES}")
logger.info("=" * 55)


def quick_start():
    """
    快速启动Web服务（不重新解析PDF，假设Milvus已有数据）
    适用于数据库已有数据，只想启动Web的情况
    """
    logger.info("快速模式: 直接启动Web服务")
    from pipeline import step5_start_web
    step5_start_web()


def run_test():
    """
    命令行交互式测试多轮对话
    使用预设的5个测试问题模拟多轮对话
    """
    from embedder import BgeM3Embedder
    from milvus_handler import MilvusHandler
    from retriever import Retriever
    from qa_generator import generate_answer
    from query_rewriter import rewrite_query
    from dialogue_manager import DialogueManager

    logger.info("命令行多轮对话测试模式")

    # 加载模型
    embedder = BgeM3Embedder()
    embedder.load_model()

    milvus = MilvusHandler()
    milvus.connect()
    milvus.create_collection()

    retriever = Retriever(embedder, milvus)
    dm = DialogueManager()
    sid = dm.create_session()

    # 预设5个测试问题（模拟多轮对话的指代消解+省略补全）
    test_questions = [
        "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
        "他参与的哪个工程荣获了国家科技进步一等奖？",
        "这个公司的法定代表人是谁？",
        "那武汉力源信息技术股份有限公司呢？",
        "武汉力源信息技术股份有限公司组织结构图中，"
        "哪个销售部的销售处最多？有哪些销售处？",
    ]

    for q in test_questions:
        print(f"\n{'='*55}")
        print(f"用户: {q}")

        # Query理解重写
        history = dm.get_history(sid)
        rewritten = rewrite_query(history, q)

        # 检索+生成
        sr = retriever.retrieve(rewritten)
        answer = generate_answer(rewritten, sr["results"])

        # 记录对话
        dm.add_turn(sid, q, answer["answer"])

        print(f"重写后: {rewritten}")
        print(f"助手: {answer['answer'][:150]}...")
        print(f"置信度: {answer['confidence']} | 耗时: {answer['response_time']}秒")


def main():
    """
    主函数：解析命令行参数并执行对应模式
    用法:
        python run.py              # 完整流水线 + Web
        python run.py --web-only   # 仅启动Web（假设数据已入库）
        python run.py --test       # 命令行测试多轮对话
    """
    parser = argparse.ArgumentParser(description="RAG工单5 - Query理解优化")
    parser.add_argument("--web-only", action="store_true", help="仅启动Web服务")
    parser.add_argument("--test", action="store_true", help="命令行测试多轮对话")
    args = parser.parse_args()

    if args.web_only:
        quick_start()
    elif args.test:
        run_test()
    else:
        from pipeline import run_pipeline
        run_pipeline()


if __name__ == "__main__":
    main()
