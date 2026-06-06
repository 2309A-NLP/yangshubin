"""
evaluator.py - RAG工单5 多轮对话评估测试运行器
工单编号: 人工智能NLP-RAG-Query理解优化任务
完成需求: 用需求文档的5个问题模拟多轮对话，自动运行+生成评估数据
功能说明: 加载模型→执行5轮多轮对话测试→调用evalreport评分→保存结果
"""

import logging  # 日志
import time     # 计时
from datetime import datetime  # 时间戳

# 导入配置
from config import OUTPUT_DIR, WORK_ORDER_ID, LOG_FORMAT, LOG_DATE_FORMAT

# 设置日志
logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT, level=logging.INFO)
logger = logging.getLogger("evaluator")

# ============================================================
# 5个测试问题（对应需求文档中的多轮对话示例）
# Q1: 完整问题（兴图新科军用收入）
# Q2: 指代消解（"他"→武汉兴图新科）
# Q3: 指代消解（"这个公司"→武汉兴图新科）
# Q4: 省略补全（"那XX呢？"→武汉力源信息）
# Q5: 完整问题（力源信息组织结构图）
# ============================================================
TEST_QUESTIONS = [
    "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
    "他参与的哪个工程荣获了国家科技进步一等奖？",
    "这个公司的法定代表人是谁？",
    "那武汉力源信息技术股份有限公司呢？",
    "武汉力源信息技术股份有限公司组织结构图中，"
    "哪个销售部的销售处最多？有哪些销售处？",
]


def load_models():
    """
    加载所有模型（嵌入、检索、对话管理器）
    返回:
        tuple: (retriever, dialogue_manager, session_id)
    """
    from embedder import BgeM3Embedder
    from milvus_handler import MilvusHandler
    from retriever import Retriever
    from dialogue_manager import DialogueManager

    logger.info("加载模型...")
    start = time.time()

    # 加载BGE-M3嵌入模型
    embedder = BgeM3Embedder()
    embedder.load_model()

    # 连接Milvus并创建集合
    milvus = MilvusHandler()
    milvus.connect()
    milvus.create_collection()

    # 创建检索器和对话管理器
    retriever = Retriever(embedder, milvus)
    dm = DialogueManager()
    sid = dm.create_session("eval_test")

    logger.info(f"模型加载完成! 耗时: {time.time()-start:.1f}秒")
    return retriever, dm, sid


def run_multi_turn_test(retriever, dm, session_id):
    """
    模拟多轮对话，依次执行5个测试问题
    每轮: Query重写 → 检索 → 生成答案 → 记录历史
    返回:
        list: 每轮的results字典
    """
    from query_rewriter import rewrite_query
    from qa_generator import generate_answer

    results = []
    total_time = 0

    print(f"\n{'='*60}")
    print(f"  📋 多轮对话测试 - {len(TEST_QUESTIONS)} 个问题")
    print(f"  🆔 工单: {WORK_ORDER_ID}")
    print(f"{'='*60}")

    for i, question in enumerate(TEST_QUESTIONS):
        turn_start = time.time()
        print(f"\n{'─'*55}")
        print(f"  [{i+1}/{len(TEST_QUESTIONS)}] 用户: {question}")

        # Step1: 获取历史，进行Query重写
        history = dm.get_history(session_id)
        rewritten = rewrite_query(history, question)

        # Step2: 检索相关文档
        search_result = retriever.retrieve(rewritten)

        # Step3: 生成答案
        if search_result["results"]:
            answer = generate_answer(rewritten, search_result["results"])
        else:
            answer = {
                "question": rewritten,
                "answer": "未找到相关的文档内容，请尝试其他问题。",
                "confidence": "low",
                "sources": [],
                "response_time": search_result["search_time"],
            }

        # Step4: 记录对话历史
        dm.add_turn(session_id, question, answer["answer"])

        elapsed = time.time() - turn_start
        total_time += elapsed

        # 打印本轮结果
        has_rewrite = rewritten != question
        print(f"  🔄 重写: {'✅ 已重写' if has_rewrite else '⏭️ 无需重写'}")
        if has_rewrite:
            print(f"       → {rewritten}")
        print(f"  🤖 答案: {answer['answer'][:200]}...")
        print(f"  🎯 置信度: {answer['confidence']} | ⏱️ {elapsed:.2f}秒")
        print(f"  📄 来源: {len(answer.get('sources', []))} 个chunks")

        # 保存结果数据
        results.append({
            "turn": i + 1,
            "original_question": question,
            "rewritten_question": rewritten,
            "has_rewrite": has_rewrite,
            "answer": answer["answer"],
            "confidence": answer["confidence"],
            "response_time": round(elapsed, 2),
            "source_count": len(answer.get("sources", [])),
            "sources": answer.get("sources", []),
        })

    print(f"\n{'='*55}")
    print(f"  测试完成! 总耗时: {total_time:.2f}秒")
    print(f"  平均: {total_time/len(TEST_QUESTIONS):.2f}秒/轮")
    print(f"{'='*55}")

    return results


def main():
    """主入口：加载模型 → 多轮测试 → 评分报告 → 保存结果"""
    overall_start = time.time()

    print(f"\n  {'='*55}")
    print(f"  RAG工单5 - 多轮对话评估测试")
    print(f"  工单: {WORK_ORDER_ID}")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  {'='*55}")

    # 第一步：加载模型
    print(f"\n  📦 正在加载模型...")
    retriever, dm, sid = load_models()

    # 第二步：执行多轮对话测试
    print(f"\n  🧪 开始多轮对话测试...")
    results = run_multi_turn_test(retriever, dm, sid)

    # 第三步：LLM评估 + 输出报告
    from evalreport import evaluate_results, print_report, save_results
    print(f"\n  📊 正在用DeepSeek评估答案质量...")
    eval_scores = evaluate_results(results)

    # 第四步：打印报告 + 保存
    print_report(results, eval_scores)
    save_results(results, eval_scores)

    total = time.time() - overall_start
    print(f"\n  🏁 全部完成! 总耗时: {total:.1f}秒")
    print(f"  📁 结果已保存: {OUTPUT_DIR / 'evaluation_results.json'}")


if __name__ == "__main__":
    main()
