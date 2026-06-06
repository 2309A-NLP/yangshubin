# -*- coding: utf-8 -*-
"""
主入口 —— 编排所有模块，提供统一的命令行接口
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统

用法：
  python main.py build       — 构建知识库（解析PDF + 生成向量 + 存入Milvus）
  python main.py query       — 交互式问答（命令行）
  python main.py eval        — 运行评估（RAG vs 纯LLM）
  python main.py web         — 启动Flask Web界面
"""

import sys
import os

# ===== 路径桥接：将所有子目录加入 sys.path =====
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ["设计", "研发", "测试", "优化", "部署"]:
    _p = os.path.join(_ROOT, _sub)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# 确保能找到同级模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_banner():
    """打印欢迎信息"""
    print("""
╔══════════════════════════════════════════════╗
║     RAG PDF 问答系统                          ║
║     基于 BGE-M3 + DeepSeek + Milvus           ║
║     工单编号：人工智能NLP-RAG-基于PDF文档的问答系统  ║
╚══════════════════════════════════════════════╝
    """)


def cmd_build():
    """构建知识库"""
    from database_builder import build_knowledge_base
    build_knowledge_base()


def cmd_query():
    """交互式问答"""
    from database_builder import ask_single_question

    print("进入交互问答模式，输入 'quit' 退出\n")
    while True:
        q = input("问题 > ").strip()
        if not q:
            continue
        if q.lower() in ("quit", "exit", "q"):
            break
        result = ask_single_question(q)
        print(f"\n🔍 RAG回答:\n{result['rag_answer']}")


def cmd_eval():
    """运行RAG评估"""
    from evaluator import RAGEvaluator, TEST_QUESTIONS
    from retriever import Retriever
    from llm_qa import LLMQA

    retriever = Retriever()
    llm = LLMQA()

    def rag_func(question):
        """RAG 模式回答函数"""
        context = retriever.retrieve_context(question)
        answer = llm.answer_with_context(question, context)
        return answer, context

    def llm_func(question):
        """纯 LLM 模式回答函数"""
        return llm.answer_without_context(question)

    evaluator = RAGEvaluator()
    results = evaluator.run_evaluation(TEST_QUESTIONS, rag_func, llm_func)
    evaluator.print_report(results)


def cmd_web():
    """启动Web界面"""
    from app import app
    print("启动 Web 界面...")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)


def cmd_all():
    """一键执行：构建 + Web"""
    cmd_build()
    print("\n知识库构建完成，启动 Web 界面...\n")
    cmd_web()


def main():
    """主函数：解析命令行参数并分发"""
    print_banner()

    if len(sys.argv) < 2:
        print("用法: python main.py [build|query|eval|web|all]")
        print("  build  — 构建知识库（PDF→切分→嵌入→Milvus）")
        print("  query  — 交互式问答（命令行）")
        print("  eval   — RAG vs 纯LLM 评估")
        print("  web    — 启动 Web 界面 (http://localhost:5000)")
        print("  all    — 构建知识库 + 启动 Web 界面")
        sys.exit(0)

    command = sys.argv[1].lower()

    commands = {
        "build": cmd_build,
        "query": cmd_query,
        "eval": cmd_eval,
        "web": cmd_web,
        "all": cmd_all,
    }

    if command in commands:
        commands[command]()
    else:
        print(f"未知命令: {command}")
        print("可用命令: build, query, eval, web, all")


if __name__ == "__main__":
    main()
