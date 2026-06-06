"""
qa_generator.py - RAG工单5 问答生成模块
工单编号: 人工智能NLP-RAG-Query理解优化任务
完成需求: 基于检索上下文调用LLM生成答案，支持中英文双语
功能说明: 构建提示词→调用DeepSeek API→解析返回→置信度评估
"""

import logging  # 日志记录
import time     # 计时
import re       # 正则检测语言

# 导入配置
from config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL,
    OUTPUT_DIR, WORK_ORDER_ID, LOG_FORMAT, LOG_DATE_FORMAT
)

# 设置日志
logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT, level=logging.INFO)
logger = logging.getLogger("qa_generator")


def get_msg_content(msg):
    """获取消息内容，MiMo模型答案在reasoning_content里（content永远为空）"""
    rc = getattr(msg, 'reasoning_content', '') or ''
    if rc.strip():
        return rc.strip()
    c = msg.content or ''
    return c.strip()


def build_prompt(question, context_results, lang="auto"):
    """
    构建问答提示词（支持中英文自动识别）
    参数:
        question: 用户问题
        context_results: 检索到的文档块列表
        lang: "auto"自动检测，"zh"中文，"en"英文
    返回:
        str: 完整的提示词
    """
    # 自动检测问题语言
    if lang == "auto":
        lang = "zh" if re.search(r'[\u4e00-\u9fff]', question) else "en"

    # 拼接检索到的上下文块
    context_parts = []
    for i, r in enumerate(context_results):
        # 显示来源页码
        source = f"第{r['page_num']}页" if lang == "zh" else f"Page {r['page_num']}"
        context_parts.append(
            f"--- {'上下文' if lang=='zh' else 'Context'} {i+1} ({source}) ---\n"
            f"{r['content']}"
        )
    context_str = "\n\n".join(context_parts)

    # 中文提示词（含工单编号）
    if lang == "zh":
        return f"""你是一个专业的PDF文档问答助手。请根据上下文信息准确回答问题。直截了当回答，不说思考过程。

工单编号: {WORK_ORDER_ID}

要求：
1. 只基于给定的上下文回答，不要编造信息
2. 如果上下文不足，请明确说明
3. 回答要简洁准确，引用具体数据和页码
4. 如果问题涉及图片/图表，结合图片描述回答

=== 上下文信息 ===
{context_str}

=== 用户问题 ===
{question}

=== 回答 ==="""

    # 英文提示词
    return f"""You are a professional PDF Q&A assistant. Answer based on the context.

Work Order: {WORK_ORDER_ID}

Rules:
1. Answer ONLY based on the given context
2. If context is insufficient, state it clearly
3. Be concise and accurate, cite data and page numbers
4. For image/chart questions, use image descriptions

=== Context ===
{context_str}

=== Question ===
{question}

=== Answer ==="""


def generate_answer(question, context_results, max_retries=2):
    """
    调用DeepSeek API生成答案
    参数:
        question: 用户问题（已重写）
        context_results: 检索到的上下文块列表
        max_retries: API失败重试次数
    返回:
        dict: {"question","answer","confidence","sources","response_time"}
    """
    # API Key为空则返回错误信息
    if not LLM_API_KEY:
        return {
            "question": question,
            "answer": "错误: LLM API Key 未配置",
            "confidence": "low", "sources": [], "response_time": 0,
        }

    start = time.time()
    logger.info(f"生成答案: {question[:40]}...")

    # 自动检测语言
    lang = "zh" if re.search(r'[\u4e00-\u9fff]', question) else "en"

    # 构建问答提示词
    prompt = build_prompt(question, context_results, lang=lang)

    # 收集来源信息（页码、分数、来源PDF）
    sources = [{
        "page_num": r["page_num"],
        "score": float(r.get("score", 0)),
        "source_pdf": r.get("source_pdf", ""),
    } for r in context_results]

    # API调用带重试机制
    for attempt in range(max_retries + 1):
        try:
            # 使用OpenAI兼容接口调用MiMo LLM
            from openai import OpenAI
            client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

            # 系统提示词
            system_msg = (
                "你是一个专业的PDF问答助手。直截了当回答，只说答案不说思考过程。"
                if lang == "zh"
                else "You are a PDF Q&A assistant. Answer directly, no reasoning."
            )

            # 调用MiMo API生成答案
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,     # 低温度确保事实性
                max_tokens=1024,     # 答案最大长度
                timeout=30,          # 30秒超时
            )

            # 解析答案文本
            answer = get_msg_content(response.choices[0].message)

            # 根据答案长度和上下文数量判断置信度
            if len(answer) < 10 or "无法" in answer[:20]:
                confidence = "low"           # 答案太短或表示无法回答
            elif len(context_results) >= 2:
                confidence = "high"          # 多个来源，答案可靠
            else:
                confidence = "medium"        # 来源较少

            elapsed = time.time() - start
            logger.info(f"答案生成完成! {elapsed:.1f}秒 置信度:{confidence}")
            return {
                "question": question,
                "answer": answer,
                "confidence": confidence,
                "sources": sources,
                "response_time": round(elapsed, 2),
            }

        except Exception as e:
            logger.warning(f"API调用失败 (第{attempt+1}次): {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)  # 指数退避

    # 所有重试失败，返回兜底信息
    elapsed = time.time() - start
    return {
        "question": question,
        "answer": "抱歉，生成答案时出错，请重试",
        "confidence": "low",
        "sources": sources,
        "response_time": round(elapsed, 2),
    }


if __name__ == "__main__":
    """单独测试问答生成"""
    ctx = [{"content": "注册资本为5,520万元", "page_num": 52, "score": 0.95}]
    r = generate_answer("武汉兴图新科注册资本是多少？", ctx)
    print(f"答案: {r['answer'][:100]}")
