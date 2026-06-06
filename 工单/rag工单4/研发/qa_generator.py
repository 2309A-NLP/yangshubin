"""
qa_generator.py - RAG工单4 问答生成模块
工单编号: 人工智能NLP-RAG-图像内容解析及检索优化
功能: 基于检索到的上下文，调用小米MiMo API生成准确答案，
      支持答案可信度评估和来源追溯
"""

import logging
import json
import os
import time
import re

from config import (
    MIMO_API_KEY, MIMO_BASE_URL, MIMO_TEXT_MODEL,
    OUTPUT_DIR, WORK_ORDER_ID, LOG_FORMAT, LOG_DATE_FORMAT
)

logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT, level=logging.INFO)
logger = logging.getLogger("qa_generator")


def build_prompt(question, context_results, lang="auto"):
    """构建问答提示词（支持中英文）"""
    if lang == "auto":
        lang = "zh" if re.search(r'[\u4e00-\u9fff]', question) else "en"

    context_parts = []
    for i, r in enumerate(context_results):
        source_tag = ""
        if r.get("has_image"):
            img_name = r.get("image_file", "")
            source_tag = f" [图片来源: {img_name}]"
        context_parts.append(
            f"--- 上下文 {i+1} (第{r['page_num']}页{source_tag}) ---\n{r['content']}"
        )
    context_str = "\n\n".join(context_parts)

    if lang == "zh":
        prompt = f"""你是一个专业的PDF文档问答助手。请根据提供的上下文信息，准确回答用户的问题。

工单编号: {WORK_ORDER_ID}

要求:
1. 只基于给定的上下文回答问题，不要编造信息
2. 如果上下文不足以回答问题，请明确说明
3. 回答要简洁准确，必要时引用具体数据
4. 提及信息来源的页码
5. 如果问题涉及图片/图表，请结合图片描述中的信息回答

=== 上下文信息 ===
{context_str}

=== 用户问题 ===
{question}

=== 回答 ===
"""
    else:
        prompt = f"""You are a professional PDF document Q&A assistant. Answer accurately based on the provided context.

Work Order: {WORK_ORDER_ID}

Requirements:
1. Answer ONLY based on the given context
2. If context is insufficient, state it clearly
3. Be concise and accurate, cite specific data
4. Mention source page numbers
5. Incorporate image description info when relevant

=== Context ===
{context_str}

=== Question ===
{question}

=== Answer ===
"""
    return prompt


def generate_answer(question, context_results, max_retries=2, lang="auto"):
    """调用小米MiMo API生成答案"""
    start_time = time.time()
    logger.info(f"生成答案: {question[:50]}...")

    if lang == "auto":
        lang = "zh" if re.search(r'[\u4e00-\u9fff]', question) else "en"

    prompt = build_prompt(question, context_results, lang=lang)

    sources = []
    for r in context_results:
        sources.append({
            "page_num": r["page_num"],
            "has_image": r.get("has_image", False),
            "image_file": r.get("image_file", ""),
            "score": float(r.get("score", 0)),
        })

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            from openai import OpenAI

            client = OpenAI(api_key=MIMO_API_KEY, base_url=MIMO_BASE_URL)
            system_msg = "你是一个专业的PDF文档问答助手，回答要简洁准确。" if lang == "zh" else "You are a professional PDF document Q&A assistant."

            response = client.chat.completions.create(
                model=MIMO_TEXT_MODEL,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=1024,
                timeout=30,
            )

            # MiMo模型可能把答案放在reasoning_content或content中
            msg = response.choices[0].message
            answer = msg.content or ""
            reasoning = getattr(msg, "reasoning_content", None) or ""
            if not answer.strip() and reasoning.strip():
                answer = reasoning

            answer = answer.strip()

            # 判断可信度
            no_answer_keywords = ["无法", "没有提供", "不足以", "未找到", "上下文不包含", "未提及"]
            if len(answer) < 10 or any(kw in answer[:30] for kw in no_answer_keywords):
                confidence = "low"
            elif len(context_results) >= 2:
                confidence = "high"
            else:
                confidence = "medium"

            elapsed = time.time() - start_time
            logger.info(f"答案生成完成! 耗时: {elapsed:.2f}秒, 置信度: {confidence}")

            return {
                "question": question,
                "answer": answer,
                "confidence": confidence,
                "sources": sources,
                "response_time": round(elapsed, 2),
            }

        except Exception as e:
            last_error = str(e)
            logger.warning(f"API调用失败 (第{attempt+1}次): {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)

    elapsed = time.time() - start_time
    logger.error(f"生成答案失败: {last_error}")
    return {
        "question": question,
        "answer": f"抱歉，生成答案时出错: {last_error}",
        "confidence": "low",
        "sources": sources,
        "response_time": round(elapsed, 2),
    }


def batch_qa(questions, retrieval_pipeline, save=True):
    """批量问答"""
    results = []
    total = len(questions)

    for idx, q_item in enumerate(questions):
        qid = q_item["id"]
        question = q_item["question"]

        logger.info(f"处理问题 [{idx+1}/{total}] (id={qid}): {question[:40]}...")
        retrieval_result = retrieval_pipeline.retrieve(question)
        answer = generate_answer(question, retrieval_result["results"])
        answer["id"] = qid
        results.append(answer)

        if idx < total - 1:
            time.sleep(0.5)

    if save:
        output_path = os.path.join(OUTPUT_DIR, "qa_results.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"问答结果已保存: {output_path}")

    return results


if __name__ == "__main__":
    test_question = "武汉力源信息技术股份有限公司本次发行股数是多少？"
    test_context = [{
        "content": "本次发行股数为2,000万股，占发行后总股本的比例为25%",
        "page_num": 1,
        "has_image": False,
        "score": 0.95,
    }]
    result = generate_answer(test_question, test_context)
    print(f"问题: {result['question']}")
    print(f"答案: {result['answer']}")
    print(f"置信度: {result['confidence']}")
