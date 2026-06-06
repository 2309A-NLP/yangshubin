"""
evalreport.py - RAG工单5 LLM评分与评估报告模块
工单编号: 人工智能NLP-RAG-Query理解优化任务
完成需求: 使用DeepSeek对答案进行4维度评分（相关性/完整性/准确性/流畅性）
功能说明: LLM评分 + 打印表格报告 + 检查响应时间达标 + 保存结果
"""

import logging  # 日志
import json     # JSON解析
import re       # 正则清理JSON
from datetime import datetime  # 时间戳

# 导入配置
from config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL,
    OUTPUT_DIR, WORK_ORDER_ID, LOG_FORMAT, LOG_DATE_FORMAT
)

# 设置日志
logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT, level=logging.INFO)
logger = logging.getLogger("evalreport")


def get_msg_content(msg):
    """获取消息内容，MiMo模型答案在reasoning_content里（content永远为空）"""
    rc = getattr(msg, 'reasoning_content', '') or ''
    if rc.strip():
        return rc.strip()
    c = msg.content or ''
    return c.strip()


def evaluate_with_llm(question, answer, context_hint=""):
    """使用LLM评估单轮答案质量（4维度评分1-10）"""
    if not LLM_API_KEY:
        return {"error": "API Key未配置"}

    # 构建评估提示词
    prompt = f"""你是一个RAG问答系统评估专家。请从以下4个维度对答案进行评分（1-10分），并给出简短说明。

工单编号: {WORK_ORDER_ID}

用户问题: {question}

系统答案: {answer}

{context_hint if context_hint else ""}

评估维度：
1. 相关性 (Relevance) — 答案是否针对问题回答，不跑题
2. 完整性 (Completeness) — 答案是否覆盖了问题的所有方面
3. 准确性 (Accuracy) — 答案中的信息是否准确、不编造
4. 流畅性 (Fluency) — 答案是否通顺、易懂、格式清晰

请以JSON格式输出，不要其他内容：
{{"relevance": {{"score": 8, "comment": "..."}},
  "completeness": {{"score": 7, "comment": "..."}},
  "accuracy": {{"score": 8, "comment": "..."}},
  "fluency": {{"score": 9, "comment": "..."}},
  "overall": {{"score": 8, "comment": "总体评价..."}}}}"""

    try:
        # 调用LLM API进行评分
        from openai import OpenAI
        client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是一个RAG问答评估专家，直截了当输出JSON评分，不说思考过程。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=512,
            timeout=15,
        )
        text = get_msg_content(response.choices[0].message)
        # 清理可能的markdown JSON包裹
        text = re.sub(r'^```json\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        return json.loads(text)
    except Exception as e:
        logger.warning(f"LLM评估失败: {e}")
        return {"error": str(e)}


def evaluate_results(results):
    """
    对每轮测试结果进行LLM评分
    参数:
        results: evaluator返回的测试结果列表
    返回:
        list: 每轮的评分字典列表
    """
    eval_scores = []
    for i, r in enumerate(results):
        print(f"     评估 Q{i+1}... ", end="")
        score = evaluate_with_llm(r["original_question"], r["answer"])
        eval_scores.append(score)
        status = "✅" if "error" not in score else "❌"
        print(f"{status}")
    return eval_scores


def print_report(results, eval_scores):
    """
    打印美观的评估报告表格
    参数:
        results: 测试结果列表
        eval_scores: LLM评分列表
    """
    print(f"\n{'='*65}")
    print(f"  📊 评估报告 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*65}")

    # 打印表头
    print(f"\n  {'轮次':<5} {'重写':<6} {'置信度':<10} {'耗时':<8} {'来源':<6} "
          f"{'相关性':<8} {'完整性':<8} {'准确性':<8} {'流畅性':<8}")
    print(f"  {'-'*65}")

    total_scores = {"relevance": 0, "completeness": 0, "accuracy": 0, "fluency": 0}

    for i, (r, ev) in enumerate(zip(results, eval_scores)):
        rewrite_mark = "✅" if r["has_rewrite"] else "⏭️"
        conf_map = {"high": "🟢高", "medium": "🟡中", "low": "🔴低"}
        conf = conf_map.get(r["confidence"], r["confidence"])

        if "error" not in ev:
            rel = ev.get("relevance", {}).get("score", 0)
            com = ev.get("completeness", {}).get("score", 0)
            acc = ev.get("accuracy", {}).get("score", 0)
            flu = ev.get("fluency", {}).get("score", 0)
            total_scores["relevance"] += rel
            total_scores["completeness"] += com
            total_scores["accuracy"] += acc
            total_scores["fluency"] += flu
        else:
            rel = com = acc = flu = "-"

        print(f"  Q{i+1:<4} {rewrite_mark:<6} {conf:<10} "
              f"{r['response_time']:<6.1f}s {r['source_count']:<6} "
              f"{str(rel):<8} {str(com):<8} {str(acc):<8} {str(flu):<8}")

    # 打印平均分
    n = len(results)
    print(f"  {'-'*65}")
    print(f"  平均分{'':<20} "
          f"{total_scores['relevance']/n:<8.1f} {total_scores['completeness']/n:<8.1f} "
          f"{total_scores['accuracy']/n:<8.1f} {total_scores['fluency']/n:<8.1f}")
    print(f"{'='*65}")

    # 检查响应时间是否达标（≤3秒）
    slow_turns = [r for r in results if r["response_time"] > 3]
    if slow_turns:
        print(f"\n⚠️  有 {len(slow_turns)} 轮响应时间超过3秒")
        for r in slow_turns:
            print(f"    Q{r['turn']}: {r['response_time']:.1f}秒")
    else:
        print(f"\n✅ 所有轮次响应时间均在3秒以内（响应时间达标）")

    # 各轮详细评估
    print(f"\n{'='*65}")
    print(f"  各轮详细评估")
    print(f"{'='*65}")
    for i, (r, ev) in enumerate(zip(results, eval_scores)):
        print(f"\n  【Q{i+1}】{r['original_question'][:60]}")
        if r["has_rewrite"]:
            print(f"     🔄 重写后: {r['rewritten_question']}")
        print(f"     🤖 答案: {r['answer'][:200]}...")
        if "error" not in ev:
            for dim in ["relevance", "completeness", "accuracy", "fluency"]:
                d = ev.get(dim, {})
                print(f"     {'📊' if dim == 'accuracy' else '📝'} {dim}: "
                      f"{d.get('score', '-')}/10 - {d.get('comment', '')}")
        print()


def save_results(results, eval_scores):
    """
    将测试结果和评分保存到output目录
    参数:
        results: 测试结果列表
        eval_scores: LLM评分列表
    返回:
        str: 保存的文件路径
    """
    output_path = OUTPUT_DIR / "evaluation_results.json"
    data = {
        "work_order": WORK_ORDER_ID,
        "test_time": datetime.now().isoformat(),
        "total_questions": len(results),
        "results": results,
        "evaluation": eval_scores,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"评估结果已保存: {output_path}")
    return output_path
