"""
query_rewriter.py - RAG工单5 Query理解重写模块（核心模块）
工单编号: 人工智能NLP-RAG-Query理解优化任务
完成需求: 多轮对话中的指代消解+省略补全
功能说明: 规则+LLM双模式重写方式，优先规则匹配，复杂情况再调MiMo API
"""

import logging  # 日志记录
import re       # 正则表达式，用于规则重写
import time     # 重试延时

# 导入配置
from config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL,
    WORK_ORDER_ID, LOG_FORMAT, LOG_DATE_FORMAT
)

# 设置日志
logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT, level=logging.INFO)
logger = logging.getLogger("query_rewriter")

# 规则重写的公司名映射表（长名在前，精确匹配防重复）
COMPANY_NAMES = [
    "武汉兴图新科电子股份有限公司", "武汉力源信息技术股份有限公司",
    "武汉兴图新科", "武汉力源信息",
    "兴图新科电子股份有限公司", "力源信息技术股份有限公司",
    "兴图新科", "力源信息",
]


def get_last_entity(history):
    """从对话历史中提取最后讨论的公司实体"""
    for turn in reversed(history):
        text = turn.get("content", "")
        for company in COMPANY_NAMES:
            if company in text:
                return company
    return None


def rule_based_rewrite(history, current_question):
    """
    基于规则的Query重写（指代消解+省略补全）
    不需调用LLM，速度快、无额外成本
    返回:
        str: 重写后的完整问题，无需重写返回原问题
    """
    # 检测是否需要重写
    need_rewrite = False
    pronouns = ['他', '她', '它', '这个', '那个', '该', '其', '这', '那', '呢']
    for p in pronouns:
        if p in current_question:
            need_rewrite = True
            break
    if len(current_question) < 10:
        need_rewrite = True
    # 如果无历史，不重写
    if len(history) < 2:
        need_rewrite = False

    if not need_rewrite:
        return current_question, False

    # 获取最后讨论的实体
    entity = get_last_entity(history)
    if not entity:
        return current_question, False

    rewritten = current_question

    # 规则1: 指代消解 "他/她/它/这个公司/该公司" → 具体公司名
    for pronoun in ['这个公司', '该公司', '他', '她', '它']:
        if pronoun in rewritten:
            rewritten = rewritten.replace(pronoun, entity)
            break

    # 规则2: "那XX呢？" 省略补全
    na_match = re.match(r'那(.+?)呢\??', rewritten)
    if na_match:
        target = na_match.group(1)
        last_user_q = ""
        for turn in reversed(history):
            if turn["role"] == "user":
                last_user_q = turn["content"]
                break
        if last_user_q:
            # 找原问题中的公司名，长名优先匹配
            found_company = None
            for company in sorted(COMPANY_NAMES, key=len, reverse=True):
                if company in last_user_q:
                    found_company = company
                    break
            if found_company:
                rewritten = last_user_q.replace(found_company, target, 1)
            else:
                rewritten = f"{target}的相关信息是什么？"

    # 规则3: 如果当前问题很短，可能是省略句，补全上次问题的模板
    if len(current_question) < 10 and current_question == rewritten:
        last_user_q = ""
        for turn in reversed(history):
            if turn["role"] == "user":
                last_user_q = turn["content"]
                break
        if last_user_q and entity:
            rewritten = f"{entity}的{current_question}"

    changed = rewritten != current_question
    return rewritten, changed


def get_msg_content(msg):
    """获取消息内容，MiMo模型答案在reasoning_content里"""
    rc = getattr(msg, 'reasoning_content', '') or ''
    if rc.strip():
        return rc.strip()
    c = msg.content or ''
    return c.strip()


def llm_rewrite(history, current_question, max_retries=2):
    """
    LLM方式的Query重写（规则无法处理时的fallback）
    直接调用MiMo API进行重写
    """
    history_text = ""
    for turn in history[-6:]:
        role = "用户" if turn["role"] == "user" else "助手"
        history_text += f"{role}: {turn['content']}\n"

    prompt = f"""把当前问题重写为独立完整的问题，只输出结果。

对话历史：
{history_text}
当前问题：{current_question}

重写结果："""

    for attempt in range(max_retries + 1):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=128, timeout=15,
            )
            rewritten = get_msg_content(response.choices[0].message)
            # 从MiMo的推理输出中提取最后一句有用结果
            lines = [l.strip() for l in rewritten.split('\n') if l.strip()]
            for line in reversed(lines):
                skip_words = ['首先', '当前', '任务', '对话历史', '我需要',
                              '根据', '从对话', '这里', '所以', '因此', '那么']
                if any(w in line[:8] for w in skip_words):
                    continue
                if len(line) > 5 and '？' in line:
                    rewritten = line
                    break
            # 清理引号
            rewritten = rewritten.strip('"').strip("'").strip('「').strip('」')
            logger.info(f"LLM重写: {current_question} → {rewritten}")
            return rewritten
        except Exception as e:
            logger.warning(f"LLM重写失败 (第{attempt+1}次): {e}")
            if attempt < max_retries:
                time.sleep(1)
    return current_question


def rewrite_query(history, current_question):
    """
    重写用户问题，优先规则重写，规则无法处理时调LLM
    返回: str 重写后的独立问题
    """
    if len(history) < 2:
        logger.info("历史不足2轮，无需重写")
        return current_question

    # 第一步：尝试规则重写
    rewritten, changed = rule_based_rewrite(history, current_question)
    if changed:
        logger.info(f"规则重写: {current_question} → {rewritten}")
        return rewritten

    # 第二步：规则无法处理，才调用LLM
    logger.info(f"规则无法处理，尝试LLM重写: {current_question}")
    rewritten = llm_rewrite(history, current_question)
    return rewritten


if __name__ == "__main__":
    """单独测试Query重写"""
    history = [
        {"role": "user", "content": "武汉兴图新科来自军用领域的收入是多少？"},
        {"role": "assistant", "content": "军用领域收入数据在招股说明书中有记载。"},
    ]
    # 测试指代消解
    r1 = rewrite_query(history, "他参与的哪个工程获得国家科技进步一等奖？")
    print(f"测试1(指代消解): {r1}")
    # 测试省略补全
    r2 = rewrite_query(history, "那力源信息呢？")
    print(f"测试2(省略补全): {r2}")
