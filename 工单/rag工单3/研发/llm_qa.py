"""
LLM 问答模块 - 基于 DeepSeek API 生成答案
此模块封装了与 DeepSeek API 的交互逻辑，提供基于检索上下文生成问答
答案的功能，包括提示词构建、上下文格式化、置信度估算和批量问答处理，
是 RAG 系统中"生成"环节的核心实现。
支持中英文双语问答。
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
"""
import os  # 导入操作系统模块，用于文件路径操作
import json  # 导入 JSON 模块，用于序列化保存结果
import time  # 导入时间模块，用于计时
import re  # 导入正则表达式模块，用于语言检测
from openai import OpenAI  # 导入 OpenAI SDK，用于调用 DeepSeek API
from config import (LLM_API_KEY, LLM_API_BASE, LLM_MODEL,  # 导入 LLM 相关配置参数
                    LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TIMEOUT,
                    TOP_K, OUTPUT_DIR, log)  # 导入输出目录和日志函数


def detect_language(question: str) -> str:
    """检测问题语言：'zh' = 中文，'en' = 英文"""
    # 检查是否包含中文字符
    if re.search(r'[\u4e00-\u9fff]', question):
        return 'zh'
    return 'en'


class DeepSeekQA:
    """DeepSeek 问答模块 - 支持中英文"""

    def __init__(self, api_key: str = LLM_API_KEY,
                 api_base: str = LLM_API_BASE,
                 model: str = LLM_MODEL):
        self.api_key = api_key  # 保存 API 密钥
        self.api_base = api_base  # 保存 API 基础地址
        self.model = model  # 保存模型名称
        self.client = OpenAI(  # 创建 OpenAI 客户端实例
            api_key=self.api_key,  # 传入 API 密钥
            base_url=self.api_base,  # 传入自定义 API 地址
            timeout=LLM_TIMEOUT,  # 设置请求超时时间
        )
        log(f"DeepSeek QA 初始化完成 (model={self.model})", "LLM")  # 记录初始化日志

    def generate_answer(self, question: str, context: list,
                        max_tokens: int = None, language: str = "auto") -> dict:
        """
        基于检索结果生成答案（支持中英文）

        Args:
            question: 用户问题
            context: 检索结果列表 [{text, score, source_type, ...}, ...]
            language: "auto" / "zh" / "en"

        Returns:
            {answer, sources, confidence, usage}
        """
        mt = max_tokens or LLM_MAX_TOKENS  # 如果未指定 max_tokens，使用默认值
        lang = language if language != "auto" else detect_language(question)  # 自动检测语言
        log(f"生成答案 (lang={lang}): {question[:40]}...", "LLM")  # 记录生成日志

        # 构建上下文文本（精简版，限制条数）
        context_text = self._format_context(context, max_items=TOP_K)  # 将检索结果格式化为文本

        # 构建 prompt（按语言选择模板）
        prompt = self._build_prompt(question, context_text, lang)  # 构造发送给 LLM 的提示词

        start = time.time()  # 记录开始时间
        try:  # 捕获 API 调用异常
            # 使用更精简的系统提示
            system_msg = ("You are a professional prospectus analyst. "
                          "Answer based ONLY on the reference material." if lang == "en"
                          else "你是一个专业的招股说明书分析助手，严格基于参考材料回答。")
            response = self.client.chat.completions.create(  # 调用 DeepSeek API
                model=self.model,  # 指定模型名称
                messages=[  # 构建消息列表
                    {"role": "system", "content": system_msg},  # 精简系统提示
                    {"role": "user", "content": prompt},  # 用户提问内容
                ],
                temperature=LLM_TEMPERATURE,  # 设置生成温度参数
                max_tokens=mt,  # 设置最大生成 token 数
            )
            elapsed = time.time() - start  # 计算 API 调用耗时

            answer = response.choices[0].message.content.strip()  # 提取生成的答案文本并去除首尾空白
            usage = {  # 构建 token 用量统计
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            }

            log(f"答案生成完成 (lang={lang}, 耗时 {elapsed:.2f}s, tokens: {usage['total_tokens']})", "LLM")

        except Exception as e:  # 捕获所有异常
            log(f"API 调用失败: {e}", "ERROR")  # 记录错误日志
            answer = f"Error generating answer: {str(e)}" if lang == "en" else f"抱歉，生成答案时出错：{str(e)}"
            usage = {}  # 用量信息清空
            elapsed = time.time() - start  # 计算异常耗时

        # 构建来源信息
        sources = []  # 初始化来源列表
        for c in context[:TOP_K]:  # 限制来源数量
            sources.append({
                "text": c.get("text", "")[:200],  # 截取前200字符的文本
                "score": c.get("score", 0),  # 相关度分数
                "source_type": c.get("source_type", "text"),  # 来源类型
                "section_title": c.get("section_title", ""),  # 章节标题
            })

        return {  # 返回完整的问答结果
            "answer": answer,  # 生成的答案
            "sources": sources,  # 参考来源列表
            "confidence": self._estimate_confidence(answer),  # 回答置信度
            "usage": usage,  # token 用量统计
            "latency": elapsed,  # 生成耗时
            "language": lang,  # 使用的语言
        }

    def _build_prompt(self, question: str, context: str, lang: str = "zh") -> str:
        """按语言构建提示词（精简版，减少token消耗）"""
        if lang == "en":
            prompt = f"""Reference material:
{context}

Question: {question}

Requirements:
1. Answer based ONLY on the reference material
2. If not found, say "Not found in reference material"
3. Quote exact numbers/amounts when applicable
4. Be concise and accurate

Answer:"""
        else:
            prompt = f"""参考材料：
{context}

问题：{question}

要求：
1. 严格基于参考材料回答
2. 未找到则说"参考材料中未找到相关信息"
3. 涉及数字准确引用
4. 简洁准确

答案："""
        return prompt.strip()

    def _format_context(self, context: list, max_items: int = 3) -> str:
        """格式化检索结果（精简版，控制token量）"""
        parts = []  # 初始化文本片段列表
        for i, c in enumerate(context[:max_items]):  # 限制最大条数
            source_type = c.get("source_type", "text")  # 获取来源类型
            score = c.get("score", 0)  # 获取相关度分数
            text = c.get("text", "")  # 获取文本内容

            # 截断过长文本（减少token消耗）
            if len(text) > 500:
                text = text[:500] + "..."

            label = f"[Table {i+1}] (score: {score:.3f})" if source_type == "table" else f"[Doc {i+1}] (score: {score:.3f})"
            parts.append(f"{label}\n{text}")

        return "\n\n".join(parts)

    def _estimate_confidence(self, answer: str) -> float:
        """估算回答置信度"""
        low_confidence_indicators = [  # 定义低置信度关键词列表
            "未找到", "没有找到", "无法回答", "不包含",
            "没有提供", "未提及", "不清楚", "可能",
            "Not found", "not found", "cannot answer",
        ]
        for indicator in low_confidence_indicators:  # 遍历低置信度关键词
            if indicator in answer:  # 如果答案中包含这些关键词
                return 0.3  # 返回低置信度 0.3
        return 0.85  # 否则返回较高置信度 0.85

    def batch_qa(self, questions: list, context_provider, language: str = "auto") -> list:
        """
        批量问答

        Args:
            questions: [{id, question}, ...]
            context_provider: 提供检索上下文的函数 (query -> [context])
            language: "auto" / "zh" / "en"

        Returns:
            [{id, question, answer, sources, confidence, latency}, ...]
        """
        results = []  # 初始化结果列表
        for q in questions:  # 遍历每个问题
            if isinstance(q, dict):  # 如果问题是字典格式
                q_text = q.get("question", "")  # 提取问题文本
                q_id = q.get("id", "")  # 提取问题编号
            else:  # 如果问题是字符串格式
                q_text = str(q)  # 直接转为字符串
                q_id = ""  # 无编号

            # 获取上下文
            ctx = context_provider(q_text)  # 调用检索函数获取上下文
            if not ctx:  # 如果未检索到相关上下文
                ctx = [{"text": "未找到相关参考信息" if language != "en" else "No relevant reference found",
                        "score": 0, "source_type": "text"}]

            # 生成答案（自动检测语言）
            result = self.generate_answer(q_text, ctx, language=language)  # 调用单题问答生成答案
            result["id"] = q_id  # 设置问题编号
            result["question"] = q_text  # 设置问题文本
            results.append(result)  # 加入结果列表

        return results  # 返回全部问答结果

    def save_results(self, results: list, filename: str = "qa_results.json"):
        """保存问答结果"""
        out_path = os.path.join(OUTPUT_DIR, filename)  # 拼接输出文件完整路径
        with open(out_path, "w", encoding="utf-8") as f:  # 以 UTF-8 写入模式打开文件
            json.dump(results, f, ensure_ascii=False, indent=2)  # 将结果序列化为 JSON 写入文件
        log(f"问答结果已保存: {out_path}", "LLM")  # 记录保存日志


def create_qa() -> DeepSeekQA:
    """工厂函数：创建 DeepSeekQA 实例"""
    return DeepSeekQA()  # 使用默认配置创建并返回问答实例


if __name__ == "__main__":  # 如果作为主程序运行
    # 测试
    qa = create_qa()  # 创建问答实例
    result = qa.generate_answer(  # 执行测试问答
        "武汉力源信息技术股份有限公司本次发行股数是多少？",  # 测试问题
        [{"text": "本次发行股数为2,000万股，占发行后总股本的25%。",
          "score": 0.95, "source_type": "text"}]
    )
    print(f"答案: {result['answer'][:200]}")  # 打印前200字符的答案
