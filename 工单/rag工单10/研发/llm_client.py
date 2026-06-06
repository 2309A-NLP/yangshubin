"""
模块功能: MiMo (小米开放平台) LLM 客户端模块
封装对 MiMo API 的调用，支持流式和非流式两种模式
使用 OpenAI 兼容接口格式调用 mimo-v2.5-pro 模型
工单编号: 人工智能NLP-RAG-金融问答系统部署
"""

import logging      # 日志记录模块，用于输出运行信息
from typing import Optional  # 类型提示，增强代码可读性
from app.config import config  # 导入全局配置对象


# 获取当前模块的日志记录器
logger = logging.getLogger("llm_client")


class MiMoClient:
    """MiMo API 客户端类，封装对小米开放平台大模型的 LLM 调用"""

    def __init__(self):
        """初始化客户端，从配置对象读取 MiMo API 参数"""
        # MiMo API 密钥（tp- 格式）
        self.api_key: str = config.MIMO_API_KEY
        # MiMo API 基础地址（OpenAI 兼容格式）
        self.api_base: str = config.MIMO_API_BASE
        # 使用的模型名称（默认 mimo-v2.5-pro）
        self.model: str = config.MIMO_MODEL
        # LLM 生成温度参数
        self.temperature: float = config.LLM_TEMPERATURE
        # 最大生成 token 数量
        self.max_tokens: int = config.LLM_MAX_TOKENS

    def generate(self, prompt: str) -> str:
        """调用 MiMo API 生成文本（非流式）

        Args:
            prompt: 完整的提示词内容（包含 system prompt 和上下文）

        Returns:
            模型生成的文本字符串，出错时返回错误提示
        """
        try:
            # 动态导入 OpenAI SDK，避免模块加载时的依赖问题
            from openai import OpenAI
            # 创建 OpenAI 兼容的客户端对象
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
            )
            # 调用 MiMo 模型的聊天完成接口
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            # 从响应中提取生成的文本，去除首尾空白
            answer: str = response.choices[0].message.content.strip()
            logger.info(f"LLM 调用成功: 生成 {len(answer)} 字符")
            return answer
        except Exception as e:
            # 捕获所有异常，返回友好错误信息
            logger.error(f"LLM 调用失败: {e}")
            return f"抱歉，调用 MiMo API 时出现错误: {str(e)}"

    def generate_stream(self, prompt: str):
        """流式调用 MiMo API 生成文本

        使用 Python 生成器逐块 yield 生成内容，
        适用于前端 SSE (Server-Sent Events) 流式展示。

        Args:
            prompt: 完整的提示词内容

        Yields:
            逐个文本块的生成结果
        """
        try:
            # 动态导入 OpenAI SDK
            from openai import OpenAI
            # 创建 OpenAI 兼容的客户端对象
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
            )
            # 发起流式聊天完成请求
            stream = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )
            # 逐块 yield 生成内容
            for chunk in stream:
                # 如果当前块有文本内容，则产出
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            # 流式调用失败，返回错误信息
            logger.error(f"LLM 流式调用失败: {e}")
            yield f"抱歉，流式调用 MiMo API 时出现错误: {str(e)}"
