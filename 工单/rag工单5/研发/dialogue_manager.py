"""
dialogue_manager.py - RAG工单5 多轮对话管理模块
工单编号: 人工智能NLP-RAG-Query理解优化任务
完成需求: 管理多轮对话状态，维护对话历史，支持上下文感知问答
功能说明: 创建多会话、添加对话轮次、提取当前实体、历史截断
"""

import logging  # 日志记录
import time     # 时间戳

# 导入配置
from config import MAX_HISTORY_TURNS, OUTPUT_DIR, LOG_FORMAT, LOG_DATE_FORMAT

# 设置日志
logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT, level=logging.INFO)
logger = logging.getLogger("dialogue_manager")


class DialogueManager:
    """
    多轮对话管理器
    支持多个独立会话，每个会话维护独立的对话历史
    自动提取当前讨论的实体（公司名），实现上下文感知
    """

    def __init__(self):
        """初始化对话管理器，创建空会话字典"""
        self.sessions = {}  # session_id -> session_data
        logger.info("对话管理器初始化完成")

    def create_session(self, session_id=None):
        """
        创建新对话会话
        参数:
            session_id: 自定义会话ID，不传则自动生成
        返回:
            str: 会话ID
        """
        if session_id is None:
            import uuid
            session_id = str(uuid.uuid4())[:8]  # 取UUID前8位

        # 初始化会话数据结构
        self.sessions[session_id] = {
            "history": [],          # 对话历史列表
            "created_at": time.time(),  # 创建时间戳
            "turn_count": 0,        # 对话轮数
            "last_question": "",    # 最后一轮问题
            "last_answer": "",      # 最后一轮答案
            "current_entity": "",   # 当前讨论的公司实体
        }
        logger.info(f"创建会话: {session_id}")
        return session_id

    def get_session(self, session_id):
        """
        获取会话，不存在则自动创建
        参数:
            session_id: 会话ID
        返回:
            dict: 会话数据
        """
        if session_id not in self.sessions:
            self.create_session(session_id)
        return self.sessions[session_id]

    def add_turn(self, session_id, question, answer):
        """
        添加一轮对话到历史记录
        参数:
            session_id: 会话ID
            question: 用户问题（原始问题，不是重写后的）
            answer: 系统生成的答案
        """
        session = self.get_session(session_id)

        # 添加用户问题和系统答案到历史
        session["history"].append({"role": "user", "content": question})
        session["history"].append({"role": "assistant", "content": answer})

        # 更新会话状态
        session["turn_count"] += 1
        session["last_question"] = question
        session["last_answer"] = answer

        # 从问题中提取当前讨论的实体（公司名）
        import re
        companies = re.findall(
            r'(武汉力源信息技术股份有限公司|武汉兴图新科电子股份有限公司'
            r'|力源信息|兴图新科)',
            question
        )
        if companies:
            session["current_entity"] = companies[-1]  # 取最后一个匹配

        # 限制历史长度，保留最近MAX_HISTORY_TURNS轮
        max_turns = MAX_HISTORY_TURNS * 2  # user + assistant = 2条记录/轮
        if len(session["history"]) > max_turns:
            session["history"] = session["history"][-max_turns:]

        logger.info(f"会话 {session_id} 第 {session['turn_count']} 轮")

    def get_history(self, session_id):
        """
        获取指定会话的对话历史
        参数:
            session_id: 会话ID
        返回:
            list: [{"role":..., "content":...}, ...]
        """
        session = self.get_session(session_id)
        return session["history"]

    def get_context_summary(self, session_id):
        """
        获取对话上下文摘要
        参数:
            session_id: 会话ID
        返回:
            str: 上下文摘要文本
        """
        session = self.get_session(session_id)
        if session["turn_count"] == 0:
            return ""

        # 生成简洁的上下文摘要
        summary_parts = []
        if session["current_entity"]:
            summary_parts.append(f"当前讨论公司: {session['current_entity']}")
        summary_parts.append(f"对话轮数: {session['turn_count']}")
        return " | ".join(summary_parts)

    def clear_session(self, session_id):
        """
        清除指定会话的所有数据
        参数:
            session_id: 会话ID
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"清除会话: {session_id}")


if __name__ == "__main__":
    """单独测试对话管理器"""
    dm = DialogueManager()
    sid = dm.create_session()
    dm.add_turn(sid, "武汉兴图新科注册资本是多少？", "5,520万元")
    dm.add_turn(sid, "他参与的哪个工程获奖？", "C4ISR工程")
    print(f"历史: {dm.get_history(sid)}")
