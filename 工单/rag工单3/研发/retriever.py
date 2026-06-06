"""
检索模块 - 混合检索（稠密向量 + 表格优先 + 公司名过滤）
此模块提供基于向量检索与规则排序相结合的混合检索能力，
支持从 Milvus 向量库中检索文档片段，并按照表格优先、公司名匹配、
关键词相关度等策略进行重排序，以提升 RAG 系统的检索准确性。
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
"""
import time  # 导入时间模块，用于计算检索耗时
import re  # 导入正则表达式模块，用于文本模式匹配
import json  # 导入 JSON 模块，用于加载文本缓存
import os  # 导入操作系统模块，用于文件路径
from typing import List, Dict  # 导入类型提示，增强代码可读性
from config import TOP_K, HYBRID_WEIGHT_DENSE, HYBRID_WEIGHT_TABLE, OUTPUT_DIR, log  # 导入配置参数与日志函数
from embedding import BGEM3Embedding  # 导入 BGE-M3 嵌入模型类
from vector_store import MilvusVectorStore  # 导入 Milvus 向量数据库操作类


# ========== 公司名识别规则 ==========
# 格式: (完整公司名, [简称列表])
COMPANY_NAMES = [  # 定义公司名称识别规则列表
    ("武汉力源信息技术股份有限公司", ["武汉力源", "力源信息", "力源"]),  # 武汉力源及其常用简称
    ("武汉兴图新科电子股份有限公司", ["武汉兴图新科", "兴图新科", "兴图"]),  # 武汉兴图新科及其常用简称
]

# 提取所有简称（用于匹配）
_COMPANY_ALIASES = {}  # 初始化别名映射字典：简称 -> 全称
for full_name, aliases in COMPANY_NAMES:  # 遍历所有公司名规则
    for alias in aliases:  # 遍历每个公司的简称列表
        _COMPANY_ALIASES[alias] = full_name  # 将简称映射到全称
    # 全称也可以匹配
    _COMPANY_ALIASES[full_name] = full_name  # 将全称也加入映射，支持全称直接匹配


def extract_company_from_query(query: str) -> str:
    """
    从查询中提取公司全称

    Args:
        query: 用户问题

    Returns:
        匹配到的公司全称，未匹配返回 ""
    """
    for alias, full_name in sorted(_COMPANY_ALIASES.items(), key=lambda x: -len(x[0])):  # 按别名长度降序遍历（长匹配优先）
        if alias in query:  # 检查别名是否出现在查询文本中
            return full_name  # 返回对应的公司全称
    return ""  # 未匹配到任何公司，返回空字符串


def _extract_core_keywords(query: str) -> list:
    """提取查询中的核心关键词（用于兜底检索判断）"""
    # 重要的金融/业务类关键词
    important_kws = [
        "收入", "金额", "比例", "比重", "占比", "股数",
        "发行", "募集", "投资", "项目", "资金", "注册",
        "资本", "持股", "控制", "关联", "供应商", "领域",
        "军用", "利润", "标准", "工程", "上游", "下游",
        "行业", "股东", "负债", "资产", "净利润",
    ]
    found = [kw for kw in important_kws if kw in query]
    return found


class HybridRetriever:
    """混合检索器 - 融合向量检索 + 表格优先 + 公司名过滤"""

    def __init__(self, embedding_model: BGEM3Embedding,
                 vector_store: MilvusVectorStore):
        self.embedding = embedding_model  # 保存 BGE-M3 嵌入模型实例
        self.vector_store = vector_store  # 保存 Milvus 向量数据库实例
        self.top_k = TOP_K  # 从配置中读取默认返回的 top-k 结果数
        self._chunks_cache = None  # 关键词检索缓存（延迟加载）

    def retrieve(self, query: str, top_k: int = None) -> list:
        """
        执行混合检索（带公司名过滤 + 高召回重排序）

        Args:
            query: 查询问题

        Returns:
            [{text, score, source_type, section_title, metadata}, ...]
            按分数降序排列
        """
        k = top_k or self.top_k  # 如果未指定 top_k，使用默认值
        log(f"检索问题: {query}", "RETRIEVE")  # 记录检索日志

        # 0. 提取问题中的公司名
        target_company = extract_company_from_query(query)  # 从用户问题中提取目标公司名称
        if target_company:  # 如果检测到目标公司
            log(f"检测到目标公司: {target_company}", "RETRIEVE")  # 记录公司检测日志

        start = time.time()  # 记录检索开始时间

        # 1. 生成查询向量
        query_vec = self.embedding.encode_query(query)  # 使用 BGE-M3 将查询文本编码为向量

        # 2. 执行 Milvus 向量检索（高召回以覆盖更多候选）
        milvus_results = self.vector_store.search(query_vec, top_k=k * 10)  # 多召回到 k*10 条

        # 3. 重排序（含公司名过滤 + 表格优先 + 金融关键词加权）
        results = self._rank_results(milvus_results, query, target_company)

        # 4. 智能兜底：如果向量检索结果置信度不足（最高分<0.5）
        #    或 top-3 结果均不包含查询的核心关键词，触发关键词搜索
        core_kws = _extract_core_keywords(query)
        results_have_core = any(
            any(kw in r.get("text", "") for kw in core_kws)
            for r in results[:3]
        ) if core_kws else True

        if (not results or results[0].get("score", 0) < 0.5) or not results_have_core:
            fallback = self._keyword_retrieve(query, top_k=k * 3)
            if fallback:
                log(f"向量检索未覆盖核心关键词，关键词兜底补充 {len(fallback)} 条", "RETRIEVE")
                # 兜底结果给合理的分数（略低于正常向量分数）
                for fb in fallback:
                    fb["score"] = 0.4 + min(fb.get("score", 0) * 0.08, 0.2)
                # 合并去重
                seen_texts = {r.get("text","")[:80] for r in results}
                added = 0
                for fb in fallback:
                    if fb.get("text","")[:80] not in seen_texts:
                        results.append(fb)
                        added += 1
                # 重新排序
                results.sort(key=lambda x: x["score"], reverse=True)
                if added > 0:
                    log(f"兜底后新增 {added} 条候选", "RETRIEVE")

        elapsed = time.time() - start  # 计算检索耗时
        log(f"检索完成 (候选{len(milvus_results)}条, 返回{len(results)}/{k}条, 耗时 {elapsed:.3f}s)", "RETRIEVE")

        return results[:k]  # 返回前 k 条结果

    def _load_chunks_cache(self):
        """延迟加载 chunks.json 用于关键词检索"""
        if self._chunks_cache is None:
            chunks_path = os.path.join(OUTPUT_DIR, "chunks.json")
            if os.path.exists(chunks_path):
                with open(chunks_path, "r", encoding="utf-8") as f:
                    self._chunks_cache = json.load(f)
                log(f"加载关键词缓存: {len(self._chunks_cache)} 个片段", "RETRIEVE")
            else:
                self._chunks_cache = []
                log("关键词缓存未找到 (chunks.json)", "WARN")
        return self._chunks_cache

    def _keyword_retrieve(self, query: str, top_k: int = 6) -> list:
        """
        关键词兜底检索：从 chunks.json 中查找包含查询关键词的文本块
        补充纯向量检索可能遗漏的内容（如财务表格数据）
        """
        chunks = self._load_chunks_cache()
        if not chunks:
            return []

        # 从查询中提取关键词（去掉停用词、公司名等）
        keywords = self._extract_keywords(query)
        # 添加查询中的重要二元词组
        for i in range(len(query) - 1):
            bigram = query[i:i+2]
            if bigram.strip() and len(bigram) > 1:
                keywords.append(bigram)

        # 对每个 chunk 计算关键词匹配得分
        scored = []
        for c in chunks:
            text = c.get("text", "")
            if not text:
                continue
            score = 0.0
            matched_kws = []
            for kw in keywords:
                if len(kw) > 1 and kw in text:
                    score += 0.15  # 每个关键词命中加 0.15
                    matched_kws.append(kw)
            # 对公司名加权重
            company = extract_company_from_query(query)
            if company and company[:4] in text:
                score += 0.3
            if score > 0:
                scored.append({
                    "text": text,
                    "score": score,
                    "source_type": c.get("type", "text"),
                    "section_title": c.get("section_title", ""),
                    "chunk_index": c.get("chunk_index", 0),
                    "metadata": c.get("metadata", {}),
                    "_kw_match": matched_kws,
                })

        # 按关键词匹配得分降序
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def _rank_results(self, results: list, query: str,
                      target_company: str = "") -> list:
        """
        结果重排序 - 表格优先 + 相关度加权 + 公司名过滤

        策略：
        - 如果检测到目标公司，过滤掉其他公司的内容（严重降权）
        - 表格类型的结果优先保留
        - 关键词匹配加分
        """
        if not results:  # 如果结果列表为空
            return []  # 直接返回空列表

        scored = []  # 初始化打分后的结果列表
        keywords = self._extract_keywords(query)  # 从查询中提取关键词

        # 获取目标公司的简称列表（用于文本匹配）
        target_aliases = []  # 初始化目标公司别名列表
        if target_company:  # 如果指定了目标公司
            for full_name, aliases in COMPANY_NAMES:  # 遍历公司名称规则表
                if full_name == target_company:  # 找到目标公司对应的规则
                    target_aliases = [full_name] + aliases  # 将全称和简称合并为列表
                    break  # 找到后跳出循环

        # 获取其他公司的全称（用于排除）
        other_companies = []  # 初始化其他公司名称列表
        if target_company:  # 如果指定了目标公司
            other_companies = [full for full, _ in COMPANY_NAMES  # 收集所有其他公司的全称
                               if full != target_company]
            for full, aliases in COMPANY_NAMES:  # 遍历所有公司规则
                if full != target_company:  # 如果是其他公司
                    other_companies.extend(aliases)  # 将其简称也加入排除列表

        for r in results:  # 遍历每条检索结果
            score = r["score"]  # 获取原始向量相似度分数
            text = r.get("text", "")  # 获取结果文本内容
            source_type = r.get("source_type", "text")  # 获取来源类型（table/text）
            metadata = r.get("metadata", {})  # 获取元数据

            # ---------- 公司名过滤（基于source_pdf元数据+文本匹配）----------
            # 从元数据中获取来源PDF（比纯文本匹配更可靠）
            source_pdf = metadata.get("source_pdf", "") if isinstance(metadata, dict) else ""

            if target_company:  # 如果检测到了目标公司
                # 检查文本是否提到目标公司（文本级匹配）
                text_contains_target = any(a in text for a in target_aliases)
                # 检查元数据中的source_pdf是否匹配目标公司
                meta_matches_target = (
                    ("力源" in source_pdf and "武汉力源" in target_company) or
                    ("兴图" in source_pdf and "武汉兴图新科" in target_company)
                )

                text_contains_other = any(  # 检查文本是否包含其他公司名称
                    oc in text for oc in other_companies
                )

                if meta_matches_target or text_contains_target:
                    # 提到目标公司（元数据或文本匹配）→ 加分
                    score += 0.2
                elif text_contains_other:
                    # 提到其他公司但没有目标公司 → 严重降权
                    score -= 0.4
                # 如果都没提到（可能只是泛泛而谈"公司"），不做处理

            # ---------- 表格类型加分 ----------
            if source_type == "table":  # 如果来源是表格
                score += HYBRID_WEIGHT_TABLE  # 加上表格优先级权重
                header = r.get("metadata", {}).get("header", "")  # 获取表格表头信息
                header_match = sum(1 for kw in keywords if kw in header)  # 统计表头中匹配的关键词数量
                score += header_match * 0.05  # 每个匹配的关键词额外加 0.05 分

            # ---------- 关键词匹配加分 ----------
            keyword_matches = sum(1 for kw in keywords if kw in text)  # 统计文本中匹配的关键词数量
            score += keyword_matches * 0.02  # 每个匹配的关键词加 0.02 分

            # ---------- 金融数值类内容加分 ----------
            finance_bonus = self._finance_score(text)  # 计算金融相关度加分
            score += finance_bonus  # 加上金融类权重

            scored.append({  # 将评分后的结果加入列表
                "text": text,  # 结果文本
                "score": max(score, 0.0),  # 分数不低于0
                "source_type": source_type,  # 来源类型
                "section_title": r.get("section_title", ""),  # 所属章节标题
                "metadata": r.get("metadata", {}),  # 元数据
                "chunk_index": r.get("chunk_index"),  # 片段索引
            })

        # 按分数降序排列
        scored.sort(key=lambda x: x["score"], reverse=True)  # 按分数从高到低排序
        return scored  # 返回排序后的结果列表

    # ========== 金融关键词权重 ==========
    FINANCE_KEYWORDS = {  # 金融/数值类关键词及其权重加成
        "收入": 0.08, "金额": 0.08, "比例": 0.07, "比重": 0.07,
        "占比": 0.07, "股数": 0.10, "发行": 0.06, "募集": 0.08,
        "投资": 0.06, "项目": 0.05, "资金": 0.07, "注册": 0.06,
        "资本": 0.06, "持股": 0.08, "控制": 0.06, "关联": 0.06,
        "供应商": 0.07, "领域": 0.05, "军用": 0.08, "利润": 0.07,
        "科技": 0.03, "电子": 0.03, "标准": 0.06, "工程": 0.05,
        "上游": 0.07, "下游": 0.07, "行业": 0.04,
    }

    def _extract_keywords(self, query: str) -> list:
        """
        从问题中提取关键词
        增强数字型/金融类问题的关键词提取

        Returns: 关键词列表
        """
        stop_words = {  # 定义停用词集合（过滤无意义的常用词）
            "的", "了", "是", "在", "什么", "多少", "哪些", "哪个",
            "如何", "怎么", "为什么", "谁", "吗", "呢", "啊", "吧",
            "与", "和", "或", "及", "并", "而", "等", "之", "为",
            "不", "也", "都", "就", "要", "将", "会", "可以", "应该",
            "本次", "股份", "公司", "有限", "技术",
        }

        words = []  # 初始化关键词列表
        for w in query:  # 遍历查询中的每个字符
            if len(w) > 1 and w not in stop_words:  # 长度大于1且不是停用词
                words.append(w)  # 加入关键词列表

        for i in range(len(query) - 1):  # 遍历所有相邻字符对（二元组）
            bigram = query[i:i+2]  # 提取两个连续字符
            if bigram not in stop_words and bigram.isalpha():  # 如果不是停用词且由字母组成
                words.append(bigram)  # 加入关键词列表

        # 提取数字型关键词（如"多少"、"占比"等金融数值类词汇）
        for i in range(len(query) - 3):
            segment = query[i:i+4]
            if "收入" in segment or "金额" in segment or "比例" in segment:
                words.append("财务数据")

        return list(set(words))  # 去重后返回关键词列表

    def _finance_score(self, text: str) -> float:
        """计算文本的金融/数值相关度加分"""
        score = 0.0
        for keyword, weight in self.FINANCE_KEYWORDS.items():
            if keyword in text:
                score += weight
        return min(score, 0.5)  # 封顶0.5避免过度加权

    def _extract_numeric_keywords(self, query: str) -> list:
        """提取问题中的数值类关键词，用于表格匹配"""
        numeric_indicators = ["多少", "占比", "比重", "比例", "收入", "金额",
                              "股数", "利润", "资本", "资金"]
        found = [kw for kw in numeric_indicators if kw in query]
        return found

    def batch_retrieve(self, questions: list, top_k: int = None) -> list:
        """批量检索多个问题"""
        all_results = []  # 初始化全部结果列表
        for q in questions:  # 遍历每个问题
            if isinstance(q, dict):  # 如果问题是字典格式
                q_text = q.get("question", "")  # 提取问题文本
                q_id = q.get("id", "")  # 提取问题编号
            else:  # 如果问题是字符串格式
                q_text = str(q)  # 直接转为字符串
                q_id = ""  # 无编号

            results = self.retrieve(q_text, top_k)  # 对当前问题执行检索
            all_results.append({  # 将结果加入总列表
                "id": q_id,  # 问题编号
                "question": q_text,  # 问题文本
                "results": results,  # 检索结果列表
            })

        return all_results  # 返回全部检索结果


def create_retriever(embedding_model: BGEM3Embedding = None,
                     vector_store: MilvusVectorStore = None) -> HybridRetriever:
    """工厂函数：创建 HybridRetriever 实例"""
    if embedding_model is None:  # 如果未提供嵌入模型
        embedding_model = BGEM3Embedding()  # 创建默认的 BGE-M3 嵌入模型
    if vector_store is None:  # 如果未提供向量存储
        vector_store = MilvusVectorStore()  # 创建默认的 Milvus 向量存储
    return HybridRetriever(embedding_model, vector_store)  # 返回混合检索器实例


if __name__ == "__main__":  # 如果作为主程序运行
    # 测试
    retriever = create_retriever()  # 创建检索器实例
    results = retriever.retrieve("武汉力源信息技术股份有限公司本次发行股数是多少？")  # 执行测试检索
    for r in results:  # 遍历检索结果
        print(f"[{r['source_type']}] 分数: {r['score']:.4f}")  # 打印来源类型和分数
        print(f"  内容: {r['text'][:100]}...")  # 打印前100字符的内容摘要
        print()  # 打印空行分隔
