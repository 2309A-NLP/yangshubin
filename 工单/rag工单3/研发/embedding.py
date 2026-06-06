"""
嵌入模块 - 基于 BGE-M3 的文本向量化
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
"""
import os  # 导入操作系统接口模块
import time  # 导入时间模块
import numpy as np  # 导入 NumPy 数值计算库
from sentence_transformers import SentenceTransformer  # 导入 SentenceTransformer 模型类
from config import (EMBEDDING_MODEL_PATH, EMBEDDING_DIM,  # 从配置模块导入模型相关配置
                    EMBEDDING_DEVICE, EMBEDDING_BATCH_SIZE,  # 导入设备类型和批处理大小
                    EMBEDDING_MAX_SEQ_LENGTH, log)  # 导入最大序列长度和日志函数


class BGEM3Embedding:  # BGE-M3 嵌入模型封装类
    """BGE-M3 嵌入模型封装，生成稠密向量和稀疏向量"""

    def __init__(self, model_path: str = EMBEDDING_MODEL_PATH,  # 构造函数，初始化嵌入模型
                 device: str = EMBEDDING_DEVICE):  # 设备参数，默认为 CUDA
        self.model_path = model_path  # 保存模型路径
        self.device = device  # 保存设备类型
        self.model = None  # 初始化模型对象为 None
        self._load_model()  # 调用内部方法加载模型

    def _load_model(self):  # 内部方法：加载 BGE-M3 模型
        """加载 BGE-M3 模型"""
        log(f"加载 BGE-M3 模型: {self.model_path}", "EMBED")  # 记录加载日志
        log(f"设备: {self.device}", "EMBED")  # 记录设备信息

        start = time.time()  # 记录开始时间
        self.model = SentenceTransformer(  # 创建 SentenceTransformer 实例
            self.model_path,  # 传入模型路径
            device=self.device,  # 指定运行设备
            trust_remote_code=True,  # 允许加载远程代码
        )  # 括号结束
        # FP16 半精度加载，显存从 ~7.5GB 降至 ~3.5GB
        if self.device == "cuda":  # 如果使用 GPU 推理
            self.model.half()  # 转换为 FP16 半精度以节省显存
            log("已启用 FP16 半精度推理（显存占用减半）", "EMBED")  # 日志记录

        self.model.max_seq_length = EMBEDDING_MAX_SEQ_LENGTH  # 设置模型最大序列长度
        elapsed = time.time() - start  # 计算加载耗时
        log(f"模型加载完成 (耗时 {elapsed:.1f}s)", "EMBED")  # 日志记录

        # 验证模型维度
        test_emb = self.model.encode(["test"], show_progress_bar=False)  # 用测试文本验证模型输出维度
        actual_dim = len(test_emb[0])  # 获取实际输出向量维度
        log(f"模型输出维度: {actual_dim}", "EMBED")  # 日志记录

    def encode_dense(self, texts: list, batch_size: int = EMBEDDING_BATCH_SIZE,  # 生成稠密向量的方法
                     normalize: bool = True) -> np.ndarray:  # 是否归一化参数，返回 numpy 数组
        """
        生成稠密向量

        Args:
            texts: 文本列表
            batch_size: 批处理大小
            normalize: 是否归一化

        Returns:
            numpy array, shape (n, dim)
        """
        if not texts:  # 如果输入文本列表为空
            return np.array([])  # 返回空数组

        log(f"编码 {len(texts)} 条文本的稠密向量 (batch_size={batch_size})", "EMBED")  # 日志记录
        start = time.time()  # 记录开始时间

        embeddings = self.model.encode(  # 调用模型进行编码
            texts,  # 输入文本列表
            batch_size=batch_size,  # 设置批处理大小
            show_progress_bar=True,  # 显示编码进度条
            normalize_embeddings=normalize,  # 是否对向量进行 L2 归一化
        )  # 括号结束

        elapsed = time.time() - start  # 计算加载耗时
        log(f"稠密向量编码完成 (耗时 {elapsed:.2f}s, {len(texts)/max(elapsed,0.01):.1f} 条/s)", "EMBED")  # 日志记录编码性能
        return np.array(embeddings, dtype=np.float32)  # 返回 float32 类型的 numpy 数组

    def encode_sparse(self, texts: list) -> list:  # 生成稀疏向量的方法
        """
        生成稀疏向量（词权重）

        Args:
            texts: 文本列表

        Returns:
            [{token_id: weight, ...}, ...] 每个文本的词权重字典
        """
        if not texts:  # 如果输入文本列表为空
            return []  # 返回空列表

        log(f"编码 {len(texts)} 条文本的稀疏向量", "EMBED")
        start = time.time()  # 记录开始时间

        # SentenceTransformer 的 BGE-M3 支持 sparse 输出
        sparse_embs = []  # 初始化稀疏向量列表
        for text in texts:  # 遍历每条文本
            output = self.model.encode(  # 编码单条文本
                text,  # 输入文本
                output_value="token_embeddings",  # 输出 token 级别的向量
                show_progress_bar=False,  # 不显示进度条
            )  # 括号结束
            # 获取词级别的权重（简化实现）
            tokens = self.model.tokenizer.tokenize(text)  # 使用 tokenizer 进行分词
            token_ids = self.model.tokenizer.convert_tokens_to_ids(tokens)  # 将 token 转换为 ID

            weights = {}  # 初始化权重字典
            for tid in set(token_ids):  # 遍历去重后的 token ID
                weights[int(tid)] = float(token_ids.count(tid)) / max(len(token_ids), 1)  # 计算词频作为稀疏权重
            sparse_embs.append(weights)  # 将该文本的稀疏向量加入列表

        elapsed = time.time() - start  # 计算加载耗时
        log(f"稀疏向量编码完成 (耗时 {elapsed:.2f}s)", "EMBED")  # 日志记录
        return sparse_embs  # 返回稀疏向量列表

    def encode_query(self, query: str) -> np.ndarray:  # 编码单个查询文本（带检索指令优化）
        """编码单个查询（带指令优化）"""
        instruction = "为这个句子生成表示以用于检索相关文档："  # BGE-M3 的检索指令前缀
        prefixed_query = f"{instruction} {query}"  # 拼接指令和查询文本
        return self.encode_dense([prefixed_query], batch_size=1)[0]  # 编码并返回第一个（唯一）向量

    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:  # 计算两个向量的余弦相似度
        """计算两个向量的余弦相似度"""
        emb1 = emb1 / (np.linalg.norm(emb1) + 1e-10)  # L2 归一化向量1
        emb2 = emb2 / (np.linalg.norm(emb2) + 1e-10)  # L2 归一化向量2
        return float(np.dot(emb1, emb2))  # 返回点积值（即余弦相似度）


def create_embedding() -> BGEM3Embedding:  # 工厂函数：创建嵌入模型实例
    """工厂函数 - 创建嵌入模型实例"""
    return BGEM3Embedding()  # 返回 BGE-M3 嵌入模型实例


if __name__ == "__main__":  # 主程序入口（测试用）
    # 测试
    embed = create_embedding()  # 创建嵌入模型实例
    texts = ["武汉力源信息技术股份有限公司本次发行股数是多少？", "招股说明书中的财务数据"]  # 测试用文本列表
    dense = embed.encode_dense(texts)  # 生成测试文本的稠密向量
    print(f"稠密向量形状: {dense.shape}")  # 打印向量维度形状
    print(f"向量前5维: {dense[0][:5]}")  # 打印第一个向量的前5维
    q_vec = embed.encode_query("发行股数是多少")  # 编码查询文本
    sim = embed.compute_similarity(dense[0], q_vec)  # 计算查询与文档的相似度
    print(f"查询相似度: {sim:.4f}")  # 打印相似度结果
