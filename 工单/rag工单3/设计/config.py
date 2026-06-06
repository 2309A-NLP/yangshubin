"""
配置模块
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
"""
import os  # 导入操作系统接口模块
import sys  # 导入系统相关功能模块
from pathlib import Path  # 导入路径处理模块

# ========== 基础路径 ==========
# 自动适配 Windows 和 WSL 路径
if sys.platform == "win32":  # 判断是否为 Windows 系统
    BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))  # Windows 下使用脚本所在目录作为基础目录
else:
    # WSL 环境
    BASE_DIR = Path("/mnt/c/Users/31326/Desktop/rag工单3")  # WSL 下使用固定路径作为基础目录

PDF_PATHS = [  # PDF 文件路径列表（支持多个 PDF）
    str(BASE_DIR / "招股说明书1.pdf"),   # 武汉力源信息技术股份有限公司
    str(BASE_DIR / "招股说明书2.pdf"),   # 武汉兴图新科电子股份有限公司
]
PDF_PATH = PDF_PATHS[0]  # 兼容旧接口指向第一个 PDF
OUTPUT_DIR = str(BASE_DIR / "output")  # 输出目录
MINERU_OUTPUT_DIR = str(BASE_DIR / "mineru_output")  # MinerU 解析输出目录
TEMPLATES_DIR = str(BASE_DIR / "templates")  # 模板文件目录
INDEX_HTML = str(BASE_DIR / "templates" / "index.html")  # 前端页面模板路径

# ========== BGE-M3 模型 ==========
if sys.platform == "win32":  # 判断是否为 Windows 系统
    EMBEDDING_MODEL_PATH = r"C:\Users\31326\Desktop\bge-m3"  # Windows 下的 BGE-M3 模型路径
else:
    EMBEDDING_MODEL_PATH = "/mnt/c/Users/31326/Desktop/bge-m3"  # WSL 下的 BGE-M3 模型路径
EMBEDDING_DIM = 1024  # BGE-M3 的向量维度
EMBEDDING_DEVICE = "cuda"  # cuda / cpu
EMBEDDING_BATCH_SIZE = 2  # 编码批处理大小
EMBEDDING_MAX_SEQ_LENGTH = 1024  # 保守设置，防止内存溢出

# ========== DeepSeek API ==========
LLM_API_KEY = "tp-czex5np7bgf6duyuyvqntmw44xcunseatmblffiw9lk9w0jy"
LLM_API_BASE = "https://token-plan-cn.xiaomimimo.com/v1"  # 小米MiMo API 地址
LLM_MODEL = "mimo-v2.5-pro"  # 小米MiMo 使用的模型名称（小写）
LLM_TEMPERATURE = 0.1  # 生成温度参数（越低越确定）
LLM_MAX_TOKENS = 2048  # 最大生成 Token 数（小米API有推理过程，需要更多token）
LLM_TIMEOUT = 180  # API 请求超时时间（秒，小米API较慢需要大超时）

# ========== Milvus 配置 ==========
MILVUS_HOST = "localhost"  # Milvus 服务主机地址
MILVUS_PORT = "19530"  # Milvus 服务端口号
MILVUS_COLLECTION = "rag_table_prospectus"  # 向量数据库集合名称
MILVUS_INDEX_TYPE = "IVF_FLAT"  # 索引类型
MILVUS_METRIC_TYPE = "COSINE"  # 距离度量方式（余弦相似度）
MILVUS_NLIST = 1024  # 聚类中心数量（IVF 索引参数）
MILVUS_NPROBE = 64  # 查询时探测的聚类数量（提高以增加召回率）

# ========== 文本切分配置 ==========
CHUNK_SIZE = 512  # 文本切分块大小（字符数）
CHUNK_OVERLAP = 64  # 文本切分块重叠大小（字符数）

# ========== 检索配置 ==========
TOP_K = 5  # 检索返回的最相似文档数量
RERANK_TOP_K = 3  # 重排序后保留的最相似文档数量
HYBRID_WEIGHT_DENSE = 0.5  # 稠密向量检索权重
HYBRID_WEIGHT_SPARSE = 0.3  # 稀疏向量检索权重
HYBRID_WEIGHT_TABLE = 0.3  # 表格检索权重（提高以优先表格内容）

# ========== 验证问题列表 ==========
TEST_QUESTIONS = [  # 测试问题列表，用于验证 RAG 检索效果
    {"id": 1, "question": "武汉力源信息技术股份有限公司本次发行股数是多少，占发行后总股本的比例是多少？"},  # 问题1：发行股数与占比
    {"id": 2, "question": "武汉力源信息技术股份有限公司本次募集资金拟投资哪些项目？"},  # 问题2：募集资金投向
    {"id": 3, "question": "与武汉力源信息技术股份有限公司存在控制关系的关联方是谁，持股比例和本公司关系是什么？"},  # 问题3：控制关系关联方
    {"id": 4, "question": "与武汉力源信息技术股份有限公司不存在控制关系的关联方企业有哪些？"},  # 问题4：非控制关系关联方
    {"id": 260, "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？"},  # 问题260：军用领域收入
    {"id": 95, "question": "武汉兴图新科电子股份有限公司参与制定了哪个技术标准？"},  # 问题95：参与技术标准
    {"id": 33, "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入占主营业务收入的比重分别是多少？"},  # 问题33：军用收入占比
    {"id": 34, "question": "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的上游涉及哪些企业？"},  # 问题34：行业上游企业
    {"id": 957, "question": "武汉兴图新科电子股份有限公司在哪个领域已经成为重要供应商？"},  # 问题957：重要供应商领域
    {"id": 793, "question": "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的下游主要包括哪些行业？"},  # 问题793：行业下游
    {"id": 795, "question": "武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖？"},  # 问题795：科技进步奖
    {"id": 543, "question": "武汉兴图新科电子股份有限公司注册资本是多少？"},  # 问题543：注册资本
    {"id": 531, "question": "武汉兴图新科电子股份有限公司法定代表人是谁？"},  # 问题531：法定代表人
    {"id": 207, "question": "武汉兴图新科电子股份有限公司计划使用本次发行募集资金的多少用于补充流动资金？"},  # 问题207：补充流动资金
]  # 列表结束


def ensure_dirs():  # 确保输出目录存在的函数
    """确保所有输出目录存在"""
    for d in [OUTPUT_DIR, MINERU_OUTPUT_DIR, TEMPLATES_DIR]:  # 遍历三个输出目录
        os.makedirs(d, exist_ok=True)  # 创建目录（已存在时不报错）


def log(msg: str, tag: str = "INFO"):  # 带时间戳的日志输出函数
    """带时间戳的日志输出"""
    from datetime import datetime  # 导入日期时间模块
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 格式化当前时间戳
    print(f"[{ts}] [{tag}] {msg}")  # 打印带时间戳和标签的日志消息


def is_windows():  # 判断是否为 Windows 系统的函数
    return sys.platform == "win32"  # 返回系统平台判断结果
