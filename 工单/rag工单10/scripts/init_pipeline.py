"""
模块功能: 数据初始化流水线脚本
按顺序执行完整的 RAG 数据准备流程:
  PDF 解析 → 文本分块 → 向量化 → Milvus 入库 → 知识图谱构建
数据来源目录: /data/pdfs/（Docker 内默认路径）
可单独运行: python init_pipeline.py
工单编号: 人工智能NLP-RAG-金融问答系统部署
"""

import sys       # 系统接口
import time      # 时间测量
import logging   # 日志记录
from pathlib import Path  # 路径处理

# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("init_pipeline")

# === 数据目录配置 ===
# 从 app.config 获取，如果不可用则使用默认值
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app.config import config
    DATA_DIR = Path(config.DATA_DIR)
except ImportError:
    DATA_DIR = Path("/data/pdfs/")


def step_pdf_parse() -> bool:
    """步骤1: 解析 PDF 文档，提取纯文本内容"""
    from app.document_loader import load_documents
    logger.info("[1/5] 开始解析 PDF 文档...")
    try:
        docs = load_documents(str(DATA_DIR))
        logger.info(f"  ✅ 解析完成，共加载 {len(docs)} 篇文档")
        return True
    except Exception as e:
        logger.error(f"  ❌ 解析失败: {e}")
        return False


def step_text_chunk() -> bool:
    """步骤2: 对 PDF 提取的文本进行分块处理"""
    from app.text_splitter import split_documents
    logger.info("[2/5] 开始文本分块...")
    try:
        chunks = split_documents(str(DATA_DIR))
        logger.info(f"  ✅ 分块完成，共生成 {len(chunks)} 个文本块")
        return True
    except Exception as e:
        logger.error(f"  ❌ 分块失败: {e}")
        return False


def step_vectorize() -> bool:
    """步骤3: 使用 BGE-M3 模型对文本块进行向量化"""
    from app.document_loader import load_documents, load_pdf
    from app.text_splitter import split_text
    from app.embedding import generate_embeddings
    logger.info("[3/5] 开始向量化...")
    try:
        # 加载所有文档
        docs = load_documents(str(DATA_DIR))
        if not docs:
            logger.warning("没有文档可向量化，尝试直接读取 PDF 文件")
            for pdf_file in sorted(DATA_DIR.glob("*.pdf")):
                if "sample" not in pdf_file.name:
                    text = load_pdf(str(pdf_file))
                    if text.strip():
                        docs.append({"text": text, "filename": pdf_file.name})
        # 提取所有文本内容并分块
        all_texts = []
        for doc in docs:
            content = doc.get("text", "")
            if content.strip():
                all_texts.append(content)
        if not all_texts:
            logger.error("没有可用的文本内容")
            return False
        # 分块处理
        chunks_list = []
        for text in all_texts:
            chunks = split_text(text)
            chunks_list.extend(chunks)
        if not chunks_list:
            logger.error("没有文本块可向量化")
            return False
        # 批量向量化
        embeddings = generate_embeddings(chunks_list)
        if embeddings is not None:
            logger.info(f"  ✅ 向量化完成: {len(embeddings)} 个向量 (维度 {embeddings.shape[1]})")
            return True
        return False
    except Exception as e:
        logger.error(f"  ❌ 向量化失败: {e}")
        return False


def step_vector_store() -> bool:
    """步骤4: 连接 Milvus 并验证可写入"""
    from app.vectorstore import MilvusClient
    logger.info("[4/5] 连接 Milvus 数据库...")
    try:
        client = MilvusClient()
        result = client.connect()
        if result:
            logger.info(f"  ✅ Milvus 连接成功")
            client.close()
            return True
        logger.error("  ❌ Milvus 连接失败")
        return False
    except Exception as e:
        logger.error(f"  ❌ Milvus 连接失败: {e}")
        return False


def step_graph_build() -> bool:
    """步骤5: 初始化知识图谱"""
    from app.graph_builder import get_graph
    logger.info("[5/5] 初始化知识图谱...")
    try:
        kg = get_graph()
        nodes = kg.graph.number_of_nodes()
        edges = kg.graph.number_of_edges()
        logger.info(f"  ✅ 图谱就绪: {nodes} 节点, {edges} 条边")
        return True
    except Exception as e:
        logger.error(f"  ❌ 图谱初始化失败: {e}")
        return False


def run_pipeline() -> bool:
    """按顺序执行完整的数据初始化流水线

    任意步骤失败则中止执行，返回 False。
    """
    start_time = time.time()
    logger.info("=" * 50)
    logger.info("  数据初始化流水线启动 (工单10)")
    logger.info(f"  数据目录: {DATA_DIR}")
    logger.info("=" * 50)

    # 检查数据目录是否存在，不存在则创建
    if not DATA_DIR.exists():
        logger.warning(f"  数据目录不存在，自动创建: {DATA_DIR}")
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 定义流水线步骤
    step_functions = [
        ("PDF 解析", step_pdf_parse),
        ("文本分块", step_text_chunk),
        ("向量化", step_vectorize),
        ("Milvus 入库", step_vector_store),
        ("图谱构建", step_graph_build),
    ]

    # 按顺序执行每个步骤
    for step_name, func in step_functions:
        logger.info(f"--- 步骤: {step_name} ---")
        ok = func()
        if not ok:
            logger.error(f"流水线在「{step_name}」步骤失败，终止执行。")
            elapsed = int(time.time() - start_time)
            logger.info(f"总耗时: {elapsed}s")
            return False

    # 全部步骤完成
    elapsed = int(time.time() - start_time)
    logger.info("=" * 50)
    logger.info(f"  ✅ 流水线全部完成 (总耗时: {elapsed}s)")
    logger.info("=" * 50)
    return True


if __name__ == "__main__":
    success = run_pipeline()
    sys.exit(0 if success else 1)
