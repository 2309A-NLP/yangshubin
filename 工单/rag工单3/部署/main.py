"""
主入口模块 - 串联所有子模块：解析→表格提取→切分→向量化→检索→问答
此模块是 RAG 系统的总入口和流水线调度器，定义了从 PDF 解析、表格提取、
文本切分、向量嵌入、Milvus 存储到检索问答的完整 6 步处理流程，
同时提供 Flask Web 服务接口供用户交互使用。
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
"""
import os  # 导入操作系统模块，用于路径操作
import sys  # 导入系统模块，用于系统参数
# ===== 路径桥接：将所有子目录加入 sys.path =====
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ["设计", "研发", "测试", "优化", "部署"]:
    _p = os.path.join(_ROOT, _sub)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import json  # 导入 JSON 模块，用于数据序列化
import time  # 导入时间模块，用于计时和日志
import argparse  # 导入命令行参数解析模块
from datetime import datetime  # 导入日期时间类，用于时间戳
from config import (  # 导入项目配置
    PDF_PATH, PDF_PATHS, OUTPUT_DIR, TEST_QUESTIONS,  # PDF路径、输出目录、测试问题集
    TOP_K, ensure_dirs, log,  # 目录保证函数和日志函数
)


def step_parse_pdf():
    """Step 1: 使用 PyMuPDF 解析 PDF"""
    log("=" * 60, "PIPELINE")  # 打印分隔线日志
    log("Step 1/6: PDF 解析 (MinerU API → PyMuPDF 降级)", "PIPELINE")  # 打印步骤标题
    log("=" * 60, "PIPELINE")  # 打印分隔线日志

    from pdf_parser import parse_pdf, save_parsed_output  # 延迟导入 PDF 解析模块
    md_content, raw_tables = parse_pdf()  # 调用解析函数，获取 Markdown 内容和原始表格数据
    save_parsed_output(md_content, raw_tables)  # 保存解析结果到文件
    log(f"PDF 解析完成，共 {len(md_content)} 字符，{len(raw_tables)} 个表格", "PIPELINE")  # 记录解析完成日志
    return md_content, raw_tables  # 返回解析结果


def step_extract_tables(raw_tables: list):
    """Step 2: 表格解析 - 将 PyMuPDF 表格转为检索用文本块"""
    log("=" * 60, "PIPELINE")  # 打印分隔线日志
    log("Step 2/6: 表格解析", "PIPELINE")  # 打印步骤标题
    log("=" * 60, "PIPELINE")  # 打印分隔线日志

    from table_parser import tables_to_text_blocks, save_tables_json, save_tables_csv  # 延迟导入表格解析模块
    table_blocks = tables_to_text_blocks(raw_tables)  # 将原始表格数据转换为检索用文本块
    save_tables_json(raw_tables)  # 保存表格 JSON 文件
    log(f"提取到 {len(raw_tables)} 个表格，生成 {len(table_blocks)} 个文本块", "PIPELINE")  # 记录表格提取日志
    return raw_tables, table_blocks  # 返回原始表格和文本块


def step_split_text(md_content: str, table_blocks: list):
    """Step 3: 文本切分"""
    log("=" * 60, "PIPELINE")  # 打印分隔线日志
    log("Step 3/6: 文本切分", "PIPELINE")  # 打印步骤标题
    log("=" * 60, "PIPELINE")  # 打印分隔线日志

    from pdf_parser import extract_content_sections  # 导入章节提取函数
    from text_splitter import (  # 导入文本切分相关函数
        split_by_sections,  # 按章节切分文本
        merge_text_and_table_chunks,  # 合并文本块和表格块
        save_chunks_json,  # 保存切分结果为 JSON
        save_chunks_text,  # 保存切分结果为文本文件
    )

    sections = extract_content_sections(md_content)  # 从 Markdown 中提取章节结构
    text_chunks = split_by_sections(sections)  # 按章节切分为文本块
    merged_chunks = merge_text_and_table_chunks(text_chunks, table_blocks)  # 合并文本块和表格块
    save_chunks_json(merged_chunks)  # 保存合并后的片断为 JSON
    save_chunks_text(merged_chunks)  # 保存合并后的片断为文本文件
    log(f"共 {len(merged_chunks)} 个片段 (文本 {len(text_chunks)} + 表格 {len(table_blocks)})", "PIPELINE")  # 记录切分统计
    return merged_chunks  # 返回合并后的片段列表


def step_build_embeddings(chunks: list):
    """Step 4: BGE-M3 向量化"""
    log("=" * 60, "PIPELINE")  # 打印分隔线日志
    log("Step 4/6: BGE-M3 向量化", "PIPELINE")  # 打印步骤标题
    log("=" * 60, "PIPELINE")  # 打印分隔线日志

    from embedding import BGEM3Embedding  # 延迟导入嵌入模型
    embedder = BGEM3Embedding()  # 创建 BGE-M3 嵌入模型实例
    texts = [c["text"] for c in chunks]  # 提取所有片段的文本内容
    embeddings = embedder.encode_dense(texts)  # 对文本进行稠密向量编码
    log(f"向量化完成，形状: {embeddings.shape}", "PIPELINE")  # 记录向量化结果形状
    return embedder, embeddings  # 返回嵌入模型和向量矩阵


def step_store_vectors(chunks: list, embeddings):
    """Step 5: 存储到 Milvus"""
    log("=" * 60, "PIPELINE")  # 打印分隔线日志
    log("Step 5/6: 向量存储 (Milvus)", "PIPELINE")  # 打印步骤标题
    log("=" * 60, "PIPELINE")  # 打印分隔线日志

    from vector_store import MilvusVectorStore  # 延迟导入向量存储模块
    store = MilvusVectorStore()  # 创建 Milvus 向量存储实例
    store.create_collection(drop_if_exists=True)  # 创建集合（若存在则删除重建）
    store.insert(chunks, embeddings)  # 将片段和向量插入集合
    stats = store.get_collection_stats()  # 获取集合kan统计信息
    log(f"Milvus 存储完成: {stats}", "PIPELINE")  # 记录存储完成日志
    return store  # 返回向量存储实例


def step_run_qa(embedder, store):
    """Step 6: 检索+问答"""
    log("=" * 60, "PIPELINE")  # 打印分隔线日志
    log("Step 6/6: 检索 + 问答", "PIPELINE")  # 打印步骤标题
    log("=" * 60, "PIPELINE")  # 打印分隔线日志

    from retriever import HybridRetriever  # 导入混合检索器
    from llm_qa import DeepSeekQA  # 导入 DeepSeek 问答模块

    retriever = HybridRetriever(embedder, store)  # 创建混合检索器实例
    qa = DeepSeekQA()  # 创建问答模块实例

    def get_context(query):  # 定义获取检索上下文的内部函数
        """获取检索上下文"""
        results = retriever.retrieve(query, top_k=5)  # 检索前 5 条结果
        return [r for r in results]  # 返回结果列表

    # 运行测试问题
    results = qa.batch_qa(TEST_QUESTIONS, get_context)  # 批量问答处理
    qa.save_results(results)  # 保存问答结果到文件

    # 打印结果摘要
    log(f"共处理 {len(results)} 个问题", "PIPELINE")  # 记录处理问题总数
    for r in results:  # 遍历每个问答结果
        conf = r.get("confidence", 0)  # 获取置信度
        lat = r.get("latency", 0)  # 获取延迟
        label = "✅" if conf > 0.5 else "⚠️"  # 根据置信度选择图标
        log(f"{label} Q{r.get('id','')}: {r.get('question','')[:40]}... | 置信度:{conf:.2f} | 延迟:{lat:.2f}s", "QA")  # 记录问答摘要
        print(f"   答案: {r.get('answer','')[:150]}")  # 打印答案前150字符
        print()  # 打印空行

    return results  # 返回问答结果列表


def run_pipeline(steps: list = None):
    """
    运行完整流水线

    Args:
        steps: 要运行的步骤列表，None=全部运行
               ['parse', 'tables', 'split', 'embed', 'store', 'qa']
    """
    all_steps = ['parse', 'tables', 'split', 'embed', 'store', 'qa']  # 定义所有可用的步骤名称
    if steps is None:  # 如果未指定步骤
        steps = all_steps  # 默认执行全部步骤

    ensure_dirs()  # 确保所有输出目录存在
    log("=" * 60, "PIPELINE")  # 打印分隔线日志
    log(f"RAG 表格解析与检索优化流水线启动", "PIPELINE")  # 打印流水线启动日志
    log(f"PDF文件 ({len(PDF_PATHS)}个):", "PIPELINE")  # 打印 PDF 文件列表
    for p in PDF_PATHS:
        log(f"  - {os.path.basename(p)}", "PIPELINE")
    log(f"输出目录: {OUTPUT_DIR}", "PIPELINE")  # 打印输出目录路径
    log(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "PIPELINE")  # 打印当前时间
    log("=" * 60, "PIPELINE")  # 打印分隔线日志

    pipeline_start = time.time()  # 记录流水线开始时间
    md_content = None  # 初始化 Markdown 内容变量
    tables = None  # 初始化表格变量
    table_blocks = None  # 初始化表格文本块变量
    chunks = None  # 初始化片段列表变量
    embedder = None  # 初始化嵌入模型变量
    embeddings = None  # 初始化向量矩阵变量
    store = None  # 初始化向量存储变量

    if 'parse' in steps:  # 如果包含解析步骤
        md_content, raw_tables = step_parse_pdf()  # 执行 PDF 解析
    if 'tables' in steps and raw_tables is not None:  # 如果包含表格解析且已有原始表格数据
        tables, table_blocks = step_extract_tables(raw_tables)  # 执行表格解析
    if 'split' in steps and md_content and table_blocks is not None:  # 如果包含切分步骤且已有 Markdown 和表格块
        chunks = step_split_text(md_content, table_blocks)  # 执行文本切分
    if 'embed' in steps and chunks:  # 如果包含向量化步骤且已有片段
        embedder, embeddings = step_build_embeddings(chunks)  # 执行向量化
    if 'store' in steps and chunks is not None and embeddings is not None:  # 如果包含存储步骤且已有片段和向量
        store = step_store_vectors(chunks, embeddings)  # 执行向量存储
    if 'qa' in steps and embedder and store:  # 如果包含问答步骤且已有嵌入模型和存储
        step_run_qa(embedder, store)  # 执行检索问答

    total_time = time.time() - pipeline_start  # 计算总耗时
    log("=" * 60, "PIPELINE")  # 打印分隔线日志
    log(f"流水线总耗时: {total_time:.1f}s", "PIPELINE")  # 记录总耗时
    log("=" * 60, "PIPELINE")  # 打印分隔线日志

    return embedder, store  # 返回嵌入模型和向量存储实例


def start_web_server(embedder=None, store=None, language="auto"):
    """启动 Flask Web 服务（支持多语言）"""
    log("启动 Web 服务器...", "WEB")  # 记录启动日志

    from flask import Flask, request, jsonify, render_template  # 延迟导入 Flask 模块
    from retriever import HybridRetriever  # 导入混合检索器
    from llm_qa import DeepSeekQA  # 导入问答模块

    app = Flask(__name__, template_folder="templates")  # 创建 Flask 应用实例，指定模板目录

    # 延迟初始化
    _embedder = embedder  # 保存外部传入的嵌入模型
    _store = store  # 保存外部传入的向量存储
    _retriever = None  # 初始化检索器为 None
    _qa = None  # 初始化问答模块为 None
    _language = language  # 保存语言设置

    def _lazy_init():  # 定义延迟初始化内部函数
        nonlocal _embedder, _store, _retriever, _qa  # 声明使用外部变量
        if _retriever is None:  # 如果检索器尚未初始化
            if _embedder is None:  # 如果嵌入模型为空
                from embedding import BGEM3Embedding  # 导入嵌入模型类
                _embedder = BGEM3Embedding()  # 创建默认嵌入模型
            if _store is None:  # 如果向量存储为空
                from vector_store import MilvusVectorStore  # 导入向量存储类
                _store = MilvusVectorStore()  # 创建默认向量存储
                _store.collection = _store.collection_name  # 设置集合名称属性
                from pymilvus import Collection  # 导入 PyMilvus 集合类
                _store.collection = Collection(_store.collection_name)  # 重新加载集合
                _store.collection.load()  # 加载集合数据到内存
            _retriever = HybridRetriever(_embedder, _store)  # 创建混合检索器
            _qa = DeepSeekQA()  # 创建问答模块实例

    @app.route("/")  # 注册根路由
    def index():
        return render_template("index.html")  # 返回 HTML 首页

    @app.route("/api/qa", methods=["POST"])  # 注册问答 API 路由（POST 方法）
    def api_qa():  # 定义问答 API 处理函数
        _lazy_init()  # 确保已初始化
        data = request.get_json() or {}  # 获取请求中的 JSON 数据
        question = data.get("question", "").strip()  # 提取并清理问题文本
        lang = data.get("language", _language)  # 提取语言参数，默认使用启动时设置
        if not question:  # 如果问题为空
            return jsonify({"error": "请输入问题" if lang != "en" else "Please enter a question"}), 400

        ctx = _retriever.retrieve(question, top_k=5 if lang == "en" else TOP_K)  # 执行检索获取上下文
        if not ctx:  # 如果未检索到结果
            empty_msg = "未找到相关参考信息"
            ctx = [{"text": empty_msg, "score": 0,
                     "source_type": "text", "section_title": ""}]

        result = _qa.generate_answer(question, ctx, language=lang)  # 生成答案
        return jsonify(result)  # 返回 JSON 格式的问答结果

    @app.route("/api/test", methods=["GET"])  # 注册测试 API 路由（GET 方法）
    def api_test():  # 定义测试 API 处理函数
        _lazy_init()  # 确保已初始化
        lang = request.args.get("language", _language)
        results = []  # 初始化结果列表
        for q in TEST_QUESTIONS:  # 遍历测试问题列表
            ctx = _retriever.retrieve(q["question"], top_k=TOP_K)  # 执行检索
            result = _qa.generate_answer(q["question"], ctx, language=lang)  # 生成答案
            result["id"] = q["id"]  # 设置问题 ID
            result["question"] = q["question"]  # 设置问题文本
            results.append(result)  # 加入结果列表
        return jsonify({"total": len(results), "results": results, "language": lang})

    app.run(host="0.0.0.0", port=5000, debug=False)  # 启动 Flask Web 服务（监听所有接口，端口 5000）


if __name__ == "__main__":  # 如果作为主程序运行
    parser = argparse.ArgumentParser(description="RAG 表格解析与检索优化系统")
    parser.add_argument("--mode", choices=["pipeline", "web", "all"],
                        default="web", help="运行模式（默认web，不跑流水线）")
    parser.add_argument("--steps", nargs="+",
                        help="流水线步骤: parse tables split embed store qa")
    parser.add_argument("--lang", choices=["auto", "zh", "en"],
                        default="auto", help="问答语言（auto=自动检测）")
    args = parser.parse_args()

    # ====== 启动校验 ======
    # 检查PDF源文件是否存在
    missing_pdfs = [p for p in PDF_PATHS if not os.path.exists(p)]
    if missing_pdfs:
        log(f"警告：以下PDF文件不存在: {missing_pdfs}", "WARN")
        log("请将招股说明书PDF文件放入项目目录", "WARN")

    if args.mode == "pipeline":
        run_pipeline(steps=args.steps)
    elif args.mode == "web":
        start_web_server(language=args.lang)
    else:
        embedder, store = run_pipeline()
        start_web_server(embedder, store, language=args.lang)
