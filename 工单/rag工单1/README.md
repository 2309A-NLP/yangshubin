# RAG PDF 问答系统

基于 BGE-M3 + DeepSeek + Milvus 的智能文档问答系统

**工单编号：** 人工智能NLP-RAG-基于PDF文档的问答系统

## 📋 系统概述

本系统基于 RAG（检索增强生成）技术，针对《招股说明书》等专业PDF文档进行智能问答。系统结合向量检索与大语言模型，能够从文档中精准提取信息并生成准确回答，有效避免纯LLM的"幻觉"问题。

## 🏗️ 系统架构

```
用户提问 → 查询理解(意图识别+消歧) → 向量编码(BGE-M3)
    → Milvus向量检索 → 上下文拼接 → DeepSeek LLM生成回答
```

### 核心模块

```
rag工单1/
├── main.py              # 主入口（build/query/eval/web/all）
├── config.py            # 全局配置（路径、模型、参数）
├── pdf_parser.py        # PDF解析模块（PyMuPDF提取文本+表格）
├── text_splitter.py     # 文本切分模块（固定窗口+段落语义）
├── embedding_model.py   # 嵌入模型（BGE-M3 1024维向量）
├── vector_store.py      # 向量存储（Milvus CRUD + 检索）
├── query_processor.py   # 查询理解（意图识别+消歧+分解）
├── retriever.py         # 检索器（整合嵌入+Milvus+查询理解）
├── llm_qa.py            # LLM问答（RAG模式 + 纯LLM模式）
├── evaluator.py         # 评估模块（RAG vs 纯LLM 对比）
├── database_builder.py  # 知识库构建（PDF→切分→嵌入→入库）
├── app.py               # Flask Web应用
├── view_data.py         # 数据查看工具
├── templates/
│   └── index.html       # Web前端页面
└── output/              # 产出物目录
    ├── 00_处理统计.txt
    ├── 01_全文提取.txt
    ├── 02_文本块总览.json
    └── 03_文本块明细.txt
```

## 🔧 技术选型

| 组件 | 技术方案 | 说明 |
|------|---------|------|
| PDF解析 | PyMuPDF (fitz) | 支持文字、表格提取 |
| 文本切分 | 自研分块器 | 固定窗口512字符 + 64字符重叠 |
| 嵌入模型 | BGE-M3 | 1024维稠密向量，支持中英文 |
| 向量数据库 | Milvus | IVF_FLAT索引，内积检索 |
| LLM引擎 | DeepSeek Chat | OpenAI兼容API |
| Web框架 | Flask | 延迟初始化，避免启动卡顿 |
| 前端 | HTML + JS | 独立模板文件，支持对比模式 |

## 🚀 使用方法

### 环境依赖

```bash
pip install pymupdf sentence-transformers pymilvus flask openai -i https://mirrors.aliyun.com/pypi/simple/
```

### 前置条件

1. **Milvus 向量数据库** — Docker 启动：
   ```bash
   docker run -d --name milvus-standalone \
     -p 19530:19530 -p 9091:9091 \
     milvusdb/milvus:v2.3.10
   ```

2. **BGE-M3 模型** — 从 ModelScope 下载到桌面：
   ```bash
   python -c "from modelscope import snapshot_download; snapshot_download('BAAI/bge-m3', local_dir='C:/Users/31326/Desktop/bge-m3')"
   ```

3. **DeepSeek API Key** — 在 `config.py` 中配置

### 运行命令

```bash
# 构建知识库（PDF解析→切分→嵌入→入库）
python main.py build

# 交互式问答（命令行）
python main.py query

# RAG vs 纯LLM 评估
python main.py eval

# 启动Web界面（http://localhost:5000）
python main.py web

# 一键执行：构建 + Web
python main.py all
```

## ⚙️ 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| CHUNK_SIZE | 512 | 每个文本块最大字符数 |
| CHUNK_OVERLAP | 64 | 相邻块重叠字符数 |
| TOP_K | 5 | 每次检索返回的候选块数量 |
| BGE_M3_DIM | 1024 | 向量维度 |
| MILVUS_COLLECTION | rag_pdf_qa | Milvus集合名称 |

## 📈 评估结果

### 测试问题（工单指定10题）

| ID | 问题 |
|----|------|
| 260 | 报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？ |
| 95 | 武汉兴图新科电子股份有限公司参与制定了哪个技术标准？ |
| 33 | 报告期内，军用领域的收入占主营业务收入的比重分别是多少？ |
| 34 | 电子信息行业的上游涉及哪些企业？ |
| 957 | 武汉兴图新科在哪个领域已经成为重要供应商？ |
| 793 | 电子信息行业的下游主要包括哪些行业？ |
| 795 | 哪个工程荣获了国家科技进步一等奖？ |
| 543 | 武汉兴图新科注册资本是多少？ |
| 531 | 法定代表人是谁？ |
| 207 | 计划使用多少募集资金补充流动资金？ |

### 评估指标

| 模式 | 准确率 | 完整性 | 相关性 | 平均分 |
|------|--------|--------|--------|--------|
| RAG模式 | 4.90/5 | 4.70/5 | 5.00/5 | **4.87/5** |
| 纯LLM模式 | 2.90/5 | 3.70/5 | 4.80/5 | 3.80/5 |

**结论：10/10 问题 RAG 模式优于纯 LLM 模式**，RAG 准确率提升 69%，有效避免幻觉问题。

## 📝 注意事项

1. **首次运行**需先执行 `python main.py build` 构建知识库
2. **Milvus** 需要 Docker 环境，确保端口 19530 可用
3. **GPU 加速**：BGE-M3 编码使用 CUDA，首次加载约需 10 秒
4. **响应时间**：热启动后 RAG 模式约 2-3 秒，冷启动（首次加载模型）约 10 秒
