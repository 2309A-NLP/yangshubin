# RAG 金融问答系统 Docker 部署文档
# 工单编号: 人工智能NLP-RAG-金融问答系统部署
# 基于 GraphRAG 架构，使用 MiMo (小米开放平台) + BGE-M3 + Milvus

## 一、项目概述

基于 GraphRAG 的金融问答系统，支持以下功能：

- **PDF 解析**：读取 CCF 年报 PDF，使用 PyMuPDF 提取结构化文本
- **向量检索**：使用 BGE-M3 模型对文本进行向量化（1024维），存入 Milvus 向量库
- **GraphRAG 问答**：结合向量检索 + 知识图谱（NetworkX）的混合检索增强生成
- **LLM 问答**：调用 MiMo API（小米开放平台，mimo-v2.5-pro）基于上下文生成回答
- **Web 界面**：Flask 提供 REST API 和前端聊天界面
- **Docker 容器化部署**：一键启动所有服务（etcd + minio + Milvus + Flask）

技术栈：Python 3.10 + Flask + Milvus + BGE-M3 + Sentence-Transformers + NetworkX + MiMo API

---

## 二、环境要求

| 组件 | 最低要求 |
|------|----------|
| Docker | 20.10+ |
| Docker Compose | v2+ |
| NVIDIA GPU | 8 GB+（用于 BGE-M3 推理） |
| 系统内存 | 16 GB+ |
| 磁盘空间 | 20 GB+（含 BGE-M3 模型 ~2.2GB） |

> ⚠️ BGE-M3 模型约 2.2 GB，请提前下载到 `C:\Users\31326\Desktop\bge-m3`。

---

## 三、目录结构

```
rag工单10/
├── Dockerfile              # Docker 镜像构建文件（轻量策略）
├── docker-compose.yml      # Docker Compose 服务编排（4个服务）
├── .dockerignore           # Docker 构建排除规则
├── requirements.txt        # Python 依赖
├── README.md               # 本部署文档
├── app/                    # 应用代码（10个模块）
│   ├── config.py           # 全局配置（MiMo + BGE-M3 + Milvus）
│   ├── llm_client.py       # MiMo API 客户端
│   ├── embedding.py        # BGE-M3 文本向量化
│   ├── document_loader.py  # PDF 解析加载
│   ├── text_splitter.py    # 文本分块
│   ├── vectorstore.py      # Milvus 向量数据库客户端
│   ├── graph_builder.py    # 知识图谱（NetworkX）
│   ├── rag_engine.py       # RAG 问答引擎
│   ├── routes.py           # Flask API 路由
│   ├── app.py              # 主入口（整合全部模块）
│   └── templates/          # HTML 模板
├── scripts/                # Docker 部署辅助脚本
│   ├── docker_entrypoint.py  # 容器入口编排
│   ├── check_env.py          # 环境检查
│   ├── wait_for_milvus.py    # 等待 Milvus 就绪
│   ├── init_pipeline.py      # 数据初始化流水线
│   └── healthcheck.py        # 健康检查
└── data/                   # PDF 数据目录
```

---

## 四、快速开始（Docker 部署）

### 4.1 确认 BGE-M3 模型

确保 BGE-M3 模型文件在以下路径：
```
/mnt/c/Users/31326/Desktop/bge-m3/
    ├── config.json
    ├── pytorch_model.bin  (~2.2GB)
    ├── tokenizer.json
    └── ...
```

### 4.2 放入 CCF 年报 PDF

将 PDF 文件放入 data 目录：
```bash
mkdir -p /mnt/c/Users/31326/Desktop/rag工单10/data
cp /path/to/ccf/*.pdf /mnt/c/Users/31326/Desktop/rag工单10/data/
```

### 4.3 一键启动全部服务

```bash
# 进入项目目录
cd /mnt/c/Users/31326/Desktop/rag工单10

# 启动所有容器（后台运行）
docker compose up -d

# 查看启动日志
docker compose logs -f

# 仅查看应用日志
docker compose logs -f rag-app
```

### 4.4 访问服务

启动成功后浏览器访问：
```
http://localhost:5008
```

API 测试命令：
```bash
# 健康检查
curl http://localhost:5008/health

# 问答
curl -X POST http://localhost:5008/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "请分析CCF年报中的营业收入变化"}'
```

---

## 五、配置说明

### 5.1 MiMo API 配置

| 环境变量 | 当前值 | 说明 |
|----------|--------|------|
| `MIMO_API_KEY` | `tp-ccb3f6yj...` | MiMo Token Plan API 密钥 |
| `MIMO_API_BASE` | `https://token-plan-cn.xiaomimimo.com/v1` | API 地址 |
| `MIMO_MODEL` | `mimo-v2.5-pro` | 使用的模型名称 |

### 5.2 数据持久化（Volumes）

| 宿主机路径 | 容器内路径 | 说明 |
|-----------|-----------|------|
| `/mnt/c/.../bge-m3` | `/models/bge-m3` | BGE-M3 模型文件（只读） |
| `rag工单10/data/` | `/data/pdfs/` | CCF 年报 PDF |
| `etcd_data` 卷 | `/etcd` | Milvus 元数据 |
| `minio_data` 卷 | `/data` | Milvus 对象存储 |
| `milvus_data` 卷 | `/var/lib/milvus` | Milvus 向量索引数据 |

### 5.3 架构说明

```
┌──────────────────────────────────────────────────────────┐
│                     Docker Network (rag-net)              │
│                                                           │
│   rag-app (Flask :5008)                                   │
│     │                                                      │
│     ├──[向量检索]──→ milvus-standalone (:19530)             │
│     │                       ├── etcd (:2379)  [元数据]      │
│     │                       └── minio (:9000) [对象存储]    │
│     │                                                      │
│     └──[LLM 问答]──→ MiMo API (mimo-v2.5-pro, 云端)       │
│                                                           │
│   BGE-M3 模型 (挂载: /models/bge-m3, 本地加载推理)           │
└──────────────────────────────────────────────────────────┘
```

---

## 六、服务管理

### 6.1 端口一览

| 服务 | 容器名 | 端口 | 说明 |
|------|--------|------|------|
| **rag-app** | `rag-app` | `5008` | Flask Web 问答服务 |
| **milvus** | `rag-milvus` | `19530` | Milvus 向量数据库 |
| **etcd** | `rag-etcd` | `2379` | Milvus 元数据存储 |
| **minio** | `rag-minio` | `9000` | Milvus 对象存储 |

### 6.2 常用命令

```bash
# 查看所有容器状态
docker compose ps

# 查看所有日志
docker compose logs -f

# 重启某个服务
docker compose restart rag-app

# 停止所有服务
docker compose stop

# 停止并删除所有容器
docker compose down

# 完全清理（含所有 Volume 数据）
docker compose down -v
```

### 6.3 使用 docker run 启动（验收标准）

```bash
# 仅启动 rag-app（假设 Milvus 已运行）
docker run -d --name rag-app \
  -p 5008:5008 \
  -e MIMO_API_KEY="tp-czex5np7bgf6duyuyvqntmw44xcunseatmblffiw9lk9w0jy" \
  -e MIMO_API_BASE="https://token-plan-cn.xiaomimimo.com/v1" \
  -e MIMO_MODEL="mimo-v2.5-pro" \
  -e MILVUS_HOST="milvus-standalone" \
  -e MILVUS_PORT="19530" \
  -e BGE_MODEL_PATH="/models/bge-m3" \
  -e DATA_DIR="/data/pdfs" \
  -v /mnt/c/Users/31326/Desktop/bge-m3:/models/bge-m3 \
  -v /mnt/c/Users/31326/Desktop/rag工单10/data:/data/pdfs \
  -v /usr/lib/python3/dist-packages:/host-packages:ro \
  --network rag-net \
  rag-app:latest
```

---

## 七、注意事项

1. **BGE-M3 模型约 2.2GB**，首次加载可能需要 10-30 秒
2. **MiMo API 需要网络连接**，确保 Docker 容器可访问外网
3. **Milvus 启动顺序**：etcd → minio → milvus-standalone（docker compose 自动处理）
4. **GPU 加速**：如需在 Docker 内使用 GPU，需安装 NVIDIA Container Toolkit
5. **数据目录放 PDF 后**需重启 rag-app 才能生效
