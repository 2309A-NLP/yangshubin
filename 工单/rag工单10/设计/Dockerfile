# Docker 镜像构建文件
# 轻量策略：小包直接安装（flask/openai/numpy等），
# 大包（torch/sentence-transformers）从宿主机挂载
# BGE-M3 模型通过 volume 挂载，不打入镜像
# 工单编号: 人工智能NLP-RAG-金融问答系统部署

# 基础镜像：Python 3.10 slim
FROM python:3.10-slim

# 配置阿里云 PyPI 镜像加速
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
RUN pip config set global.trusted-host mirrors.aliyun.com

# 安装所有轻量 Web/API 依赖（不含 torch/sentence-transformers）
RUN pip install --no-cache-dir --timeout=120 flask flask-cors openai
RUN pip install --no-cache-dir --timeout=120 numpy pymilvus
RUN pip install --no-cache-dir --timeout=120 PyMuPDF networkx

# 设置工作目录
WORKDIR /app

# 复制应用代码
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY requirements.txt ./

# 无 PYTHONPATH 全局设置（避免宿主机 numpy 版本冲突）
# 宿主机大包（torch/sentence-transformers）在 app/app.py 中动态导入

# 默认启动：Docker 入口脚本
CMD ["python", "scripts/docker_entrypoint.py"]
