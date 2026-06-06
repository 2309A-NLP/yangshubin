#!/usr/bin/env python3
"""
入口文件: RAG 金融问答系统启动器
拖入 PyCharm 时默认打开此文件，而非 README.md
默认启动 Web 对话界面（无需额外参数）
用法:
  python 0_run.py             Web 对话界面（默认）
  python 0_run.py --pipeline  完整评估流水线
  python 0_run.py --test      测试模式（2 个问题快速验证）
  python 0_run.py --rebuild   强制重建数据
工单编号: 人工智能NLP-RAG-金融问答系统部署
"""

import sys
from pathlib import Path

# 第一步: 将项目根目录加入 Python 搜索路径，确保 from app.xxx 能正常导入
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# 第二步: 导入主入口模块
from app.app import main

if __name__ == "__main__":
    # 默认走 Web 对话模式，传参则覆盖
    if len(sys.argv) == 1:
        sys.argv.append("--web-only")
    main()
