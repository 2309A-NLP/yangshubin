# -*- coding: utf-8 -*-
"""
路径桥接文件 —— 将所有子目录加入 sys.path
在 main.py 最先 import 此文件，之后 from config import ... 等原 import 全部不用改
"""
import sys
import os

# 项目根目录
ROOT = os.path.dirname(os.path.abspath(__file__))

# 将根目录和5个子目录全部加入 sys.path
for sub in ["设计", "研发", "测试", "优化", "部署"]:
    p = os.path.join(ROOT, sub)
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

# 根目录本身也要加入（config.py 在 设计/ 里，但 main.py 在 部署/ 里）
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
