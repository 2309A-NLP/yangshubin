"""
模块功能: 等待 Milvus 服务就绪脚本
轮询 Milvus 连接，每 2 秒一次，最多 60 次（120 秒超时）
成功返回退出码 0，超时返回 1
可单独运行: python wait_for_milvus.py
工单编号: 人工智能NLP-RAG-金融问答系统部署
"""

import sys       # 系统接口
import time      # 时间测量
from pymilvus import connections, utility  # Milvus 客户端

# === 配置区域 ===
# 读取环境变量，Docker 内默认 milvus-standalone
MILVUS_HOST = "milvus-standalone"
MILVUS_PORT = 19530
MAX_RETRIES = 60       # 最大轮询次数
RETRY_INTERVAL = 2     # 每次轮询间隔（秒）
TIMEOUT_SECONDS = MAX_RETRIES * RETRY_INTERVAL  # 总超时时间 120s


def wait_for_milvus() -> int:
    """轮询等待 Milvus 服务就绪

    Returns:
        0: 就绪成功
        1: 超时未就绪
    """
    print(f"等待 Milvus 服务就绪...")
    print(f"  地址: {MILVUS_HOST}:{MILVUS_PORT}")
    print(f"  超时: {TIMEOUT_SECONDS}s ({MAX_RETRIES} 次, 每 {RETRY_INTERVAL}s)")

    start_time = time.time()

    # 循环轮询，直到连接成功或超时
    for attempt in range(1, MAX_RETRIES + 1):
        elapsed = int(time.time() - start_time)
        try:
            # 尝试连接 Milvus 服务
            connections.connect(
                alias="wait_check",
                host=MILVUS_HOST,
                port=MILVUS_PORT,
                timeout=RETRY_INTERVAL,
            )
            # 验证服务是否真正响应
            if utility.ping_server():
                connections.disconnect("wait_check")
                print(f"\n  ✅ Milvus 已就绪! (耗时: {elapsed}s, 尝试: {attempt} 次)")
                return 0
            else:
                connections.disconnect("wait_check")
                print(f"  [{attempt}/{MAX_RETRIES}] 服务未响应, {RETRY_INTERVAL}s 后重试...")
        except Exception as e:
            print(f"  [{attempt}/{MAX_RETRIES}] 连接失败: {e}, {RETRY_INTERVAL}s 后重试...")

        # 最后一次尝试不再 sleep
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_INTERVAL)

    # 超时退出
    print(f"\n  ❌ Milvus 在 {TIMEOUT_SECONDS}s 内未就绪，启动终止。")
    return 1


if __name__ == "__main__":
    exit_code = wait_for_milvus()
    sys.exit(exit_code)
