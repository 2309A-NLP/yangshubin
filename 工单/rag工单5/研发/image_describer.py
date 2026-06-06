"""
image_describer.py - RAG工单5 图片描述生成模块（使用MiMo多模态API）
工单编号: 人工智能NLP-RAG-Query理解优化任务
完成需求: 对PDF提取的图片（组织结构图/表格等）生成文字描述，转为可检索文本块
功能说明: 使用MiMo mimo-v2-omni模型描述图片，结果缓存到文件
"""

import logging  # 日志
import json     # JSON缓存
import os       # 文件路径
import time     # 计时和重试延时
import base64   # 图片base64编码
from pathlib import Path  # 路径处理

# 导入配置
from config import (
    MIMO_API_KEY, MIMO_BASE_URL, MIMO_VL_MODEL,
    PROJECT_DIR, OUTPUT_DIR, LOG_FORMAT, LOG_DATE_FORMAT
)

# 设置日志
logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT, level=logging.INFO)
logger = logging.getLogger("image_describer")

# 图片目录和缓存路径
EXTRACTED_IMAGES_DIR = PROJECT_DIR / "extracted_images"
DESCRIPTION_CACHE_PATH = OUTPUT_DIR / "image_descriptions.json"


def encode_image_base64(image_path):
    """
    将图片文件编码为base64 data URI
    参数:
        image_path: Path对象，图片文件路径
    返回:
        str: data:image/xxx;base64,xxxx 格式的URI
    """
    # 根据扩展名获取MIME类型
    ext = image_path.suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".gif": "image/gif",
        ".bmp": "image/bmp", ".webp": "image/webp",
    }
    mime = mime_map.get(ext, "image/jpeg")

    # 读取图片并base64编码
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{data}"


def parse_image_filename(filename):
    """
    从文件名解析来源PDF和页码
    文件名格式: 招股说明书1_p0546_i0029.jpeg
    返回:
        dict: {source_pdf, page_num} 或 None
    """
    name = filename.stem  # 去掉扩展名
    parts = name.split("_")
    if len(parts) >= 3:
        source_pdf = parts[0] + ".pdf"
        page_part = parts[1]
        page_num = int(page_part[1:]) if page_part.startswith("p") else 0
        return {"source_pdf": source_pdf, "page_num": page_num}
    return None


def describe_single_image(image_path, retries=2):
    """调用MiMo多模态API描述单张图片"""
    image_uri = encode_image_base64(image_path)

    # 描述提示词：重点关注表格数据和组织结构
    prompt_text = (
        "请用中文详细描述这张图片的全部内容。重点包括：\n"
        "1. 如果是表格：列出所有行列数据和数字\n"
        "2. 如果是组织结构图：列出所有部门名称和层级关系\n"
        "3. 如果是流程图：描述流程步骤和箭头指向\n"
        "4. 所有文字内容、数字、百分比等数据信息"
    )
    # 带重试的API调用
    for attempt in range(retries + 1):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=MIMO_API_KEY, base_url=MIMO_BASE_URL)

            # 构造多模态消息：文本+图片base64
            response = client.chat.completions.create(
                model=MIMO_VL_MODEL,   # mimo-v2-omni 多模态模型
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": image_uri}},
                    ],
                }],
                max_tokens=512,
                temperature=0.1,
                timeout=30,
            )

            # 提取返回的描述文本
            text = response.choices[0].message.content.strip()
            if text:
                return text
            else:
                logger.warning(f"API返回空内容: {image_path.name}")

        except Exception as e:
            logger.warning(f"描述失败 (第{attempt+1}次): {image_path.name} - {e}")
            if attempt < retries:
                time.sleep(2)

    return None


def _save_descriptions(descriptions):
    """保存描述结果到缓存JSON文件"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(DESCRIPTION_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(descriptions, f, ensure_ascii=False, indent=2)


def _descriptions_to_chunks(descriptions):
    """将图片描述列表转为chunk格式（与text_chunker兼容）"""
    chunks = []
    for i, desc in enumerate(descriptions):
        content = f"[图片描述 - {desc['img_filename']}] {desc['description']}"
        chunks.append({
            "content": content, "index": i,
            "source_pdf": desc["source_pdf"],
            "page_num": desc["page_num"],
            "is_image_desc": True,
        })
    return chunks


def describe_all_images(force=False):
    """描述所有提取的图片，返回图片描述文本块"""
    if not force and DESCRIPTION_CACHE_PATH.exists():
        try:
            with open(DESCRIPTION_CACHE_PATH, "r", encoding="utf-8") as f:
                cached = json.load(f)
            logger.info(f"使用缓存: {len(cached)} 条图片描述")
            return _descriptions_to_chunks(cached)
        except Exception:
            logger.info("缓存读取失败，重新生成")

    # 扫描图片目录
    if not EXTRACTED_IMAGES_DIR.exists():
        logger.warning(f"图片目录不存在: {EXTRACTED_IMAGES_DIR}")
        return []

    image_files = sorted(Path(EXTRACTED_IMAGES_DIR).glob("*"))
    exts = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    image_files = [f for f in image_files if f.suffix.lower() in exts]
    logger.info(f"共发现 {len(image_files)} 张图片，开始生成描述...")

    descriptions = []
    total = len(image_files)

    for i, img_path in enumerate(image_files):
        file_info = parse_image_filename(img_path)
        if not file_info:
            logger.warning(f"无法解析文件名: {img_path.name}")
            continue

        logger.info(f"[{i+1}/{total}] 描述: {img_path.name}")
        desc = describe_single_image(img_path)

        if desc:
            descriptions.append({
                "img_filename": img_path.name,
                "source_pdf": file_info["source_pdf"],
                "page_num": file_info["page_num"],
                "description": desc,
                "description_length": len(desc),
            })
            logger.info(f"  ✅ 描述成功: {len(desc)} 字")
        else:
            logger.warning(f"  ❌ 描述失败: {img_path.name}")

        # 每10张保存一次进度
        if (i + 1) % 10 == 0:
            _save_descriptions(descriptions)

    # 最终保存
    _save_descriptions(descriptions)
    logger.info(f"图片描述完成! 成功: {len(descriptions)}/{total}")

    return _descriptions_to_chunks(descriptions)


if __name__ == "__main__":
    """单独测试图片描述功能"""
    chunks = describe_all_images(force=True)
    print(f"\n共生成 {len(chunks)} 个图片描述chunks:")
    for c in chunks[:3]:
        print(f"  [页{c['page_num']}][{c['source_pdf']}] {c['content'][:80]}...")
