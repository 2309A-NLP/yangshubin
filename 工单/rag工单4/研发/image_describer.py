"""
image_describer.py - RAG工单4 图像描述模块
工单编号: 人工智能NLP-RAG-图像内容解析及检索优化
功能: 使用小米MiMo多模态模型(mimo-v2-omni)对PDF提取的图片
      进行语义描述，将图像内容转化为文本描述，便于后续检索
"""

import os
import base64
import logging
import json
import time
from pathlib import Path

from config import (
    MIMO_API_KEY, MIMO_BASE_URL, MIMO_VL_MODEL,
    EXTRACTED_IMAGES_DIR, OUTPUT_DIR,
    LOG_FORMAT, LOG_DATE_FORMAT
)

logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT, level=logging.INFO)
logger = logging.getLogger("image_describer")


def encode_image_to_base64(image_path):
    """将图片文件编码为Base64字符串"""
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png" if ext == "png" else "image/png"
    b64_str = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64_str}"


def call_mimo_vl(messages, max_tokens=1024, timeout=60):
    """调用小米MiMo多模态模型"""
    from openai import OpenAI
    client = OpenAI(api_key=MIMO_API_KEY, base_url=MIMO_BASE_URL)
    response = client.chat.completions.create(
        model=MIMO_VL_MODEL, messages=messages,
        max_tokens=max_tokens, timeout=timeout,
    )
    msg = response.choices[0].message
    content = msg.content or ""
    reasoning = getattr(msg, "reasoning_content", None) or ""
    if not content.strip() and reasoning.strip():
        content = reasoning
    return content


def describe_image(image_path, custom_prompt=None):
    """使用MiMo v2-omni描述单张图片内容"""
    logger.info(f"调用MiMo描述图片: {os.path.basename(image_path)}")
    try:
        image_b64 = encode_image_to_base64(image_path)
        prompt = custom_prompt or (
            "请详细描述这张图片中的内容。如果图片包含：\n"
            "1. 组织结构图：请列出所有部门名称和层级关系，统计具体的部门数量\n"
            "2. 图表/曲线图：请描述数据趋势、坐标轴含义、关键数值，标明增长率最高的和最低的行业\n"
            "3. 表格：请提取表格中的关键数据，逐行列出\n"
            "4. 流程图：请描述流程步骤\n"
            "5. 普通图片：描述图片中的场景和对象\n"
            "请用中文回答，尽可能详细和精确。"
        )
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_b64}},
                {"type": "text", "text": prompt}
            ]
        }]
        description = call_mimo_vl(messages)
        logger.info(f"图片描述成功: {os.path.basename(image_path)}")
        return description
    except Exception as e:
        logger.error(f"描述图片时出错: {e}")
        return f"[图像描述出错: {str(e)}]"


def describe_all_images(images_info, max_images=50):
    """批量描述所有图片"""
    if not images_info:
        logger.warning("没有图片需要描述")
        return []

    logger.info(f"开始批量描述图片，共 {len(images_info)} 张，上限 {max_images} 张")
    sorted_images = sorted(images_info, key=lambda x: x["width"] * x["height"], reverse=True)
    images_to_describe = sorted_images[:max_images]

    descriptions = []
    for idx, img_info in enumerate(images_to_describe):
        prompt = _build_prompt_by_image_type(img_info)
        logger.info(f"描述第 {idx+1}/{len(images_to_describe)} 张: {img_info['filename']}")
        description = describe_image(img_info["path"], custom_prompt=prompt)
        descriptions.append({
            "page_num": img_info["page_num"],
            "filename": img_info["filename"],
            "img_index": img_info["img_index"],
            "width": img_info["width"],
            "height": img_info["height"],
            "description": description,
            "source_pdf": img_info.get("source_pdf", "未知"),
        })
        if idx < len(images_to_describe) - 1:
            time.sleep(0.5)

    output_path = os.path.join(OUTPUT_DIR, "image_descriptions.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(descriptions, f, ensure_ascii=False, indent=2)

    logger.info(f"图片描述完成！共描述 {len(descriptions)} 张图片，结果已保存到 {output_path}")
    return descriptions


def _build_prompt_by_image_type(img_info):
    """根据图片特征构建针对性提示词"""
    w, h = img_info["width"], img_info["height"]

    if w > 500 and h > 300:
        return (
            "请详细描述这张图片中的所有文字信息、结构关系和具体数据。\n\n"
            "如果是组织结构图：列出所有部门名称，注明每一层级的部门和子部门数量，"
            "例如「销售部下设X个部门，包括：A、B、C...」，并说明各子部门的数量。\n\n"
            "如果是图表/数据图：精确描述图中各行业/类别的增长率数值，"
            "明确指出增长率最高的行业及其增长率百分比，负增长/下降的行业及其下降幅度。"
            "请给出具体的数字和行业名称，不要用模糊表述。\n\n"
            "如果是表格：逐行提取所有数据。\n\n"
            "请尽量精确，给出具体的数字和名称。"
        )
    else:
        return "请描述这张图片的主要内容，提取其中的文字信息。"


if __name__ == "__main__":
    test_img = list(Path(EXTRACTED_IMAGES_DIR).glob("*.png")) + list(Path(EXTRACTED_IMAGES_DIR).glob("*.jpg"))
    if test_img:
        desc = describe_image(str(test_img[0]))
        print(f"图片: {test_img[0].name}")
        print(f"描述: {desc[:300]}...")
    else:
        print("没有找到测试图片")
