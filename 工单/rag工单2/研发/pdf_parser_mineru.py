# -*- coding: utf-8 -*-
"""
PDF解析模块 —— 使用 MinerU 进行版面分析+表格提取+Markdown输出
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Optional


class MinerUParser:
    """
    MinerU PDF 解析器
    功能：版面分析 → 表格识别 → 公式识别 → Markdown输出
    依赖：magic-pdf 包和预训练模型
    """

    def __init__(self, models_dir: str = None):
        self.models_dir = models_dir or "/home/zzy/.magic-pd-models"
        self._ready = False

    def check_ready(self) -> bool:
        """检查模型是否已下载"""
        required = [
            os.path.join(self.models_dir, "Layout/LayoutLMv3/model_final.pth"),
        ]
        for f in required:
            if os.path.exists(f):
                self._ready = True
                return True
        # 检查是否有任何模型文件
        layout_dir = os.path.join(self.models_dir, "Layout")
        if os.path.exists(layout_dir) and any(
            f.endswith(".pth") or f.endswith(".pt")
            for root, _, files in os.walk(layout_dir) for f in files
        ):
            self._ready = True
            return True
        return False

    def parse_to_markdown(self, pdf_path: str, output_dir: str = None) -> str:
        """
        用 MinerU 解析 PDF，返回 Markdown 文本

        Args:
            pdf_path: PDF文件路径
            output_dir: MinerU输出目录（默认为 pdf_path 同级的 mineru_output）

        Returns:
            Markdown 格式的文本内容
        """
        if output_dir is None:
            output_dir = str(Path(pdf_path).parent / "mineru_output")
        os.makedirs(output_dir, exist_ok=True)

        # 配置 MinerU
        import tempfile
        config = {
            "bucket_info": {},
            "models-dir": self.models_dir,
            "device-mode": "cuda",
            "enable_table": True,
            "enable_formula": False,
        }
        config_path = os.path.join(tempfile.gettempdir(), "magic-pdf.json")
        with open(config_path, "w") as f:
            json.dump(config, f)

        # 保存原配置并替换
        home_config = os.path.expanduser("~/magic-pdf.json")
        old_config = None
        if os.path.exists(home_config):
            with open(home_config) as f:
                old_config = f.read()

        with open(home_config, "w") as f:
            json.dump(config, f)

        try:
            # 调用 MinerU CLI
            import subprocess
            result = subprocess.run(
                ["magic-pdf", "-p", pdf_path, "-o", output_dir, "-m", "auto"],
                capture_output=True, text=True, timeout=600
            )
            if result.returncode != 0:
                # 尝试 txt 模式
                print(f"[MinerU] auto模式失败，尝试txt模式: {result.stderr[-200:]}")
                result = subprocess.run(
                    ["magic-pdf", "-p", pdf_path, "-o", output_dir, "-m", "txt"],
                    capture_output=True, text=True, timeout=600
                )

            # 查找输出的 Markdown 文件
            md_files = list(Path(output_dir).rglob("*.md"))
            if md_files:
                md_path = str(md_files[0])
                with open(md_path, "r", encoding="utf-8") as f:
                    md_content = f.read()
                print(f"[MinerU] 解析完成: {md_path} ({len(md_content)}字符)")
                return md_content
            else:
                print(f"[MinerU] 未找到Markdown输出，检查目录: {output_dir}")
                return ""

        finally:
            # 恢复原配置
            if old_config:
                with open(home_config, "w") as f:
                    f.write(old_config)
            elif os.path.exists(home_config):
                os.remove(home_config)

    def parse_markdown_to_plaintext(self, md_text: str) -> str:
        """
        将 MinerU 输出的 Markdown 转为纯文本

        Args:
            md_text: Markdown 文本

        Returns:
            清洗后的纯文本
        """
        if not md_text:
            return ""

        text = md_text

        # 移除代码块
        text = re.sub(r"```[\s\S]*?```", "", text)

        # 移除图片引用
        text = re.sub(r"!\[.*?\]\(.*?\)", "", text)

        # 保留表格（转为文字描述）
        # 表格的 Markdown 格式保留
        text = re.sub(r"\|", " | ", text)
        text = re.sub(r"\|?\s*:?-+:?\s*\|?", "", text)  # 移除表格分隔行

        # 保留标题标记
        text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)

        # 移除链接但保留文字
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

        # 移除加粗/斜体标记
        text = re.sub(r"\*{1,3【报错】}([^*]+)\*{1,3【报错】}", r"\1", text)

        # 清理多余空行
        text = re.sub(r"\n{3【报错】,}", "\n\n", text)

        return text.strip()


def parse_with_mineru(pdf_path: str, output_dir: str = None) -> str:
    """
    快捷函数：用 MinerU 解析 PDF 并返回清洗后的文本

    Args:
        pdf_path: PDF文件路径
        output_dir: 输出目录

    Returns:
        清洗后的纯文本
    """
    parser = MinerUParser()
    if not parser.check_ready():
        print("[MinerU] 模型未就绪，请先下载模型")
        return ""

    md = parser.parse_to_markdown(pdf_path, output_dir)
    return parser.parse_markdown_to_plaintext(md)


if __name__ == "__main__":
    from config import PDF_PATH

    parser = MinerUParser()
    ready = parser.check_ready()
    print(f"模型就绪: {ready}")

    if ready:
        text = parse_with_mineru(PDF_PATH)
        print(f"解析文本长度: {len(text)}")
        print(f"前500字:\n{text[:500]}")
    else:
        print("模型还在下载中，请稍后再试")
        print(f"模型目录: {parser.models_dir}")
        # 列出已下载的文件
        for root, dirs, files in os.walk(parser.models_dir):
            for f in files:
                print(f"  {os.path.join(root, f)}")
