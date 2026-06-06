"""
表格解析模块 - 从 MinerU 输出的 markdown 中提取表格
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

本模块专注于从 Markdown 格式文本中提取和解析表格数据。
主要功能：
1. 检测 Markdown 中的表格行（以 | 包裹的行）
2. 解析表头、分隔行和数据行，构建结构化表格数据
3. 将表格转换为自然语言文本描述，便于后续语义检索
4. 支持将表格保存为 JSON 和 CSV 格式文件

主要函数：
- extract_tables_from_markdown(): 从 Markdown 中提取所有表格
- parse_table_block(): 解析单个表格块的详细结构
- tables_to_text_blocks(): 将表格转为文本块用于向量检索
- save_tables_json() / save_tables_csv(): 保存表格到文件
"""
import re    # 导入正则表达式模块，用于匹配表格分隔行等模式
import json    # 导入 JSON 模块，用于表格数据的序列化保存
import csv    # 导入 CSV 模块，用于表格数据的 CSV 格式导出
import os    # 导入操作系统接口模块，用于文件路径操作
from config import OUTPUT_DIR, log    # 从配置模块导入输出目录路径和日志函数


def extract_tables_from_markdown(md_content: str) -> list:
    """
    从 markdown 文本中提取所有表格

    Returns:
        [{table_index: int, header: [str], rows: [[str]], raw: str, context_before: str}, ...]
    """
    tables = []    # 初始化表格列表，用于存储所有提取到的表格
    lines = md_content.split("\n")    # 按换行符将 Markdown 文本分割为行列表
    i = 0    # 行索引指针，初始化为 0
    table_index = 0    # 表格计数器，初始化为 0

    while i < len(lines):    # 遍历所有行
        line = lines[i]    # 获取当前行
        # 检测表格行（以 | 开头和结尾）    # 判断当前行是否为 Markdown 表格的行
        if is_table_row(line):    # 调用辅助函数检测表格行
            table_lines = []    # 初始化当前表格的原始行列表
            context_lines = []    # 初始化表格上方的上下文行列表

            # 向前收集上下文（最多 5 行非空非表行）    # 在表格之前收集描述性文本作为上下文
            ctx_start = max(0, i - 6)    # 计算上下文收集的起始位置（最多向前 6 行）
            for j in range(ctx_start, i):    # 从起始位置遍历到当前行
                tl = lines[j].strip()    # 去除行首尾空白
                if tl and not is_table_row(tl) and not tl.startswith("|---"):    # 如果是非空、非表格行、非分隔行
                    context_lines.append(tl)    # 添加到上下文列表

            # 收集整个表格    # 从当前行开始，持续收集所有连续的表格行
            while i < len(lines) and is_table_row(lines[i]):    # 当还有行且为表格行时继续
                table_lines.append(lines[i].strip())    # 添加当前行到表格行列表（去除空白）
                i += 1    # 行指针后移

            if len(table_lines) >= 3:  # 至少要有表头+分隔行+数据行    # 判断表格是否有效
                table = parse_table_block(table_lines)    # 调用解析函数解析表格块
                if table["header"]:    # 如果成功解析出表头
                    table["table_index"] = table_index    # 设置表格索引编号
                    table["context_before"] = "\n".join(context_lines[-3:])    # 取最近 3 行上下文作为表格说明
                    tables.append(table)    # 将解析后的表格添加到结果列表
                    table_index += 1    # 表格计数器递增
        else:    # 如果当前行不是表格行
            i += 1    # 行指针后移，继续处理下一行

    log(f"从 markdown 中提取到 {len(tables)} 个表格", "TABLE")    # 记录提取到的表格数量
    return tables    # 返回所有提取到的表格


def is_table_row(line: str) -> bool:
    """判断是否为表格行"""    # 检测一行是否属于 Markdown 表格格式
    line = line.strip()    # 去除行首尾空白
    return line.startswith("|") and line.endswith("|") and "|" in line[1:-1]    # 以 | 开头、以 | 结尾，且内部有 |


def parse_table_block(table_lines: list) -> dict:
    """
    解析一个表格块

    Returns:
        {header: [str], rows: [[str]], raw: str}
    """
    if not table_lines:    # 如果表格行列表为空
        return {"header": [], "rows": [], "raw": ""}    # 返回空数据结构

    # 解析表头（第一行）    # 将表格的第一行解析为表头
    header_cells = parse_table_row(table_lines[0])    # 调用行解析函数处理首行

    # 跳过分隔行（|---|...）    # 识别并跳过 --- 分隔行
    data_start = 1    # 数据起始行索引，从第 2 行开始
    for j in range(1, len(table_lines)):    # 从第二行开始遍历
        if re.match(r"^\|[\s\-:]+\|\s*$", table_lines[j]):    # 使用正则匹配分隔行格式（| --- | 或 | :--- | 等）
            data_start = j + 1    # 继续将起始下标后移
        else:    # 如果当前行不是分隔行
            break    # 跳出分隔行检测循环

    # 解析数据行    # 从数据起始行开始解析表格的实际数据
    rows = []    # 初始化数据行列表
    for j in range(data_start, len(table_lines)):    # 从数据起始行到结尾遍历
        row = parse_table_row(table_lines[j])    # 解析当前行为单元格列表
        if row and any(cell.strip() for cell in row):    # 如果解析成功且至少有一个非空单元格
            rows.append(row)    # 添加到数据行列表

    return {    # 返回解析后的表格数据结构
        "header": header_cells,    # 表头单元格列表
        "rows": rows,    # 数据行列表
        "raw": "\n".join(table_lines),    # 原始表格行的字符串表示
    }


def parse_table_row(row_line: str) -> list:
    """解析单行表格，返回单元格列表"""    # 将一行表格文本解析为单个单元格的列表
    row_line = row_line.strip()    # 去除行首尾空白
    if not (row_line.startswith("|") and row_line.endswith("|")):    # 如果不是以 | 包裹的格式
        return []    # 返回空列表

    # 去掉首尾 | 并分割    # 手动按 | 分割单元格内容
    inner = row_line[1:-1]    # 去掉首尾的 | 符号
    cells = []    # 初始化单元格列表
    current = ""    # 当前正在构建的单元格内容
    for ch in inner:    # 遍历内部字符串的每个字符
        if ch == "|":    # 遇到 | 分隔符
            cells.append(current.strip())    # 将当前累积的单元格内容去除空白后添加
            current = ""    # 重置当前单元格内容
        else:    # 如果是普通字符
            current += ch    # 累积到当前单元格内容中
    if current:    # 处理最后一个单元格（| 符号后的剩余内容）
        cells.append(current.strip())    # 去除空白后添加

    return cells    # 返回解析后的单元格列表


def tables_to_text_blocks(tables: list) -> list:
    """
    将表格转换为文本描述块，用于后续检索

    Returns:
        [{text: str, table_index: int, metadata: dict}, ...]
    """
    blocks = []    # 初始化文本块列表
    for t in tables:    # 遍历每个表格
        # 构建表格的文本描述    # 将结构化表格转为自然语言描述文本
        text_parts = []    # 初始化文本片段列表
        if t.get("context_before"):    # 如果表格前有上下文说明
            text_parts.append(f"上下文：{t['context_before']}")    # 添加上下文文本

        header = t.get("header") or []    # 获取表头，默认空列表
        rows = t.get("rows") or []    # 获取数据行，默认空列表

        text_parts.append(f"表格标题：{' '.join([str(h or '') for h in header])}")    # 用空格连接表头作为表格标题
        text_parts.append("表格内容：")    # 添加内容标记
        for row in rows:    # 遍历每一行数据
            row_text = " | ".join(    # 用 | 分隔每个字段描述
                [f"{header[i] if i < len(header) else '列{i+1}'}: {str(cell or '')}"    # 格式："列名: 值"
                 for i, cell in enumerate(row)]    # 枚举每个单元格及其列索引
            )
            text_parts.append(row_text)    # 添加到文本片段列表

        blocks.append({    # 构建文本块对象
            "text": "\n".join(text_parts),    # 用换行连接所有文本片段
            "table_index": t.get("table_index", 0),    # 表格索引
            "type": "table",    # 内容类型标记为表格
            "metadata": {    # 元数据信息
                "type": "table",    # 内容类型标记为表格
                "header": " ".join([str(h or "") for h in header]),    # 表头字符串
                "row_count": len(rows),    # 数据行数
                "source_pdf": t.get("source_pdf", ""),    # 来源 PDF 文件名
            },
        })

    return blocks    # 返回所有文本块


def save_tables_json(tables: list, filename: str = "tables.json"):
    """保存表格数据到 JSON 文件"""    # 将表格列表序列化并写入 JSON 文件
    out_path = os.path.join(OUTPUT_DIR, filename)    # 构造输出文件路径
    data = []    # 初始化待保存的数据列表
    for t in tables:    # 遍历每个表格
        data.append({    # 提取关键字段，构建序列化数据
            "table_index": t.get("table_index"),    # 表格索引
            "header": t.get("header") or [],    # 表头
            "rows": t.get("rows") or [],    # 数据行
            "num_rows": t.get("num_rows"),    # 行数
            "num_cols": t.get("num_cols"),    # 列数
        })
    with open(out_path, "w", encoding="utf-8") as f:    # 以 UTF-8 编码打开文件
        json.dump(data, f, ensure_ascii=False, indent=2)    # 序列化为格式化 JSON，保留中文
    log(f"表格数据已保存: {out_path}", "TABLE")    # 记录保存日志


def save_tables_csv(tables: list, filename_prefix: str = "table"):
    """将每个表格保存为单独的 CSV 文件"""    # 为每个表格单独创建一个 CSV 文件
    for t in tables:    # 遍历每个表格
        idx = t.get("table_index", 0)    # 获取表格索引
        out_path = os.path.join(OUTPUT_DIR, f"{filename_prefix}_{idx}.csv")    # 构造文件路径（如 table_0.csv）
        header = t.get("header") or []    # 获取表头
        rows = t.get("rows") or []    # 获取数据行
        with open(out_path, "w", encoding="utf-8", newline="") as f:    # 以 UTF-8 编码写入，避免空行
            writer = csv.writer(f)    # 创建 CSV 写入器
            if header:    # 如果有表头
                writer.writerow(header)    # 写入表头行
            for row in rows:    # 遍历数据行
                writer.writerow(row)    # 写入数据行
        log(f"表格已保存: {out_path}", "TABLE")    # 记录保存日志


if __name__ == "__main__":
    # 测试表格提取    # 主程序入口：测试表格提取功能
    test_md = """    # 定义测试用的 Markdown 文本
| 项目 | 金额 |
| --- | --- |
| 营业收入 | 1000万 |
| 净利润 | 200万 |
"""    # 多行字符串包含一个简单的测试表格
    tables = extract_tables_from_markdown(test_md)    # 调用表格提取函数
    blocks = tables_to_text_blocks(tables)    # 将表格转为文本块
    for b in blocks:    # 遍历并打印每个文本块
        print(b["text"])    # 输出文本块内容
    save_tables_json(tables)    # 保存表格到 JSON 文件
