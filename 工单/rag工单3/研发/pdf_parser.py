"""
PDF 解析模块 - 优先使用 MinerU Docker API，失败自动降级到 PyMuPDF
自带表格检测，支持双平台路径
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

本模块是系统的 PDF 解析核心，提供两套解析方案：
1. MinerU Docker API（优先）：通过本地部署的 MinerU 服务进行异步 PDF 解析，
   自动检测并提取表格数据，返回 Markdown 格式文本和结构化表格列表。
2. PyMuPDF 降级方案（备用）：当 MinerU 服务不可用时，使用 PyMuPDF (fitz) 
   直接解析 PDF 文本和表格，保证系统在任何环境下都能正常工作。

主要函数：
- parse_pdf(): 主入口，自动优先尝试 MinerU API，失败则降级到 PyMuPDF
- try_mineru_api(): 通过 HTTP 调用 MinerU Docker 服务进行异步解析
- _parse_mineru_result(): 解析 MinerU 返回结果（支持多种返回格式）
- parse_pdf_with_fitz(): 使用 PyMuPDF 实现 PDF 文本与表格提取
- save_parsed_output(): 将解析结果保存为 Markdown、JSON 和 CSV 格式
"""
import os          # 导入操作系统接口模块，用于文件路径和目录操作
import json        # 导入 JSON 模块，用于处理 JSON 数据的序列化和反序列化
import csv         # 导入 CSV 模块，用于将表格数据保存为 CSV 格式文件
import time        # 导入时间模块，用于等待和计时操作
import requests    # 导入 HTTP 请求库，用于调用 MinerU Docker API
import fitz  # PyMuPDF    # 导入 PyMuPDF 库，用于 PDF 文件的直接解析（降级方案）
from pathlib import Path    # 从 pathlib 导入 Path 类，用于跨平台路径处理
from config import PDF_PATH, PDF_PATHS, OUTPUT_DIR, log, ensure_dirs    # 从配置模块导入 PDF 路径、输出目录、日志函数和目录创建函数

# ========== MinerU API 配置 ==========    # MinerU Docker 服务 API 配置区域
MINERU_API_BASE = "http://localhost:8000"    # MinerU 服务的基础 URL，默认监听本地 8000 端口
MINERU_TIMEOUT_SYNC = 300      # 同步解析超时（秒）    # MinerU 同步解析请求的超时时间
MINERU_POLL_INTERVAL = 10      # 轮询间隔（秒）    # 轮询 MinerU 异步任务状态的间隔时间
MINERU_MAX_POLL_TIME = 600     # 最大等待时间（秒）    # 等待 MinerU 异步任务完成的最大时间


def parse_pdf(pdf_path: str = None) -> tuple:
    """
    主入口：支持单个或批量 PDF 解析
    - 如果传入 pdf_path，解析单个 PDF
    - 如果传入 None，自动使用 PDF_PATHS 列表解析所有 PDF 并合并

    Returns:
        (merged_md_content, merged_tables)
        其中 merged_tables 中每个表格附带 source_pdf 字段
    """
    paths = []
    if pdf_path:
        paths = [pdf_path]
    else:
        paths = PDF_PATHS  # 使用配置中的多 PDF 列表

    if len(paths) == 1:
        # 单 PDF：保持原有逻辑
        p = paths[0]
        log(f"开始解析 PDF: {p}", "PARSE")
        md_content, tables = try_mineru_api(p)
        if md_content:
            log(f"MinerU API 解析成功: {len(md_content)} 字符, {len(tables)} 个表格", "PARSE")
            return md_content, tables
        log("MinerU API 解析失败或不可用，降级到 PyMuPDF", "PARSE")
        return parse_pdf_with_fitz(p)

    # 多 PDF：逐个解析并合并
    log(f"检测到 {len(paths)} 个 PDF，开始批量解析", "PARSE")
    for p in paths:
        log(f"  - {os.path.basename(p)}", "PARSE")

    all_md_parts = []
    all_tables = []
    table_offset = 0
    total_chars = 0
    total_tables = 0

    for p in paths:
        pdf_name = os.path.basename(p)
        log(f"解析 {pdf_name} ...", "PARSE")
        md_content, tables = _parse_single_pdf_fallback(p)

        if not md_content:
            log(f"  ⚠️ {pdf_name} 解析失败，跳过", "WARN")
            continue

        # 在 Markdown 中标记来源
        all_md_parts.append(f"<!-- source: {pdf_name} -->\n{md_content}")

        # 为表格附加上来源信息（调整表格索引）
        for t in tables:
            t["table_index"] = table_offset
            table_offset += 1
            t["source_pdf"] = pdf_name
        all_tables.extend(tables)

        total_chars += len(md_content)
        total_tables += len(tables)
        log(f"  ✅ {pdf_name}: {len(md_content)} 字符, {len(tables)} 个表格", "PARSE")

    merged_md = "\n\n".join(all_md_parts)
    log(f"多 PDF 解析完成: 合计 {total_chars} 字符, {total_tables} 个表格", "PARSE")
    return merged_md, all_tables


def _parse_single_pdf_fallback(pdf_path: str) -> tuple:
    """解析单个 PDF，MinerU 优先 → PyMuPDF 降级"""
    md_content, tables = try_mineru_api(pdf_path)
    if md_content:
        return md_content, tables
    return parse_pdf_with_fitz(pdf_path)


def try_mineru_api(pdf_path: str) -> tuple:
    """尝试通过 MinerU Docker API 解析 PDF，失败返回 (None, [])"""    # 尝试 MinerU API 解析，失败时返回空结果
    # 检查 MinerU 服务是否可用    # 先通过健康检查端点判断服务状态
    try:    # 捕获网络请求异常
        r = requests.get(f"{MINERU_API_BASE}/health", timeout=5)    # 发送 GET 请求到 /health 端点，超时 5 秒
        if r.status_code != 200:    # 如果返回状态码不是 200（OK）
            log(f"MinerU 服务状态异常: HTTP {r.status_code}", "WARN")    # 记录警告日志
            return None, []    # 返回空结果，触发降级
    except requests.exceptions.ConnectionError:    # 捕获连接拒绝异常（服务未启动）
        log("MinerU 服务未运行 (localhost:8000 连接失败)", "WARN")    # 记录警告日志
        return None, []    # 返回空结果
    except requests.exceptions.Timeout:    # 捕获请求超时异常
        log("MinerU 健康检查超时", "WARN")    # 记录警告日志
        return None, []    # 返回空结果

    log("MinerU 服务可用，提交异步解析任务", "PARSE")    # 记录服务可用并准备提交任务

    # 提交异步任务    # 向 MinerU 服务提交 PDF 异步解析任务
    file_name = os.path.basename(pdf_path)    # 从完整路径中提取 PDF 文件名
    try:    # 捕获文件操作和网络请求异常
        with open(pdf_path, "rb") as f:    # 以二进制读模式打开 PDF 文件
            resp = requests.post(    # 发送 POST 请求提交文件
                f"{MINERU_API_BASE}/tasks",    # 目标 URL：/tasks 端点
                files={"files": (file_name, f)},    # 以 multipart/form-data 格式上传文件
                timeout=30,    # 上传请求超时时间 30 秒
            )
        if resp.status_code not in (200, 202):    # 如果返回状态码不是 200 或 202（任务已接受）
            log(f"提交任务失败: HTTP {resp.status_code} - {resp.text[:200]}", "ERROR")    # 记录错误日志并截取前 200 字符的响应内容
            return None, []    # 返回空结果

        task_data = resp.json()    # 将响应解析为 JSON 对象
        task_id = task_data.get("task_id")    # 从 JSON 中提取任务 ID
        if not task_id:    # 如果响应中没有任务 ID
            log(f"返回数据无 task_id: {task_data}", "ERROR")    # 记录错误日志
            return None, []    # 返回空结果

        log(f"任务已提交: {task_id}", "PARSE")    # 记录任务提交成功的日志

    except Exception as e:    # 捕获所有其他异常
        log(f"提交任务异常: {e}", "ERROR")    # 记录异常日志
        return None, []    # 返回空结果

    # 轮询等待任务完成    # 循环检查 MinerU 任务的完成状态
    start_time = time.time()    # 记录轮询开始时间
    while time.time() - start_time < MINERU_MAX_POLL_TIME:    # 在最大等待时间内循环
        elapsed = int(time.time() - start_time)    # 计算已等待时间（秒）
        try:    # 捕获轮询过程中的异常
            status_resp = requests.get(    # 发送 GET 请求查询任务状态
                f"{MINERU_API_BASE}/tasks/{task_id}", timeout=10    # 目标 URL：/tasks/{task_id}，超时 10 秒
            )
            if status_resp.status_code == 404:    # 如果任务不存在（404）
                log(f"任务 {task_id} 已失效（可能容器重启）", "ERROR")    # 记录错误日志
                return None, []    # 返回空结果

            status_data = status_resp.json()    # 将状态响应解析为 JSON
            status = status_data.get("status", "")    # 提取任务状态字段

            if status == "completed":    # 如果任务已完成
                # 获取结果    # 从 MinerU 获取解析结果
                result_resp = requests.get(    # 发送 GET 请求获取结果
                    f"{MINERU_API_BASE}/tasks/{task_id}/result", timeout=30    # 目标 URL：/tasks/{task_id}/result，超时 30 秒
                )
                if result_resp.status_code != 200:    # 如果获取结果失败
                    log(f"获取结果失败: HTTP {result_resp.status_code}", "ERROR")    # 记录错误日志
                    return None, []    # 返回空结果
                return _parse_mineru_result(result_resp.json())    # 解析 MinerU 返回结果并返回

            elif status == "failed":    # 如果任务执行失败
                err_msg = status_data.get("error", "未知错误")    # 提取错误信息，默认"未知错误"
                log(f"任务失败: {err_msg}", "ERROR")    # 记录错误日志
                return None, []    # 返回空结果

            elif status == "pending" or status == "processing":    # 如果任务还在排队或处理中
                if elapsed % 60 < MINERU_POLL_INTERVAL and elapsed > 0:    # 每分钟输出一次进度日志（避免日志过多）
                    log(f"等待 MinerU 解析... ({elapsed}s)", "PARSE")    # 记录等待日志
                time.sleep(MINERU_POLL_INTERVAL)    # 按配置的间隔时间休眠等待

            else:    # 处理未知状态
                log(f"未知状态: {status}", "WARN")    # 记录警告日志
                time.sleep(MINERU_POLL_INTERVAL)    # 按配置的间隔时间休眠等待

        except requests.exceptions.Timeout:    # 捕获轮询请求超时
            log(f"轮询超时 (已等待 {elapsed}s)", "WARN")    # 记录警告日志
            continue    # 继续下一轮轮询
        except Exception as e:    # 捕获轮询过程中的其他异常
            log(f"轮询异常: {e}", "ERROR")    # 记录错误日志
            return None, []    # 返回空结果

    log(f"任务超时（超过 {MINERU_MAX_POLL_TIME}s）", "ERROR")    # 记录超时错误日志
    return None, []    # 返回空结果


def _parse_mineru_result(result_data: dict) -> tuple:
    """
    解析 MinerU 返回结果，提取 markdown 内容和表格
    MinerU 返回格式不固定，尝试多种字段名
    """
    # 尝试不同的字段名    # 兼容 MinerU 不同版本的返回格式差异
    result = result_data.get("result") or result_data    # 优先取 "result" 字段，若无则使用整个数据

    if isinstance(result, str):    # 如果结果直接是字符串
        # 直接是 markdown 字符串    # 该字符串为完整的 Markdown 内容
        return _extract_tables_from_md(result), _extract_tables_from_md(result, return_tables=True)    # 从 Markdown 中提取文本和表格

    if isinstance(result, dict):    # 如果结果是字典类型
        # 尝试常见字段名    # 兼容 MinerU 不同版本的字段命名差异
        md_content = (    # 按优先级尝试多个可能的 Markdown 字段名
            result.get("md_content")    # 常见字段名：md_content
            or result.get("markdown")    # 备选字段名：markdown
            or result.get("text")    # 备选字段名：text
            or result.get("full_content")    # 备选字段名：full_content
            or ""    # 如果以上字段都不存在，返回空字符串
        )

        if md_content:    # 如果成功获取到 Markdown 内容
            return md_content, _extract_tables_from_md(md_content, return_tables=True)    # 返回文本和从 Markdown 中提取的表格

        # 尝试 pdf_info 格式（MinerU 旧版）    # 兼容 MinerU 旧版返回的 pdf_info 格式
        pdf_info = result.get("pdf_info") or []    # 尝试获取 pdf_info 列表
        if pdf_info:    # 如果存在 pdf_info 数据
            md_parts = []    # 初始化 Markdown 文本片段列表
            tables = []    # 初始化表格列表
            table_idx = 0    # 表格计数器初始化为 0
            for page in pdf_info:    # 遍历每一页的数据
                page_num = page.get("page_num", 0)    # 获取页码
                md_parts.append(f"<!-- page {page_num} -->")    # 添加页眉注释到 Markdown
                for block in page.get("blocks", []):    # 遍历页面中的每个块（block）
                    block_type = block.get("type", "")    # 获取块类型（text 或 table）
                    if block_type == "text":    # 如果是文本块
                        for line in block.get("lines", []):    # 遍历文本块中的每一行
                            for span in line.get("spans", []):    # 遍历行中的每个文本片段（span）
                                md_parts.append(span.get("content", ""))    # 将文本内容添加到 Markdown
                    elif block_type == "table":    # 如果是表格块
                        table_data = block.get("table", {})    # 获取表格数据
                        header = table_data.get("header", [])    # 获取表格表头
                        rows = table_data.get("rows", [])    # 获取表格数据行
                        if not header and rows:    # 如果没有显式表头但有数据行
                            header = rows[0]    # 将第一行作为表头
                            rows = rows[1:]    # 剩余行作为数据行
                        tables.append({    # 将格式化后的表格数据添加到表格列表
                            "table_index": table_idx,    # 表格索引编号
                            "header": [str(c or "") for c in header],    # 表头，将所有元素转为字符串，None 转为空字符串
                            "rows": [[str(c or "") for c in row] for row in rows],    # 数据行，同样进行字符串转换
                            "num_rows": len(rows),    # 数据行数
                            "num_cols": max(len(header), len(rows[0]) if rows else 0),    # 列数（取表头和第一行中的最大值）
                            "raw": format_table_markdown(header, rows),    # 表格的 Markdown 格式原始字符串
                        })
                        table_idx += 1    # 表格计数器递增
                        md_parts.append(f"\n[表格 {table_idx}]\n")    # 在 Markdown 中添加表格占位标记

            return "\n\n".join(md_parts), tables    # 将所有 Markdown 片段用双换行连接，与表格列表一并返回

    # 兜底：尝试 JSON 序列化作为文本    # 最后的降级处理：将结果转为 JSON 文本
    try:    # 尝试 JSON 序列化
        text = json.dumps(result, ensure_ascii=False, indent=2)    # 将结果转为格式化的 JSON 字符串
        log(f"MinerU 返回格式未知，已转为 JSON 文本 ({len(text)} 字符)", "WARN")    # 记录警告日志
        return text, []    # 返回 JSON 文本和空表格列表
    except Exception:    # 如果 JSON 序列化失败
        return str(result), []    # 直接转为字符串返回


def _extract_tables_from_md(md_text: str, return_tables: bool = False) -> list:
    """
    从 markdown 文本中提取表格，用于 MinerU 返回的 markdown 解析
    如果 return_tables=False，返回 md_text 本身
    """
    if not return_tables:    # 如果不需要提取表格
        return md_text    # 直接返回原始 Markdown 文本

    tables = []    # 初始化表格列表
    lines = md_text.split("\n")    # 按换行符分割 Markdown 文本为行列表
    i = 0    # 行索引指针初始化为 0
    table_idx = 0    # 表格计数器初始化为 0

    while i < len(lines):    # 遍历所有行
        line = lines[i].strip()    # 去除当前行首尾空白
        if line.startswith("|") and line.endswith("|") and "|" in line[1:-1]:    # 判断是否为 Markdown 表格行（以 | 包裹）
            table_lines = []    # 初始化当前表格的行列表
            while i < len(lines):    # 持续收集表格行直到非表格行
                l = lines[i].strip()    # 去除行首尾空白
                if l.startswith("|") and l.endswith("|"):    # 如果是表格行
                    table_lines.append(l)    # 添加到表格行列表
                    i += 1    # 指针后移
                else:    # 如果不是表格行
                    break    # 跳出表格收集循环

            if len(table_lines) >= 3:  # 表头+分隔线+数据    # 表格至少需要表头行、分隔行和数据行
                header = []    # 初始化表头列表
                rows = []    # 初始化数据行列表
                for idx, tl in enumerate(table_lines):    # 遍历表格的每一行
                    cells = [c.strip() for c in tl.strip("|").split("|")]    # 去除首尾 |，按 | 分割，去除每个单元格空白
                    if idx == 0:    # 第一行为表头
                        header = cells    # 保存表头
                    elif idx == 1:    # 第二行为分隔行
                        continue  # 跳过分隔行    # 分隔行格式为 |---|---|，跳过不处理
                    else:    # 其余为数据行
                        if any(c for c in cells):    # 如果该行至少有一个非空单元格
                            rows.append(cells)    # 添加到数据行列表

                if header:    # 如果成功提取到表头
                    tables.append({    # 将表格的格式化数据添加到结果列表
                        "table_index": table_idx,    # 表格索引
                        "header": header,    # 表头
                        "rows": rows,    # 数据行
                        "num_rows": len(rows),    # 数据行数
                        "num_cols": max(len(header), len(rows[0]) if rows else 0),    # 列数
                        "raw": "\n".join(table_lines),    # 原始 Markdown 表格字符串
                    })
                    table_idx += 1    # 表格计数器递增
        else:    # 如果当前行不是表格行
            i += 1    # 指针后移，继续处理下一行

    return tables    # 返回提取到的所有表格


# ==================== PyMuPDF 降级方案（原有逻辑） ====================    # 当 MinerU 服务不可用时的降级解析方案


def parse_pdf_with_fitz(pdf_path: str = PDF_PATH) -> tuple:
    """
    使用 PyMuPDF 解析 PDF，返回 markdown 格式文本
    （MinerU 不可用时的降级方案）

    Returns:
        (markdown_content, tables_list)
    """
    ensure_dirs()    # 确保输出目录存在
    log(f"PyMuPDF 解析: {pdf_path}", "PARSE")    # 记录开始使用 PyMuPDF 解析的日志

    if not os.path.exists(pdf_path):    # 检查 PDF 文件是否存在
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")    # 如果文件不存在，抛出文件未找到异常

    doc = fitz.open(pdf_path)    # 使用 PyMuPDF 打开 PDF 文档
    total_pages = len(doc)    # 获取 PDF 总页数
    log(f"PDF 共 {total_pages} 页", "PARSE")    # 记录 PDF 页数信息

    md_lines = []    # 初始化 Markdown 行列表
    all_tables = []    # 初始化所有表格列表
    table_index = 0    # 表格计数器初始化为 0

    for page_num in range(total_pages):    # 逐页遍历 PDF
        page = doc[page_num]    # 获取当前页对象

        # 提取页内表格    # 使用 PyMuPDF 的表格检测功能
        page_tables = page.find_tables()    # 查找当前页面中所有的表格
        table_regions = []    # 初始化表格区域列表，用于记录表格占据的页面区域

        for t in page_tables:    # 遍历当前页找到的每个表格
            table_data = extract_table_data(t, table_index)    # 从 PyMuPDF 表格对象提取结构化数据
            table_index += 1    # 表格计数器递增
            all_tables.append(table_data)    # 添加到总表格列表
            table_regions.append((t.bbox, table_data))    # 记录表格边界框和对应数据

        # 提取文本，跳过表格区域    # 提取页面的纯文本内容
        page_text = page.get_text("text")    # 以纯文本模式提取页面文本
        page_md = text_to_markdown(page_text, page_num + 1)    # 将文本转为 Markdown 格式
        md_lines.append(page_md)    # 添加到 Markdown 行列表

    doc.close()    # 关闭 PDF 文档，释放资源
    md_content = "\n\n".join(md_lines)    # 将所有页的 Markdown 用双换行拼接
    log(f"PyMuPDF 解析完成: {len(md_content)} 字符, {len(all_tables)} 个表格", "PARSE")    # 记录解析完成日志

    return md_content, all_tables    # 返回 Markdown 内容和表格列表


def extract_table_data(pymupdf_table, idx: int) -> dict:
    """从 PyMuPDF 表格对象提取结构化数据"""    # 将 PyMuPDF 表格对象转换为统一的字典格式
    try:    # 捕获表头提取异常
        header = pymupdf_table.header.names if pymupdf_table.header else []    # 尝试获取 PyMuPDF 表格的表头名称
    except Exception:    # 如果获取表头失败
        header = []    # 表头置为空列表

    rows = []    # 初始化数据行列表
    for row in pymupdf_table.extract():    # 提取表格的所有行数据
        rows.append([str(cell or "") for cell in row])    # 将每个单元格转为字符串，None 转为空字符串

    # 合并表头检测: 第一行可能是表头    # 如果 PyMuPDF 未识别出表头，将第一行作为表头处理
    if not header and rows:    # 如果没有识别出表头但有数据行
        header = rows[0]    # 将第一行作为表头
        rows = rows[1:]    # 剩余行作为数据行

    return {    # 返回结构化的表格数据
        "table_index": idx,    # 表格索引编号
        "header": header,    # 表头列表
        "rows": rows,    # 数据行列表
        "num_rows": len(rows),    # 数据行数
        "num_cols": max(len(header), len(rows[0]) if rows else 0),    # 列数（表头和数据行的最大值）
        "raw": format_table_markdown(header, rows),    # 表格的 Markdown 格式表示
    }


def _clean_cells(cells: list) -> list:
    """清理表格单元格中的 None 值"""    # 确保所有单元格值都是非 None 的字符串
    return [str(c or "") for c in cells]    # 将每个元素转为字符串，None 或空值转为空字符串


def format_table_markdown(header: list, rows: list) -> str:
    """将表格格式化为 markdown"""    # 将表头和数据行渲染为标准的 Markdown 表格格式
    header = _clean_cells(header)    # 清理表头中的 None 值

    if not header and not rows:    # 如果既没有表头也没有数据行
        return ""    # 返回空字符串

    if not header:    # 如果没有表头但有数据行
        header = [f"列{i+1}" for i in range(len(rows[0]))] if rows else []    # 自动生成列名：列1、列2...

    md = "| " + " | ".join(header) + " |\n"    # 构建 Markdown 表头行
    md += "| " + " | ".join(["---"] * len(header)) + " |\n"    # 构建 Markdown 分隔行（--- 表示列对齐方式）

    for row in rows:    # 遍历每一行数据
        row = _clean_cells(row)    # 清理行中的 None 值
        while len(row) < len(header):    # 如果数据行列数少于表头列数
            row.append("")    # 补充空字符串以对齐
        md += "| " + " | ".join(row[:len(header)]) + " |\n"    # 构建 Markdown 数据行（截断到表头长度）

    return md    # 返回完整的 Markdown 表格字符串


def text_to_markdown(text: str, page_num: int) -> str:
    """将提取的文本转为 markdown 格式"""    # 将纯文本智能转换为 Markdown 格式，自动识别标题
    lines = text.split("\n")    # 按换行符分割文本为行列表
    md_parts = [f"<!-- page {page_num} -->"]    # 初始化 Markdown 片段列表，以 HTML 注释标记页码开头

    for line in lines:    # 遍历每一行文本
        stripped = line.strip()    # 去除行首尾空白
        if not stripped:    # 如果是空行
            continue    # 跳过不处理

        if len(stripped) < 50 and stripped[-1] not in "。；，、":    # 如果行较短（<50字符）且不以中文标点结尾，可能是标题
            if not stripped.startswith("#"):    # 如果还不是 Markdown 标题格式
                if any(stripped.startswith(f"{i}.") for i in range(1, 20)):    # 如果以"数字."开头（如"1."、"2."）
                    md_parts.append(f"### {stripped}")    # 转为三级标题
                elif any(stripped.startswith(f"第{i}") for i in "一二三四五六七八九十"):    # 如果以"第X"开头（如"第一章"）
                    md_parts.append(f"## {stripped}")    # 转为二级标题
                elif stripped[0].isalpha() and stripped[1:2] in "。、":    # 如果首字符是字母且第二个字符是中文标点
                    md_parts.append(f"### {stripped}")    # 转为三级标题
                else:    # 其他短行
                    md_parts.append(stripped)    # 直接保留原样
            else:    # 如果已经是 Markdown 标题格式
                md_parts.append(stripped)    # 直接保留原样
        else:    # 长行文本，一般为段落内容
            md_parts.append(stripped)    # 直接保留原样

    return "\n\n".join(md_parts)    # 用双换行连接所有 Markdown 片段


def save_parsed_output(md_content: str, tables: list):
    """保存解析结果 - 合并所有表格到单个 CSV"""    # 将解析结果保存为三种格式：Markdown、JSON 和 CSV
    ensure_dirs()    # 确保输出目录存在

    # 1. 保存完整 markdown    # 将完整的 Markdown 文本保存到文件
    md_path = os.path.join(OUTPUT_DIR, "parsed_content.md")    # 构造 Markdown 文件路径
    with open(md_path, "w", encoding="utf-8") as f:    # 以 UTF-8 编码写入文件
        f.write(md_content)    # 写入 Markdown 内容
    log(f"解析内容已保存: {md_path} ({len(md_content)} 字符)", "PARSE")    # 记录保存日志

    # 2. 保存表格 JSON    # 将表格数据序列化为 JSON 格式保存
    tables_out = []    # 初始化输出表格列表
    for t in tables:    # 遍历所有表格
        tables_out.append({    # 提取关键字段，添加 raw_markdown 字段
            "table_index": t["table_index"],    # 表格索引
            "header": t["header"],    # 表头
            "rows": t["rows"],    # 数据行
            "num_rows": t["num_rows"],    # 行数
            "num_cols": t["num_cols"],    # 列数
            "raw_markdown": t["raw"],    # 原始 Markdown 字符串
            "source_pdf": t.get("source_pdf", ""),    # 来源 PDF 文件名
        })

    tables_path = os.path.join(OUTPUT_DIR, "tables.json")    # 构造 JSON 文件路径
    with open(tables_path, "w", encoding="utf-8") as f:    # 以 UTF-8 编码写入文件
        json.dump(tables_out, f, ensure_ascii=False, indent=2)    # 序列化为带缩进的 JSON 格式，保留中文
    log(f"表格 JSON 已保存: {tables_path} ({len(tables)} 个)", "PARSE")    # 记录保存日志

    # 3. 所有表格合并到单个 CSV（用空行分隔）    # 将所有表格合并写入同一个 CSV 文件
    csv_path = os.path.join(OUTPUT_DIR, "all_tables.csv")    # 构造 CSV 文件路径
    with open(csv_path, "w", encoding="utf-8", newline="") as f:    # 以 UTF-8 编码写入，newline="" 避免空行
        writer = csv.writer(f)    # 创建 CSV 写入器
        for t in tables:    # 遍历所有表格
            # 写表头行 + 表格编号作为注释    # 在 CSV 中用注释行标记表格编号和尺寸
            writer.writerow([f"# 表格 {t['table_index']} - {t['num_rows']}行×{t['num_cols']}列"])    # 写入表格注释行
            if t["header"]:    # 如果有表头
                writer.writerow(t["header"])    # 写入表头行
            for row in t["rows"]:    # 遍历数据行
                writer.writerow(row)    # 写入数据行
            writer.writerow([])  # 空行分隔    # 写入空行分隔不同表格
    log(f"所有表格已合并保存: {csv_path} ({len(tables)} 个表格)", "PARSE")    # 记录保存日志


def extract_content_sections(md_content: str) -> list:
    """提取章节结构"""    # 从 Markdown 内容中提取标题层级结构
    sections = []    # 初始化章节列表
    lines = md_content.split("\n")    # 按换行符分割 Markdown 内容
    current = {"level": 0, "title": "ROOT", "content": []}    # 初始化当前章节对象，根节点为 ROOT

    for line in lines:    # 遍历每一行
        if line.startswith("#"):    # 如果是 Markdown 标题行
            if current["content"]:    # 如果当前章节已有内容
                sections.append(current)    # 将当前章节保存到列表
            level = len(line) - len(line.lstrip("#"))    # 计算标题级别（# 的数量）
            title = line.strip("#").strip()    # 提取标题文本，去除 # 符号和首尾空白
            current = {"level": level, "title": title, "content": []}    # 创建新的章节对象
        else:    # 如果是非标题行
            current["content"].append(line)    # 将行添加到当前章节的内容列表中

    if current["content"]:    # 处理最后一个章节：如果最后一个章节有内容
        sections.append(current)    # 将最后一个章节保存到列表

    log(f"提取到 {len(sections)} 个章节", "PARSE")    # 记录提取到的章节数量
    return sections    # 返回章节列表


if __name__ == "__main__":
    md_content, tables = parse_pdf()    # 入口：调用主解析函数，获取 Markdown 内容和表格
    save_parsed_output(md_content, tables)    # 保存解析结果到文件
    sections = extract_content_sections(md_content)    # 提取章节结构
    print(f"章节数: {len(sections)}")    # 打印章节数量
    print(f"表格数: {len(tables)}")    # 打印表格数量
    if tables:    # 如果有表格数据
        print(f"第一个表格: {tables[0]['header']}")    # 打印第一个表格的表头
