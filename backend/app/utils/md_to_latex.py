"""Markdown → LaTeX 转换器。

将 WriterAgent 输出的 Markdown（含 HTML 表格）转为完整 LaTeX 文档。
BS4 解析 HTML 表格 → tabular+booktabs，正则处理其余元素。
"""

import re
import os


def _escape_body_text(text: str) -> str:
    """转义正文中的 LaTeX 特殊字符（& % #）。"""
    text = text.replace("&", "\\&")
    text = text.replace("%", "\\%")
    text = text.replace("#", "\\#")
    return text


def convert(md_path: str, tex_path: str, title: str = "论文标题") -> str:
    """将 Markdown 文件转为可编译的 LaTeX 文件。

    Args:
        md_path: 输入 .md 路径。
        tex_path: 输出 .tex 路径。
        title: 论文标题。

    Returns:
        LaTeX 源码字符串。
    """
    with open(md_path, "r", encoding="utf-8") as f:
        md = f.read()

    base_dir = os.path.dirname(os.path.abspath(md_path))
    tex = _md_to_latex(md, title, base_dir)

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex)

    return tex


def _md_to_latex(md: str, title: str, base_dir: str = ".") -> str:
    """核心转换逻辑。

    Args:
        md: Markdown 文本。
        title: 论文标题。
        base_dir: 图片文件查找的基准目录（默认为当前目录）。
    """

    # ==== Step 0: 预处理 ====
    # 提取 HTML 表格（优先从 Markdown，其次外部注入的 tables_data）
    import sys
    _mod = sys.modules[__name__]
    tables_latex = []
    if hasattr(_mod, '_external_tables') and _mod._external_tables:
        md = _extract_html_tables_from_list(md, _mod._external_tables, tables_latex)
        _mod._external_tables = None
    else:
        md = _extract_html_tables(md, tables_latex)

    # ==== Step 1: 标题转换 ====
    # 去掉数字前缀（如 "1. 一、问题重述" → "一、问题重述"）
    def _clean_heading(m):
        text = m.group(1)
        # 去掉多级编号: "5.1.1 标题" → "标题"
        text = re.sub(r"^\d+(?:\.\d+)*\s+", "", text)
        return text
    # ###/#### subsub: 保留 "5.4.1" 编号（有上下文意义）
    md = re.sub(r"^\s*####?\s+(.+)$", r"\\subsubsection{\1}", md, flags=re.MULTILINE)
    md = re.sub(r"^\s*###\s+(.+)$", r"\\subsubsection{\1}", md, flags=re.MULTILINE)
    # ## subsection: 保留 "1.1" style编号
    md = re.sub(r"^\s*##\s+(.+)$", r"\\subsection{\1}", md, flags=re.MULTILINE)
    # # section: 只清除 "4 四、" 这类数字+中文的冲突前缀
    md = re.sub(
        r"^\s*#\s+\d+(?:\.\d+)*\s+(.+)",
        lambda m: "\\section{" + m.group(1) + "}",
        md, flags=re.MULTILINE,
    )
    md = re.sub(r"^\s*#\s+(.+)", lambda m: "\\section{" + m.group(1) + "}", md, flags=re.MULTILINE)

    # ==== Step 2: 加粗 ====
    md = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", md)

    # ==== Step 3: 公式 ====
    # $$...$$ → \begin{equation}...\end{equation}
    # 使用非贪婪匹配，DOTALL 模式确保跨行公式完整匹配
    md = re.sub(
        r"\$\$\s*((?:[^$]|\$(?!\$))*?)\s*\$\$",
        r"\\begin{equation}\n\1\n\\end{equation}",
        md, flags=re.DOTALL,
    )

    # ==== Step 4: 图片 ====
    def _fig_replacer(m):
        alt = m.group(1)
        src = m.group(2).replace("\\", "/")  # Windows backslash → forward slash
        full_path = os.path.join(base_dir, src) if not os.path.isabs(src) else src
        if os.path.exists(full_path):
            return (
                r"\begin{figure}[htbp]" + "\n"
                r"\centering" + "\n"
                rf"\includegraphics[width=0.85\textwidth]{{{src}}}" + "\n"
                rf"\caption{{{alt}}}" + "\n"
                r"\end{figure}"
            )
        else:
            # 图片文件不存在，用简单占位文本避免 xelatex 崩溃
            return (
                r"\begin{figure}[htbp]" + "\n"
                r"\centering" + "\n"
                rf"\texttt{{[图片缺失: {src}]}}" + "\n"
                rf"\caption{{{alt}}}" + "\n"
                r"\end{figure}"
            )
    md = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", _fig_replacer, md)

    # Step 4.1: 移除 figure 后紧跟的重复图注（**图X: ...**）
    md = re.sub(
        r"(\\end\{figure\})\s*\n\s*\\textbf\{图\d+[：:][^}]*\}",
        r"\1",
        md,
    )

    # ==== Step 4.5: Unicode 下标替换 ====
    unicode_subs = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    def _fix_unicode_sub(m):
        inner = m.group(1).translate(unicode_subs)
        return f"$_{{{inner}}}$"
    md = re.sub(r"([₀₁₂₃₄₅₆₇₈₉]+)", _fix_unicode_sub, md)

    # ==== Step 5: 列表 ====
    # - item → \item
    md = re.sub(r"^\s*-\s+(.+)", r"\\item \1", md, flags=re.MULTILINE)
    # 连续的 \item 用 enumerate 包裹
    md = _wrap_items(md)

    # ==== Step 6: 转义特殊字符 ====
    # 公式环境内（$...$, $$...$$, \begin...\end）不转义
    # 只转义正文中的 & % _ #
    # 分割出公式和正文：正则匹配 $...$, $$...$$, \begin{...}...\end{...}
    body_parts = re.split(
        r"(\$\$(?:[^$]|\$(?!\$))*?\$\$|\$(?!\$)[^$]*?\$|\\begin\{[^}]*\}.*?\\end\{[^}]*\})",
        md, flags=re.DOTALL,
    )
    md = "".join(
        _escape_body_text(part) if i % 2 == 0 else part
        for i, part in enumerate(body_parts)
    )

    # ==== Step 7: 元数据提取 ====
    # 提取标题：第一个 \section{...} 作为论文标题（不以中文数字开头）
    title_line = title  # 默认使用传入的标题
    title_match = re.search(
        r"\\section\{((?!一、|二、|三、|四、|五、|六、|七、|八、|九、|十、).+?)\}",
        md
    )
    if title_match:
        title_line = title_match.group(1)
        md = md[:title_match.start()] + md[title_match.end():]

    # 提取摘要内容：从 \subsection{摘要} 或 ## 摘要 之后到下一个 \section
    abstract = ""
    abs_match = re.search(r"\\subsection\{摘要\}", md)
    if not abs_match:
        abs_match = re.search(r"\\section\{摘要\}", md)

    if abs_match:
        abs_start = abs_match.end()
        abs_end = md.find("\\section{", abs_start)
        if abs_end < 0:
            abs_end = len(md)
        abstract = md[abs_start:abs_end].strip()
        # 移除关键词行
        kw_match = re.search(r"\\textbf\{关键词[：:]\}|关键词[：:]", abstract)
        if kw_match:
            abstract = abstract[:kw_match.start()].strip()
        md = md[:abs_match.start()] + md[abs_end:]

    # 清理残留的 \textbf{标题：...} 和 \textbf{摘要}
    md = re.sub(r"\\textbf\{(?:标题|摘要)[：:][^}]*\}", "", md)

    # ==== Step 8: 组装完整文档 ====
    doc_title = title_line or title
    preamble = (
        r"\documentclass[12pt,a4paper]{ctexart}" + "\n"
        r"\usepackage{amsmath,amssymb}" + "\n"
        r"\usepackage{booktabs}" + "\n"
        r"\usepackage{graphicx}" + "\n"
        r"\usepackage{grffile}" + "\n"
        r"\usepackage{geometry}" + "\n"
        r"\usepackage{hyperref}" + "\n"
        r"\usepackage{enumitem}" + "\n"
        r"\geometry{left=2.5cm,right=2.5cm,top=2.5cm,bottom=2.5cm}" + "\n"
        "\n"
        rf"\title{{{doc_title}}}" + "\n"
        r"\author{}" + "\n"
        r"\date{}" + "\n"
        "\n"
        r"\begin{document}" + "\n"
        r"\maketitle" + "\n"
        "\n"
        r"\begin{abstract}" + "\n"
        f"{abstract}\n"
        r"\end{abstract}" + "\n"
        "\n"
        f"{md}\n"
        "\n"
        r"\end{document}"
    )
    latex = preamble

    # ==== Step 9: 回填 HTML 表格 + 清理重复表名 ====
    for i, tbl_tex in enumerate(tables_latex):
        placeholder = f"[TABLE_{i}]"
        # 移除占位符前重复的 \textbf{表X: ...}
        latex = re.sub(
            r"\\textbf\{表\d+[：:][^}]*\}[ \t]*\n?[ \t]*\[TABLE_" + str(i) + r"\]",
            f"[TABLE_{i}]",
            latex,
        )
        latex = latex.replace(placeholder, tbl_tex)

    # 抑制 ctexart 自动编号（标题自带中文数字）
    latex = latex.replace(
        "\\begin{document}",
        "\\renewcommand{\\thesection}{}\n"
        "\\renewcommand{\\thesubsection}{}\n"
        "\\renewcommand{\\thesubsubsection}{}\n"
        "\\begin{document}",
    )

    return latex


def _extract_html_tables_from_list(md: str, html_list: list[str], tables_out: list) -> str:
    """从外部传入的 HTML 表格列表生成 LaTeX tabular。

    当 res.md 被 DOCX 流程替换为 [TABLE_X] 占位符后，
    表格内容可从 res.json 的 response_content 中恢复。
    """
    if not html_list:
        return md

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return md

    for i, html in enumerate(html_list):
        if not html.strip():
            continue
        soup = BeautifulSoup(html, "lxml")
        table = soup.find("table")
        if not table:
            continue

        caption = ""
        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["th", "td"])]
            if cells:
                rows.append(cells)

        if not rows:
            continue
        headers, data_rows = rows[0], rows[1:]
        num_cols = max(len(r) for r in rows)

        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
        ]
        if caption:
            lines.append(rf"\\caption{{{caption}}}")
        if num_cols > 5:
            lines.append("\\footnotesize")
        lines += [
            f"\\begin{{tabular}}{{{'c' * num_cols}}}",
            "\\toprule",
        ]

        for r_idx, row in enumerate([headers] + data_rows):
            while len(row) < num_cols:
                row.append("")
            safe = []
            for c in row:
                if any(ch in c for ch in "_^{}\\"):
                    c = f"${c}$" if not c.startswith("$") else c
                c = c.replace("&", "\\&").replace("%", "\\%")
                safe.append(c)
            line = " & ".join(
                f"\\textbf{{{c}}}" if r_idx == 0 else c for c in safe
            ) + " \\\\"
            lines.append(line)
            if r_idx == 0:
                lines.append("\\midrule")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")

        tbl_tex = "\n".join(lines)
        tables_out.append(tbl_tex)
        # 保留 [TABLE_X] 占位符，由 Step 9 统一替换

    return md


def _extract_html_tables(md: str, tables_out: list) -> str:
    """BS4 解析 HTML <table> → LaTeX tabular+booktabs。
    同时提取表前的 **表X: xxx** 作为 \\caption{}。
    """
    html_tables = re.findall(r"<table>(.*?)</table>", md, re.DOTALL)
    if not html_tables:
        return md

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return md

    for i, html in enumerate(html_tables):
        soup = BeautifulSoup(f"<table>{html}</table>", "lxml")
        table = soup.find("table")
        if not table:
            continue

        # 找 HTML <caption> 作为表名（去掉 "表X:" 前缀，让 LaTeX 自动编号）
        caption = ""
        cap_elem = soup.find("caption")
        if cap_elem:
            caption = cap_elem.get_text(strip=True)
            # 去掉 LLM 写的 "表X:" 前缀（LaTeX 会自动加编号）
            caption = re.sub(r"^表\d+[：:]\s*", "", caption)
            caption = caption.replace("%", "\\%").replace("&", "\\&")

        headers, rows = [], []
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["th", "td"])]
            if not cells:
                continue
            if not headers and tr.find("th"):
                headers = cells
            else:
                rows.append(cells)

        if not headers and rows:
            headers, rows = rows[0], rows[1:]
        all_rows = ([headers] if headers else []) + rows
        num_cols = max(len(r) for r in all_rows)

        # 构建 LaTeX tabular（宽表自动缩小）
        col_spec = "c" * num_cols
        size_cmd = ""
        if num_cols > 5:
            size_cmd = "\\footnotesize\n"
        elif num_cols > 4:
            size_cmd = "\\small\n"

        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
        ]
        if caption:
            lines.append(rf"\caption{{{caption}}}")
        if size_cmd.strip():
            lines.append(size_cmd.strip())
        lines += [
            f"\\begin{{tabular}}{{{col_spec}}}",
            "\\toprule",
        ]

        for r_idx, row in enumerate(all_rows):
            while len(row) < num_cols:
                row.append("")
            safe_cells = []
            for c in row:
                if any(ch in c for ch in "_^{}\\"):
                    c = f"${c}$" if not c.startswith("$") else c
                c = c.replace("&", "\\&").replace("%", "\\%")
                safe_cells.append(c)
            line = " & ".join(
                f"\\textbf{{{c}}}" if r_idx == 0 else c for c in safe_cells
            )
            line += " \\\\"
            lines.append(line)
            if r_idx == 0:
                lines.append("\\midrule")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")

        tbl_tex = "\n".join(lines)
        tables_out.append(tbl_tex)

        placeholder = f"[TABLE_{i}]"
        md = md.replace(f"<table>{html}</table>", placeholder, 1)

    return md


def _wrap_items(md: str) -> str:
    """将连续的 \\item 行用 enumerate 包裹。"""
    lines = md.split("\n")
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("\\item "):
            result.append("\\begin{enumerate}[label=(\\arabic*)]")
            while i < len(lines) and lines[i].strip().startswith("\\item "):
                result.append(lines[i])
                i += 1
            result.append("\\end{enumerate}")
        else:
            result.append(line)
            i += 1
    return "\n".join(result)
