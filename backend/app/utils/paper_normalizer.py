"""论文格式标准化模块。

自动修复论文 Markdown 格式中的常见错误：
- 标题层级标准化
- 表格格式标准化
- 图片引用标准化
- 章节顺序调整
- 添加缺失的表号图号标注
"""

import re
from bs4 import BeautifulSoup
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NormalizationResult:
    """标准化结果。"""
    original_content: str
    normalized_content: str
    fixes_applied: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    
    @property
    def has_changes(self) -> bool:
        return self.original_content != self.normalized_content


_NUMERIC_CELL = re.compile(r"[\s\d.,%\-–—/()（）·:：]+")


def _caption_from_html(table_html: str, max_parts: int = 4, max_len: int = 40) -> str:
    """从表头行推导一个描述性题注。

    原实现给所有无题注的表格写死 "数据表格"，实测 24 张表全部叫这个名字：
    等于没有题注，又容易被误当成真实题注读过去。表头通常已经说明了表格
    内容，直接采用它比占位词有用得多。

    有些表格把数值也写成 `<th>`（如「轮次 | 5 | 10 | 20 | 40」），故先剔除
    纯数值格；若剔除后只剩一个词，则补上数值跨度（如「轮次 5～40」），
    否则「表15: 轮次」这种题注信息量太低。
    """
    m = re.search(r"<tr[^>]*>(.*?)</tr>", table_html, re.S | re.I)
    if not m:
        return "数据表"
    cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", m.group(1), re.S | re.I)
    parts = []
    for c in cells:
        t = re.sub(r"<[^>]+>", "", c)
        t = re.sub(r"\s+", " ", t).strip()
        if t:
            parts.append(t)
    if not parts:
        return "数据表"

    plain = [p for p in parts if not _NUMERIC_CELL.fullmatch(p)]
    nums = [p for p in parts if _NUMERIC_CELL.fullmatch(p)]

    # 「双列对」表头（如「编号|理想坐标|编号|理想坐标」）会重复，
    # 去重后才是这张表真正在讲什么。
    plain = list(dict.fromkeys(plain))

    if len(plain) == 1 and len(nums) >= 2:
        cap = f"{plain[0]} {nums[0]}～{nums[-1]}"
    else:
        use = plain or parts
        cap = "、".join(use[:max_parts])
        if len(use) > max_parts:
            cap += " 等"
    if len(cap) > max_len:
        cap = cap[:max_len] + "…"
    return cap or "数据表"


def _caption_from_following(lines, i, image_pattern, max_len: int = 48) -> str:
    """用图片后紧跟的分析文字首句作图注。

    原实现写死 "图片分析"，与表题注原先写死的 "数据表格" 是同一类缺陷：
    看着像题注、却不传递任何信息。图片后面的首句通常正是该图的说明，
    直接采用它比占位词有用。确实取不到时给一个显式的待补标记，
    而不是一句读起来像真题注的话。
    """
    for k in range(i + 1, min(i + 6, len(lines))):
        t = lines[k].strip()
        if not t or t.startswith("#") or image_pattern.search(t):
            continue
        t = re.sub(r"[*`$]", "", t)
        # 跳过「结果分析。」这类过短的引导标签，它不构成题注
        sents = [x.strip() for x in re.split(r"[。；;]", t) if x.strip()]
        cap = next((x for x in sents if len(x) >= 8), sents[0] if sents else t)
        if len(cap) > max_len:
            cap = cap[:max_len] + "…"
        return cap or "（图注待补）"
    return "（图注待补）"


class PaperNormalizer:
    """论文格式标准化器。"""
    
    # 中文数字到阿拉伯数字映射
    CN_NUMBER_MAP = {
        "一": "1", "二": "2", "三": "3", "四": "4", "五": "5",
        "六": "6", "七": "7", "八": "8", "九": "9", "十": "10"
    }
    
    # 一级标题模式（一、xxx）
    HEADING_1_PATTERN = re.compile(r"^#?\s*([一二三四五六七八九十]+)、(.+)")
    
    # 二级标题模式（1.1 xxx 或 1.1.xxx）
    HEADING_2_PATTERN = re.compile(r"^##?\s*(\d+\.[\d]+)\s+(.+)")
    
    # 三级标题模式（1.1.1 xxx）
    HEADING_3_PATTERN = re.compile(r"^###?\s*(\d+\.[\d]+\.[\d]+)\s+(.+)")
    
    # Markdown 一级标题
    MD_HEADING_1 = re.compile(r"^#\s+(.+)")
    
    # 表号标注模式
    TABLE_CAPTION_PATTERN = re.compile(r"^\*\*?表\s*(\d+)[：:]\s*(.+)\*\*?\s*$", re.MULTILINE)
    
    # 图号标注模式
    FIGURE_CAPTION_PATTERN = re.compile(r"^\*\*?图\s*(\d+)[：:]\s*(.+)\*\*?\s*$", re.MULTILINE)
    
    # 图片引用模式
    IMAGE_PATTERN = re.compile(r"!\[([^\]]*)\]\(([^)]+\.(?:png|jpg|jpeg|gif|bmp|webp))\)")
    
    def __init__(self):
        self.result = NormalizationResult("", "")
        self._table_counter = 0
        self._figure_counter = 0
        self._last_section_number = 0
    
    def normalize(self, content: str) -> NormalizationResult:
        """执行完整标准化流程。
        
        Args:
            content: Markdown 论文内容。
            
        Returns:
            NormalizationResult: 标准化结果。
        """
        self.result = NormalizationResult(original_content=content, normalized_content=content)
        self._table_counter = 0
        self._figure_counter = 0
        
        self._normalize_line_endings()
        self._normalize_tables()
        self._normalize_headings()
        self._normalize_figure_captions()
        self._normalize_table_captions()
        self._remove_duplicate_blank_lines()
        
        return self.result
    
    def _normalize_line_endings(self) -> None:
        """统一行尾换行符。"""
        content = self.result.normalized_content
        
        # 确保使用一致的换行符
        content = content.replace("\r\n", "\n").replace("\r", "\n")
        
        # 移除行尾多余空格
        lines = [line.rstrip() for line in content.split("\n")]
        content = "\n".join(lines)
        
        if content != self.result.original_content:
            self.result.normalized_content = content
            self.result.fixes_applied.append("统一行尾换行符和空格")
    
    def _normalize_tables(self) -> None:
        """标准化 HTML 表格格式。"""
        content = self.result.normalized_content
        
        def fix_table(match):
            html = match.group(0)
            soup = BeautifulSoup(html, "lxml")
            table = soup.find("table")
            
            if not table:
                return html
            
            # 确保有 thead 和 tbody
            if not table.find("thead"):
                first_row = table.find("tr")
                if first_row and first_row.find("th"):
                    thead = soup.new_tag("thead")
                    thead.append(first_row.extract())
                    table.insert(0, thead)
            
            if not table.find("tbody"):
                tbody = soup.new_tag("tbody")
                for child in list(table.children):
                    if child.name and child.name not in ["thead"]:
                        tbody.append(child.extract())
                table.append(tbody)
            
            # 确保单元格格式正确
            for tag in ["th", "td"]:
                for cell in table.find_all(tag):
                    # 清理单元格内容，保留基本格式
                    text = cell.get_text()
                    # 移除多余的空白字符
                    text = re.sub(r"\s+", " ", text).strip()
                    cell.string = text
            
            fixed = str(soup)
            if fixed != html:
                self.result.fixes_applied.append(f"修复 HTML 表格格式")
            
            return fixed
        
        pattern = re.compile(r"<table>.*?</table>", re.DOTALL | re.IGNORECASE)
        content = pattern.sub(fix_table, content)
        self.result.normalized_content = content
    
    def _normalize_headings(self) -> None:
        """标准化标题格式。"""
        lines = self.result.normalized_content.split("\n")
        normalized_lines = []
        current_chapter = 0
        
        for i, line in enumerate(lines):
            original_line = line
            line = line.strip()
            
            # 检查一级标题（一、xxx）
            match1 = self.HEADING_1_PATTERN.match(line)
            if match1:
                chapter_num = match1.group(1)
                title = match1.group(2)
                
                # 转换中文数字到阿拉伯数字
                if chapter_num in self.CN_NUMBER_MAP:
                    current_chapter = int(self.CN_NUMBER_MAP[chapter_num])
                
                # 移除 # 前缀，统一格式
                line = f"# {chapter_num}、{title}"
                self.result.fixes_applied.append(f"标准化一级标题: {chapter_num}、{title[:20]}...")
            
            # 检查二级标题（1.1 xxx）- 确保格式一致
            match2 = self.HEADING_2_PATTERN.match(line)
            if match2:
                num = match2.group(1)
                title = match2.group(2)
                # 确保编号格式为 x.x
                if "." in num:
                    parts = num.split(".")
                    if len(parts) > 2:
                        num = f"{parts[0]}.{parts[1]}"
                        line = f"## {num} {title}"
                        self.result.fixes_applied.append(f"修复二级标题编号: {title[:20]}...")
                    elif len(parts) == 2:
                        line = f"## {num} {title}"
            
            # 检查三级标题
            match3 = self.HEADING_3_PATTERN.match(line)
            if match3:
                num = match3.group(1)
                title = match3.group(2)
                # 确保编号格式为 x.x.x
                parts = num.split(".")
                if len(parts) == 3:
                    line = f"### {num} {title}"
                elif len(parts) > 3:
                    num = f"{parts[0]}.{parts[1]}.{parts[2]}"
                    line = f"### {num} {title}"
                    self.result.fixes_applied.append(f"修复三级标题编号: {title[:20]}...")
            
            # 处理 Markdown 格式的一级标题
            match_md1 = self.MD_HEADING_1.match(line)
            if match_md1:
                title = match_md1.group(1)
                # 检查是否应该是中文数字格式
                for cn, ar in self.CN_NUMBER_MAP.items():
                    if title.startswith(f"{cn}、") or title.startswith(f"{ar}、"):
                        # 已经是正确格式
                        pass
            
            normalized_lines.append(line if line == original_line.strip() else line)
        
        self.result.normalized_content = "\n".join(normalized_lines)
    
    def _normalize_figure_captions(self) -> None:
        """确保图片有正确的图号标注。"""
        content = self.result.normalized_content
        lines = content.split("\n")
        normalized_lines = []
        figure_counter = 0
        
        i = 0
        while i < len(lines):
            line = lines[i]
            normalized_lines.append(line)
            
            # 检测图片引用
            if self.IMAGE_PATTERN.search(line):
                figure_counter += 1

                # 判断是否已有图号标注。
                # 本项目的图注约定是「图片在前、图注在后」（见 flows.py 与
                # _renumber_tables_figures），因此必须同时向前后看。
                #
                # 原实现有两处错误，会导致给**每一张**已有图注的图片都插入一条
                # 占位图注"**图N: 图片分析**"：
                #   1) 取的是 normalized_lines[-1]，而该行正是刚追加的图片行本身，
                #      对图片行匹配图注正则必然失败；
                #   2) 只看前一行，而约定中图注在图片之后。
                caption_re = re.compile(r"^\*{0,2}图\s*\d+[：:]")
                prev_line = normalized_lines[-2].strip() if len(normalized_lines) >= 2 else ""
                has_caption = bool(caption_re.match(prev_line))
                if not has_caption:
                    for k in range(i + 1, min(i + 4, len(lines))):
                        nxt = lines[k].strip()
                        if not nxt:
                            continue
                        if nxt.startswith("#") or self.IMAGE_PATTERN.search(nxt):
                            break
                        has_caption = bool(caption_re.match(nxt))
                        break

                if not has_caption:
                    cap = _caption_from_following(lines, i, self.IMAGE_PATTERN)
                    normalized_lines[-1] = (
                        f"**图{figure_counter}: {cap}**\n\n{line}")
                    self.result.fixes_applied.append(
                        f"添加图号标注: 图{figure_counter}")
                
                # 检查后续内容是否有图注（至少50字）
                j = i + 1
                caption_length = 0
                while j < len(lines) and j < i + 10:
                    next_line = lines[j].strip()
                    if next_line.startswith("#") or self.IMAGE_PATTERN.search(next_line):
                        break
                    caption_length += len(next_line)
                    if caption_length > 50:
                        break
                    j += 1
                
                if caption_length < 10:
                    self.result.warnings.append(
                        f"图{figure_counter} 的图注文字可能过短，建议至少50字分析说明"
                    )
            
            i += 1
        
        self.result.normalized_content = "\n".join(normalized_lines)
    
    def _normalize_table_captions(self) -> None:
        """为表格自动编号，补充缺失的表号标注。"""
        content = self.result.normalized_content

        table_pattern = re.compile(r"(<table>.*?</table>)", re.DOTALL | re.IGNORECASE)
        tables = list(table_pattern.finditer(content))

        if not tables:
            return

        # 收集已有表号作为起始偏移
        existing_captions = re.findall(r"\*\*?表\s*(\d+)[：:]", content)
        max_existing = max((int(n) for n in existing_captions), default=0)

        result_parts = []
        last_end = 0
        added = 0
        for idx, match in enumerate(tables):
            table_html = match.group(1)
            start, end = match.start(), match.end()

            # 表格前 300 字符内查找已有表号
            before = content[max(0, start - 300):start]
            cap_match = re.search(r"\*\*?表\s*(\d+)[：:]\s*([^*]*)\*\*?", before)

            if cap_match:
                # 已有题注：它本来就在 content[last_end:start] 这一段里，
                # 直接保留即可。原实现在这里也走"重新插入 label"的分支，
                # 于是每张已有题注的表都会多出一条重复题注（实测 21 张表 42 条）。
                continue

            added += 1
            label = f"**表{max_existing + added}: {_caption_from_html(table_html)}**"
            result_parts.append(content[last_end:start])
            result_parts.append(f"\n\n{label}\n\n{table_html}")
            last_end = end

        result_parts.append(content[last_end:])
        self.result.normalized_content = "".join(result_parts)

        if len(tables) > 0:
            self.result.fixes_applied.append(f"为 {len(tables)} 个表格自动编号")
    
    def _remove_duplicate_blank_lines(self) -> None:
        """移除重复的空行。"""
        content = self.result.normalized_content
        
        # 将多个连续空行替换为单个空行
        content = re.sub(r"\n{3,}", "\n\n", content)
        
        if content != self.result.normalized_content:
            self.result.normalized_content = content
            self.result.fixes_applied.append("移除重复空行")


def normalize_paper(content: str) -> NormalizationResult:
    """便捷函数：标准化论文格式。
    
    Args:
        content: Markdown 论文内容。
        
    Returns:
        NormalizationResult: 标准化结果。
    """
    normalizer = PaperNormalizer()
    return normalizer.normalize(content)
