"""论文格式验证模块。

验证论文 Markdown 格式的完整性和正确性：
- 标题层级验证
- 表格格式验证（HTML <table>）
- 图片引用验证
- 公式格式验证
- 章节编号连续性验证
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ValidationError:
    """验证错误信息。"""
    section: str
    error_type: str
    message: str
    line_number: Optional[int] = None
    fix_suggestion: str = ""


@dataclass
class ValidationResult:
    """验证结果。"""
    passed: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    score: float = 0.0
    
    @property
    def error_count(self) -> int:
        return len(self.errors)
    
    @property
    def warning_count(self) -> int:
        return len(self.warnings)
    
    def add_error(self, section: str, error_type: str, message: str, 
                  line_number: Optional[int] = None, fix_suggestion: str = ""):
        self.errors.append(ValidationError(
            section=section, error_type=error_type,
            message=message, line_number=line_number,
            fix_suggestion=fix_suggestion
        ))
        self.passed = False
    
    def add_warning(self, message: str):
        self.warnings.append(message)
    
    def calculate_score(self) -> float:
        """计算格式完整性评分（0-100）。"""
        if not self.errors:
            return 100.0
        
        # 基础分数
        score = 100.0
        
        # 错误扣分（按严重程度）
        error_weights = {
            "missing_section": 15,      # 缺少章节
            "invalid_heading": 10,      # 无效标题
            "invalid_table": 8,         # 表格格式错误
            "missing_image": 5,         # 图片缺失
            "invalid_formula": 3,       # 公式格式错误
            "inconsistent_numbering": 5, # 编号不一致
            "empty_section": 2,        # 空章节
        }
        
        for error in self.errors:
            weight = error_weights.get(error.error_type, 5)
            score -= weight
        
        return max(0.0, score)


class PaperValidator:
    """论文格式验证器。"""
    
    # 章节顺序定义（按论文标准结构）
    REQUIRED_SECTIONS = [
        "摘要",
        "问题重述",
        "问题分析",
        "模型假设",
        "符号说明",
        "数据预处理",
        "模型建立与求解",
        "敏感性分析",
        "模型评价与改进",
        "参考文献",
    ]
    
    # 一级标题模式（一、xxx）
    HEADING_1_PATTERN = re.compile(r"^#?\s*([一二三四五六七八九十]+)、(.+)")
    
    # 二级标题模式（1.1 xxx）
    HEADING_2_PATTERN = re.compile(r"^##?\s*(\d+\.\d+)\s+(.+)")
    
    # 三级标题模式（1.1.1 xxx）
    HEADING_3_PATTERN = re.compile(r"^###?\s*(\d+\.\d+\.\d+)\s+(.+)")
    
    # Markdown 一级标题（# xxx）
    MD_HEADING_1 = re.compile(r"^#\s+(.+)")
    
    # Markdown 二级标题（## xxx）
    MD_HEADING_2 = re.compile(r"^##\s+(.+)")
    
    # Markdown 三级标题（### xxx）
    MD_HEADING_3 = re.compile(r"^###\s+(.+)")
    
    # HTML 表格模式
    HTML_TABLE_PATTERN = re.compile(r"<table>.*?</table>", re.DOTALL | re.IGNORECASE)
    
    # 图片引用模式
    IMAGE_PATTERN = re.compile(r"!\[([^\]]*)\]\(([^)]+\.(?:png|jpg|jpeg|gif|bmp|webp))\)")
    
    # 公式模式（块级）
    BLOCK_FORMULA_PATTERN = re.compile(r"\$\$[\s\S]+?\$\$")
    
    # 内联公式模式
    INLINE_FORMULA_PATTERN = re.compile(r"(?<!\$)\$(?!\$)[^$\n]+\$(?!\$)")
    
    def __init__(self, work_dir: str | Path):
        self.work_dir = Path(work_dir)
        self.result = ValidationResult(passed=True)
    
    def validate(self, markdown_content: str) -> ValidationResult:
        """执行完整验证流程。
        
        Args:
            markdown_content: Markdown 论文内容。
            
        Returns:
            ValidationResult: 验证结果。
        """
        self.result = ValidationResult(passed=True)
        
        lines = markdown_content.split("\n")
        
        self._validate_headings(lines)
        self._validate_tables(markdown_content)
        self._validate_images(markdown_content)
        self._validate_formulas(markdown_content)
        self._validate_section_sequencing(lines)
        self._validate_required_sections(markdown_content)
        self._validate_references(markdown_content)
        self._validate_content_completeness(lines)
        
        self.result.score = self.result.calculate_score()
        return self.result
    
    def _validate_headings(self, lines: list[str]) -> None:
        """验证标题格式。"""
        heading_stack = []
        prev_level = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # 检查一级标题（一、xxx）
            match1 = self.HEADING_1_PATTERN.match(line)
            if match1:
                heading_stack = [match1.group(1)]
                prev_level = 1
                continue
            
            # 检查 Markdown 一级标题
            match_md1 = self.MD_HEADING_1.match(line)
            if match_md1:
                # 检查是否缺少 # 前缀但应该有一级标题格式
                if any(char in line for char in "一二三四五六七八九十"):
                    self.result.add_warning(f"第{i+1}行：使用中文数字编号但缺少 '#' 前缀")
                prev_level = 1
                continue
            
            # 检查二级标题
            match2 = self.HEADING_2_PATTERN.match(line)
            if match2:
                heading_stack.append(match2.group(1))
                if prev_level > 1:
                    self.result.add_error(
                        "标题层级",
                        "invalid_heading",
                        f"第{i+1}行：二级标题前缺少对应的一级标题",
                        i + 1,
                        "确保存在上级标题"
                    )
                prev_level = 2
                continue
            
            match_md2 = self.MD_HEADING_2.match(line)
            if match_md2:
                prev_level = 2
                continue
            
            # 检查三级标题
            match3 = self.HEADING_3_PATTERN.match(line)
            if match3:
                if prev_level < 2:
                    self.result.add_error(
                        "标题层级",
                        "invalid_heading",
                        f"第{i+1}行：三级标题前缺少二级标题",
                        i + 1,
                        "确保存在 '1.1 xxx' 格式的二级标题"
                    )
                prev_level = 3
                continue
            
            match_md3 = self.MD_HEADING_3.match(line)
            if match_md3:
                prev_level = 3
                continue
    
    def _validate_tables(self, content: str) -> None:
        """验证表格格式。"""
        tables = list(self.HTML_TABLE_PATTERN.finditer(content))
        
        for idx, match in enumerate(tables):
            table_html = match.group(0)
            position = match.start()
            
            # 验证表头
            if "<th>" not in table_html.lower() and "<thead>" not in table_html.lower():
                if "<tr>" not in table_html:
                    self.result.add_error(
                        "表格格式",
                        "invalid_table",
                        f"表格 {idx+1}：缺少表头行（<th> 或 <thead>）",
                        fix_suggestion="使用 <th> 标签定义表头单元格"
                    )
            
            # 验证数据行
            rows = re.findall(r"<tr>.*?</tr>", table_html, re.DOTALL | re.IGNORECASE)
            if len(rows) < 2:
                self.result.add_error(
                    "表格格式",
                    "invalid_table",
                    f"表格 {idx+1}：缺少数据行（至少需要表头+1行数据）",
                    fix_suggestion="确保表格至少有表头行和1行数据"
                )
            
            # 验证单元格完整性
            for row_idx, row_html in enumerate(rows):
                cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row_html, re.DOTALL | re.IGNORECASE)
                if not cells:
                    self.result.add_error(
                        "表格格式",
                        "invalid_table",
                        f"表格 {idx+1} 第{row_idx+1}行：缺少单元格",
                        fix_suggestion="确保每行有正确数量的单元格"
                    )
            
            # 检查表格前后是否有表号标注
            before = content[max(0, position-200):position]
            if not re.search(r"表\s*\d+", before):
                self.result.add_warning(f"表格 {idx+1}：缺少表号标注（如 **表1: xxx**）")
    
    def _validate_images(self, content: str) -> None:
        """验证图片引用。"""
        images = list(self.IMAGE_PATTERN.finditer(content))
        
        for idx, match in enumerate(images):
            alt_text = match.group(1)
            image_path = match.group(2)
            
            # 检查图片是否存在
            if not image_path.startswith(("http://", "https://", "data:")):
                image_file = self.work_dir / image_path
                if not image_file.exists():
                    self.result.add_error(
                        "图片引用",
                        "missing_image",
                        f"图片 {idx+1}：'{image_path}' 不存在",
                        fix_suggestion=f"确保图片文件存在于 {self.work_dir}"
                    )
            
            # 检查是否有图号标注
            position = match.start()
            before = content[max(0, position-150):position]
            if not re.search(r"图\s*\d+", before):
                self.result.add_warning(f"图片 {idx+1}：缺少图号标注（如 **图1: xxx**）")
            
            # 检查是否有图注文字
            after = content[match.end():min(match.end()+200, len(content))]
            if len(alt_text) < 5:
                self.result.add_warning(f"图片 {idx+1}：图注文字过短，可能缺少分析说明")
    
    def _validate_formulas(self, content: str) -> None:
        """验证公式格式。"""
        block_formulas = list(self.BLOCK_FORMULA_PATTERN.finditer(content))
        
        for idx, match in enumerate(block_formulas):
            formula = match.group(0)
            
            # 检查公式是否完整
            if formula.count("$") != 4:
                self.result.add_error(
                    "公式格式",
                    "invalid_formula",
                    f"块级公式 {idx+1}：格式不完整（应使用 $$...$$）",
                    fix_suggestion="使用 $$ 包裹块级公式"
                )
            
            # 检查公式内容是否为空
            formula_content = formula.replace("$$", "").strip()
            if len(formula_content) < 3:
                self.result.add_error(
                    "公式格式",
                    "invalid_formula",
                    f"块级公式 {idx+1}：公式内容为空",
                    fix_suggestion="确保公式有实际内容"
                )
        
        # 检查内联公式
        inline_formulas = list(self.INLINE_FORMULA_PATTERN.finditer(content))
        for idx, match in enumerate(inline_formulas):
            formula = match.group(0)
            if formula.count("$") != 2:
                self.result.add_error(
                    "公式格式",
                    "invalid_formula",
                    f"内联公式 {idx+1}：格式不完整",
                    fix_suggestion="使用 $...$ 包裹内联公式"
                )
    
    def _validate_section_sequencing(self, lines: list[str]) -> None:
        """验证章节编号连续性。"""
        section_numbers = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # 提取中文数字编号
            match = self.HEADING_1_PATTERN.match(line)
            if match:
                section_numbers.append((match.group(1), i + 1))
                continue
            
            # 提取阿拉伯数字编号
            match2 = self.HEADING_2_PATTERN.match(line)
            if match2:
                section_numbers.append((match2.group(1), i + 1))
                continue
            
            match3 = self.HEADING_3_PATTERN.match(line)
            if match3:
                section_numbers.append((match3.group(1), i + 1))
        
        # 检查编号是否连续（一、二、三...）
        expected_order = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]
        seen_order = []
        
        for num, _ in section_numbers:
            if num in expected_order and len(num) == 1:
                seen_order.append(num)
        
        for i in range(len(seen_order) - 1):
            curr_idx = expected_order.index(seen_order[i]) if seen_order[i] in expected_order else -1
            next_idx = expected_order.index(seen_order[i+1]) if seen_order[i+1] in expected_order else -1
            
            if curr_idx != -1 and next_idx != -1 and next_idx < curr_idx:
                self.result.add_error(
                    "章节编号",
                    "inconsistent_numbering",
                    f"章节编号顺序错误：'{seen_order[i]}' 后面不应是 '{seen_order[i+1]}'",
                    fix_suggestion="按一、二、三...顺序排列章节"
                )
    
    def _validate_required_sections(self, content: str) -> None:
        """验证必需章节是否存在。"""
        for section in self.REQUIRED_SECTIONS:
            if section not in content:
                if section in ["摘要", "参考文献"]:
                    self.result.add_error(
                        "章节完整性",
                        "missing_section",
                        f"缺少必需章节：{section}",
                        fix_suggestion=f"添加 {section} 章节"
                    )
                else:
                    self.result.add_warning(f"缺少建议章节：{section}")

    def _validate_references(self, content: str) -> None:
        """验证参考文献格式（GB/T 7714-2025）。"""
        import re
        
        ref_section_pattern = re.compile(r"#.*参考文献", re.IGNORECASE)
        if not ref_section_pattern.search(content):
            self.result.add_warning("缺少参考文献章节，建议添加")
            return
        
        ref_pattern = re.compile(r"\[(\d+)\]\s+\S+.*?\.\s*[\[J\]]?[\[C\]]?[\[M\]]?", re.DOTALL)
        refs = ref_pattern.findall(content)
        
        if len(refs) < 3:
            self.result.add_warning(f"参考文献数量较少（{len(refs)}条），建议至少引用5-10条")
        
        for match in ref_pattern.finditer(content):
            ref_text = match.group(0)
            if not re.search(r"\d{4}", ref_text):
                self.result.add_warning(f"参考文献可能缺少年份：{ref_text[:50]}...")
            if not re.search(r"\[J\]|\[C\]|\[M\]|\[D\]", ref_text, re.IGNORECASE):
                self.result.add_warning(f"参考文献可能缺少文献类型标识：{ref_text[:50]}...")
    
    def _validate_content_completeness(self, lines: list[str]) -> None:
        """验证内容完整性（段落长度等）。"""
        current_section = ""
        section_lines = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # 检测章节标题
            match = self.HEADING_1_PATTERN.match(line)
            if match:
                # 检查上一个章节的内容
                if current_section and len(section_lines) < 3:
                    self.result.add_warning(
                        f"章节 '{current_section}' 内容过少（{len(section_lines)} 行）"
                    )
                current_section = match.group(1)
                section_lines = []
                continue
            
            # 跳过空行和标题
            if line and not line.startswith("#"):
                section_lines.append(line)
        
        # 检查最后一个章节
        if current_section and len(section_lines) < 3:
            self.result.add_warning(
                f"章节 '{current_section}' 内容过少（{len(section_lines)} 行）"
            )


def validate_paper(markdown_content: str, work_dir: str | Path) -> ValidationResult:
    """便捷函数：验证论文格式。
    
    Args:
        markdown_content: Markdown 论文内容。
        work_dir: 工作目录（用于检查图片文件是否存在）。
        
    Returns:
        ValidationResult: 验证结果。
    """
    validator = PaperValidator(work_dir)
    return validator.validate(markdown_content)
