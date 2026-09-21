"""生成数学建模论文的 Word 参考模板（templates/论文模板.docx）。

背景：`common_utils.md_2_docx` 依赖 `templates/论文模板.docx` 作为 pandoc 的
`--reference-doc`，但该文件在本仓库中从未存在过，于是 pandoc 一直使用内置默认
样式——中文字体、行距、首行缩进全部缺失，导出的 DOCX 排版不像学术论文。

做法：取 pandoc 自带的 `reference.docx`（其样式名与 pandoc 输出严格对应，
不必自己猜名字），逐项改为国内数学建模论文的通行排版，另存为模板。

排版参数依据《全国大学生数学建模竞赛论文格式规范》及多所高校的细化要求
（官方对字体字号不做硬性规定，以下为通行做法）：
  页面    A4，上下边距 2.5 cm，左右 3.0 cm
  正文    宋体小四（12pt）+ 西文 Times New Roman，1.5 倍行距，
          首行缩进 2 字符，两端对齐，段前段后 0
  一级标题 黑体三号（16pt）居中
  二级标题 黑体四号（14pt）左对齐
  三级标题 黑体小四（12pt）左对齐
  图表题注 宋体五号（10.5pt）居中
  页脚    居中连续页码

运行：
    ./.venv/Scripts/python.exe scripts/build_paper_template.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
OUT = os.path.join(BACKEND, "templates", "论文模板.docx")
SRC = os.path.join(BACKEND, "templates", "_pandoc_default.docx")

SONG = "宋体"
HEI = "黑体"
LATIN = "Times New Roman"


def ensure_pandoc_reference() -> None:
    """取 pandoc 自带 reference.docx 作为样式底本。"""
    if os.path.exists(SRC):
        return
    os.makedirs(os.path.dirname(SRC), exist_ok=True)
    import pypandoc
    pandoc = pypandoc.get_pandoc_path()
    subprocess.run(
        [pandoc, "--print-default-data-file", "reference.docx"],
        stdout=open(SRC, "wb"), check=True,
    )
    print(f"已获取 pandoc 默认 reference: {SRC}")


def set_fonts(style, latin: str, east: str, size_pt: float, bold: bool | None = None):
    """设置样式的西文/中文字体、字号与颜色。中文必须写 w:eastAsia，否则不生效。

    颜色必须显式设为黑：pandoc 自带 reference 的标题样式带主题强调色，
    不覆盖的话中文论文里会出现蓝色标题。
    """
    f = style.font
    f.name = latin
    f.size = Pt(size_pt)
    f.color.rgb = RGBColor(0, 0, 0)
    if bold is not None:
        f.bold = bold
    rPr = style.element.get_or_add_rPr()
    # 同时清掉主题色引用（w:themeColor 的优先级高于 w:val）
    for tag in ("w:themeColor", "w:themeTint", "w:themeShade"):
        el = rPr.find(qn(tag))
        if el is not None:
            rPr.remove(el)
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = rPr.makeelement(qn("w:rFonts"), {})
        rPr.insert(0, rFonts)
    rFonts.set(qn("w:ascii"), latin)
    rFonts.set(qn("w:hAnsi"), latin)
    rFonts.set(qn("w:eastAsia"), east)


def set_para(style, *, align=None, before=None, after=None,
             line=None, first_line_chars=None, left_chars=None):
    """设置段落格式。首行缩进用 firstLineChars（字符数×100），随字号自适应。"""
    pPr = style.element.get_or_add_pPr()
    if align is not None:
        style.paragraph_format.alignment = align
    if before is not None:
        style.paragraph_format.space_before = Pt(before)
    if after is not None:
        style.paragraph_format.space_after = Pt(after)
    if line is not None:
        style.paragraph_format.line_spacing = line

    ind = pPr.find(qn("w:ind"))
    if first_line_chars is not None or left_chars is not None:
        if ind is None:
            ind = pPr.makeelement(qn("w:ind"), {})
            pPr.append(ind)
        if first_line_chars is not None:
            ind.set(qn("w:firstLineChars"), str(int(first_line_chars * 100)))
            ind.set(qn("w:firstLine"), "0")
        if left_chars is not None:
            ind.set(qn("w:leftChars"), str(int(left_chars * 100)))
    # 中文段落默认禁止孤行
    if pPr.find(qn("w:widowControl")) is None:
        wc = pPr.makeelement(qn("w:widowControl"), {})
        wc.set(qn("w:val"), "1")
        pPr.append(wc)


def add_page_number_footer(section) -> None:
    """页脚居中插入连续页码域。"""
    footer = section.footer
    p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    for el, attrs, text in (
        ("w:fldChar", {"w:fldCharType": "begin"}, None),
        ("w:instrText", {"xml:space": "preserve"}, " PAGE "),
        ("w:fldChar", {"w:fldCharType": "end"}, None),
    ):
        e = run._element.makeelement(qn(el), {qn(k): v for k, v in attrs.items()})
        if text:
            e.text = text
        run._element.append(e)
    set_fonts_el(run, LATIN, SONG, 10.5)


def set_fonts_el(run, latin, east, size_pt):
    run.font.name = latin
    run.font.size = Pt(size_pt)
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = rPr.makeelement(qn("w:rFonts"), {})
        rPr.insert(0, rFonts)
    rFonts.set(qn("w:ascii"), latin)
    rFonts.set(qn("w:hAnsi"), latin)
    rFonts.set(qn("w:eastAsia"), east)


def build() -> None:
    ensure_pandoc_reference()
    d = Document(SRC)
    S = d.styles

    # ── 页面 ──
    sec = d.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)       # A4
    sec.top_margin = sec.bottom_margin = Cm(2.5)
    sec.left_margin = sec.right_margin = Cm(3.0)
    add_page_number_footer(sec)

    # ── 正文 ──
    # 注意 Body Text 与 First Paragraph 都要设：pandoc 对"标题后的第一段"
    # 用 First Paragraph，对其余正文用 Body Text。只设一个会导致缩进时有时无。
    for name in ("Body Text", "First Paragraph"):
        st = S[name]
        set_fonts(st, LATIN, SONG, 12)
        set_para(st, align=WD_ALIGN_PARAGRAPH.JUSTIFY, before=0, after=0,
                 line=1.5, first_line_chars=2)

    # Compact 用于紧凑列表项，列表不应有首行缩进
    set_fonts(S["Compact"], LATIN, SONG, 12)
    set_para(S["Compact"], align=WD_ALIGN_PARAGRAPH.JUSTIFY, before=0, after=0,
             line=1.5, first_line_chars=0, left_chars=2)

    set_fonts(S["Normal"], LATIN, SONG, 12)
    set_para(S["Normal"], line=1.5, before=0, after=0)

    # ── 标题 ──
    set_fonts(S["Heading 1"], LATIN, HEI, 16, bold=False)
    set_para(S["Heading 1"], align=WD_ALIGN_PARAGRAPH.CENTER,
             before=12, after=12, line=1.5, first_line_chars=0)
    set_fonts(S["Heading 2"], LATIN, HEI, 14, bold=False)
    set_para(S["Heading 2"], align=WD_ALIGN_PARAGRAPH.LEFT,
             before=10, after=6, line=1.5, first_line_chars=0)
    set_fonts(S["Heading 3"], LATIN, HEI, 12, bold=False)
    set_para(S["Heading 3"], align=WD_ALIGN_PARAGRAPH.LEFT,
             before=8, after=4, line=1.5, first_line_chars=0)
    set_fonts(S["Heading 4"], LATIN, SONG, 12, bold=True)
    set_para(S["Heading 4"], align=WD_ALIGN_PARAGRAPH.LEFT,
             before=6, after=4, line=1.5, first_line_chars=0)
    for n in ("Heading 5", "Heading 6"):
        set_fonts(S[n], LATIN, SONG, 12, bold=True)
        set_para(S[n], align=WD_ALIGN_PARAGRAPH.LEFT, before=4, after=2,
                 line=1.5, first_line_chars=0)

    # ── 图题 / 表题：五号居中，不加缩进 ──
    for n in ("Image Caption", "Table Caption", "Caption"):
        set_fonts(S[n], LATIN, SONG, 10.5)
        set_para(S[n], align=WD_ALIGN_PARAGRAPH.CENTER,
                 before=3, after=6, line=1.25, first_line_chars=0)

    # ── 引文块 ──
    set_fonts(S["Block Text"], LATIN, SONG, 10.5)
    set_para(S["Block Text"], align=WD_ALIGN_PARAGRAPH.JUSTIFY,
             line=1.25, first_line_chars=0, left_chars=2)

    # ── 表格内的文字用五号，否则表格会显得臃肿 ──
    try:
        for st in S:
            if st.name and "Table" in st.name and st.type is not None:
                set_fonts(st, LATIN, SONG, 10.5)
                set_para(st, line=1.0, before=1, after=1, first_line_chars=0)
    except Exception as e:
        print(f"  （表格样式调整跳过: {e}）")

    # ── 参考文献条目 ──
    if "Bibliography" in [s.name for s in S]:
        set_fonts(S["Bibliography"], LATIN, SONG, 10.5)
        set_para(S["Bibliography"], align=WD_ALIGN_PARAGRAPH.JUSTIFY,
                 line=1.25, before=0, after=3, first_line_chars=0, left_chars=2)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    d.save(OUT)
    print(f"已生成模板: {OUT}  ({os.path.getsize(OUT)} 字节)")


if __name__ == "__main__":
    build()
