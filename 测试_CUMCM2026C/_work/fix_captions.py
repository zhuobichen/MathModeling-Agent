# -*- coding: utf-8 -*-
"""把 paper.md 里的表题注按「正文出现顺序」重编成表 1..表 15。

起因有两处：

1) `_normalize_table_captions` 只给**没有**题注的表补编号，编号从
   「已有题注的最大号 + 1」起算，它不是按位置发的。本文手写的三条题注
   被题目固定成「表 1/表 2/表 3」（购电量、充放电量、紧急购电量），于是
   顺序变成 表1、表2、表4（符号说明）、表5（汇总）、表3（紧急购电量）
   —— 表 3 排到了表 4、表 5 后面。

2) 归一化只认 HTML 表。2.1 节那张「层次/不确定性来源/决策结构/对应问题」
   是 markdown 竖线表，pandoc 会正常渲染成 Word 表，但它不在
   `<table>` 计数里，于是整篇唯一一张没有题注的表就是它。

这里改为：全部 15 张表都由本文自己按顺序写好题注，归一化阶段就无事可做；
题目要求的表 1/2/3 在题注里注明对应关系，避免读者把论文的顺位编号
当成题目里的表号。
"""
from __future__ import annotations

import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.abspath(os.path.join(HERE, "..", "_交付", "paper.md"))

CAPTIONS = [
    "表 1: 四个问题的层次划分",
    "表 2: 主要符号及其含义",
    "表 3: 微网在指定时间段的购电量及全天的购电量和购电费"
    "（问题 1，即题目要求的表 1）",
    "表 4: 储能设备在指定时间段的充放电量及 0:00 和 24:00 的储电量"
    "（问题 1，即题目要求的表 2）",
    "表 5: 问题 2 的全年汇总指标",
    "表 6: 微网在指定日期的紧急购电量（问题 2，即题目要求的表 3）",
    "表 7: 问题 1 与问题 2 的关键指标对比",
    "表 8: 光伏预报误差的统计特征（按发布时刻与预报步长）",
    "表 9: 问题 3 中三种处理方式的费用对比",
    "表 10: 问题 4 四种情形的费用对比",
    "表 11: LP 求解器的双算法交叉验证",
    "表 12: 全部约束的回代核验结果",
    "表 13: 松弛检验：逐步放开约束后的目标值",
    "表 14: 由结果文件独立反算的核验结果",
    "表 15: 支撑材料文件列表",
]

CAP_RE = re.compile(r"^\*\*表\s*\d+\s*[:：]")
HTML_RE = re.compile(r"^\s*<table", re.I)
SEP_RE = re.compile(r"^\s*\|[\s:\-|]+\|\s*$")


def is_md_table_start(lines, i):
    """竖线表的第一行：本行以 | 开头，且下一行是 |---| 分隔行。"""
    return (re.match(r"^\s*\|", lines[i]) and i + 1 < len(lines)
            and SEP_RE.match(lines[i + 1]))


def prev_nonblank(lines, i):
    j = i - 1
    while j >= 0 and not lines[j].strip():
        j -= 1
    return j


def main():
    with open(PAPER, encoding="utf-8") as f:
        lines = f.read().split("\n")

    out = []
    n = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        starts = HTML_RE.match(line) or is_md_table_start(lines, i)
        if starts:
            cap = CAPTIONS[n]
            n += 1
            j = prev_nonblank(out, len(out))
            if j >= 0 and CAP_RE.match(out[j].strip()):
                print(f"  替换 {out[j].strip()[:30]} → {cap}")
                out[j] = f"**{cap}**"
            else:
                print(f"  新增 {cap}")
                if out and out[-1].strip():
                    out.append("")
                out.append(f"**{cap}**")
                out.append("")
        out.append(line)
        i += 1

    if n != len(CAPTIONS):
        raise SystemExit(f"表数不符：文中 {n} 张，题注写了 {len(CAPTIONS)} 条")

    text = "\n".join(out)

    # 正文里"表 1 中 10:00、14:00、20:00 三个时段"指的是问题 1 的购电量表，
    # 重编号后它是表 3。其余"按表 1、表 2 的格式"是复述题目，保持原样。
    old = "**（3）购电集中在高价时段。** 表 1 中 10:00、14:00、20:00"
    new = "**（3）购电集中在高价时段。** 表 3 中 10:00、14:00、20:00"
    if old not in text:
        raise SystemExit("找不到「购电集中在高价时段」那句，正文引用未改")
    text = text.replace(old, new)

    # 附录里提到的是生成脚本针对题目表 1/2/3，明确写清
    text = text.replace("论文表 1/表 2/表 3 的生成", "题目表 1/表 2/表 3 的生成")
    text = text.replace("生成论文表 1/表 2/表 3", "生成题目表 1/表 2/表 3")

    with open(PAPER, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"共 {n} 张表，已写回 {PAPER}")


if __name__ == "__main__":
    main()
