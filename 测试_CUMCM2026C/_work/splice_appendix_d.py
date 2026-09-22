# -*- coding: utf-8 -*-
"""把 c_tables.py 生成的 14 张表拼成论文的附录 D（幂等：重复运行只保留一份）。"""
from __future__ import annotations

import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.abspath(os.path.join(HERE, "..", "_交付", "paper.md"))
SRC = os.path.join(HERE, "appendix_tables.md")
MARK = "\n# 附录 D 指定日期的详细结果"

INTRO = """
# 附录 D 指定日期的详细结果

题面要求：问题 1 需给出指定时间段的购电量与充放电量，问题 2、问题 3 需给出
**2025-03-20、06-21、09-23、12-21**这四天的结果，问题 4 需给出重算结果。
问题 1 的表 1 / 表 2 格式结果已在正文（表 3、表 4）给出；问题 2–4 的四天逐项结果
按题面**表 1 / 表 2 / 表 3 的格式**集中在本附录，正文表 5–表 10 只给汇总指标。
四天的完整逐时段数据见对应的 result*.xlsx。

为便于排版，附录里的表以**日期为列**（题面原表以时段为列、日期并排），
字段一项不少；表 2 的「充电量」与「放电量」拆成两张表，避免 9 列在 A4 纵向
里挤到表头读不出来。

"""


def main():
    with open(PAPER, encoding="utf-8") as f:
        s = f.read()
    if MARK in s:                       # 幂等：先切掉旧的一份
        s = s.split(MARK)[0].rstrip() + "\n"
    with open(SRC, encoding="utf-8") as f:
        body = f.read()
    s = s.rstrip() + "\n" + INTRO + body.strip() + "\n"
    with open(PAPER, "w", encoding="utf-8", newline="\n") as f:
        f.write(s)
    print(f"已拼入附录 D：{len(body)} 字符，论文共 {len(s)} 字符")


if __name__ == "__main__":
    main()
