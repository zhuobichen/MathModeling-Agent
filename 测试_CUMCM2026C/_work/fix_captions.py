# -*- coding: utf-8 -*-
"""检查 paper.md 里的表题注：按正文出现顺序应为「表 1」…「表 N」，一张不漏。

这个脚本前身是「按出现顺序给表重编题注」。现在 29 张表（正文 15 张 +
附录 D 14 张）的题注都由 c_tables.py 与自己写好，重编已经没有必要，
改成**校验**更合适——它能在编译 docx 前抓住两类问题：

  1. 某张 <table> 前面没有题注。归一化会给它补一条「表{已有最大号+1}: …」，
     自动推导的题注（如「表4: 符号、含义、单位」）读起来不像表名，
     而且编号是按「已有最大号」发的、不按位置，顺序会错乱。
  2. 题注编号与正文顺序不符（缺号、跳号、重号）。

注意归一化补题注用的是 `表\\s*(\\d+)[：:]`，**认不出「表 D-1」这类带字母的
编号**，会把它们当成没有题注的表另补一条。所以附录里的表号必须顺着正文
往下排（表 16…表 29），不能另起 D- 序号。
"""
from __future__ import annotations

import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.abspath(os.path.join(HERE, "..", "_交付", "paper.md"))

CAP_RE = re.compile(r"^\*\*表\s*(\d+)\s*[:：]")
HTML_RE = re.compile(r"^\s*<table", re.I)
SEP_RE = re.compile(r"^\s*\|[\s:\-|]+\|\s*$")


def is_md_table_start(lines, i):
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

    bad, seen, n = [], [], 0
    for i, line in enumerate(lines):
        if not (HTML_RE.match(line) or is_md_table_start(lines, i)):
            continue
        n += 1
        j = prev_nonblank(lines, i)
        m = CAP_RE.match(lines[j].strip()) if j >= 0 else None
        if not m:
            bad.append((i + 1, lines[j].strip()[:50] if j >= 0 else ""))
            continue
        seen.append(int(m.group(1)))

    print(f"共 {n} 张表；题注编号 = {seen}")
    ok = True
    if bad:
        ok = False
        print("**以下表缺题注**（归一化会自动编号，导致顺序错乱）：")
        for ln, prev in bad:
            print(f"  第 {ln} 行前一行是：{prev!r}")
    expect = list(range(1, n + 1))
    if seen != expect:
        ok = False
        print(f"**题注编号与正文顺序不符**：期望 {expect}")
    extra = re.findall(r"^## (D\.\d+ .*)$", "\n".join(lines), re.M)
    print(f"附录 D 小节：{extra}")
    print("题注校验：", "通过" if ok else "**未通过**")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
