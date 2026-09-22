# -*- coding: utf-8 -*-
"""按重算后的结果同步论文里的数值（幂等）。

起因：给结果文件的紧急购电量加了 1e-3 kWh/步 的阈值（滤掉线代求解器在
「计划刚好够用」时留下的残差），问题 3 / 4-3 的紧急购电费各降了约 5 元：
  14961323.00 → 14961318.24 元
  15626113.24 → 15626108.38 元
  1742418.11  → 1742413.35 元
  1758707.21  → 1758702.35 元
  405165.5 / 405166 → 405164.6 / 405165 kWh
"""
from __future__ import annotations

import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.abspath(os.path.join(HERE, "..", "_交付", "paper.md"))

PAIRS = [
    ("14961323.00", "14961318.24"),
    ("15626113.24", "15626108.38"),
    ("1742418.11", "1742413.35"),
    ("1758707.21", "1758702.35"),
    ("405165.5", "405164.6"),
    ("<td>405166</td>", "<td>405165</td>"),
    ("405166 kWh", "405165 kWh"),
]


def main():
    with open(PAPER, encoding="utf-8") as f:
        s = f.read()
    n = 0
    for a, b in PAIRS:
        c = s.count(a)
        if c:
            s = s.replace(a, b)
            n += c
            print(f"  {a} → {b}  ×{c}")
    with open(PAPER, "w", encoding="utf-8", newline="\n") as f:
        f.write(s)
    print(f"共替换 {n} 处")


if __name__ == "__main__":
    main()
