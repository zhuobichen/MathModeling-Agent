# -*- coding: utf-8 -*-
"""论文数值一致性审计：把论文里声称的数，逐个对回**生成它们的 JSON / 日志**。

为什么需要：这份论文的数值经过七轮改动，每一轮都是用脚本做批量替换。批量替换
最容易留下的不是"错数"，而是**改了一半**——比如正文改了、表里没改，或者某个
数在两处出现、只更新了一处。这类不一致不会被编译发现，但评委一眼能看出。

本脚本只做一件事：**列出论文里出现的每个关键数，并把它与该数的权威来源比对**。
权威来源是 `out_*.json` 与 `logs_*.txt`（都是本轮实际跑出来的）。
"""
from __future__ import annotations

import io
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(HERE, "..", "_交付", "paper.md")


def load(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return json.load(f)


def main():
    s = io.open(PAPER, encoding="utf-8").read()
    facts = []

    q2 = load("out_q2.json")["summary"]
    q3 = load("out_q3.json")
    bd = load("out_bounds.json")
    k2 = load("out_q4_k2opt.json")
    k3 = load("out_q4_k3opt.json")
    sm = load("out_summary.json")

    # (论文里该怎么写, 权威值, 说明)
    checks = [
        ("问题二架次数", "36", str(q2["n_sorties"])),
        ("问题二完成时间", "8188", f"{q2['t_done']:.0f}"),
        ("问题二能耗", "90.58", f"{q2['E']:.2f}"),
        ("问题二首批违约", "0", str(q2["viol_first"])),
        ("问题二医疗违约", "0", str(q2["viol_med"])),
        ("问题三缺口数", "51", str(q3["n_gaps"])),
        ("问题三站位数", "3", str(len(q3["stations"]))),
        ("问题三中继能耗", "7.280", f"{q3['relay_E']:.3f}"),
        ("问题三所需中继", "3", str(q3["n_relay_min"])),
        ("问题三运输推迟", "35", f"{q3['t_shift']:.0f}"),
        ("能耗下界", "78.54", f"{bd['E_min']:.2f}"),
        ("完成时间下界", "8187", f"{bd['T_lb_fleet']:.0f}"),
        ("机队利用率", "93.6", f"{bd['util']['U01'] * 0 + sum(bd['util'].values()) / len(bd['util']) * 100:.1f}"),
        ("K=2 最优完成", "10881", f"{k2['best_score'][5]:.0f}"),
        ("K=2 最优极差", "2", str(k2["best_score"][4])),
        ("K=2 分法数", "16383", str(k2["n_partitions"])),
        ("K=3 最优完成", "11829", f"{k3['best_score'][5]:.0f}"),
        ("K=3 分法数", "2375101", str(k3["n_partitions"])),
        ("Q1 架次数", "18", str(sm["q1"]["n"])),
    ]
    print("=" * 78)
    print("论文数值一致性审计")
    print("=" * 78)
    print(f"{'项':<20}{'论文里应出现':>14}{'权威值':>14}   结论")
    bad = 0
    for name, want, got in checks:
        ok = (want == got)
        # 论文里是否真的出现这个数
        present = want in s
        tag = "一致" if ok else "**不一致**"
        if not present:
            tag += "（论文里找不到）"
        if not ok or not present:
            bad += 1
        print(f"{name:<20}{want:>14}{got:>14}   {tag}")
    print()
    print(f"不一致或缺失：{bad} 项")

    # 反向扫：论文里不该再出现的陈旧数值
    stale = ["107.41", "101.55", "10064", "10973", "10847", "12384",
             "6.515", "5.184", "7.976", "8660", "7.324", "9153", "64 个缺口", "29.8%", "30.6%",
             "2.4 亿", "9732", "90.17", "80.39", "92.16", "12882"]
    print()
    print("陈旧数值扫描（论文里若出现即为残留）：")
    hit = 0
    for t in stale:
        for m in re.finditer(re.escape(t), s):
            line = s[:m.start()].count("\n") + 1
            ctx = s[max(0, m.start() - 40):m.end() + 30].replace("\n", " ")
            print(f"  L{line}: …{ctx}…")
            hit += 1
    print(f"  命中 {hit} 处")
    return bad, hit


if __name__ == "__main__":
    b, h = main()
    print()
    print("审计结束：", "全部一致" if (b == 0 and h == 0) else "见上面标记")
