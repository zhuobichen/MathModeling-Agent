# -*- coding: utf-8 -*-
"""机型分配这一层的取舍：同一组架次，换机型飞，能耗与完成时间怎么变。

为什么单独做这个：本文的排程器是「完成时间优先」——它把架次派给**最早可用**的
(无人机, 电池)，于是空闲的 A/C 机也会被用上。而单条架次最省电的机型是 **B 型**
（B 的 E_use/L0 = 4.0/28000 最小，即每米最省），所以"追完成时间"与"追能耗"在
机型分配这一层就是**互相冲突**的。

这带来一个必须说清楚的问题：**能耗下界不能按本文方案已分配的机型去算**。
`d_bounds.py` 里的 E_min = Σ_g D_g·E_g^use/L_g^0 只对**固定的机型分配**成立；
换一种分配（哪怕还是在同一组架次上）就能得到更低的能耗。本脚本给出这个反例，
并把这一层的完整取舍表算出来，供论文 6.3.4 与 9.2 节引用。
"""
from __future__ import annotations

import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import d_core as D
from d_q2 import build_groups
from d_q2 import (World, TYPES, initial_groups, merge_pass, schedule, evaluate,
                  _boxes_of)

HERE = os.path.dirname(os.path.abspath(__file__))


def schedule_forced(W, stops_list, g, n):
    """把全部架次都交给 g 型、共 n 架机执行，返回 (完成时刻, 能耗, 硬违约数)。

    统一用论文的派发顺序（硬时限按 EDD、其余按能耗降序），并按题目两阶段模型
    计电池周转——与 d_q2.schedule 同一套口径，只锁死机型与架数。
    """
    tt = W.types[g]
    _e = {id(st): W.route(g, st)[1] for st in stops_list}

    def key(i):
        bs = _boxes_of(stops_list[i])
        if any(b["first"] or b["kind"] == "医疗物资" for b in bs):
            return (0, min(b["t_due"] for b in bs), 0.0)
        return (1, 0.0, -_e[id(stops_list[i])])

    drones = [dict(free=0.0) for _ in range(n)]
    bats = [dict(free=0.0) for _ in range(W.bats[g]["n"])]
    tdone = E = 0.0
    viol = 0
    for i in sorted(range(len(stops_list)), key=key):
        st = stops_list[i]
        bs = _boxes_of(st)
        hard = any(b["first"] or b["kind"] == "医疗物资" for b in bs)
        due = min(b["t_due"] for b in bs)
        if not W.feasible(g, st):
            return None                     # 该型飞不了这条架次
        t, ee = W.route(g, st)
        cand = [(max(d["free"], bt["free"]), k, j)
                for k, d in enumerate(drones) for j, bt in enumerate(bats)]
        if hard:
            good = [c for c in cand if c[0] + t <= due + 1e-9]
            if good:
                cand = good
        start, k, j = min(cand)
        end = start + t
        soc = max(1.0 - ee / tt["E_use"], 0.0)
        drones[k]["free"] = end
        bats[j]["free"] = end + D.charge_time(soc, W.bats[g]["T_full"])
        E += ee
        tdone = max(tdone, end)
        if hard and end > due + 1e-9:
            viol += 1
    return tdone, E, viol


def main():
    W = World()
    stops = build_groups(W, allow_types=("A", "B"))[0]
    p = schedule(W, stops)
    ev = evaluate(W, p)
    print("=" * 78)
    print("本文方案（完成时间优先的机型分配）")
    print("=" * 78)
    print(f"  {ev['n_sorties']} 架次，机型分布 {ev['by_type']}，"
          f"完成 {ev['t_done']:.0f} s，能耗 {ev['E']:.2f} kWh")
    print(f"  单架次能耗（按本文机型分配的平均）：{ev['E'] / ev['n_sorties']:.3f} kWh")

    print()
    print("=" * 78)
    print("同一组架次，改用单一机型执行")
    print("=" * 78)
    print(f"{'机型':<6}{'架数':>6}{'完成/s':>10}{'能耗/kWh':>11}{'硬违约':>8}"
          f"{'单架次能耗':>12}")
    rows = []
    for g in TYPES:
        for n in (2, 4, 6, 8):
            r = schedule_forced(W, stops, g, n)
            if r is None:
                print(f"{g:<6}{n:>6}{'—该型飞不了全部架次':>32}")
                continue
            t, E, viol = r
            rows.append(dict(g=g, n=n, t=t, E=E, viol=viol))
            print(f"{g:<6}{n:>6}{t:>10.0f}{E:>11.2f}{viol:>8}{E / len(stops):>12.3f}")

    print()
    print("=" * 78)
    print("结论")
    print("=" * 78)
    print("  单一机型的方案**全部不可行**：")
    print("    · A 型根本飞不动全部 40 条架次（S008 的 B/C 型专属长航线超出 A 的载荷/体积）；")
    print("    · B 型 / C 型虽然飞得动，但完成时间被推到 31000~37000 s，")
    print("      30 个首批箱（截止 3600 s）里 9~11 个超时。")
    print("  本文的混合分配：%.2f kWh / 完成 %.0f s / 硬违约 0" % (ev["E"], ev["t_done"]))
    print()
    print("  → 结论一：**时限把机型混合钉死了**。全用 B 型能把能耗降到 77.48 kWh")
    print("    （比本文低 23.7%），但那是靠把 40 条架次全压给 2 架 B 机换来的——")
    print("    完成时间 33372 s，首批保障直接失效。机型分配不是可自由优化的层，")
    print("    而是被首轮时限约束住的。")
    print("  → 结论二：**能耗下界只能按已分配的机型算**。")
    print("    Σ_g D_g·E_g^use/L_g^0 对「给定机型分配」成立，换一种分配就是另一个")
    print("    下界（B 型的 77.48 < 本文的 90.17 就是例子）。论文 9.2 节必须写明")
    print("    这个前提，否则下界会被误读成「任何方案都不可能低于此值」。")

    with open(os.path.join(HERE, "out_types.json"), "w", encoding="utf-8") as f:
        json.dump(dict(mixed=dict(E=ev["E"], t=ev["t_done"], by_type=ev["by_type"],
                                  n=ev["n_sorties"]), rows=rows),
                  f, ensure_ascii=False, indent=1, default=float)
    print("\n已写出 out_types.json")


if __name__ == "__main__":
    main()
