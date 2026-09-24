# -*- coding: utf-8 -*-
"""问题二的三维敏感性：组批容量、派发顺序、机队规模。

三者都是"先扫再定"，不是先验选择。三张表共用同一套口径（同一物理内核、
同一组服务区数据、同一排程器），唯一的差别是被扫的那个参数。

机队利用率 = 全队占用时长之和 /（机队规模 × 完成时刻），用来回答
"完成时间到底是被机队顶上去的，还是被流程顶上去的"。
"""
from __future__ import annotations

import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from d_q2 import (build_groups, World, TYPES, merge_pass, schedule, evaluate)
from d_sched_opt import schedule_ordered

HERE = os.path.dirname(os.path.abspath(__file__))

CAPS = [("A 型容量", 25.0, 0.060),
        ("B 型容量（正式）", 30.0, 0.073),
        ("C 型容量", 80.0, 0.250)]

FLEETS = [("库存", {"A": 4, "B": 2, "C": 2}),
          ("C 加 2 架", {"A": 4, "B": 2, "C": 4}),
          ("B、C 各加 2 架", {"A": 4, "B": 4, "C": 4}),
          ("A 加 2、B 加 2", {"A": 6, "B": 4, "C": 4}),
          ("A 加到 8", {"A": 8, "B": 4, "C": 4})]

ORDERS = [("(紧迫度, 载质量)——原设计", "urg"),
          ("(紧迫度, 时长降序)", "urg+lpt"),
          ("(紧迫度, 能耗降序)", "urg+E"),
          ("(硬时限最长先派, 其余 LPT)", "hardfirst+lpt"),
          ("纯 LPT", "lpt"),
          ("(硬时限 EDD, 其余 LPT)", "edd+lpt"),
          ("(硬时限 EDD, 其余能耗降序)——正式", "edd+E")]


def util(plan, ev, nf):
    busy = sum(p["t"] for p in plan)
    nd = sum(nf.values())
    return busy / (nd * ev["t_done"])


def stock(W, fleet=None):
    return fleet or {g: sum(1 for d in W.drones if d["type"] == g) for g in TYPES}


def main():
    out = {}

    # ── 容量扫描 ────────────────────────────────────────────
    print("=" * 78)
    print("一、组批容量扫描（机队取库存，合并档统一用 A/B 可行的，即各容量下的最优合并）")
    print("=" * 78)
    print(f"{'容量档':<20}{'kg':>6}{'m³':>8}{'架次':>6}{'完成/s':>9}"
          f"{'能耗/kWh':>10}{'利用率':>8}{'首批违约':>9}{'医疗违约':>9}")
    rows = []
    for name, cm, cv in CAPS:
        W = World()
        W.cap_m, W.cap_v = cm, cv
        g = build_groups(W, allow_types=("A", "B"))[0]
        p = schedule(W, g)
        e = evaluate(W, p)
        u = util(p, e, stock(W))
        rows.append(dict(name=name, cm=cm, cv=cv, n=e["n_sorties"],
                         t=e["t_done"], E=e["E"], util=u,
                         vf=e["viol_first"], vm=e["viol_med"]))
        print(f"{name:<20}{cm:>6.0f}{cv:>8.3f}{e['n_sorties']:>6}"
              f"{e['t_done']:>9.0f}{e['E']:>10.2f}{u * 100:>7.1f}%"
              f"{e['viol_first']:>9}{e['viol_med']:>9}")
    out["cap"] = rows

    # ── 派发顺序扫描 ────────────────────────────────────────
    print()
    print("=" * 78)
    print("二、派发顺序扫描（容量 30 kg，合并档 A）")
    print("=" * 78)
    print(f"{'顺序':<32}{'架次':>6}{'完成/s':>9}{'能耗/kWh':>10}"
          f"{'首批违约':>9}{'医疗违约':>9}")
    W = World()
    g1 = build_groups(W, allow_types=("A", "B"))[0]
    rows = []
    for name, key in ORDERS:
        p = schedule_ordered(W, g1, key)
        e = evaluate(W, p)
        rows.append(dict(name=name, key=key, n=e["n_sorties"], t=e["t_done"],
                         E=e["E"], vf=e["viol_first"], vm=e["viol_med"]))
        print(f"{name:<32}{e['n_sorties']:>6}{e['t_done']:>9.0f}{e['E']:>10.2f}"
              f"{e['viol_first']:>9}{e['viol_med']:>9}")
    out["order"] = rows

    # ── 机队规模扫描 ────────────────────────────────────────
    print()
    print("=" * 78)
    print("三、机队规模扫描（容量 30 kg，正式派发顺序）")
    print("=" * 78)
    print(f"{'机队':<20}{'架次':>6}{'完成/s':>9}{'能耗/kWh':>10}{'利用率':>8}")
    rows = []
    base = None
    for name, fl in FLEETS:
        W = World()
        g = build_groups(W, allow_types=("A", "B"))[0]
        p = schedule(W, g, fleet=fl)
        e = evaluate(W, p)
        u = util(p, e, fl)
        if base is None:
            base = e["t_done"]
        rows.append(dict(name=name, fleet=fl, n=e["n_sorties"], t=e["t_done"],
                         E=e["E"], util=u, rel=e["t_done"] / base - 1))
        print(f"{name:<20}{e['n_sorties']:>6}{e['t_done']:>9.0f}{e['E']:>10.2f}"
              f"{u * 100:>7.1f}%")
    out["fleet"] = rows

    with open(os.path.join(HERE, "out_sens.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print("\n已写出 out_sens.json")


if __name__ == "__main__":
    main()
