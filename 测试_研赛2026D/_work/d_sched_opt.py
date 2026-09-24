# -*- coding: utf-8 -*-
"""排程顺序的实验：固定组批与路线，只改「架次派发的先后顺序」。

为什么值得做：本文的完成时间由**最忙的那架**决定，而 schedule() 是按
（紧迫度, 载质量）派发的——紧迫度对全部软时限架次都取同一个值，于是软架次
之间的先后完全由"箱数/质量"这个与机器负载无关的量决定。经典并行机排程里
min-makespan 的列表规则是 **LPT（最长的任务先派）**，与本文的顺序不同。

本脚本复用 d_q2 的 World / initial_groups / merge_pass，把 schedule() 照抄
一份并外挂排序键，比较几种顺序。**不改任何物理口径**，所以结果可直接对比。

若某个顺序严格更优（完成时间更短且硬约束零违反），才回填进 d_q2.py。
"""
from __future__ import annotations

import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import d_core as D
from d_q2 import (World, TYPES, initial_groups, merge_pass, evaluate,
                  _boxes_of)


def schedule_ordered(W, stops_list, key_name):
    """schedule() 的副本，唯一的差别是 order 的排序键。"""
    # 先算每个架次在「最优机型」下的估计时长，供 LPT 使用
    dur, eest = [], []
    for st in stops_list:
        ts = W.ok_types(st)
        if not ts:
            raise RuntimeError(f"架次 {st} 无可用机型")
        cands = [(W.route(g, st)[0], W.route(g, st)[1], g) for g in ts]
        tmin, emin, _ = min(cands, key=lambda x: (x[0], x[1]))
        dur.append(tmin)
        eest.append(emin)
    hard = [any(b["first"] or b["kind"] == "医疗物资" for b in _boxes_of(st))
            for st in stops_list]

    def order_key(i):
        if key_name == "urg":                       # 本文原顺序
            return (W.urgency(stops_list[i]),
                    sum(b["mass"] for _, g in stops_list[i] for b in g))
        if key_name == "urg+lpt":                   # 同紧迫度内最长先派
            return (W.urgency(stops_list[i]), -dur[i])
        if key_name == "hardfirst+lpt":             # 硬时限先派，其余 LPT
            return (0 if hard[i] else 1, -dur[i])
        if key_name == "lpt":                       # 完全 LPT
            return (-dur[i],)
        if key_name == "urg+E":                     # 同紧迫度内能耗大的先派
            return (W.urgency(stops_list[i]), -eest[i])
        if key_name == "edd+lpt":    # 硬时限按 EDD，其余 LPT（经典"截止期感知"规则）
            return (0, min(b["t_due"] for b in _boxes_of(stops_list[i])), 0.0) \
                if hard[i] else (1, 0.0, -dur[i])
        if key_name == "edd+E":
            return (0, min(b["t_due"] for b in _boxes_of(stops_list[i])), 0.0) \
                if hard[i] else (1, 0.0, -eest[i])
        raise ValueError(key_name)

    order = sorted(range(len(stops_list)), key=order_key)

    nf = {g: sum(1 for d in W.drones if d["type"] == g) for g in TYPES}
    pool = {g: [dict(id=f"{g}{i+1:02d}", free=0.0) for i in range(W.bats[g]["n"])]
            for g in TYPES}
    inv = {g: [d["id"] for d in W.drones if d["type"] == g] for g in TYPES}
    drones = {}
    for g in TYPES:
        for i in range(nf.get(g, 0)):
            did = inv[g][i] if i < len(inv[g]) else f"{g}-EXT{i + 1:02d}"
            drones[did] = dict(id=did, type=g, free=0.0)

    plan = [None] * len(stops_list)
    for i in order:
        st = stops_list[i]
        due = min(b["t_due"] for b in _boxes_of(st))
        cand = []
        for g in W.ok_types(st):
            t, E = W.route(g, st)
            for did, d in drones.items():
                if d["type"] != g:
                    continue
                for bt in pool[g]:
                    cand.append((max(d["free"], bt["free"]), E, did, bt, g, t))
        if hard[i]:
            good = [c for c in cand if c[0] + c[5] <= due + 1e-9]
            if good:
                cand = good
        start, E, did, bt, g, t = min(cand, key=lambda x: (x[0], x[1]))
        end = start + t
        soc = max(1.0 - E / W.types[g]["E_use"], 0.0)
        chg = D.charge_time(soc, W.bats[g]["T_full"])
        deliv, node = {}, "O01"
        cur = (start + W.types[g]["t_prep"]
               + sum(len(b2) for _, g2 in st for b2 in g2) * W.types[g]["t_load"])
        q = sum(b2["mass"] for _, g2 in st for b2 in g2)
        for s, gp in st:
            cur += W.T.seg(g, node, s, q)["t"]
            cur += W.types[g]["t_hand0"] + len(gp) * W.types[g]["t_hand1"]
            for b2 in gp:
                deliv[b2["box"]] = cur
            q -= sum(b2["mass"] for b2 in gp)
            node = s
        plan[i] = dict(stops=st, g=g, drone=did, battery=bt["id"], start=start,
                       end=end, t=t, E=E, soc=soc, chg=chg,
                       boxes=[b2["box"] for _, gp in st for b2 in gp],
                       deliv=deliv)
        drones[did]["free"] = end
        bt["free"] = end + chg
    return plan


def main():
    W = World()
    g0 = initial_groups(W)
    for allow in (("A",), ("A", "B")):
        g = merge_pass(W, g0, allow_types=allow)
        print("=" * 78)
        print(f"合并档 {'+'.join(allow)}：{len(g)} 架次")
        print("=" * 78)
        print(f"{'顺序':<16}{'架次':>5}{'完成/s':>9}{'能耗/kWh':>10}"
              f"{'首批违约':>9}{'医疗违约':>9}{'机型':>22}")
        best = None
        for k in ("urg", "urg+lpt", "urg+E", "hardfirst+lpt", "lpt",
                  "edd+lpt", "edd+E"):
            p = schedule_ordered(W, g, k)
            e = evaluate(W, p)
            flag = ""
            if e["viol_first"] == 0 and e["viol_med"] == 0:
                if best is None or e["t_done"] < best[1]:
                    best = (k, e["t_done"], e["n_sorties"])
                    flag = "  ←"
            print(f"{k:<16}{e['n_sorties']:>5}{e['t_done']:>9.0f}{e['E']:>10.2f}"
                  f"{e['viol_first']:>9}{e['viol_med']:>9}"
                  f"{str(e['by_type']):>22}{flag}")
        if best:
            print(f"  → 该档下完成时间最短且硬约束零违反的顺序：{best[0]}"
                  f"（{best[2]} 架次 / {best[1]:.0f} s）")
        print()


if __name__ == "__main__":
    main()
