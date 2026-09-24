# -*- coding: utf-8 -*-
"""问题二：排程顺序的**局部搜索**（论文 6.3.2 节的补充）。

## 为什么

本文的排程器是"按某个顺序逐个派发，每个派给最早可用的（同型无人机, 已换好的
电池）"。所以**结果只取决于派发顺序**——顺序定了，时间表就定了。

前面只试过 7 个**固定规则**（紧迫度、LPT、EDD……），最好的是 9153 s，而
给定架次集合的完成时间下界是 8660 s（差 5.7%）。这 5.7% 的差距是不是规则
选得不好？没人试过**在顺序空间里搜索**。

## 做法

对 36 个架次的全排列做局部搜索：

- 邻域 = 交换顺序中的任意两个架次（36×35/2 = 630 个邻居）；
- 每步重跑一次排程模拟，做**最好改进**（best-improvement）；
- 接受条件：完成时间变短 **且**硬约束仍零违反；
- 直到没有改进为止。

一次模拟约 36 次派发，630 个邻居一轮几秒钟，可以跑很多轮。
"""
from __future__ import annotations

import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import d_core as D
from d_q2 import (World, TYPES, build_groups, evaluate, _boxes_of)

HERE = os.path.dirname(os.path.abspath(__file__))


def simulate(W, stops_list, order, delta=0.0):
    """按给定顺序派发，返回 (完成时刻, 计划)。口径与 d_q2.schedule 一致。"""
    nf = {g: sum(1 for d in W.drones if d["type"] == g) for g in TYPES}
    pool = {g: [dict(id=f"{g}{i+1:02d}", free=0.0)
                for i in range(W.bats[g]["n"])] for g in TYPES}
    inv = {g: [d["id"] for d in W.drones if d["type"] == g] for g in TYPES}
    drones = {}
    for g in TYPES:
        for i in range(nf.get(g, 0)):
            did = inv[g][i] if i < len(inv[g]) else f"{g}-EXT{i+1:02d}"
            drones[did] = dict(id=did, type=g, free=0.0)

    plan, viol = [], 0
    for i in order:
        st = stops_list[i]
        bs = _boxes_of(st)
        hard = any(b["first"] or b["kind"] == "医疗物资" for b in bs)
        due = min(b["t_due"] for b in bs)
        cand = []
        for g in W.ok_types(st):
            t, E = W.route(g, st)
            for did, d in drones.items():
                if d["type"] != g:
                    continue
                for bt in pool[g]:
                    cand.append((max(d["free"], bt["free"]), E, did, bt, g, t))
        if not cand:
            return None
        if hard:
            good = [c for c in cand if c[0] + c[5] + delta <= due + 1e-9]
            if good:
                cand = good
        start, E, did, bt, g, t = min(cand, key=lambda x: (x[0], x[1]))
        end = start + t
        soc = max(1.0 - E / W.types[g]["E_use"], 0.0)
        chg = D.charge_time(soc, W.bats[g]["T_full"])
        deliv, node = {}, "O01"
        cur = (start + W.types[g]["t_prep"]
               + len(bs) * W.types[g]["t_load"])
        q = sum(b2["mass"] for b2 in bs)
        for s, gp in st:
            cur += W.T.seg(g, node, s, q)["t"]
            cur += W.types[g]["t_hand0"] + len(gp) * W.types[g]["t_hand1"]
            for b2 in gp:
                deliv[b2["box"]] = cur
            q -= sum(b2["mass"] for b2 in gp)
            node = s
        plan.append(dict(stops=st, g=g, drone=did, battery=bt["id"],
                         start=start, end=end, t=t, E=E, soc=soc, chg=chg,
                         boxes=[b2["box"] for _, gp in st for b2 in gp],
                         deliv=deliv))
        drones[did]["free"] = end
        bt["free"] = end + chg
        if hard and end > due + 1e-9:
            viol += 1
    if viol:
        return None
    return max(p["end"] for p in plan), plan


def base_order(W, stops_list):
    """本文正式采用的顺序：硬时限按 EDD，其余按估计能耗降序。"""
    _e = {id(st): min(W.route(g, st)[1] for g in W.ok_types(st))
          for st in stops_list}

    def key(i):
        bs = _boxes_of(stops_list[i])
        if any(b["first"] or b["kind"] == "医疗物资" for b in bs):
            return (0, min(b["t_due"] for b in bs), 0.0)
        return (1, 0.0, -_e[id(stops_list[i])])
    return sorted(range(len(stops_list)), key=key)


def main():
    W = World()
    g1, p1, e1 = build_groups(W, allow_types=("A", "B"))
    order = base_order(W, g1)
    t0, _ = simulate(W, g1, order)
    print(f"起点（EDD + 能耗降序）：{len(g1)} 架次 / 完成 {t0:.0f} s / "
          f"能耗 {e1['E']:.2f} kWh")
    print(f"给定架次集合的完成时间下界 = 8660 s（见 9.2 节）\n")

    cur, cur_t = list(order), t0
    it = 0
    while True:
        it += 1
        best = None
        n = len(cur)
        for a in range(n):
            for b in range(a + 1, n):
                trial = list(cur)
                trial[a], trial[b] = trial[b], trial[a]
                r = simulate(W, g1, trial)
                if r is None:
                    continue
                t, _ = r
                if t < cur_t - 1e-9 and (best is None or t < best[0]):
                    best = (t, trial)
        if best is None:
            break
        cur_t, cur = best
        print(f"  第 {it} 轮：交换后 {cur_t:.0f} s（−{t0 - cur_t:.0f} s）")
        if it >= 40:
            print("  （达 40 轮上限，停）")
            break
    print()
    print("=" * 78)
    print(f"局部搜索结束：{t0:.0f} s → {cur_t:.0f} s（改善 {t0 - cur_t:.0f} s，"
          f"{(t0 - cur_t) / t0 * 100:.2f}%）")
    print(f"距下界 8660 s 还差 {cur_t - 8660:.0f} s（{cur_t / 8660 - 1:.1%}）")
    print("=" * 78)
    _, plan = simulate(W, g1, cur)
    ev = evaluate(W, plan)
    print(f"  校验：首批违约 {ev['viol_first']}，医疗违约 {ev['viol_med']}，"
          f"能耗 {ev['E']:.2f} kWh")
    with open(os.path.join(HERE, "out_q2_sched_ls.json"), "w",
              encoding="utf-8") as f:
        json.dump(dict(t0=t0, t_best=cur_t, order=cur, improv=t0 - cur_t,
                       E=ev["E"], viol_first=ev["viol_first"],
                       viol_med=ev["viol_med"]), f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
