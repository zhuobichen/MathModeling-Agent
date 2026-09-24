# -*- coding: utf-8 -*-
"""问题二：**机型分配的重平衡**（论文 6.3.5 节）。

## 为什么

排程已经压到它自己的下界上了：8188 s 对 8187 s。但这个下界本身是

$$T\\ \\ge\\ \\max_g\\ \\frac{1}{n_g}\\sum_{r:\\ g_r=g} t_r(g_r)$$

**它取决于机型分配 $g_r$**——而机型分配是决策。现在三型的负载是
A 29104/4、B 16373/2、C 15861/2，即 7276 / **8187** / 7930：**B 型顶着**。
把 B 上的一部分架次挪给 A 或 C，下界就跟着降。理论上三型均分能到
总工作量/8。

## 做法

在**机型分配**上做局部搜索：目标直接取上面那个下界（它便宜、可 O(1) 更新），
把每条架次在可行机型之间挪，取最好改进；搜完再用顺序局部搜索把真实完成时间
压到新下界附近。最后校验真实完成时间与硬约束。
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

import d_core as D
from d_q2 import (World, TYPES, build_groups, evaluate, optimise_order,
                  simulate_order, pack_min_sorties, repair_grouping)


def main():
    W = World()
    g1, p1, e1 = build_groups(W, allow_types=("A", "B"))
    nf = {g: sum(1 for d in W.drones if d["type"] == g) for g in TYPES}

    # 每条架次的可行机型与时长
    feas = [W.ok_types(st) for st in g1]
    dur = [{g: W.route(g, st)[0] for g in W.ok_types(st)} for st in g1]
    # **注意**：build_groups 返回的 plan 是按派发顺序排的，与 g1 的下标不对应。
    # 必须用 stops 对象本身去建立映射（simulate_order 存的正是 stops_list 里的元素）。
    idx = {id(x["stops"]): k for k, x in enumerate(p1)}
    cur = [None] * len(g1)
    for i, st in enumerate(g1):
        cur[i] = p1[idx[id(st)]]["g"]

    def bound(asg):
        w = {g: 0.0 for g in TYPES}
        for i, g in enumerate(asg):
            w[g] += dur[i][g]
        return max(w[g] / nf[g] for g in TYPES), w

    b0, w0 = bound(cur)
    b_start = b0
    print(f"起点（排程器给的分配）：下界 {b0:.0f} s；各型工作量 "
          f"{ {g: round(w0[g]) for g in TYPES} }，机队 {nf}")

    it = 0
    while True:
        it += 1
        best = None
        for i in range(len(cur)):
            for g in feas[i]:
                if g == cur[i]:
                    continue
                trial = list(cur)
                trial[i] = g
                b, w = bound(trial)
                if b < b0 - 1e-9 and (best is None or b < best[0]):
                    best = (b, trial, w)
        if best is None:
            break
        b0, cur, w0 = best
        print(f"  第 {it} 轮：改一条架次的机型 → 下界 {b0:.0f} s"
              f"（{ {g: round(w0[g]) for g in TYPES} }）")
        if it >= 200:
            break

    print()
    print("=" * 78)
    print(f"机型分配重平衡：下界 {b_start:.0f} → {b0:.0f} s")
    print("=" * 78)
    print(f"  各型架次数 { {g: cur.count(g) for g in TYPES} }")
    print(f"  （锁定该分配后重跑顺序搜索）")
    print(f"  各型工作量 { {g: round(w0[g]) for g in TYPES} }（机队 {nf}）")

    # 用这个分配去排：把不合适的架次锁死机型需要改写 simulate_order，
    # 这里先看"顺序搜索能否把真实完成时间压到新下界附近"——用现有的顺序搜索。
    lock = {id(g1[i]): cur[i] for i in range(len(g1))}
    p2 = optimise_order(W, g1, lock=lock)
    e2 = evaluate(W, p2)
    print(f"  顺序搜索后的实际排程：完成 {e2['t_done']:.0f} s / "
          f"能耗 {e2['E']:.2f} kWh / 违约 {e2['viol_first']},{e2['viol_med']}")
    print(f"  机型分布（排程器自选）{e2['by_type']}")
    print(f"  → 重平衡给出的下界是 {b0:.0f} s；能不能达到，取决于排程器"
          f"是否真按这个分配去派")

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "out_q2_type_ls.json"), "w", encoding="utf-8") as f:
        json.dump(dict(bound_start=b_start,
                       bound_best=b0, asg=cur,
                       work={g: w0[g] for g in TYPES},
                       t_real=e2["t_done"], E=e2["E"],
                       by_type=e2["by_type"]), f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
