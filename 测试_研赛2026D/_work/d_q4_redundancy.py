# -*- coding: utf-8 -*-
"""问题四：**冗余**的量化（论文 8.1.2 / 8.4.3 节）。

## 缺的是什么

题目问题四的原话是「比较**资源规模、冗余、均衡性与库存缺口**」。本文此前把
**规模**（各组用几架机）、**缺口**（需求 − 库存）、**均衡**（架次极差）都算了，
**但"冗余"一直没定义、也没算过**——只在文字里提过一句"冗余为 0"，那说的是
运输机需求恰好等于库存，是**总账**上的冗余，不是**分组内部**的冗余。

## 定义

下面这个定义是可算的，而且与"分组独立核算"的要求直接对应：

对第 $j$ 组，先把它的架次排到最优，得到**完成全部硬时限所需的最少机队** $\mathbf n_j^{\min}$
（就是 `min_fleet`）；而它实际分到的是 $\mathbf n_j^{\text{alloc}}$。于是

$$\\text{冗余}_j=\\frac{\\sum_g (n_{g,j}^{\\text{alloc}}-n_{g,j}^{\\min})}{\\sum_g n_{g,j}^{\\text{alloc}}}$$

即**分到的机队里，有多少是"用不上的"**。这个量回答的是"这组的资源配得紧不紧"，
与"缺口"（配少了）正好是一对：缺口为负、冗余为正。

**要把两件事分清楚**：
- 各组**自备峰值机队**时的冗余（`min_fleet` 给的就是本组自己顶住时限所需的架数）；
- 各组**共享库存**时的冗余（本文 8.1.1 的分配模型允许把库存按机型分给各组，
  但一架机不能同时服务两组，所以仍要逐组顶住时限）。

本文两者都算，因为题目的"冗余"没指明口径——**这本身也是一条要说清楚的口径**。
"""
from __future__ import annotations

import itertools
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import d_core as D
from d_q2 import World, TYPES, build_groups
import d_q3 as Q3
import d_q4 as Q4
import d_relay

HERE = os.path.dirname(os.path.abspath(__file__))
INV = Q4.INV

CASES = {
    "K=2 地理均衡型": [["S001","S002","S004","S005","S008","S009","S010","S011","S012","S013","S014"],
                     ["S003","S006","S007","S015"]],
    "K=2 资源缺口型（全局最优）": [
        ["S001","S004","S006","S010","S011","S012","S013","S014"],
        ["S002","S003","S005","S007","S008","S009","S015"]],
    "K=3 地理均衡型": [["S001","S002","S005","S009","S010","S011","S012","S013","S014"],
                     ["S003","S006","S007","S015"], ["S004","S008"]],
    "K=3 资源缺口型（全局最优）": [
        ["S001","S002","S003","S005","S007","S008","S009","S015"],
        ["S004","S006","S013"], ["S010","S011","S012","S014"]],
}


def main():
    W = World()
    rt, W.relays, W.rbats = D.load_relays()
    C = D.Comms(W.geo, W.nodes, D.load_comm())
    g1, p1, e1 = build_groups(W, allow_types=("A", "B"))
    q3 = json.load(open(os.path.join(HERE, "out_q3.json"), encoding="utf-8"))
    st = q3["stations"]
    T_SHIFT = float(q3.get("t_shift", 0.0))

    # 每条架次需要哪些站位（算中继冗余用）
    smask = {}
    for i, x in enumerate(p1):
        m = 0
        for (t0, t1, smp) in Q3.gaps_of(W, C, x):
            for j, s in enumerate(st):
                if d_relay.gap_covered(W, C, smp, tuple(s["point"])):
                    m |= 1 << j
                    break
        smask[i] = m

    print(f"运输 {len(p1)} 架次；δ={T_SHIFT:.0f} s\n")
    out = {}
    for name, groups in CASES.items():
        gid = {s: i for i, gp in enumerate(groups) for s in gp}
        idxs = [[k for k, x in enumerate(p1) if gid[x["stops"][0][0]] == i]
                for i in range(len(groups))]
        subs = [[p1[k]["stops"] for k in gg] for gg in idxs]
        alloc = Q4.best_allocation(W, subs, t_shift=T_SHIFT)
        print(f"{name}  分组 {[len(g) for g in groups]} 站，架次 "
              f"{[len(gg) for gg in idxs]}")
        rows = []
        for j, gg in enumerate(idxs):
            # **最小机队要与分配同口径**：`min_fleet` 只允许每架是一种机型、且逐架
            # 新增，所以它算出的"最少"在混合机型下并不最少（实测会出现"冗余 −33%"
            # 这种负值，那不是冗余为负，是拿错了尺子）。这里改成在同一套混合机型
            # 配置里搜最小总数：与分配同一目标（零违反），时间容差也一致。
            nalloc = alloc["per_group"][j]["cap"]
            tot_a = sum(nalloc.values())
            t_alloc = alloc["per_group"][j]["t"]
            best_m = None
            for a2 in range(INV["drone"]["A"] + 1):
                for b2 in range(INV["drone"]["B"] + 1):
                    for c2 in range(INV["drone"]["C"] + 1):
                        if a2 + b2 + c2 == 0 or a2 + b2 + c2 > tot_a:
                            continue
                        rr = Q4.schedule_fleet_capped(
                            W, subs[j], {"A": a2, "B": b2, "C": c2})
                        if rr is None or rr["viol"] > 0:
                            continue
                        if rr["t_done"] + T_SHIFT > t_alloc * 1.05:
                            continue
                        if best_m is None or a2 + b2 + c2 < best_m[0]:
                            best_m = (a2 + b2 + c2, {"A": a2, "B": b2, "C": c2})
            tot_m = best_m[0] if best_m else tot_a
            nfmin = best_m[1] if best_m else nalloc
            red = (tot_a - tot_m) / tot_a if tot_a else 0.0
            # 中继冗余：分到的站位数 − 本组真正需要的站位数
            m = 0
            for k in gg:
                m |= smask[k]
            need_st = bin(m).count("1")
            rows.append(dict(j=j + 1, n=len(gg), alloc=nalloc, minf=nfmin,
                             red=red, need_st=need_st))
            print(f"    组{j+1}：分到 { {g: nalloc[g] for g in TYPES if nalloc[g]} }"
                  f"，本组最少需 { {g: nfmin[g] for g in TYPES if nfmin[g]} }"
                  f"（{tot_m} 架） → **冗余 {red * 100:.0f}%**"
                  f"；中继需 {need_st} 架")
        red_all = (sum(sum(r["alloc"].values()) for r in rows)
                   - sum(sum(r["minf"].values()) for r in rows)) \
            / sum(sum(r["alloc"].values()) for r in rows)
        print(f"    分组总冗余：{red_all * 100:.0f}%"
              f"（分到 {sum(sum(r['alloc'].values()) for r in rows)} 架，"
              f"最少需 {sum(sum(r['minf'].values()) for r in rows)} 架）\n")
        out[name] = dict(groups=groups, rows=rows, red_all=red_all)

    with open(os.path.join(HERE, "out_q4_redundancy.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print("已写出 out_q4_redundancy.json")


if __name__ == "__main__":
    main()
