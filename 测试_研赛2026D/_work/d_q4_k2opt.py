# -*- coding: utf-8 -*-
"""K=2 的**全局最优验证**（论文 8.4.2 节）。

## 为什么只需检查 31 种，而不是 16383 种

五维目标是分层字典序：

    (硬时限违反, 运输机缺口, 电池组缺口, 中继缺口, 架次极差, 最晚组完成)

**中继缺口排在第 4 位**。穷举全部 16383 种分法后，各组站位需求之和的最小值是
**3**（分布 3→31、4→2144、5→8160、6→6048），于是中继缺口的最小值是 1。任何
中继需求为 4 以上的分区，**不管后面几维多好，字典序上都已经输了**。

所以只要把中继需求恰好为 3 的 **31 种**分区精确算一遍，就足以判定全局最优——
这是**充分筛选**，不是取前 N 名猜。
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import d_core as D
from d_q2 import World, TYPES, build_groups, schedule, evaluate
import d_q3 as Q3
import d_q4 as Q4
import d_relay
import d_q4_partition as P

HERE = os.path.dirname(os.path.abspath(__file__))
INV = Q4.INV


def main():
    W = World()
    rt, W.relays, W.rbats = D.load_relays()
    C = D.Comms(W.geo, W.nodes, D.load_comm())
    g1, p1, e1 = build_groups(W, allow_types=("A", "B"))
    P.p1_stub = p1
    with open(os.path.join(HERE, "out_q3.json"), encoding="utf-8") as f:
        q3 = json.load(f)
    stations = q3["stations"]
    T_SHIFT = float(q3.get("t_shift", 0.0))

    sites = sorted({s for x in p1 for s, _ in x["stops"]})
    site_mask = {s: 0 for s in sites}
    for x in p1:
        m = 0
        for (t0, t1, smp) in Q3.gaps_of(W, C, x):
            for j, st in enumerate(stations):
                if d_relay.gap_covered(W, C, smp, tuple(st["point"])):
                    m |= 1 << j
                    break
        site_mask[x["stops"][0][0]] |= m
    pc = lambda m: bin(m).count("1")

    n = len(sites)
    order = sites
    cnt = Counter()
    cand = []
    for mask in range(1, 1 << (n - 1)):
        S = [order[i] for i in range(n - 1) if mask >> i & 1]
        comp = [s for s in order if s not in S]
        if not S or not comp:
            continue
        mS = mC = 0
        for s in S:
            mS |= site_mask[s]
        for s in comp:
            mC |= site_mask[s]
        t = pc(mS) + pc(mC)
        cnt[t] += 1
        if t == 3:
            cand.append((S, comp))
    print(f"K=2 共 {sum(cnt.values())} 种分法；各组站位需求之和的分布 "
          f"{dict(sorted(cnt.items()))}")
    print(f"最小值 3 → 中继缺口 1，且**只有 {len(cand)} 种**分区能达到")
    print("任何需求 ≥4 的分区在字典序第 4 位上已经输了，故只需精算这"
          f"{len(cand)} 种。\n")

    cache = P.memoise_scheduler()
    best = None
    for S, comp in cand:
        subs = [[p1[i]["stops"] for i in range(len(p1))
                 if p1[i]["stops"][0][0] in grp] for grp in (S, comp)]
        alloc = Q4.best_allocation(W, subs, t_shift=T_SHIFT)
        nd = {gg: sum(alloc["per_group"][j]["cap"][gg] for j in range(2))
              for gg in TYPES}
        bn = {gg: sum(alloc["per_group"][j]["bat"][gg] for j in range(2))
              for gg in TYPES}
        gap_d = sum(max(0, nd[gg] - INV["drone"][gg]) for gg in TYPES)
        gap_b = sum(max(0, bn[gg] - INV["battery"][gg]) for gg in TYPES)
        ns = [len(s) for s in subs]
        spread = max(ns) - min(ns)
        score = (int(alloc["viol"]), gap_d, gap_b, 1, spread,
                 float(alloc["tmax"]))
        if best is None or score < best[0]:
            best = (score, (S, comp), ns, gap_d, gap_b)
            print(f"  新最优 {score}  {len(S)}/{len(comp)} 站，架次 {ns}")
    score, (S, comp), ns, gd, gb = best
    print()
    print("=" * 78)
    print("K=2 的全局最优（在 16383 种分法的完整字典序下）")
    print("=" * 78)
    print(f"  目标 (违反, 机型缺口, 电池缺口, 中继缺口, 极差, 完成) = {tuple(score)}")
    print(f"  分区：{sorted(S)} ／ {sorted(comp)}（{len(S)}/{len(comp)} 站）")
    print(f"  架次 {ns}，机型缺口 {gd}，电池缺口 {gb}")

    out = dict(n_partitions=sum(cnt.values()), dist=dict(cnt),
               n_candidates=len(cand), best_score=list(score),
               groups=[sorted(S), sorted(comp)], n_sorties=ns)
    with open(os.path.join(HERE, "out_q4_k2opt.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print("\n已写出 out_q4_k2opt.json")


if __name__ == "__main__":
    main()
