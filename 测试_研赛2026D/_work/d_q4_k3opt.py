# -*- coding: utf-8 -*-
"""K=3 的**全局最优验证**（论文 8.4.2 节）。

## 先说一个数

15 个服务区分成 3 组，分法数是第二类斯特林数

$$S(15,3)=\\frac{3^{15}-3\\cdot2^{15}+3}{6}=2\\,375\\,101$$

**不是"2.4 亿"**。论文第一版把这个数写大了一百倍（写成 2.4 亿），据此判定
"K=3 无法穷举、只能给局部最优"。改成 238 万之后，**K=3 也可以穷举**。

## 怎么穷举得动

和 K=2 一样，先用一个**充分条件**把候选砍到极少数：

1. 每个站位需要哪些中继站位是纯位运算，于是对任意子集 $S$ 可以先算出
   $\\mathrm{mask}(S)$（32768 个子集，DP 一遍即可）；
2. 于是任意一个三分区的"各组中继需求之和"
   $=\\mathrm{popcount}(\\mathrm{mask}(A))+\\mathrm{popcount}(\\mathrm{mask}(B))
   +\\mathrm{popcount}(\\mathrm{mask}(C))$，**O(1)**；
3. 目标里"中继缺口"排在第 4 位，所以**只有中继需求取到最小值的那些分区**
   值得精算，其余在字典序上已经输了。

238 万次 O(1) 计算，几秒钟就能跑完；再对达标的那一小批做精确资源分配。
"""
from __future__ import annotations

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


def main():
    W = World()
    rt, W.relays, W.rbats = D.load_relays()
    C = D.Comms(W.geo, W.nodes, D.load_comm())
    g1, p1, e1 = build_groups(W, allow_types=("A", "B"))
    with open(os.path.join(HERE, "out_q3.json"), encoding="utf-8") as f:
        q3 = json.load(f)
    stations = q3["stations"]
    sidx = {s: i for i, s in enumerate(sorted({x["stops"][0][0] for x in p1}))}
    sites = sorted(sidx)
    n = len(sites)
    site_mask = [0] * n
    for x in p1:
        m = 0
        for (t0, t1, smp) in Q3.gaps_of(W, C, x):
            for j, st in enumerate(stations):
                if d_relay.gap_covered(W, C, smp, tuple(st["point"])):
                    m |= 1 << j
                    break
        site_mask[sidx[x["stops"][0][0]]] |= m
    print(f"站点 {n} 个；各站需要的站位掩码 {[bin(m) for m in site_mask]}")

    # mask[S] = 子集 S 的并集（DP）
    size = 1 << n
    mk = [0] * size
    for S in range(1, size):
        low = S & (-S)
        i = low.bit_length() - 1
        mk[S] = mk[S ^ low] | site_mask[i]
    pc = int.bit_count
    full = size - 1

    # 枚举三分区：要求 min(A) < min(B) < min(C)，每个分法只数一次
    print("枚举 238 万个三分区（用 popcount 求各组中继需求之和）...")
    dist = {}
    best_need = 99
    achievers = []
    cnt = 0
    for A in range(1, full):
        if A & 1 == 0:              # 站点 0 必须在 A 组（否则会重复计数）
            continue
        rem = full ^ A
        if rem == 0:
            continue
        # B 必须含 rem 的最低位
        low = rem & (-rem)
        B0 = rem
        B = B0
        while B:
            if B & low and B != rem:
                C = rem ^ B
                if C:
                    cnt += 1
                    t = pc(mk[A]) + pc(mk[B]) + pc(mk[C])
                    dist[t] = dist.get(t, 0) + 1
                    if t < best_need:
                        best_need = t
                        achievers = [(A, B, C)]
                    elif t == best_need:
                        achievers.append((A, B, C))
            B = (B - 1) & B0
    print(f"三分区总数 {cnt}（应为 S(15,3)=2375101）")
    print(f"各组中继需求之和的分布 {dict(sorted(dist.items()))}")
    print(f"最小值 = {best_need} → 中继缺口 = {max(0, best_need - 2)}，"
          f"达标的只有 {len(achievers)} 种")

    # 精算达标者
    print(f"\n对达标的 {len(achievers)} 种做精确资源分配...")
    best = None
    for A, B, C in achievers:
        groups = [[sites[i] for i in range(n) if m >> i & 1]
                  for m in (A, B, C)]
        idxs = [[k for k, x in enumerate(p1) if x["stops"][0][0] in gp]
                for gp in groups]
        subs = [[p1[k]["stops"] for k in gg] for gg in idxs]
        alloc = Q4.best_allocation(W, subs, t_shift=0.0)
        nd = {g: sum(alloc["per_group"][j]["cap"][g] for j in range(3))
              for g in TYPES}
        gap_d = sum(max(0, nd[g] - INV["drone"][g]) for g in TYPES)
        gap_b = sum(alloc["gap_bat"].values())
        ns = [len(s) for s in subs]
        score = (int(alloc["viol"]), gap_d, gap_b, max(0, best_need - 2),
                 max(ns) - min(ns), float(alloc["tmax"]))
        if best is None or score < best[0]:
            best = (score, groups, ns, nd, alloc["bat_tot"], alloc["tmax"])
            print(f"  新最优 {score}  架次 {ns}")
    if best is None:
        print("  无达标分区（不应发生）")
        return
    score, groups, ns, nd, bat, tmax = best
    print()
    print("=" * 78)
    print("K=3 的全局最优")
    print("=" * 78)
    print(f"  目标 (违反, 机型缺口, 电池缺口, 中继缺口, 极差, 完成) = {tuple(score)}")
    for j, gp in enumerate(groups):
        print(f"    组{j+1}：{len(gp)} 站 {sorted(gp)}，{ns[j]} 架次")
    print(f"  机型 {nd}  电池 {bat}  最晚组 {tmax:.0f} s")

    out = dict(n_partitions=cnt, dist=dict(dist), min_relay_need=best_need,
               n_achievers=len(achievers), best_score=list(score),
               groups=[sorted(g) for g in groups], n_sorties=ns)
    with open(os.path.join(HERE, "out_q4_k3opt.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print("\n已写出 out_q4_k3opt.json")


if __name__ == "__main__":
    main()
