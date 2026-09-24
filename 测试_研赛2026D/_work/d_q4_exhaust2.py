# -*- coding: utf-8 -*-
"""K=2 的**精确穷举**：1023 种二分法全部按 (8-1) 的最优分配评估一遍。

为什么单独跑：`d_q4_partition.py` 里的局部搜索只保证局部最优，而"这个分区是
K=2 下的最优"这句话对论文的分量完全不同。K=2 的分区空间只有 2^10−1 = 1023
种，直接穷举得出来。

## 与完整目标的差别（必须写清楚）

完整目标含**电池组缺口**，而算电池缺口要对每个分区再跑一遍 `min_batteries`
（每组最多 7×5×5 种电池向量）——代价高一个量级。本脚本的排序用

    (硬时限违反, 运输机缺口, 中继缺口, 架次极差, 最晚组完成)

**不含电池缺口**；电池缺口只对筛出的最优候选单独算。也就是说本脚本能证明
"在运输机/中继缺口、均衡性、完成时间上，1023 种里没有更好的"，不能证明
"电池缺口也最小"。
"""
from __future__ import annotations

import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import d_core as D
from d_q2 import build_groups, World, TYPES, initial_groups, merge_pass, schedule
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
    g1 = build_groups(W, allow_types=("A", "B"))[0]   # 顺序搜索用不上（这里只做分区枚举）
    p1 = schedule(W, g1)
    P.p1_stub = p1
    with open(os.path.join(HERE, "out_q3.json"), encoding="utf-8") as f:
        q3res = json.load(f)
    T_SHIFT = float(q3res.get("t_shift", 0.0))

    cache = P.memoise_scheduler()
    comps, comp_of, by_comp = P.sorties_by_component(W, p1)
    smask = P.station_mask_per_sortie(W, C, p1, q3res["stations"])
    ncomp = len(comps)
    print(f"共现分量 {ncomp} 个；K=2 的分法 {2 ** (ncomp - 1) - 1} 种（固定分量0在A组）")

    def rneed_of(gidx):
        m = 0
        for i in gidx:
            m |= smask[i]
        return bin(m).count("1")

    best = None
    cands = []
    n = 0
    for mask in range(1, 1 << (ncomp - 1)):
        A = [c for c in range(ncomp) if (mask >> c) & 1]
        B = [c for c in range(ncomp) if not ((mask >> c) & 1)]
        if not A or not B:
            continue
        gs = [[i for c in grp for i in by_comp[c]] for grp in (A, B)]
        subs = [[p1[i]["stops"] for i in g] for g in gs]
        alloc = Q4.best_allocation(W, subs, t_shift=T_SHIFT)
        nd = {g: sum(alloc["per_group"][j]["cap"][g] for j in range(2))
              for g in TYPES}
        gap_d = sum(max(0, nd[g] - INV["drone"][g]) for g in TYPES)
        rel = sum(rneed_of(g) for g in gs)
        gap_r = max(0, rel - INV["relay"])
        ns = [len(g) for g in gs]
        score = (int(alloc["viol"]), gap_d, gap_r, max(ns) - min(ns),
                 float(alloc["tmax"]))
        n += 1
        cands.append((score, (A, B), nd, rel, alloc))
        if best is None or score < best[0]:
            best = (score, (A, B), nd, rel, alloc)
            print(f"  [{n:>4}/{1023}] 新最优 {score}  分区 "
                  f"{[sorted(x for c in grp for x in comps[c]) for grp in (A, B)]}"
                  f"  缓存 {len(cache)}")
    score, AB, nd, rel, alloc = best

    # ── 补上电池缺口：全量算太贵，只对"在其余四维上已经很好"的那些候选精算 ──
    # 判据：中继缺口不劣于最优、且 (极差, 完成时间) 都在前列。剩下的排序再按
    # 完整五维目标（含电池缺口）来。
    # 五维目标的比较规则是 (违反, 机型缺口, 电池缺口, 中继缺口, 极差, 完成)。
    # 已有最优是 (0,0,0,1,20,11778)，所以要打败它必须同时满足：
    #   违反 0、机型缺口 0、电池缺口 0，且中继缺口 = 1（这是全局最小值），
    #   然后极差 < 20 或极差 = 20 且完成更早。
    # 因此只需对「中继缺口已到最小(1)、机型缺口 0、极差 < 20」的那些补算电池缺口
    # ——这是**充分的筛选**，不是取前 N 名猜。
    min_r = min(c[0][2] for c in cands)          # 全部 1023 种里中继缺口的最小值
    k0_r, k0_s = 1, 20        # 待验证候选的 (中继缺口, 极差)
    cand_pool = [c for c in cands
                 if c[0][0] == 0 and c[0][1] == 0 and c[0][2] <= min_r
                 and c[0][3] <= k0_s]
    print()
    print(f"1023 种里中继缺口最小 = {min_r}")
    print(f"可能打败候选的分区（违反0、机型缺口0、中继缺口={min_r}、极差≤{k0_s}）："
          f"{len(cand_pool)} 种，全部补算电池缺口")
    ranked = cand_pool
    best5 = None
    for sc, ABc, ndc, relc, ga in ranked:
        gsc = [[i for c in grp for i in by_comp[c]] for grp in ABc]
        ssc = [[p1[i]["stops"] for i in g] for g in gsc]
        full = Q4.best_allocation(W, ssc, t_shift=T_SHIFT)
        bnc = {g: sum(full["per_group"][j]["bat"][g] for j in range(2))
               for g in TYPES}
        gb = sum(max(0, bnc[g] - INV["battery"][g]) for g in TYPES)
        score5 = (sc[0], sum(max(0, ndc[g2] - INV["drone"][g2]) for g2 in TYPES),
                  gb, sc[2], sc[3], sc[4])
        if best5 is None or score5 < best5[0]:
            best5 = (score5, ABc, ndc, relc, bnc, full)
    s5, AB5, nd5, rel5, bn5, full5 = best5
    print(f"  五维最优 (违反, 机型缺口, 电池缺口, 中继缺口, 极差, 完成) = {tuple(s5)}")
    print(f"  分区：{[[sorted(x for c in grp for x in comps[c]) for grp in AB5]]}")
    print(f"  机型 {nd5}  电池 {bn5}  中继 {rel5} 架")

    score, AB, nd, rel, alloc = best
    print()
    print("=" * 78)
    print(f"K=2 精确穷举完成：{n} 种")
    print("=" * 78)
    print(f"  最优目标 (违反, 运输机缺口, 中继缺口, 架次极差, 最晚组完成) = {score}")
    print(f"  分区：{[[sorted(x for c in grp for x in comps[c]) for grp in AB]]}")
    print(f"  运输机需求 {nd}（库存 {INV['drone']}）  中继需求 {rel} 架")
    # 电池缺口单独精算（贵，只算最优这一个）
    gs = [[i for c in grp for i in by_comp[c]] for grp in AB]
    subs = [[p1[i]["stops"] for i in g] for g in gs]
    full = Q4.best_allocation(W, subs, t_shift=T_SHIFT)
    bn = {g: sum(full["per_group"][j]["bat"][g] for j in range(2)) for g in TYPES}
    gap_b = sum(max(0, bn[g] - INV["battery"][g]) for g in TYPES)
    print(f"  电池需求 {bn}（库存 {INV['battery']}）→ 缺口 {gap_b}")

    out = dict(n_partitions=n, best_score=list(score),
               best5_score=list(s5),
               best5_groups=[sorted(x for c in grp for x in comps[c]) for grp in AB5],
               best5_need_bat=bn5,
               best_groups=[sorted(x for c in grp for x in comps[c]) for grp in AB],
               need_drone=nd, need_relay=rel, need_bat=bn, gap_bat=gap_b,
               per_group=[dict(cap=full["per_group"][j]["cap"],
                               bat=full["per_group"][j]["bat"],
                               t=full["per_group"][j]["t"],
                               n=len(gs[j])) for j in range(2)])
    with open(os.path.join(HERE, "out_q4_exhaust2.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print("已写出 out_q4_exhaust2.json")


if __name__ == "__main__":
    main()
