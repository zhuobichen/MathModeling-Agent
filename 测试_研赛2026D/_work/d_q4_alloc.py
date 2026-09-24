# -*- coding: utf-8 -*-
"""问题四：库存约束下的取舍优化与资源缺口的重新核算（论文 8.3 节）。

d_q4.py 原来只做"评价"：算出各组独立执行需要几架机，与库存一比得出缺口。
但那个需求是用 **min_fleet（每组只配单一机型）** 估出来的，得到"两种分区都缺
3 架 B 型"。库存里其实有 A、B、C 三种机型——按机型**混合分配**之后库存刚好够。
所以原先那个"缺口"是**建模口径造成的假缺口**。

本脚本把两件事分开算清楚：

  ① 运输机：库存 {A4,B2,C2} 分配给各组，$\min\sum_j \mathrm{Viol}_j$，
     约束 $\sum_j n_{g,j}\le N_g$。穷举求解（见 d_q4.best_allocation），
     取全局最优，而不是贪心。
  ② 中继：题目要求"各组独立核算、期间资源不得跨组调配"，所以第 $j$ 组需要
     |stations(j)| 架中继，合计 $\sum_j|\text{stations}(j)|$。这才是**真正的缺口**。

顺带回答一个更根本的问题：**库存到底够不够？** 不分区的 8 架机共享是零违反的
（问题二就是），分区之后才出现缺口——缺口的根源是"并行度"，不是总量。
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import d_core as D
from d_q2 import World, TYPES, initial_groups, merge_pass, schedule, evaluate
import d_q3 as Q3
import d_q4 as Q4


def main():
    W = World()
    rt, W.relays, W.rbats = D.load_relays()
    C = D.Comms(W.geo, W.nodes, D.load_comm())
    g1 = merge_pass(W, initial_groups(W), allow_types=("A", "B"))
    p1 = schedule(W, g1)
    e1 = evaluate(W, p1)
    inv = Q4.INV

    print("=" * 78)
    print("基线：不分区（8 架机共享同一池）")
    print("=" * 78)
    pooled = Q4.schedule_fleet_capped(W, g1, dict(inv["drone"]))
    print(f"  40 架次共享库存 {inv['drone']}：硬时限违反 {pooled['viol']:.0f} 个，"
          f"完成 {pooled['t_done']:.0f} s，能耗 {pooled['E']:.2f} kWh")
    print("  → **不分区的库存是够的**。缺口只在分区之后才出现——")
    print("    分区把「共享的峰值」变成「每组各自的峰值」，这才是缺口的成因。")

    out = {"pooled": pooled}

    # 问题三的中继站位（2 站，全覆盖）——与 d_q4 同一套
    demands = []
    for p in p1:
        for (t0, t1, smp) in Q3.gaps_of(W, C, p):
            pt = Q3.pick_point(W, C, smp, Q3.candidate_points(W, C, 20), rt)
            if pt:
                demands.append((pt, t0, t1))
    centers = Q3.cluster_stations(W, demands, 2)

    for K in (2, 3):
        groups = Q4.group_sites(W, K, g1)
        gid = {s: i for i, gp in enumerate(groups) for s in gp}
        subs = [[st for st in g1 if all(gid[x] == i2 for x, _ in st)]
                for i2 in range(len(groups))]
        print()
        print("=" * 78)
        print(f"K = {K}：{len(groups)} 组（"
              f"{' / '.join(str(len(g)) for g in groups)} 站）")
        print("=" * 78)
        assert Q4.check_partition(W, g1, groups) == 0

        alloc = Q4.best_allocation(W, subs, verbose=True)
        need = {g: alloc["alloc"][g] for g in TYPES}
        gap = {g: max(0, need[g] - inv["drone"][g]) for g in TYPES}
        print(f"  运输机：需求 {need} / 库存 { {g: inv['drone'][g] for g in TYPES} }"
              f" → 缺口 {gap}"
              f"{'（库存刚好够）' if not any(gap.values()) else ''}")

        # 中继：各组独立核算（题目要求资源不得跨组调配）
        s2st = defaultdict(set)
        for p in p1:
            ids = {gid[x] for x, _ in p["stops"]}
            if len(ids) != 1:
                continue
            gi = ids.pop()
            for (t0, t1, smp) in Q3.gaps_of(W, C, p):
                pt = Q3.pick_point(W, C, smp, Q3.candidate_points(W, C, 20), rt)
                if pt is None:
                    continue
                j = min(range(len(centers)),
                        key=lambda jj: W.geo.dist(pt[0], pt[1],
                                                 centers[jj][0], centers[jj][1]))
                s2st[gi].add(j)
        rel_per = {i2: len(v) for i2, v in sorted(s2st.items())}
        rel_need = sum(rel_per.values())
        print(f"  中继：各组需 {rel_per}，合计 {rel_need} 架 / 库存 {inv['relay']} 架"
              f" → **缺口 {max(0, rel_need - inv['relay'])} 架**")
        print(f"        （本题只有 {len(centers)} 个站位；不分区分时 2 架就够——"
              f"中继缺口完全是「不得跨组调配」这条要求造成的，不是覆盖能力不够。）")

        print(f"  完成时间：最晚组 {alloc['tmax']:.0f} s，"
              f"不分区基准 {e1['t_done']:.0f} s "
              f"（{alloc['tmax'] / e1['t_done'] - 1:+.1%}）")

        out[f"K{K}"] = dict(groups=groups, alloc=alloc, gap_drone=gap,
                            relay_per_group=rel_per, relay_need=rel_need,
                            gap_relay=max(0, rel_need - inv["relay"]),
                            tmax=alloc["tmax"], t_ref=e1["t_done"])

    with open(os.path.join(Q4.HERE, "out_q4_alloc.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print("\n已写出 out_q4_alloc.json")


if __name__ == "__main__":
    main()
