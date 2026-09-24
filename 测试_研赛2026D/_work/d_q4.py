# -*- coding: utf-8 -*-
"""问题四：救援任务分区与资源配置优化。

把 15 个服务区划成 K（2 或 3）个任务组，**保持问题三已定的货箱组批、访问顺序、
中继任务与通信保障关系不变**，只把资源按组切开，各自独立执行：
  · 一架次涉及多个服务区时，这些服务区必须划进同一组（题目要求）；
  · 各组只承担本组的运输与通信保障，期间资源不得跨组调配。

因此分区的自由度只剩下「哪些服务区归一组」。判据：
  · 资源规模：各组所需运输无人机（分机型）、共享电池、中继无人机与能源组件之和；
  · 资源冗余：峰值需求与配置数之差；
  · 组间均衡：各组工作量（架次、能耗、时间跨度）的离散程度；
  · 与库存的缺口：现有 4×A + 2×B + 2×C 运输机、A/B/C 电池 6/4/4 组、
    2 架中继 + 6 组能源组件。
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import d_core as D
from d_q2 import World, TYPES, SITES, initial_groups, merge_pass, schedule, evaluate
import d_q3 as Q3

HERE = os.path.dirname(os.path.abspath(__file__))
INV = dict(drone={"A": 4, "B": 2, "C": 2},
           battery={"A": 6, "B": 4, "C": 4},
           relay=2, relay_mod=6)


# ══════════════════════════════════════════════════════════
def components(W, sorties):
    """服务区的「共现连通分量」：同一架次里出现的服务区必须同组。

    15 个服务区实测连成 **5 个分量**（多站架次把它们绑在一起）。分区的自由度
    只在这些分量之间——直接对服务区聚类一定会产出跨组架次（题目不允许）。
    """
    par = {s: s for s in SITES}

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    for st in sorties:
        ss = [x for x, _ in st]
        for a in ss[1:]:
            ra, rb = find(ss[0]), find(a)
            if ra != rb:
                par[ra] = rb
    grp = defaultdict(list)
    for s in SITES:
        grp[find(s)].append(s)
    return [sorted(v) for v in grp.values()]


def group_sites(W, k, sorties=None, seed_off=0):
    """把 15 个服务区划成 k 组：**先在共现分量之间聚类**，再按地理就近合并。

    分量内不可拆（拆了就产生跨组架次），所以 k 必须 ≥ 分量数才能给出 k 组；
    实际只有 5 个分量，k=2/3 都要求把多个分量并成一组。
    """
    comps = components(W, sorties) if sorties is not None else         [[s] for s in SITES]
    cen = []
    for c in comps:
        cen.append((float(np.mean([W.nodes[s]["lon"] for s in c])),
                    float(np.mean([W.nodes[s]["lat"] for s in c]))))
    if k <= 1 or len(comps) <= k:
        return [sorted(x for c in comps for x in c)] if k == 1 else [list(c)
                                                                   for c in comps]
    # 贪心最大最小距离选初始簇心（在分量上）
    pick = [comps[seed_off % len(comps)]]
    while len(pick) < k:
        best, bd = None, -1
        for c in comps:
            if c in pick:
                continue
            d = min(W.geo.dist(W.nodes[c[0]]["lon"], W.nodes[c[0]]["lat"],
                               W.nodes[pk[0]]["lon"], W.nodes[pk[0]]["lat"])
                    for pk in pick)
            if d > bd:
                bd, best = d, c
        pick.append(best)
    assign = [[c] for c in pick]
    for c in comps:
        if c in pick:
            continue
        i = min(range(len(assign)),
                key=lambda j: W.geo.dist(
                    W.nodes[c[0]]["lon"], W.nodes[c[0]]["lat"],
                    float(np.mean([W.nodes[x]["lon"] for x in assign[j][0]])),
                    float(np.mean([W.nodes[x]["lat"] for x in assign[j][0]]))))
        assign[i].append(c)
    return [sorted(x for c in g for x in c) for g in assign if g]


def check_partition(W, sorties, groups):
    """题目要求：同一架次涉及的服务区必须同组。"""
    gid = {s: i for i, gp in enumerate(groups) for s in gp}
    bad = 0
    for st in sorties:
        ids = {gid[s] for s, _ in st}
        if len(ids) > 1:
            bad += 1
    return bad


def schedule_fleet(W, stops_list, n):
    """用至多 n 架运输机独立执行一组架次，返回 (violations, 机型用量, E)。"""
    drones = []                       # [dict(type, free)]
    order = sorted(range(len(stops_list)), key=lambda i: W.urgency(stops_list[i]))
    viol = E = 0.0
    for i in order:
        st = stops_list[i]
        due = min(b["t_due"] for b in _boxes(st))
        hard = any(b["first"] or b["kind"] == "医疗物资" for b in _boxes(st))
        oks = W.ok_types(st)
        if not oks:
            raise RuntimeError("组内无机型可用")
        cand = []
        for g in oks:
            t, EE = W.route(g, st)
            for k, d in enumerate(drones):
                cand.append((max(d["free"], 0.0), EE, k, g, t, False))
            # 允许新开一架（只有在已有空闲机都不满足时限时才用）
            if len(drones) < n:
                cand.append((0.0, EE, len(drones), g, t, True))
        if hard:
            good = [c for c in cand if c[0] + c[4] <= due + 1e-9]
            if good:
                cand = good
        # 先挑不在时限内超时的，再挑最早可用的；新开一架要付出「多用一架」的代价
        cand.sort(key=lambda c: (c[0] + c[4] > due + 1e-9 if hard else False,
                                 c[5], c[0], c[1]))
        start, EE, k, g, t, isnew = cand[0]
        if isnew:
            drones.append(dict(type=g, free=start + t))
        else:
            drones[k]["free"] = start + t
        E += EE
        if hard and start + t > due + 1e-9:
            viol += 1
    need = defaultdict(int)
    for d in drones:
        need[d["type"]] += 1
    return viol, {g: int(need[g]) for g in TYPES}, E


def min_fleet(W, stops_list, cap=12):
    """该组独立执行所需的最小机队：满足全部硬时限的最少架数（二分）。"""
    lo, hi = 1, cap
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        viol, need, E = schedule_fleet(W, stops_list, mid)
        if viol == 0:
            best = (mid, need, E)
            hi = mid - 1
        else:
            lo = mid + 1
    return best if best else (cap, *schedule_fleet(W, stops_list, cap)[1:])


def _boxes(stops):
    return [b for _, gp in stops for b in gp]


def main():
    W = World()
    rt, W.relays, W.rbats = D.load_relays()
    C = D.Comms(W.geo, W.nodes, D.load_comm())
    g1 = merge_pass(W, initial_groups(W), allow_types=("A", "B"))
    p1 = schedule(W, g1)
    e1 = evaluate(W, p1)

    # 问题三的中继站位（2 站，全覆盖）
    demands = []
    for p in p1:
        for (t0, t1, smp) in Q3.gaps_of(W, C, p):
            pt = Q3.pick_point(W, C, smp, Q3.candidate_points(W, C, 20), rt)
            if pt:
                demands.append((pt, t0, t1))
    centers = Q3.cluster_stations(W, demands, 2)
    st = Q3.station_plan(W, rt, demands, centers, C)
    # 每个站位服务哪些服务区（按覆盖的缺口归属）
    def stations_for_site(s):
        out = set()
        for p in p1:
            if s not in [x for x, _ in p["stops"]]:
                continue
            for (t0, t1, smp) in Q3.gaps_of(W, C, p):
                pt = Q3.pick_point(W, C, smp, Q3.candidate_points(W, C, 20), rt)
                if pt is None:
                    continue
                i = min(range(len(centers)),
                        key=lambda j: W.geo.dist(pt[0], pt[1],
                                                 centers[j][0], centers[j][1]))
                out.add(i)
        return out

    print("=" * 78)
    print("问题三的共享资源基线（全部 15 个服务区不分区）")
    print("=" * 78)
    print(f"  运输 {len(p1)} 架次 / 能耗 {e1['E']:.2f} kWh / 完成 {e1['t_done']:.0f}s"
          f"；中继 {len(st)} 站 / 能耗 {sum(x['E'] for x in st):.2f} kWh / "
          f"能源组件 {sum(x['n_bat'] for x in st)} 组")
    print(f"  实际使用：运输机 "
          f"{ {g: sum(1 for p in p1 if p['g'] == g) for g in TYPES} }（按机型计架次）")

    print()
    print("=" * 78)
    print("分区方案对比")
    print("=" * 78)
    out = {}
    for K in (2, 3):
        groups = group_sites(W, K, g1)
        gid = {s: i for i, gp in enumerate(groups) for s in gp}
        bad = check_partition(W, g1, groups)
        print(f"\n  K = {K}：分区 {['+'.join(gp) for gp in groups]}")
        print(f"    跨组架次（题目不允许，需先拆分）：{bad}")
        need_d = {g: 0 for g in TYPES}
        tot_sortie = tot_E = 0
        spans = []
        rel_need = 0
        for i2, gp in enumerate(groups):
            sub = [s2 for s2 in g1 if all(gid[x] == i2 for x, _ in s2)]
            if not sub:
                continue
            n_min, nf, EE = min_fleet(W, sub)
            for g in TYPES:
                need_d[g] += nf[g]
            tot_sortie += len(sub)
            tot_E += EE
            spans.append(n_min)
            print(f"    组{i2+1}（{len(gp)} 站）：{len(sub):>2} 架次 | "
                  f"能耗 {EE:>7.2f} kWh | 最小机队 {n_min} 架 "
                  f"{ {g: nf[g] for g in TYPES if nf[g]} }")
        # 中继：按「站位覆盖的缺口属于哪一组」分配
        s2st = defaultdict(set)
        for k2, p2 in enumerate(p1):
            gidx = {gid[x] for x, _ in p2["stops"]}
            if len(gidx) != 1:
                continue
            gi = gidx.pop()
            for (g0, g1_, smp) in Q3.gaps_of(W, C, p2):
                pt = Q3.pick_point(W, C, smp, Q3.candidate_points(W, C, 20), rt)
                if pt is None:
                    continue
                j2 = min(range(len(centers)),
                         key=lambda jj: W.geo.dist(pt[0], pt[1],
                                                  centers[jj][0], centers[jj][1]))
                s2st[gi].add(j2)
        rel_per = {i2: len(v) for i2, v in s2st.items()}
        rel_need = sum(rel_per.values())
        print(f"    中继：各组需 {rel_per}，合计 {rel_need} 架（库存 {INV['relay']}）"
              f"  → 缺口 {max(0, rel_need - INV['relay'])} 架")
        print(f"    合计需运输机 { {g: need_d[g] for g in TYPES} }（库存 "
              f"{ {g: INV['drone'][g] for g in TYPES} }）"
              f"  → 缺口 { {g: max(0, need_d[g]-INV['drone'][g]) for g in TYPES} }")
        ns = [len([s2 for s2 in g1 if all(gid[x] == i2 for x, _ in s2)])
              for i2 in range(len(groups))]
        print(f"    组间均衡：架次 {ns}（极差 {max(ns)-min(ns)}）"
              f"  最小机队 {spans}（极差 {max(spans)-min(spans)}）")

        out[K] = dict(groups=groups, cross=bad,
                      need_drone={g: int(need_d[g]) for g in TYPES},
                      need_relay=int(rel_need),
                      gap_drone={g: int(max(0, need_d[g] - INV["drone"][g]))
                                 for g in TYPES},
                      gap_relay=int(max(0, rel_need - INV["relay"])),
                      n_sortie=tot_sortie, E=tot_E)

    with open(os.path.join(HERE, "out_q4.json"), "w", encoding="utf-8") as f:
        json.dump(dict(inventory=INV, schemes=out,
                       baseline=dict(n_sortie=len(p1), E=e1["E"],
                                     t_done=e1["t_done"])),
                  f, ensure_ascii=False, indent=1, default=float)
    print("\n已写出 out_q4.json")


if __name__ == "__main__":
    main()
