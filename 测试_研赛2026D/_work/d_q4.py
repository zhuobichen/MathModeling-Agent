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

import itertools
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import d_core as D
import d_relay
from d_q2 import build_groups, World, TYPES, SITES, initial_groups, merge_pass, schedule, evaluate
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


# ══════════════════════════════════════════════════════════
def schedule_fleet_capped(W, stops_list, cap, bat_cap=None):
    """在**指定机型与电池配置**下独立执行一组架次。

    与 schedule_fleet 同一套口径（时限感知、两阶段充电），差别只有三点：
    机型用量受 cap 约束；电池池也受 bat_cap 约束（默认给满池）；返回完成时刻。
    cap 里某型为 0 表示该组没有这种机。
    """
    # **先把架次列表按内容排序**：schedule 的派发键在并列时靠 sorted 的稳定性
    # 打破，于是"同一个架次集合、不同的输入顺序"会排出不同的时间表——实测
    # 同一个分区经两条代码路径算出 12384 s 与 12032 s 两个结果。排序之后，
    # 本函数成为**架次集合的纯函数**，记忆化缓存也才真正安全。
    stops_list = sorted(
        stops_list,
        key=lambda st: tuple((s, tuple(sorted(b["box"] for b in bs)))
                             for s, bs in st))
    _e = {}
    for st in stops_list:
        oks = [g for g in W.ok_types(st) if cap.get(g, 0) > 0]
        if not oks:
            return None                     # 该配置下有架次无可用机型
        _e[id(st)] = min(W.route(g, st)[1] for g in oks)

    def key(i):
        bs = _boxes(stops_list[i])
        if any(b["first"] or b["kind"] == "医疗物资" for b in bs):
            return (0, min(b["t_due"] for b in bs), 0.0)
        return (1, 0.0, -_e[id(stops_list[i])])

    nb = {g: (bat_cap or {}).get(g, W.bats[g]["n"]) for g in TYPES}
    if any(cap.get(g, 0) > 0 and nb[g] <= 0 for g in TYPES):
        return None
    drones = {g: [dict(free=0.0) for _ in range(cap.get(g, 0))] for g in TYPES}
    bats = {g: [dict(free=0.0) for _ in range(nb[g])] for g in TYPES}
    viol = E = tdone = 0.0
    for i in sorted(range(len(stops_list)), key=key):
        st = stops_list[i]
        bs = _boxes(st)
        hard = any(b["first"] or b["kind"] == "医疗物资" for b in bs)
        due = min(b["t_due"] for b in bs)
        cand = []
        for g in W.ok_types(st):
            if cap.get(g, 0) <= 0:
                continue
            t, ee = W.route(g, st)
            for k, d in enumerate(drones[g]):
                for j, bt in enumerate(bats[g]):
                    cand.append((max(d["free"], bt["free"]), ee, g, k, j, t))
        if not cand:
            return None
        if hard:
            good = [c for c in cand if c[0] + c[5] <= due + 1e-9]
            if good:
                cand = good
        start, ee, g, k, j, t = min(cand, key=lambda c: (c[0], c[1]))
        end = start + t
        soc = max(1.0 - ee / W.types[g]["E_use"], 0.0)
        drones[g][k]["free"] = end
        bats[g][j]["free"] = end + D.charge_time(soc, W.bats[g]["T_full"])
        E += ee
        tdone = max(tdone, end)
        if hard and end > due + 1e-9:
            viol += 1
    return dict(viol=viol, E=E, t_done=tdone)


def min_batteries(W, stops_list, cap, t_ref, tol=0.05):
    """该组在机型配置 cap 下**所需的最少电池组数**。

    题目要求「各组独立核算资源、期间资源不得跨组调配」，电池组也要按组分；
    缺省的"每组都给满池"会让各组的电池数之和远超库存，不是可行配置。

    **判据不能只看"零违反"**：少给电池会让无人机排队等充电，首批箱照样按时
    送到，但完成时间被拖长。所以要求同时满足
    `viol == 0 且 t_done ≤ t_ref × (1 + tol)`（t_ref 取该组满池时的完成时刻）。

    **必须联合搜索而不是逐型二分**：逐型分别压到极限、再叠加起来，总效应会
    远超单型 5% 的容差（实测 9065 s 被叠到 11004 s，多出 21%）。这里直接枚举
    该组的全部电池向量（≤ 7×5×5），取"满足判据且组数最少"的那个，
    同组数时优先少用 A 型——A 型电池是最紧的资源。
    """
    lim = t_ref * (1.0 + tol)
    inv = INV["battery"]
    # 按**组数从小到大**枚举，命中即停——这样第一个可行的就是最优的。
    # （原来的按字母序枚举要跑满 7×5×5 = 175 次，改成升序后通常几次就命中；
    #   这是让"每组×每种配置的电池表"能整张算出来、从而不必筛候选的关键。）
    vecs = sorted(((a, b, c)
                   for a in range(inv["A"] + 1)
                   for b in range(inv["B"] + 1)
                   for c in range(inv["C"] + 1)),
                  key=lambda v: (sum(v), v))
    for a, b, c in vecs:
        if any(cap.get(g, 0) > 0 and v <= 0
               for g, v in (("A", a), ("B", b), ("C", c))):
            continue
        bc = {"A": a, "B": b, "C": c}
        r = schedule_fleet_capped(W, stops_list, cap, bc)
        if r is not None and r["viol"] == 0 and r["t_done"] <= lim:
            return bc, r
    bc = dict(inv)                         # 满池是保底的可行解
    return bc, schedule_fleet_capped(W, stops_list, cap, bc)


def all_caps(inv=INV["drone"]):
    """所有不超过库存的机型配置（全枚举，规模 = (nA+1)(nB+1)(nC+1) - 1）。"""
    out = []
    for a in range(inv["A"] + 1):
        for b in range(inv["B"] + 1):
            for c in range(inv["C"] + 1):
                if a + b + c:
                    out.append({"A": a, "B": b, "C": c})
    return out


def best_allocation(W, subs, verbose=False, t_shift=0.0):
    """**库存约束下的最优分配**：把库存按机型分给各组，使硬时限违反数最少。

    $$\min \sum_j \mathrm{Viol}_j(\mathbf n_j)\quad
      \mathrm{s.t.}\ \sum_j n_{g,j}\le N_g\ \forall g$$

    每组 × 每种配置先算好缓存（K=3 时 3×44 = 132 次排程），再穷举分配组合，
    所以取到的是**全局最优**而不只是贪心解。

    **这一步改变了问题四的结论**：d_q4.py 原来用一个「每组只配单一机型」的
    min_fleet 去估需求，得到"两种分区都缺 3 架 B 型"；但库存里有 A、B、C
    三种机型，按机型混合分配后**库存刚好够、零违反**。原先那个"缺口"是
    建模口径造成的假缺口——详见论文 8.3 节。
    """
    inv = INV["drone"]
    caps = all_caps(inv)
    cache = []
    for sl in subs:
        tab = {}
        for cp in caps:
            tab[tuple(sorted(cp.items()))] = schedule_fleet_capped(W, sl, cp)
        cache.append(tab)
    # 先按 (违反数, 运输机缺口, 完成时间) 粗排，再对最优的几组精算电池缺口。
    # **不能只按 (违反数, 完成时间) 排**：那样选出的分配可能完成时间最短、
    # 却多用了一组电池——而外层分区的目标里"电池组缺口"排在完成时间之前。
    # 内层与外层的目标不一致，会让求出的分区在真实目标下不是最优。实测同一
    # 分区曾出现 12032 s / 电池缺 1 组 与 12384 s / 电池缺 0 组两个结果，
    # 只按完成时间排就会选错。
    # **先把"每组 × 每种配置"的电池表整张算出来**，再穷举分配。
    # 这样目标里的"电池组缺口"对**每一个**组合都是精确的，不必靠筛候选去近似
    # ——筛候选是不可靠的：实测按完成时间取前 8 名与 30 名，都把"电池缺口 0"
    # 的那个组合挤掉了，于是内层给出的解在外层目标下并非最优。
    batab = {}
    for j in range(len(subs)):
        for i, cp in enumerate(caps):
            r = cache[j][tuple(sorted(cp.items()))]
            if r is None:
                batab[(j, i)] = None
                continue
            bc, _ = min_batteries(W, subs[j], cp, t_ref=r["t_done"])
            batab[(j, i)] = bc

    rows = []
    n_eval = 0
    for combo in itertools.product(range(len(caps)), repeat=len(subs)):
        tot = {g: sum(caps[i][g] for i in combo) for g in TYPES}
        if any(tot[g] > inv[g] for g in TYPES):
            continue
        res = [cache[j][tuple(sorted(caps[i].items()))]
               for j, i in enumerate(combo)]
        if any(r is None for r in res):
            continue
        if any(batab[(j, i)] is None for j, i in enumerate(combo)):
            continue
        n_eval += 1
        gap_d = sum(max(0, tot[g] - inv[g]) for g in TYPES)
        bat_tot = {g: sum(batab[(j, i)][g] for j, i in enumerate(combo))
                   for g in TYPES}
        gap_b = sum(max(0, bat_tot[g] - INV["battery"][g]) for g in TYPES)
        key = (sum(r["viol"] for r in res), gap_d, gap_b,
               max(r["t_done"] for r in res))
        rows.append((key, combo, res, tot, bat_tot, gap_b))
    if not rows:
        raise RuntimeError("无可行的资源分配")
    rows.sort(key=lambda x: x[0])
    key, combo, res, tot, bat_tot, gap_b = rows[0]
    per = []
    for j, i in enumerate(combo):
        r = res[j]
        bc = batab[(j, i)]
        rr = schedule_fleet_capped(W, subs[j], caps[i], bc) or r
        per.append(dict(cap=caps[i], viol=rr["viol"], t=rr["t_done"] + t_shift,
                        E=rr["E"], bat=bc, t_full=r["t_done"] + t_shift))
    viol, gap_d, tmax = key[0], key[1], key[3]
    tmax = max(p["t"] for p in per)          # 最终配置下的最晚组完成时刻
    out = dict(alloc=tot, viol=viol, tmax=tmax, n_eval=n_eval,
               per_group=per, bat_tot=bat_tot,
               gap_bat={g: max(0, bat_tot[g] - INV["battery"][g]) for g in TYPES})
    if verbose:
        print(f"    库存约束下的最优分配（穷举 {n_eval} 个可行组合）："
              f"{ {g: tot[g] for g in TYPES} } → 违反 {viol:.0f} 个，"
              f"最晚组完成 {tmax:.0f} s")
        for j, i in enumerate(combo):
            cp = caps[i]
            b = per[j]["bat"]
            print(f"      组{j+1}：{cp['A']}A+{cp['B']}B+{cp['C']}C，"
                  f"电池 {b['A']}/{b['B']}/{b['C']} 组"
                  f" → 违反 {per[j]['viol']:.0f}，"
                  f"完成 {per[j]['t']:.0f} s（满池时 {per[j]['t_full']:.0f} s）")
        print(f"    电池组合计 {bat_tot}（库存 {INV['battery']}）"
              f" → 缺口 {out['gap_bat']}")
    return out


def main():
    W = World()
    rt, W.relays, W.rbats = D.load_relays()
    C = D.Comms(W.geo, W.nodes, D.load_comm())
    g1, p1, e1 = build_groups(W, allow_types=("A", "B"))

    # 问题三的中继站位：直接读 out_q3.json 的**最终 3 站方案**。
    # 不再用 cluster_stations + station_plan 现算——那两个函数的覆盖判据
    # 有高估（拿缺口自己的悬停点当无人机位置判链路，见 d_relay.py），
    # 现在以 d_relay 的站位选择为准，问题四必须与问题三同一套站位。
    with open(os.path.join(HERE, "out_q3.json"), encoding="utf-8") as f:
        q3res = json.load(f)
    st = q3res["stations"]
    centers = [tuple(x["point"]) for x in st]
    # 问题三要求中继先到点，运输因此整体推迟 t_shift 秒；问题四以问题三方案
    # 为基础，各组同样受这一条约束，故每组的完成时刻都要加上这个推迟量。
    T_SHIFT = float(q3res.get("t_shift", 0.0))
    print(f"  问题三的中继站位 {len(st)} 个（读自 out_q3.json）；"
          f"中继先到点导致的运输推迟 {T_SHIFT:.0f}s（各组继承）")
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
    print(f"  （中继站位数由 d_relay 的修正判据给出：k=2 穷举证明只能覆盖 58/64，"
          f"故需 {len(st)} 站）")
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
        # 中继：按「哪个站位覆盖了这一组的缺口」分配（判据与 d_relay 一致）
        s2st = defaultdict(set)
        for k2, p2 in enumerate(p1):
            gidx = {gid[x] for x, _ in p2["stops"]}
            if len(gidx) != 1:
                continue
            gi = gidx.pop()
            for (g0, g1_, smp) in Q3.gaps_of(W, C, p2):
                for j2, x2 in enumerate(st):
                    if d_relay.gap_covered(W, C, smp, tuple(x2["point"])):
                        s2st[gi].add(j2)
                        break
        rel_per = {i2: len(v) for i2, v in s2st.items()}
        # 各组需要的能源组件数 = 该组分到的那些站位的组件数之和
        rel_mod = {i2: int(sum(st[j]["n_bat"] for j in v))
                   for i2, v in s2st.items()}
        rel_need = sum(rel_per.values())
        mod_need = sum(rel_mod.values())
        print(f"    中继：各组需 {rel_per}，合计 {rel_need} 架（库存 {INV['relay']}）"
              f"  → 缺口 {max(0, rel_need - INV['relay'])} 架")
        print(f"    对照·各组按单一机型自备峰值机队：合计 "
              f"{ {g: need_d[g] for g in TYPES} }（库存 "
              f"{ {g: INV['drone'][g] for g in TYPES} }）"
              f"  → 表面上缺 { {g: max(0, need_d[g]-INV['drone'][g]) for g in TYPES} }")
        print("      ↑ 这个「缺口」是建模口径造成的：它只允许每组配一种机型。")
        ns = [len([s2 for s2 in g1 if all(gid[x] == i2 for x, _ in s2)])
              for i2 in range(len(groups))]
        print(f"    组间均衡：架次 {ns}（极差 {max(ns)-min(ns)}）"
              f"  最小机队 {spans}（极差 {max(spans)-min(spans)}）")

        # 库存约束下的最优分配（把库存按机型分给各组，而不是"每组自备峰值机队"）
        subs = [[st for st in g1 if all(gid[x] == i2 for x, _ in st)]
                for i2 in range(len(groups))]
        alloc = best_allocation(W, subs, verbose=True, t_shift=T_SHIFT)
        print(f"    对照·每组按单一机型自备峰值机队：合计 "
              f"{ {g: need_d[g] for g in TYPES} }"
              f"（这才是原先那句「缺 3 架 B 型」的来源，是建模口径造成的假缺口）")

        out[K] = dict(groups=groups, cross=bad,
                      need_drone={g: int(need_d[g]) for g in TYPES},
                      need_relay=int(rel_need),
                      gap_drone={g: int(max(0, need_d[g] - INV["drone"][g]))
                                 for g in TYPES},
                      gap_relay=int(max(0, rel_need - INV["relay"])),
                      alloc=alloc,
                      relay_per_group={int(k2): v for k2, v in rel_per.items()},
                      relay_mod_per_group={int(k2): v for k2, v in rel_mod.items()},
                      need_relay_mod=int(mod_need),
                      n_sortie=tot_sortie, E=tot_E)

    with open(os.path.join(HERE, "out_q4.json"), "w", encoding="utf-8") as f:
        json.dump(dict(inventory=INV, schemes=out,
                       baseline=dict(n_sortie=len(p1), E=e1["E"],
                                     t_done=e1["t_done"])),
                  f, ensure_ascii=False, indent=1, default=float)
    print("\n已写出 out_q4.json")


if __name__ == "__main__":
    main()
