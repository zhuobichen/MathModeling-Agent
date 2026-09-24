# -*- coding: utf-8 -*-
"""问题一：单点往返运输能力与货箱组批。

不涉及实体无人机与共享电池，每个服务区独立：O01 → Si → O01 直接往返，
同一服务区可分多个架次，不跨服务区组批。
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import d_core as D

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "_交付"))

SITES = [f"S{i:03d}" for i in range(1, 16)]


# ══════════════════════════════════════════════════════════
def sortie(g, site, boxes, T, rho=None):
    """一趟 O01→Si→O01 的时间与能耗。boxes 为本次投送的货箱列表。"""
    tt = T.types[g]
    r = rho if rho is not None else tt["rho"]
    q = sum(b["mass"] for b in boxes)
    out = T.seg(g, "O01", site, q)
    back = T.seg(g, site, "O01", 0.0)
    k = len(boxes)
    t = (out["t"] + back["t"] + tt["t_prep"] + k * tt["t_load"]
         + tt["t_hand0"] + k * tt["t_hand1"])
    E = out["E"] + back["E"]
    return dict(t=t, E=E, q=q, vol=sum(b["vol"] for b in boxes),
                ok=E <= (1 - r) * tt["E_use"] + 1e-12,
                soc=1 - E / tt["E_use"])


def max_payload(g, site, T, rho=None, lo=0.0, hi=None, tol=1e-6):
    """最大安全载荷：最大的 q 使 E(O→S,q)+E(S→O,0) <= (1-rho)E_use。

    E 关于 q 单调不减（L(q) 单调减、爬升质量单调增），故二分即可。
    """
    tt = T.types[g]
    hi = tt["Q"] if hi is None else hi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        e = T.seg(g, "O01", site, mid)["E"] + T.seg(g, site, "O01", 0.0)["E"]
        if e <= (1 - (rho if rho is not None else tt["rho"])) * tt["E_use"]:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return max(lo, 0.0)


# ══════════════════════════════════════════════════════════
def feasible_subsets(mass, vol, cap_m, cap_v):
    """返回所有满足质量/体积约束的子集（按位掩码），只取含最低位的那些。"""
    n = len(mass)
    ok = []
    for mask in range(1, 1 << n):
        m = v = 0.0
        good = True
        for i in range(n):
            if mask >> i & 1:
                m += mass[i]
                v += vol[i]
                if m > cap_m + 1e-9 or v > cap_v + 1e-9:
                    good = False
                    break
        if good:
            ok.append(mask)
    return ok


_BINS_CACHE = {}


def min_bins(mass, vol, cap_m, cap_v, max_boxes=None):
    """装箱：最少几个架次能把全部货箱送完。返回 (n_bins, [mask,...])。

    标准「一次填一箱」DP：f[mask] = 装完 mask 中货箱所需的最少架次，
    枚举含最低位的可行子集 sub 作为「最后一箱」，f[mask] = 1 + f[mask^sub]。
    子集枚举总量是 3^n，n=15 时约 1400 万次——必须是纯 Python 位运算，
    用 numpy 标量反而慢十倍。结果按 (容量, 体积) 缓存，返航余量敏感性
    那一轮会反复用同一组容量。
    """
    n = len(mass)
    if n == 0:
        return 0, []
    if n > 18:
        return _ffd(mass, vol, cap_m, cap_v)
    key = (round(cap_m, 6), round(cap_v, 9), max_boxes, tuple(mass), tuple(vol))
    if key in _BINS_CACHE:
        return _BINS_CACHE[key]
    N = 1 << n
    sm = [0.0] * N
    sv = [0.0] * N
    pc = [0] * N
    for mask in range(1, N):
        lb = mask & -mask
        r = mask ^ lb
        i = lb.bit_length() - 1
        sm[mask] = sm[r] + mass[i]
        sv[mask] = sv[r] + vol[i]
        pc[mask] = pc[r] + 1
    feas = bytearray(N)
    for mask in range(1, N):
        if (sm[mask] <= cap_m + 1e-9 and sv[mask] <= cap_v + 1e-9
                and (max_boxes is None or pc[mask] <= max_boxes)):
            feas[mask] = 1
    INF = 99
    f = [INF] * N
    f[0] = 0
    ch = [0] * N
    for mask in range(1, N):
        low = mask & -mask
        best, bsub = INF, 0
        sub = mask
        while sub:
            if (sub & low) and feas[sub]:
                v = f[mask ^ sub] + 1
                if v < best:
                    best, bsub = v, sub
                    if best == 1:
                        break
            sub = (sub - 1) & mask
        f[mask] = best
        ch[mask] = bsub
    if f[N - 1] >= INF:
        # 有货箱单独就超过容量（返航余量放大时会遇到），此时无可行组批；
        # 必须显式返回，否则重建路径时 ch[m]=0 会让 while 死循环。
        out = (10 ** 6, [])
        _BINS_CACHE[key] = out
        return out
    masks, m = [], N - 1
    while m:
        s = ch[m]
        masks.append(s)
        m ^= s
    out = (int(f[N - 1]), masks)
    _BINS_CACHE[key] = out
    return out


def _ffd(mass, vol, cap_m, cap_v):
    idx = sorted(range(len(mass)), key=lambda i: -max(mass[i], vol[i] * 1e3))
    bins = []
    for i in idx:
        for b in bins:
            if b["m"] + mass[i] <= cap_m + 1e-9 and b["v"] + vol[i] <= cap_v + 1e-9:
                b["m"] += mass[i]
                b["v"] += vol[i]
                b["items"].append(i)
                break
        else:
            bins.append(dict(m=mass[i], v=vol[i], items=[i]))
    masks = []
    for b in bins:
        ms = 0
        for i in b["items"]:
            ms |= 1 << i
        masks.append(ms)
    return len(bins), masks


# ══════════════════════════════════════════════════════════
def run(geo, nodes, T, boxes):
    by_site = defaultdict(list)
    for b in boxes:
        by_site[b["site"]].append(b)

    # ── (1) 最大安全载荷 ──
    cap = {}
    for g in ("A", "B", "C"):
        for s in SITES:
            cap[(g, s)] = min(max_payload(g, s, T), T.types[g]["Q"])

    # ── (2)(3) 逐服务区的组批与三目标 ──
    plan = {}
    for s in SITES:
        bs = by_site[s]
        mass = [b["mass"] for b in bs]
        vol = [b["vol"] for b in bs]
        opts = []
        for g in ("A", "B", "C"):
            tt = T.types[g]
            cm = min(cap[(g, s)], tt["Q"])
            cv = tt["vol"]
            n, masks = min_bins(mass, vol, cm, cv)
            if n >= 10 ** 6:
                continue
            groups = [[bs[i] for i in range(len(bs)) if m >> i & 1] for m in masks]
            ss = [sortie(g, s, gp, T) for gp in groups]
            opts.append(dict(g=g, n=n, groups=groups, sorties=ss,
                             E=sum(x["E"] for x in ss),
                             time=sum(x["t"] for x in ss),
                             cap=cm, vol=cv))
        plan[s] = opts
    return cap, plan


def main():
    geo = D.Geo()
    nodes = D.load_nodes()
    geo.set_origin(nodes["O01"]["lon"], nodes["O01"]["lat"])
    types, drones, bats = D.load_drones()
    T = D.Transport(geo, nodes, types)
    boxes = D.load_boxes()
    cap, plan = run(geo, nodes, T, boxes)

    print("=" * 78)
    print("（1）三种机型在各服务区的最大安全载荷 / kg（返航余量 20%）")
    print("=" * 78)
    print(f"{'服务区':<8}{'A(≤25)':>10}{'B(≤30)':>10}{'C(≤80)':>10}   受限机型")
    for s in SITES:
        row = [cap[(g, s)] for g in "ABC"]
        lim = "ABC"[int(np.argmin([row[0] / 25, row[1] / 30, row[2] / 80]))]
        print(f"{s:<8}{row[0]:>10.2f}{row[1]:>10.2f}{row[2]:>10.2f}   受限于 {lim}")

    print()
    print("=" * 78)
    print("（2）（3）逐服务区组批：各机型方案的架次数 / 能耗 / 作业时间")
    print("=" * 78)
    print(f"{'服务区':<8}{'箱数':>5}  " +
          "  ".join(f"{g}:架次/能耗kWh/时间min" for g in "ABC"))
    tot = dict(n={}, E={}, t={})
    for g in "ABC":
        tot["n"][g] = tot["E"][g] = tot["t"][g] = 0.0
    best = {}
    for s in SITES:
        nb = len([b for b in boxes if b["site"] == s])
        line = f"{s:<8}{nb:>5}  "
        cells = []
        for o in plan[s]:
            cells.append(f"{o['n']:>3}/{o['E']:>6.3f}/{o['time']/60:>6.2f}")
        # 推荐：先比架次数，再比能耗，最后比时间
        o = min(plan[s], key=lambda x: (x["n"], x["E"], x["time"]))
        best[s] = o
        tot["n"][o["g"]] += o["n"]
        tot["E"][o["g"]] += o["E"]
        tot["t"][o["g"]] += o["time"]
        line += "  ".join(f"{c:>18}" for c in cells)
        print(line + f"   →选 {o['g']} 机")
    print("-" * 78)
    ns = sum(o["n"] for o in best.values())
    Es = sum(o["E"] for o in best.values())
    ts = sum(o["time"] for o in best.values())
    print(f"推荐方案合计：{ns} 架次，总能耗 {Es:.3f} kWh，"
          f"累计作业时间 {ts/60:.2f} min = {ts/3600:.2f} h")
    print("各机型承担架次：", {g: int(tot['n'][g]) for g in "ABC"})

    # ── (3) 三条优先准则下的总指标（权衡关系）──
    print()
    print("=" * 78)
    print("（3）三种优先准则下的总指标")
    print("=" * 78)
    crit = {
        "最少架次": lambda o: (o["n"], o["E"], o["time"]),
        "最少能耗": lambda o: (o["E"], o["n"], o["time"]),
        "最短作业时间": lambda o: (o["time"], o["n"], o["E"]),
    }
    trade = {}
    for name, key in crit.items():
        pick = {s: min(plan[s], key=key) for s in SITES}
        n_ = sum(o["n"] for o in pick.values())
        E_ = sum(o["E"] for o in pick.values())
        t_ = sum(o["time"] for o in pick.values())
        gs = {g: sum(1 for o in pick.values() if o["g"] == g) for g in "ABC"}
        trade[name] = dict(n=n_, E=E_, t=t_, by_type=gs,
                           by_site={s: pick[s]["g"] for s in SITES})
        print(f"  {name:<8} 架次 {n_:>3}   能耗 {E_:>7.3f} kWh   "
              f"作业时间 {t_/60:>7.2f} min   机型分布 {gs}")
    print()
    print("  权衡的典型例子（S002）：C 机 2 架次 / 11.017 kWh vs "
          "B 机 4 架次 / 10.319 kWh")
    print("  —— 少 2 架次要多吃 0.70 kWh，这就是「架次数」与「能耗」的冲突点。")

    # ── (4) 返航安全余量敏感性 ──
    print()
    print("=" * 78)
    print("（4）返航安全余量 ρ 的影响")
    print("=" * 78)
    print(f"{'ρ':>5}  {'A 最大载荷范围':>18}{'B':>14}{'C':>16}"
          f"   推荐方案架次/能耗")
    sens = {}
    for rr in (0.0, 0.10, 0.20, 0.30, 0.40):
        caps, n_, E_ = {}, 0, 0.0
        for g in "ABC":
            caps[g] = [min(max_payload(g, s, T, rho=rr), T.types[g]["Q"])
                       for s in SITES]
        by_site = defaultdict(list)
        for b in boxes:
            by_site[b["site"]].append(b)
        for s in SITES:
            bs = by_site[s]
            mass = [b["mass"] for b in bs]
            vol = [b["vol"] for b in bs]
            besto = None
            for g in "ABC":
                cm = min(caps[g][SITES.index(s)], T.types[g]["Q"])
                nn, masks = min_bins(mass, vol, cm, T.types[g]["vol"])
                if nn >= 10 ** 6:
                    continue
                groups = [[bs[i] for i in range(len(bs)) if m >> i & 1]
                          for m in masks]
                ss = [sortie(g, s, gp, T, rho=rr) for gp in groups]
                o = dict(g=g, n=nn, E=sum(x["E"] for x in ss))
                if besto is None or (o["n"], o["E"]) < (besto["n"], besto["E"]):
                    besto = o
            if besto:
                n_ += besto["n"]
                E_ += besto["E"]
        rg = lambda v: f"{min(v):.1f}~{max(v):.1f}"
        print(f"{rr*100:>4.0f}%  {rg(caps['A']):>18}{rg(caps['B']):>14}"
              f"{rg(caps['C']):>16}   {n_:>3} 架次 / {E_:.3f} kWh")
        sens[rr] = dict(capA=caps["A"], capB=caps["B"], capC=caps["C"],
                        n=n_, E=E_)

    import json
    res = dict(cap={f"{g}|{s}": cap[(g, s)] for g in "ABC" for s in SITES},
               best={s: dict(g=o["g"], n=o["n"], E=o["E"], t=o["time"],
                             groups=[[b["box"] for b in gp]
                                     for gp in o["groups"]],
                             soc=[sortie(o["g"], s, gp, T)["soc"] * 100
                                  for gp in o["groups"]])
                     for s, o in best.items()},
               trade=trade, sens={str(k): v for k, v in sens.items()},
               totals=dict(n=int(ns), E=Es, t=ts))
    with open(os.path.join(HERE, "out_q1.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("\n已写出 out_q1.json")
    return res


if __name__ == "__main__":
    main()
