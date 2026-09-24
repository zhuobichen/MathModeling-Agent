# -*- coding: utf-8 -*-
"""问题三：通信约束下的运输与中继联合调度。

运输无人机在爬升/巡航/下降/投送**全过程**都要保持通信：与 G01 直连可用记
「直连」，否则要靠一架中继（U—R 接入 + R—G01 回传**同时**可用）。

步骤：
  1. 把问题二的每一条运输架次展开成随时间变化的三维位置，逐时刻判直连，
     得到「通信缺口」时段（实测 30.6% 的采样点无直连，28/41 条架次有缺口）；
  2. 候选悬停点取 DEM 里的高地格点（离地固定 300 m，越高越容易越过山体）；
  3. 每个缺口时段挑一个候选点使 U—R 与 R—G01 同时可用；
  4. 把「(悬停点, 时段)」需求排给 2 架中继无人机：按时间顺序贪心，
     同一悬停点上连续的需求合并成一段驻留，含往返飞行与能源组件充电周转。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import d_core as D
from d_q2 import World, TYPES, SITES, initial_groups, merge_pass, schedule, evaluate

HERE = os.path.dirname(os.path.abspath(__file__))
STEP = 10.0
HOVER_AGL = 300.0


# ══════════════════════════════════════════════════════════
def leg_track(W, g, i, j, q, t0):
    """一个航段的三维轨迹：爬升 → 水平巡航 → 下降，各段匀速。"""
    T = W.T
    a, b = W.nodes[i], W.nodes[j]
    d = W.geo.dist(a["lon"], a["lat"], b["lon"], b["lat"])
    zc = T.cruise_alt(i, j)
    z0, z1 = T.op_alt(i), T.op_alt(j)
    h_up, h_dn = max(zc - z0, 0.0), max(zc - z1, 0.0)
    vc = T.types[g]["vc"]
    tu, tc, td = (h_up / T.types[g]["v_up"], d / vc,
                  h_dn / T.types[g]["v_dn"])
    pts = []
    n_up = max(int(tu / STEP), 1)
    for k in range(n_up):
        f = k / n_up
        pts.append((t0 + f * tu, a["lon"], a["lat"], z0 + h_up * f))
    n_c = max(int(tc / STEP), 1)
    for k in range(n_c + 1):
        f = k / n_c
        pts.append((t0 + tu + f * tc, a["lon"] + (b["lon"] - a["lon"]) * f,
                    a["lat"] + (b["lat"] - a["lat"]) * f, zc))
    n_d = max(int(td / STEP), 1)
    for k in range(1, n_d + 1):
        f = k / n_d
        pts.append((t0 + tu + tc + f * td, b["lon"], b["lat"], zc - h_dn * f))
    return pts, tu + tc + td


def sortie_track(W, p):
    """把一条排好程的运输架次展开成 [(t, lon, lat, z, phase)]。"""
    g, tt = p["g"], W.types[p["g"]]
    pts = []
    t = p["start"] + tt["t_prep"] + len(p["boxes"]) * tt["t_load"]
    node, q = "O01", sum(b["mass"] for _, gp in p["stops"] for b in gp)
    for s, gp in p["stops"]:
        seg, dur = leg_track(W, g, node, s, q, t)
        pts += [(x[0], x[1], x[2], x[3], "fly") for x in seg]
        t += dur
        pts.append((t, W.nodes[s]["lon"], W.nodes[s]["lat"], W.T.op_alt(s),
                    "hover"))
        t += tt["t_hand0"] + len(gp) * tt["t_hand1"]
        q -= sum(b["mass"] for b in gp)
        node = s
    seg, dur = leg_track(W, g, node, "O01", 0.0, t)
    pts += [(x[0], x[1], x[2], x[3], "fly") for x in seg]
    return pts


# ══════════════════════════════════════════════════════════
def gw_pos(W, C):
    return (W.nodes["O01"]["lon"], W.nodes["O01"]["lat"],
            W.nodes["O01"]["alt"] + C.h_gw)


def direct_ok(C, W, lon, lat, z):
    return C.link_ok("U", (lon, lat, z), "G01", gw_pos(W, C))[0]


def relay_covers(C, W, u, p):
    g01 = gw_pos(W, C)
    return (C.link_ok("U", u, "Ra", p)[0]
            and C.link_ok("Rb", p, "G01", g01)[0])


def candidate_points(W, C, n_side=26):
    """候选悬停点：DEM 范围内按高程取前若干高地格点（离地固定 300 m）。

    只取高地是因为悬停高度越低越容易被山体挡住；离地 300 m 是题目给的上限，
    统一取上限即可，剩下的自由度只有平面位置。
    """
    geo = W.geo
    dem = geo.dem
    rows, cols = np.where(dem > geo.nodata + 1)
    z = dem[rows, cols]
    lon = geo.lon0 + cols * geo.dlon
    lat = geo.lat0 - rows * geo.dlat
    # 粗网格化：每个约 1 km 的格子里取最高点，保证空间铺开（键要转成元组）
    kx = (lon * 100).astype(int)
    ky = (lat * 100).astype(int)
    best = {}
    for a, b2, zz, lo, la in zip(kx, ky, z, lon, lat):
        k = (int(a), int(b2))
        if k not in best or zz > best[k][2]:
            best[k] = (float(lo), float(la), float(zz))
    pts = sorted(best.values(), key=lambda x: -x[2])[:n_side * n_side]
    return [(float(lo), float(la), float(zz + HOVER_AGL)) for lo, la, zz in pts]


# ══════════════════════════════════════════════════════════
def gaps_of(W, C, p):
    """一条运输架次的通信缺口：返回 [(t0, t1, [(t,lon,lat,z), ...]), ...]。"""
    tr = sortie_track(W, p)
    bad = [(t, lo, la, z) for (t, lo, la, z, ph) in tr
           if not direct_ok(C, W, lo, la, z)]
    if not bad:
        return []
    out, cur = [], [bad[0]]
    for x in bad[1:]:
        if x[0] - cur[-1][0] <= STEP * 1.5:
            cur.append(x)
        else:
            out.append(cur)
            cur = [x]
    out.append(cur)
    return [(c[0][0], c[-1][0] + STEP, c) for c in out]


def pick_point(W, C, samples, cands, rt=None):
    """为一个缺口挑悬停点。

    评分 = 中继从 O01 飞到该点的时间（不是到缺口的水平距离）：中继必须在
    缺口**开始之前**到点。按距离挑会让点偏远、到场迟 40~100 s，通信保障
    就断在最开始那一段。
    """
    scored = []
    for p in cands:
        if not all(relay_covers(C, W, (s[1], s[2], s[3]), p) for s in samples):
            continue
        if rt is not None:
            tt = relay_flight(W, rt, p)[0]
        else:
            tt = W.geo.dist(W.nodes["O01"]["lon"], W.nodes["O01"]["lat"],
                            p[0], p[1])
        scored.append((tt, p))
    if not scored:
        return None
    return min(scored)[1]


# ══════════════════════════════════════════════════════════
def relay_flight(W, rt, point):
    """中继往返一个悬停点的时间与能耗（题目附录 2）。"""
    o = W.nodes["O01"]
    zc = W.geo.elev_line(o["lon"], o["lat"], point[0], point[1]) + 50.0
    d = W.geo.dist(o["lon"], o["lat"], point[0], point[1])
    z0 = o["alt"]
    h_up = max(zc - z0, 0.0)
    h_dn = max(zc - point[2], 0.0)
    t_fly = h_up / rt["v_up"] + d / rt["vc"] + h_dn / rt["v_dn"]
    # 水平能耗 = 巡航功率 × 巡航时间；爬升附加 = 势能/效率
    E_hor = rt["P_cruise"] * (d / rt["vc"]) / 3600.0
    E_up = rt["m_takeoff"] * 9.81 * h_up / rt["eta_up"] / 3.6e6
    return t_fly, E_hor + E_up


def cluster_stations(W, demands, k):
    """把各缺口的悬停点聚成 k 个中继站位（贪心最大最小距离）。"""
    pts = [d[0] for d in demands]
    uniq = list({(round(p[0], 5), round(p[1], 5), round(p[2], 1)) for p in pts})
    if k >= len(uniq):
        return list(uniq)
    centers = [uniq[0]]
    while len(centers) < k:
        best, bd = None, -1
        for p in uniq:
            d = min(W.geo.dist(p[0], p[1], c[0], c[1]) for c in centers)
            if d > bd:
                bd, best = d, p
        centers.append(best)
    # 用一轮 k-means 式再分配
    for _ in range(8):
        groups = [[] for _ in centers]
        for p in uniq:
            i = min(range(len(centers)),
                    key=lambda j: W.geo.dist(p[0], p[1], centers[j][0],
                                             centers[j][1]))
            groups[i].append(p)
        new_c = []
        for j, gp in enumerate(groups):
            if not gp:
                new_c.append(centers[j])
                continue
            lo = sum(x[0] for x in gp) / len(gp)
            la = sum(x[1] for x in gp) / len(gp)
            zz = max(x[2] for x in gp)
            new_c.append((lo, la, zz))
        centers = new_c
    return centers


def station_plan(W, rt, demands, centers, C=None):
    """按站位排中继：每站驻留覆盖分配给它的缺口时间窗。

    与「每个缺口飞一趟」相比，驻留把 62 次往返压成几段长驻留——后者在
    2 架中继上是根本排不下的（62×约 1000s 的往返 / 2 架 ≫ 运输窗口）。
    """
    assign = {i: [] for i in range(len(centers))}
    for (pt, t0, t1) in demands:
        i = min(range(len(centers)),
                key=lambda j: W.geo.dist(pt[0], pt[1], centers[j][0],
                                        centers[j][1]))
        assign[i].append((pt, t0, t1))
    out = []
    for i, c in enumerate(centers):
        ds = assign[i]
        if not ds:
            continue
        # 站位用簇内最高的那个悬停点，保证仍能覆盖簇内全部缺口
        st = max([d[0] for d in ds], key=lambda p: p[2])
        # 必须逐条验证该站位真能覆盖分到的缺口——聚类只负责分组，
        # 不代表站位点仍在几何上覆盖得住
        cov = sum(1 for (pt, t0, t1) in ds if relay_covers(C, W, pt, st))
        t_fly, e_fly = relay_flight(W, rt, st)
        win = (min(d[1] for d in ds), max(d[2] for d in ds))
        hover = win[1] - win[0]
        E = e_fly * 2 + (rt["P_hover"] + rt["P_comm"]) * hover / 3600.0
        usable = (1 - rt["rho"]) * rt["E_use"]
        n_bat = max(1, int(np.ceil(E / usable)))
        out.append(dict(station=i, point=st, n_gap=len(ds), n_cover=cov,
                        win=win, hover=hover, E=E, n_bat=n_bat, t_fly=t_fly,
                        t_out=rt["t_prep"] + t_fly, t_back=win[1] + t_fly))
    return out


def main():
    W = World()
    rt, W.relays, W.rbats = D.load_relays()
    C = D.Comms(W.geo, W.nodes, D.load_comm())

    g1 = merge_pass(W, initial_groups(W), allow_types=("A", "B"))
    p1 = schedule(W, g1)
    e1 = evaluate(W, p1)

    print("=" * 78)
    print("第一步：直连链路的缺口")
    print("=" * 78)
    all_gaps = []
    for k, p in enumerate(p1):
        gs = gaps_of(W, C, p)
        for (t0, t1, smp) in gs:
            all_gaps.append((k, t0, t1, smp, p))
    tot_pts = sum(len(sortie_track(W, p)) for p in p1)
    bad_pts = sum(len(s) for _, _, _, s, _ in all_gaps)
    print(f"  运输架次 {len(p1)} 条；采样点 {tot_pts}；无直连 {bad_pts}"
          f"（{bad_pts/tot_pts*100:.1f}%）；缺口时段 {len(all_gaps)} 段")

    print()
    print("=" * 78)
    print("第二步：中继悬停点")
    print("=" * 78)
    cands = candidate_points(W, C, n_side=20)
    print(f"  候选悬停点 {len(cands)} 个（每约 1 km 格子的最高点，离地 {HOVER_AGL:.0f} m）")
    demands, picked = [], 0
    for (k, t0, t1, smp, p) in all_gaps:
        pt = pick_point(W, C, smp, cands, rt)
        if pt is None:
            print(f"    ✗ 架次{k} 时段 {t0:.0f}-{t1:.0f} 未找到覆盖点")
            continue
        picked += 1
        demands.append((pt, t0, t1))
    print(f"  覆盖 {picked}/{len(all_gaps)} 个缺口时段")

    # ── 中继必须先到点：算出运输需要整体推迟多少才来得及 ──
    delta = 0.0
    for (pt, t0, t1) in demands:
        tf = relay_flight(W, rt, pt)[0]
        delta = max(delta, rt["t_prep"] + tf - t0)
    delta = max(delta, 0.0)
    if delta > 0:
        print(f"  中继到场最早需 {rt['t_prep']:.0f}s(准备) + 飞行；"
              f"为保证运输全程有链路，运输整体推迟 {delta:.0f}s 起飞"
              f"（中继先到点）")
        for pp in p1:
            pp["start"] += delta
            pp["end"] += delta
            pp["deliv"] = {k: v + delta for k, v in pp["deliv"].items()}
        all_gaps2 = []
        for k2, pp in enumerate(p1):
            for (g0, g1, smp) in gaps_of(W, C, pp):
                all_gaps2.append((g0, g1, smp))
        demands = []
        for (g0, g1, smp) in all_gaps2:
            pt = pick_point(W, C, smp, cands, rt)
            if pt is not None:
                demands.append((pt, g0, g1))
        e1 = evaluate(W, p1)
        print(f"  推迟后：完成 {e1['t_done']:.0f}s，首批违约 "
              f"{e1['viol_first']}，医疗违约 {e1['viol_med']}，"
              f"缺口时段 {len(demands)}")

    print()
    print("=" * 78)
    print("第三步：按站位驻留的中继排程")
    print("=" * 78)
    print("  各站位：只用一个悬停点、在覆盖窗口内持续驻留；"
          "能源组件不足时按需换电")
    sweep = {}
    best_plan = None
    for nr in (1, 2, 3, 4):
        centers = cluster_stations(W, demands, nr)
        st = station_plan(W, rt, demands, centers, C)
        E = sum(x["E"] for x in st)
        nbat = sum(x["n_bat"] for x in st)
        t_joint = max(max(p["end"] for p in p1),
                      max([x["t_back"] for x in st] or [0]))
        ncov = sum(x["n_cover"] for x in st)
        sweep[nr] = dict(n_station=len(st), E=E, n_bat=nbat, t_joint=t_joint,
                         hover=sum(x["hover"] for x in st),
                         n_gap=sum(x["n_gap"] for x in st), n_cover=ncov)
        print(f"  {nr} 站：覆盖 {ncov}/{len(demands)} 个缺口 | "
              f"驻留窗口合计 {sum(x['hover'] for x in st):>6.0f}s | "
              f"中继能耗 {E:>6.3f} kWh | 需能源组件 {nbat} 个（库 6） | "
              f"联合完成 {t_joint:>6.0f}s")
        # 只在**全覆盖**的方案里选能耗最低的；覆盖不全的方案再便宜也不能用
        if ncov == len(demands) and (best_plan is None or E < best_plan[0]):
            best_plan = (E, nr, st, centers)
    if best_plan is None:
        raise RuntimeError("现有中继数量下无法全覆盖")
    E, nr, st, centers = best_plan
    print(f"  → 取 {nr} 站方案（全覆盖中能耗最低）")
    print()
    print(f"{'站':>3}{'缺口':>6}{'覆盖':>6}{'覆盖窗口/s':>18}{'驻留/s':>9}"
          f"{'能耗/kWh':>10}{'组件':>5}   悬停点 (经,纬,海拔)")
    for x in st:
        w = "{:.0f}-{:.0f}".format(x["win"][0], x["win"][1])
        print(f"{x['station']+1:>3}{x['n_gap']:>6}{x['n_cover']:>6}{w:>18}"
              f"{x['hover']:>9.0f}"
              f"{x['E']:>10.3f}{x['n_bat']:>5}   "
              f"({x['point'][0]:.4f},{x['point'][1]:.4f},{x['point'][2]:.0f})")

    res = dict(n_sorties=len(p1), n_gaps=len(all_gaps), bad_pts=bad_pts,
               tot_pts=tot_pts, coverage=picked / max(len(all_gaps), 1),
               relay=[dict(station=x["station"], lon=x["point"][0],
                           lat=x["point"][1], alt=x["point"][2],
                           t_out=x["t_out"], t_back=x["t_back"],
                           win=list(x["win"]), hover=x["hover"],
                           E=x["E"], n_bat=x["n_bat"], n_gap=x["n_gap"],
                           n_cover=x["n_cover"]) for x in st],
               relay_E=E, uncovered=len(demands)-sum(x["n_cover"] for x in st),
               t_joint=max(max(p["end"] for p in p1),
                           max([x["t_back"] for x in st] or [0])),
               t_transport=e1["t_done"], transport_E=e1["E"],
               transport_sorties=len(p1), sweep=sweep,
               stations=st)
    with open(os.path.join(HERE, "out_q3.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    print("\n已写出 out_q3.json")
    return res


if __name__ == "__main__":
    main()
