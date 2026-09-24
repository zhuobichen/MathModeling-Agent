# -*- coding: utf-8 -*-
"""中继悬停点的**连续平面搜索**：一个站位到底能不能覆盖全部缺口？

本文原来的做法（d_q3.candidate_points）只取「每约 1 km 格子里 DEM 最高的那个点」
——那是在**高地集合**里挑，不是在**平面**上优化。坑在于：覆盖缺口的判据是
"中继同时看得见缺口和网关"，它由**几何连线**决定，最高点未必是最会"看"的点
（横向位置同样重要）。

于是本文把站位位置当成连续变量，做两阶段局部搜索：

  ① 在原有 400 个高地候选里按覆盖数排序，取前若干个；
  ② 在每个好点周围 ±400 m 的平面上按 100 m 步长细搜，取覆盖数最大者。

这一步直接改变问题三与问题四的结论：**如果 1 个站位就能全覆盖，问题三只需
1 架中继（能耗接近减半），问题四的中继缺口也跟着缩小。**
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import d_core as D
from d_q2 import World, initial_groups, merge_pass, schedule
import d_q3 as Q3

HERE = os.path.dirname(os.path.abspath(__file__))
HOVER = Q3.HOVER_AGL


def gap_samples(W, C, p1):
    """全部通信缺口，每个缺口带自己的采样点集合。"""
    out = []
    for p in p1:
        for (t0, t1, smp) in Q3.gaps_of(W, C, p):
            out.append((t0, t1, smp))
    return out


def n_covered(W, C, gaps, pt):
    """一个悬停点能覆盖多少个**完整缺口**（缺口的每个采样点都要能建链）。"""
    n = 0
    for (t0, t1, smp) in gaps:
        if all(Q3.relay_covers(C, W, (s[1], s[2], s[3]), pt) for s in smp):
            n += 1
    return n


def pt_at(W, lon, lat):
    """在 (lon, lat) 处、离地 300 m 的悬停点。"""
    return (lon, lat, W.geo.elev(lon, lat) + HOVER)


def main():
    W = World()
    rt, W.relays, W.rbats = D.load_relays()
    C = D.Comms(W.geo, W.nodes, D.load_comm())
    g1 = merge_pass(W, initial_groups(W), allow_types=("A", "B"))
    p1 = schedule(W, g1)
    gaps = gap_samples(W, C, p1)
    print(f"缺口总数 {len(gaps)}；采样点合计 {sum(len(g[2]) for g in gaps)}")

    cands = Q3.candidate_points(W, C, 20)
    print(f"原有高地候选 {len(cands)} 个")

    t0 = time.time()
    scored = sorted(((n_covered(W, C, gaps, c), c) for c in cands),
                    key=lambda x: -x[0])
    print(f"  逐个评估耗时 {time.time() - t0:.1f} s；"
          f"覆盖数分布 {sorted({s[0] for s in scored}, reverse=True)[:8]} ...")
    print(f"  最好的高地候选覆盖 {scored[0][0]}/{len(gaps)}")

    # ② 在最好的几个点周围做平面细搜
    print()
    print("平面细搜（每点 ±400 m，100 m 步长，离地保持 300 m）")
    best = (scored[0][0], scored[0][1], "高地候选")
    t0 = time.time()
    for cov, c in scored[:5]:
        lo0, la0 = c[0], c[1]
        for dx in np.arange(-400.0, 400.1, 100.0):
            for dy in np.arange(-400.0, 400.1, 100.0):
                # 经度按纬度缩尺换算成米
                k = 111320.0
                lo = lo0 + dx / (k * np.cos(np.radians(la0)))
                la = la0 + dy / k
                pt = pt_at(W, lo, la)
                n = n_covered(W, C, gaps, pt)
                if n > best[0]:
                    best = (n, pt, f"细搜自 ({lo0:.4f},{la0:.4f}) 偏移 ({dx:+.0f},{dy:+.0f}) m")
    print(f"  细搜耗时 {time.time() - t0:.1f} s")

    n, pt, how = best
    print()
    print("=" * 78)
    print("结果")
    print("=" * 78)
    print(f"  单站位最大覆盖：**{n}/{len(gaps)}**")
    print(f"  悬停点：经 {pt[0]:.5f}° 纬 {pt[1]:.5f}° 海拔 {pt[2]:.1f} m"
          f"（地面 {pt[2] - HOVER:.1f} m，离地 {HOVER:.0f} m）")
    print(f"  来源：{how}")
    if n == len(gaps):
        print("  → **一个站位即可全覆盖**：问题三只需 1 架中继。")
    else:
        print(f"  → 单站位**无法**全覆盖（差 {len(gaps) - n} 个缺口），"
              f"问题三的 2 站方案是必要的。")

    # 对照：原来的 1 站方案覆盖多少
    st1 = Q3.station_plan(W, rt, [(Q3.pick_point(W, C, s, cands, rt), a, b)
                                  for (a, b, s) in gaps if
                                  Q3.pick_point(W, C, s, cands, rt)],
                          Q3.cluster_stations(
                              W, [(Q3.pick_point(W, C, s, cands, rt), a, b)
                                  for (a, b, s) in gaps if
                                  Q3.pick_point(W, C, s, cands, rt)], 1), C)
    print(f"  对照·d_q3 原有 1 站方案覆盖 {st1[0]['n_cover']}/{len(gaps)}，"
          f"能耗 {st1[0]['E']:.3f} kWh")

    out = dict(n_gap=len(gaps), best_cover=n,
               best_point=[float(pt[0]), float(pt[1]), float(pt[2])],
               how=how,
               old_single=dict(cover=int(st1[0]["n_cover"]),
                               E=float(st1[0]["E"])))
    with open(os.path.join(HERE, "out_relay_opt.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n已写出 out_relay_opt.json")


if __name__ == "__main__":
    main()
