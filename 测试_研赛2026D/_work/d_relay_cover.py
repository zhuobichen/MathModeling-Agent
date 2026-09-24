# -*- coding: utf-8 -*-
"""通信覆盖的**修正核算**：把覆盖面重新算一遍。

发现的问题：`d_q3.station_plan` 里验证站位覆盖用的是

    for (pt, t0, t1) in ds:  if relay_covers(C, W, pt, st)

其中 `pt` 是该缺口**自己挑的那个悬停点**（离地 300 m 的高点），不是无人机
的真实位置。而 `relay_covers(C, W, u, p)` 的第一个参数是**无人机位置**
（它判的是 U—Ra 链路）。拿一个"高高在上的悬停点"当无人机位置来判链路，
会比真实情况容易通过——**于是覆盖数被高估**。

正确的判据只有一条：**缺口里的每一个采样点（无人机的真实位置）都要能被
某个站位建立起 U—Ra—G01 的链路。** 本脚本按这条重算，并对站位做重新搜索。
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import d_core as D
from d_q2 import World, initial_groups, merge_pass, schedule
import d_q3 as Q3

HERE = os.path.dirname(os.path.abspath(__file__))


def gaps(W, C, p1):
    out = []
    for p in p1:
        for (t0, t1, smp) in Q3.gaps_of(W, C, p):
            out.append((t0, t1, smp))
    return out


def covers_point(W, C, smp, pt):
    return all(Q3.relay_covers(C, W, (s[1], s[2], s[3]), pt) for s in smp)


def covered_by_set(W, C, gs, pts):
    """多少个缺口能被 pts 中**至少一个**站位完整覆盖。"""
    return sum(1 for (a, b, smp) in gs
               if any(covers_point(W, C, smp, pt) for pt in pts))


def main():
    W = World()
    rt, W.relays, W.rbats = D.load_relays()
    C = D.Comms(W.geo, W.nodes, D.load_comm())
    g1 = merge_pass(W, initial_groups(W), allow_types=("A", "B"))
    p1 = schedule(W, g1)
    gs = gaps(W, C, p1)
    print(f"缺口 {len(gs)} 个，采样点 {sum(len(g[2]) for g in gs)} 个")

    # 现有 2 站方案的两个悬停点（来自当前 out_q3.json）
    cur = json.load(open(os.path.join(HERE, "out_q3.json"), encoding="utf-8"))
    pts_cur = [tuple(s["point"]) for s in cur["stations"]]
    print()
    print("=" * 78)
    print("对照：现有 2 站方案，按**修正判据**重算覆盖")
    print("=" * 78)
    for i, pt in enumerate(pts_cur, 1):
        print(f"  站{i} 单独覆盖 {sum(1 for (a,b,smp) in gs if covers_point(W,C,smp,pt))}"
              f"/{len(gs)}   点 ({pt[0]:.4f}, {pt[1]:.4f}, {pt[2]:.1f})")
    print(f"  两站合计覆盖 **{covered_by_set(W, C, gs, pts_cur)}/{len(gs)}**"
          f"（原报表写的是 {cur['n_gaps']}/{cur['n_gaps']}）")

    # 重新搜站：候选池 = 400 个高地 + 细搜邻域
    cands = Q3.candidate_points(W, C, 20)
    sc = sorted(((sum(1 for (a, b, smp) in gs if covers_point(W, C, smp, c)), c)
                 for c in cands), key=lambda x: -x[0])
    print()
    print("=" * 78)
    print("重新搜索站位（先取覆盖数最高的一批候选，再两两贪心）")
    print("=" * 78)
    print(f"  候选 {len(cands)} 个；最高覆盖 {sc[0][0]}/{len(gs)}")
    print(f"  覆盖数前 10：{[x[0] for x in sc[:10]]}")

    top = [c for _, c in sc[:60]]
    best = None
    for a, b in itertools.combinations(top, 2):
        n = covered_by_set(W, C, gs, [a, b])
        if best is None or n > best[0]:
            best = (n, a, b)
    n2, pa, pb = best
    print(f"  两站最优：**{n2}/{len(gs)}**")
    for i, pt in enumerate((pa, pb), 1):
        print(f"    站{i} 点 ({pt[0]:.4f}, {pt[1]:.4f}, {pt[2]:.1f})，"
              f"单独覆盖 {sum(1 for (a,b2,s2) in gs if covers_point(W,C,s2,pt))}")

    # 贪心加到全覆盖，看看几个站才够
    print()
    print("  贪心增站（每次取能多覆盖最多的那个）")
    sel, remaining, hist = [], list(range(len(gs))), []
    while remaining and len(sel) < 8:
        bestc = None
        for c in cands:
            k = sum(1 for i in remaining if covers_point(W, C, gs[i][2], c))
            if bestc is None or k > bestc[0]:
                bestc = (k, c)
        if bestc[0] == 0:
            break
        sel.append(bestc[1])
        remaining = [i for i in remaining
                     if not covers_point(W, C, gs[i][2], bestc[1])]
        hist.append(len(gs) - len(remaining))
        print(f"    加第 {len(sel)} 站 → 累计覆盖 {len(gs) - len(remaining)}/{len(gs)}")
    print(f"  → **{len(sel)} 站可全覆盖**；剩 {len(remaining)} 个缺口覆盖不到")

    out = dict(n_gap=len(gs), n_station_greedy=len(sel),
               cov_hist=hist,
               cur_two_station=covered_by_set(W, C, gs, pts_cur),
               best_two=n2, best_two_pts=[list(pa), list(pb)],
               old_reported=cur["n_gaps"])
    with open(os.path.join(HERE, "out_relay_cover.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n已写出 out_relay_cover.json")


if __name__ == "__main__":
    main()
