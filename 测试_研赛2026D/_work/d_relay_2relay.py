# -*- coding: utf-8 -*-
"""问题三：**能不能只用 2 架中继覆盖全部缺口？**（论文 7.4 节）

## 问题

本文的 3 站方案里，三个站位的缺口区间在某一时刻**同时需要服务**，一架中继同一
时刻只能在一个站位上，所以要 3 架——而库存只有 2 架。但这个"并发 3"是当前站位
与分配下的值，不是下界。两个自由度可以动：

  ① **换站位**：另一组 3 个站位同样能全覆盖，缺口在时间上可能错得开；
  ② **换分配**：一个缺口常被多个站位覆盖，派给谁可以选——把重叠的缺口挪走，
     就可能让某两个站位的服务区间**互不相交**。

## 判据

3 个站位配 2 架中继，必然是"一架负责两个站位 + 一架负责一个"。一架中继要负责
两个站位，那两个站位的服务区间必须错开，**且中间要留够转场飞行时间**：

$$\\text{2 架够用}\\iff\\exists\\ \\text{一对站位}(j,k)\\ \\text{与一种分配}：
\\ \\mathrm{sep}(j,k)\\ \\ge\\ t_{\\text{fly}}(j\\to k)$$

其中 $t_{\\text{fly}}$ 按两站间的水平距离与巡航速度算（本题站间 4.2~8.7 km，
即 277~578 s），$\\mathrm{sep}$ 是两串服务区间之间的最小间隔。

**第一版把转场余量写成固定的 60 s，那是错的**——它和两站的实际距离无关，
会把"其实够不着"的方案判成可行。
"""
from __future__ import annotations

import itertools
import json
import os
import random
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import d_core as D
from d_q2 import World, build_groups, schedule
import d_q3 as Q3
import d_relay

HERE = os.path.dirname(os.path.abspath(__file__))
BUFFER = 1.10          # 转场留 10% 余量


def merge_intervals(ivs):
    out = []
    for a, b in sorted(ivs):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def pair_sep(ivA, ivB):
    """两串区间之间的**最小间隔**（秒）。有交集时返回负的交叠深度。"""
    A, B = merge_intervals(ivA), merge_intervals(ivB)
    if not A or not B:
        return float("inf")
    best = float("inf")
    for a0, a1 in A:
        for b0, b1 in B:
            if a1 < b0:
                best = min(best, b0 - a1)
            elif b1 < a0:
                best = min(best, a0 - b1)
            else:
                best = min(best, -min(a1, b1) + max(a0, b0))
    return best


def flatten(assign, gaps, n_st):
    ivs = [[] for _ in range(n_st)]
    for i, j in enumerate(assign):
        ivs[j].append((gaps[i][0], gaps[i][1]))
    return ivs


def best_pair(assign, gaps, n_st, req):
    """返回 (最优的余量, 是哪一对)。余量 = sep - 所需转场时间。"""
    ivs = flatten(assign, gaps, n_st)
    best = (-1e18, None)
    for a, b in itertools.combinations(range(n_st), 2):
        s = pair_sep(ivs[a], ivs[b]) - req[(a, b)]
        if s > best[0]:
            best = (s, (a, b))
    return best


def climb(gaps, cover, n_st, req, restarts=10, iters=1500, seed=0):
    rnd = random.Random(seed)
    n = len(gaps)
    best = (-1e18, None)
    for r in range(restarts):
        assign = [rnd.choice(cover[i]) for i in range(n)]
        cur, _ = best_pair(assign, gaps, n_st, req)
        for _ in range(iters // restarts):
            i = rnd.randrange(n)
            if len(cover[i]) < 2:
                continue
            old = assign[i]
            assign[i] = rnd.choice([j for j in cover[i] if j != old])
            new, _ = best_pair(assign, gaps, n_st, req)
            if new >= cur:
                cur = new
            else:
                assign[i] = old
        if cur > best[0]:
            best = (cur, list(assign))
    return best


def main():
    W = World()
    rt, W.relays, W.rbats = D.load_relays()
    C = D.Comms(W.geo, W.nodes, D.load_comm())
    g1, p1, e1 = build_groups(W, allow_types=("A", "B"))
    with open(os.path.join(HERE, "out_q3.json"), encoding="utf-8") as f:
        q3 = json.load(f)
    print(f"运输 {len(p1)} 架次；当前站位 {len(q3['stations'])} 个")

    all_gaps = []
    for p in p1:
        for (t0, t1, smp) in Q3.gaps_of(W, C, p):
            all_gaps.append((t0, t1, smp))
    ngap = len(all_gaps)
    cands = Q3.candidate_points(W, C, 20)
    masks = d_relay.cover_masks(W, C, [g[2] for g in all_gaps], cands)
    print(f"缺口 {ngap} 个；候选悬停点 {len(cands)} 个")

    # ── 枚举"能全覆盖"的 3 站集合（在前 40 个候选里取三元组） ──
    top = sorted(range(len(cands)), key=lambda i: -bin(masks[i]).count("1"))[:40]
    full = (1 << ngap) - 1
    triples = []
    for t in itertools.combinations(top, 3):
        m = masks[t[0]] | masks[t[1]] | masks[t[2]]
        if m == full:
            triples.append(t)
    print(f"前 40 个候选中能全覆盖的三元组：{len(triples)} 个")

    def req_of(t):
        r = {}
        for a, b in itertools.combinations(range(3), 2):
            pa, pb = cands[t[a]], cands[t[b]]
            d = W.geo.dist(pa[0], pa[1], pb[0], pb[1])
            r[(a, b)] = d / rt["vc"] * BUFFER
        return r

    def cover_of(t):
        return [[j for j in range(3) if masks[t[j]] >> i & 1] for i in range(ngap)]

    # 第一轮：便宜的贪心估计（按开始时刻最早优先，选当前最空闲的覆盖站位）
    ranked = []
    for t in triples:
        cov = cover_of(t)
        if any(not c for c in cov):
            continue
        req = req_of(t)
        # 贪心：按缺口起点排序，依次派给"能覆盖且当前余量最大"的站位
        order = sorted(range(ngap), key=lambda i: all_gaps[i][0])
        ivs = [[] for _ in range(3)]
        for i in order:
            bestj, bestsep = None, -1e18
            for j in cov[i]:
                trial = ivs[j] + [(all_gaps[i][0], all_gaps[i][1])]
                # 与其它站的分离度取最差的那一对
                worst = min(pair_sep(trial, ivs[k]) - req[(min(j, k), max(j, k))]
                            for k in range(3) if k != j)
                if worst > bestsep:
                    bestsep, bestj = worst, j
            ivs[bestj].append((all_gaps[i][0], all_gaps[i][1]))
        gap = max(pair_sep(ivs[a], ivs[b]) - req[(a, b)]
                  for a, b in itertools.combinations(range(3), 2))
        ranked.append((gap, t))
    ranked.sort(key=lambda x: -x[0])
    print(f"贪心估计最好的 5 个：{[(round(g), t) for g, t in ranked[:5]]}")

    # 第二轮：对最好的若干做爬山
    best = (-1e18, None, None)
    for gap0, t in ranked[:40]:
        cov = cover_of(t)
        req = req_of(t)
        g, assign = climb(all_gaps, cov, 3, req, seed=hash(t) & 0xFFFF)
        if g > best[0]:
            best = (g, t, assign)
    g, t, assign = best
    print()
    print("=" * 78)
    print(f"在 {len(triples)} 个能全覆盖的 3 站集合上搜索（贪心 + 爬山）")
    print("=" * 78)
    print(f"  最好余量 = {g:+.0f} s"
          f"（= 服务区间的最小间隔 − 该对站位的转场飞行时间）")
    ivs = flatten(assign, all_gaps, 3)
    for a, b in itertools.combinations(range(3), 2):
        pa, pb = cands[t[a]], cands[t[b]]
        d = W.geo.dist(pa[0], pa[1], pb[0], pb[1])
        need = d / rt["vc"] * BUFFER
        s = pair_sep(ivs[a], ivs[b])
        print(f"    站{a+1} × 站{b+1}：间隔 {s:+.0f} s，需转场 {need:.0f} s"
              f"（{d/1000:.1f} km） → {'够' if s >= need else '不够'}")
    if g >= 0:
        print(f"\n  → **2 架中继够用**：站{best[1]} 的那一对可以共用一架")
    else:
        print(f"\n  → **2 架不够**：所枚举的能全覆盖的 3 站集合里，**没有任何一对**"
              f"站位的服务区间能错开到转场所需的 {abs(g):.0f} s 以上")
    for j in range(3):
        m = merge_intervals(ivs[j])
        print(f"    站{j+1}：{len(ivs[j])} 个缺口 → {len(m)} 段，合计 "
              f"{sum(b-a for a, b in m):.0f} s，点 "
              f"({cands[t[j]][0]:.4f},{cands[t[j]][1]:.4f},{cands[t[j]][2]:.0f})")

    out = dict(n_gap=ngap, n_triples=len(triples), best_slack=g,
               two_relay_enough=bool(g >= 0),
               points=[[cands[i][0], cands[i][1], cands[i][2]] for i in t],
               assign=assign,
               per_station=[[list(x) for x in merge_intervals(ivs[j])]
                            for j in range(3)])
    with open(os.path.join(HERE, "out_relay_2relay.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print("\n已写出 out_relay_2relay.json")


if __name__ == "__main__":
    main()
