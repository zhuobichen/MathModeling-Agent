# -*- coding: utf-8 -*-
"""问题三续：**4 个站位配 2 架中继**行不行？（论文 7.4 节的第二个自由度）

3 站配 2 架需要"有一对站位的区间能错开"；上面搜了 609 个能全覆盖的三元组，
最好的一对只错开 55 s，而两站之间转场要 179 s —— 差 125 s。

但还有一条路没试：**多设一个站位**。站位多了，每个站分摊到的缺口更少、区间更
稀疏，两两错开的机会反而更大。此时 2 架中继 = 把 4 个站位**配成两对**，
每对内的两个站位区间必须错开（各自留够转场）。

$$\\text{2 架够用}\\iff\\exists\\ \\text{4 个站位的配对}\\{(a,b),(c,d)\\}:\\
\\mathrm{sep}(a,b)\\ge t_{\\text{fly}}(a\\!\\to\\!b)\\ \\wedge\\
\\mathrm{sep}(c,d)\\ge t_{\\text{fly}}(c\\!\\to\\!d)$$

配对的三种分法（4 个里挑 2 个做一对，剩下一对）都要试。
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
from d_q2 import World, build_groups
import d_q3 as Q3
import d_relay
from d_relay_2relay import merge_intervals, pair_sep, BUFFER

HERE = os.path.dirname(os.path.abspath(__file__))

PAIRINGS4 = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]


def pairing_slack(ivs, req):
    """4 个站位配对成两对，返回最好那一种配法的余量（取两对里较差的那对）。"""
    best = -1e18
    for pa, pb in PAIRINGS4:
        s = min(pair_sep(ivs[pa[0]], ivs[pa[1]]) - req[pa],
                pair_sep(ivs[pb[0]], ivs[pb[1]]) - req[pb])
        best = max(best, s)
    return best


def climb4(gaps, cover, req, restarts=8, iters=1200, seed=0):
    rnd = random.Random(seed)
    n = len(gaps)
    best = (-1e18, None)
    for _ in range(restarts):
        assign = [rnd.choice(cover[i]) for i in range(n)]
        ivs = [[] for _ in range(4)]
        for i, j in enumerate(assign):
            ivs[j].append((gaps[i][0], gaps[i][1]))
        cur = pairing_slack(ivs, req)
        for _ in range(iters // restarts):
            i = rnd.randrange(n)
            if len(cover[i]) < 2:
                continue
            old = assign[i]
            assign[i] = rnd.choice([j for j in cover[i] if j != old])
            ivs = [[] for _ in range(4)]
            for k, j in enumerate(assign):
                ivs[j].append((gaps[k][0], gaps[k][1]))
            new = pairing_slack(ivs, req)
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
    all_gaps = []
    for p in p1:
        for (t0, t1, smp) in Q3.gaps_of(W, C, p):
            all_gaps.append((t0, t1, smp))
    ngap = len(all_gaps)
    cands = Q3.candidate_points(W, C, 20)
    masks = d_relay.cover_masks(W, C, [g[2] for g in all_gaps], cands)
    full = (1 << ngap) - 1

    # 起点：3 站搜索里最好的那一组（站1 是大簇，站2/3 是可分离的那对）
    with open(os.path.join(HERE, "out_relay_2relay.json"), encoding="utf-8") as f:
        prev = json.load(f)
    base3 = [cands.index(tuple(p)) for p in prev["points"]]
    print(f"基础 3 站：{base3}，覆盖 "
          f"{bin(masks[base3[0]] | masks[base3[1]] | masks[base3[2]]).count('1')}/{ngap}")

    top = sorted(range(len(cands)), key=lambda i: -bin(masks[i]).count("1"))[:40]
    tried = []
    for c4 in top:
        if c4 in base3:
            continue
        t = list(base3) + [c4]
        if (masks[t[0]] | masks[t[1]] | masks[t[2]] | masks[t[3]]) != full:
            continue
        tried.append(t)
    print(f"加上第 4 个站位后仍能全覆盖的候选：{len(tried)} 个（来自前 40 个候选点）")

    best = (-1e18, None, None)
    for t in tried:
        cov = [[j for j in range(4) if masks[t[j]] >> i & 1] for i in range(ngap)]
        if any(not c for c in cov):
            continue
        req = {}
        for a, b in itertools.combinations(range(4), 2):
            d = W.geo.dist(cands[t[a]][0], cands[t[a]][1],
                           cands[t[b]][0], cands[t[b]][1])
            req[(a, b)] = d / rt["vc"] * BUFFER
        g, assign = climb4(all_gaps, cov, req, seed=sum(t) * 7919 % 65536)
        if g > best[0]:
            best = (g, t, assign)
    g, t, assign = best
    print()
    print("=" * 78)
    print(f"4 站 × 2 架：在 {len(tried)} 个组合上搜索（爬山）")
    print("=" * 78)
    print(f"  最好余量 = {g:+.0f} s")
    ivs = [[] for _ in range(4)]
    for i, j in enumerate(assign):
        ivs[j].append((all_gaps[i][0], all_gaps[i][1]))
    for pa, pb in PAIRINGS4:
        sa = pair_sep(ivs[pa[0]], ivs[pa[1]]) - req_of_pair(W, rt, cands, t, pa)
        sb = pair_sep(ivs[pb[0]], ivs[pb[1]]) - req_of_pair(W, rt, cands, t, pb)
        print(f"    配对 ({pa[0]+1},{pa[1]+1})+({pb[0]+1},{pb[1]+1})："
              f"余量 {min(sa, sb):+.0f} s")
    if g >= 0:
        print(f"\n  → **4 站 + 2 架中继够用**")
    else:
        print(f"\n  → **4 站 + 2 架也不够**：最好的配对里仍差 {-g:.0f} s")
    for j in range(4):
        m = merge_intervals(ivs[j])
        print(f"    站{j+1}：{len(ivs[j])} 个缺口 → {len(m)} 段，合计 "
              f"{sum(b-a for a, b in m):.0f} s，点 "
              f"({cands[t[j]][0]:.4f},{cands[t[j]][1]:.4f},{cands[t[j]][2]:.0f})")

    out = dict(n_gap=ngap, n_tried=len(tried), best_slack=g,
               two_relay_enough=bool(g >= 0),
               points=[[cands[i][0], cands[i][1], cands[i][2]] for i in t],
               per_station=[[list(x) for x in merge_intervals(ivs[j])]
                            for j in range(4)])
    with open(os.path.join(HERE, "out_relay_2relay4.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print("\n已写出 out_relay_2relay4.json")


def req_of_pair(W, rt, cands, t, pair):
    a, b = pair
    d = W.geo.dist(cands[t[a]][0], cands[t[a]][1],
                   cands[t[b]][0], cands[t[b]][1])
    return d / rt["vc"] * BUFFER


if __name__ == "__main__":
    main()
