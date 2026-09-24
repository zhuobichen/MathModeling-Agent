# -*- coding: utf-8 -*-
"""问题四：**以资源缺口为目标**的分区搜索（论文 8.4 节）。

前面的 d_q4.py 是在共现分量上按**架次数均衡**聚类得到分区的，然后才把资源分配
做成优化。但题目问的是「比较资源规模、冗余、均衡性与库存缺口」——**缺口本身
应该进目标函数**。本文把它补上：

    在共现分量之间搜索分区，使资源缺口（运输机 / 电池组 / 中继）最小，
    硬时限违反数仍为 0。

## 目标函数（分层字典序）

$$\\min\\ \\Big(\\underbrace{\\textstyle\\sum_t \\text{Gap}^{\\text{drone}}_t}_{\\text{机型缺口之和}},
\\ \\sum_t \\text{Gap}^{\\text{bat}}_t,\\ \\text{Gap}^{\\text{relay}},\\ \\max_j T_j\\Big)$$

其中每一层的缺口都是 (8-1) 的**最优分配**算出来的——也就是说，评判一个分区好
不好，用的是"这个分区在库存约束下最优能配成什么样"，而不是某一套具体的配法。

## 搜索

15 个服务区连成 11 个共现分量，分量内不可拆。于是分区空间是"把 11 个分量分成
$k$ 组"。分量数小，但 $k=3$ 时仍有 28501 种分法，而每次评估要跑一遍资源分配的
穷举——所以用**局部搜索**：从一个地理聚类解出发，反复尝试"把某个分量挪到另一组"，
接受能改善目标的移动，直到没有改进为止；再换几个起点重跑，取最好。

排程结果按 (架次集合, 机型配置) 缓存，所以一轮邻域评估里绝大多数调用是命中的。
"""
from __future__ import annotations

import itertools
import json
import os
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import d_core as D
from d_q2 import build_groups, World, TYPES, SITES, initial_groups, merge_pass, schedule, evaluate
import d_q3 as Q3
import d_q4 as Q4
import d_relay

HERE = os.path.dirname(os.path.abspath(__file__))
INV = Q4.INV


def memoise_scheduler():
    """把 Q4.schedule_fleet_capped 换成带缓存的版本。

    best_allocation / min_batteries 都是在 d_q4 的模块命名空间里查这个名字的，
    所以在这里覆盖即可，不必改它们的代码。
    """
    raw = Q4.schedule_fleet_capped
    cache = {}

    def key_of(stops, cap, bat_cap):
        # 架次的站点顺序与箱集都要进键（路线不同 → 结果不同）；
        # 键里不放可变对象，所以摊平成字符串元组。
        s = tuple(
            tuple((s2, tuple(sorted(b["box"] for b in bs))) for s2, bs in st)
            for st in stops)
        c = tuple(sorted((cap or {}).items()))
        b = tuple(sorted((bat_cap or {}).items()))
        return (s, c, b)

    def wrapped(W, stops_list, cap, bat_cap=None):
        k = key_of(stops_list, cap, bat_cap)
        if k not in cache:
            cache[k] = raw(W, stops_list, cap, bat_cap)
        return cache[k]

    Q4.schedule_fleet_capped = wrapped
    return cache


def sorties_by_component(W, p1):
    """每个共现分量包含哪些架次（同一架次的站点必在同一分量内）。"""
    comps = Q4.components(W, [x["stops"] for x in p1])
    comp_of = {s: i for i, c in enumerate(comps) for s in c}
    by_comp = defaultdict(list)
    for i, x in enumerate(p1):
        cid = comp_of[x["stops"][0][0]]
        assert all(comp_of[s] == cid for s, _ in x["stops"]), "架次跨了分量"
        by_comp[cid].append(i)
    return comps, comp_of, by_comp


def station_mask_per_sortie(W, C, p1, stations):
    """每条架次需要哪些中继站位（它至少有一个缺口被该站位覆盖）。"""
    out = []
    for x in p1:
        m = 0
        for (t0, t1, smp) in Q3.gaps_of(W, C, x):
            for j, s in enumerate(stations):
                if d_relay.gap_covered(W, C, smp, tuple(s["point"])):
                    m |= 1 << j
                    break
        out.append(m)
    return out


def evaluate_partition(W, groups_of_sorties, rneed, t_shift, n_comp_ok):
    """一个分区的目标向量 + 详细结果。groups_of_sorties: [[架次下标], ...]"""
    subs = [[p1_stub[i]["stops"] for i in g] for g in groups_of_sorties]
    alloc = Q4.best_allocation(W, subs, t_shift=t_shift)
    nd = {g: sum(alloc["per_group"][j]["cap"][g] for j in range(len(subs)))
          for g in TYPES}
    bn = {g: sum(alloc["per_group"][j]["bat"][g] for j in range(len(subs)))
          for g in TYPES}
    gap_d = sum(max(0, nd[g] - INV["drone"][g]) for g in TYPES)
    gap_b = sum(max(0, bn[g] - INV["battery"][g]) for g in TYPES)
    rel = sum(rneed[j] for j in range(len(subs)))
    gap_r = max(0, rel - INV["relay"])
    ns = [len(g) for g in groups_of_sorties]
    spread = max(ns) - min(ns)          # 组间均衡性（架次极差）
    # 目标分层：先消掉硬约束违反，再按「资源缺口 → 均衡性 → 完成时间」比。
    # 题目问的是「资源规模、冗余、均衡性与库存缺口」，所以均衡性要进目标，
    # 不能像第一版那样只当结果报出去——那一版搜出了"1 站 vs 14 站"这种
    # 缺口最小但完全失衡的分区。
    score = (int(alloc["viol"]), gap_d, gap_b, gap_r, spread,
             float(alloc["tmax"]))
    return dict(score=score, viol=alloc["viol"], need_drone=nd, need_bat=bn,
                need_relay=rel, gap_drone=gap_d, gap_bat=gap_b, gap_relay=gap_r,
                spread=spread, n_sorties=ns, tmax=alloc["tmax"],
                per_group=alloc["per_group"])


p1_stub = None            # 由 main 注入（架次列表）


def main():
    global p1_stub
    W = World()
    rt, W.relays, W.rbats = D.load_relays()
    C = D.Comms(W.geo, W.nodes, D.load_comm())
    g1 = build_groups(W, allow_types=("A", "B"))[0]
    p1 = schedule(W, g1)
    p1_stub = p1
    with open(os.path.join(HERE, "out_q3.json"), encoding="utf-8") as f:
        q3res = json.load(f)
    stations = q3res["stations"]
    T_SHIFT = float(q3res.get("t_shift", 0.0))

    cache = memoise_scheduler()
    comps, comp_of, by_comp = sorties_by_component(W, p1)
    print(f"共现分量 {len(comps)} 个，各自 { [len(by_comp[i]) for i in range(len(comps))] } 条架次")
    smask = station_mask_per_sortie(W, C, p1, stations)
    nst = len(stations)
    print(f"中继站位 {nst} 个；每条架次需要的站位掩码分布 "
          f"{ {bin(m).count('1'): sum(1 for x in smask if x == m) for m in sorted(set(smask))} }")

    def rneed_of(gidx):
        m = 0
        for i in gidx:
            m |= smask[i]
        return bin(m).count("1")

    def evaluate(groups_by_comp):
        gs = [[i for c in grp for i in by_comp[c]] for grp in groups_by_comp]
        if any(not g for g in gs):
            return None
        rn = [rneed_of(g) for g in gs]
        r = evaluate_partition(W, gs, rn, T_SHIFT, comps)
        # 分区是从共现分量拼出来的，跨组架次按构造成立必为 0——但要**验一遍**：
        # 这类"按构造成立"的断言最容易被后来的改动悄悄破坏。
        cid = {c: j for j, grp in enumerate(groups_by_comp) for c in grp}
        r["cross"] = sum(
            1 for x in p1
            if len({cid[comp_of[s]] for s, _ in x["stops"]}) > 1)
        return r

    ncomp = len(comps)

    def local_search(init, tag):
        cur = [list(g) for g in init]
        cur_res = evaluate(cur)
        improved = True
        steps = 0
        while improved:
            improved = False
            best = None
            for i, grp in enumerate(cur):          # 把分量 i 挪到别的组
                for c in list(grp):
                    for j in range(len(cur)):
                        if j == i or (len(grp) == 1):
                            continue
                        trial = [list(g) for g in cur]
                        trial[i].remove(c)
                        trial[j].append(c)
                        trial = [g for g in trial if g]
                        if len(trial) != len(cur):
                            continue
                        r = evaluate(trial)
                        if r is None:
                            continue
                        if best is None or r["score"] < best[0]["score"]:
                            best = (r, trial)
            if best and best[0]["score"] < cur_res["score"]:
                cur, cur_res = best[1], best[0]
                improved = True
                steps += 1
        print(f"  [{tag}] 局部搜索 {steps} 步收敛：目标 {cur_res['score']}")
        return cur, cur_res

    out = {}
    for K in (2, 3):
        print()
        print("=" * 78)
        print(f"K = {K}：以资源缺口为目标的分区搜索")
        print("=" * 78)
        # 起点 1：地理聚类（d_q4 的做法）
        geo = Q4.group_sites(W, K, g1)
        comp_of_site = {s: i for i, c in enumerate(comps) for s in c}
        init_geo = []
        for gp in geo:
            cs = sorted({comp_of_site[s] for s in gp})
            init_geo.append(cs)
        # 起点 2..：轮流用单分量作种子
        starts = [("地理聚类", init_geo)]
        for seed in range(min(ncomp, 4)):
            rest = [c for c in range(ncomp) if c != seed]
            if K == 2:
                starts.append((f"种子 {seed}", [[seed], rest]))
            else:
                h = len(rest) // 2
                starts.append((f"种子 {seed}", [[seed], rest[:h], rest[h:]]))

        best_all = None
        for tag, st in starts:
            if any(not g for g in st) or len(st) != K:
                continue
            g, r = local_search(st, tag)
            if best_all is None or r["score"] < best_all[1]["score"]:
                best_all = (g, r, tag)
        g, r, tag = best_all
        print(f"  → 最优分区（起点：{tag}）")
        print(f"    目标向量 (违反, 机型缺口, 电池缺口, 中继缺口, 最晚组完成) = {r['score']}")
        for j, grp in enumerate(g):
            sites = sorted(s for c in grp for s in comps[c])
            rn_j = rneed_of([i for c in grp for i in by_comp[c]])
            print(f"    组{j+1}：{len(sites)} 站 {sites}")
            print(f"      机型 { {x: r['per_group'][j]['cap'][x] for x in TYPES if r['per_group'][j]['cap'][x]} }"
                  f"  电池 { {x: r['per_group'][j]['bat'][x] for x in TYPES if r['per_group'][j]['bat'][x]} }"
                  f"  中继 {rn_j} 架"
                  f"  架次 {r['n_sorties'][j]}"
                  f"  完成 {r['per_group'][j]['t']:.0f}s")
        print(f"    合计：机型 {r['need_drone']}（缺口 {r['gap_drone']}）"
              f"  电池 {r['need_bat']}（缺口 {r['gap_bat']}）"
              f"  中继 {r['need_relay']} 架（缺口 {r['gap_relay']}）"
              f"  架次极差 {r['spread']}  跨组架次 {r['cross']}"
              f"  最晚组 {r['tmax']:.0f}s")
        # 对照：原来按架次均衡聚类的分区
        # 注意 set()：一个分量含多个站，按站取分量必须去重，否则重复计入架次
        base = evaluate([sorted({comp_of_site[s] for s in gp}) for gp in geo])
        print(f"    对照·地理聚类分区（按架次数均衡聚类）：目标 {base['score']}"
              f"  机型需求 {base['need_drone']}  电池 {base['need_bat']}"
              f"  中继 {base['need_relay']}  架次极差 {base['spread']}"
              f"  跨组架次 {base['cross']}"
              f"  最晚组 {base['tmax']:.0f}s")
        out[K] = dict(groups=[[sorted(s for c in grp for s in comps[c])] for grp in g],
                      best=[g2 for g2 in g], score=list(r["score"]),
                      detail={k: v for k, v in r.items() if k != "per_group"},
                      baseline=dict(score=list(base["score"]),
                                    need_drone=base["need_drone"],
                                    need_bat=base["need_bat"],
                                    need_relay=base["need_relay"]),
                      start=tag)

    # ── K=2 的穷举核对：1023 种二分法 ─────────────────────────────
    # 局部搜索只保证局部最优。K=2 的分区空间只有 2^10-1 = 1023 种，值得穷举核一遍。
    # 但每种都跑一遍完整的最优分配太贵（每次 ~130 次排程），所以分两级：
    #   ① 便宜的代理指标：每组跑一遍 min_fleet（单机型二分），把各组的机型需求
    #      相加得到"库存缺口代理"，按它排序；
    #   ② 取前若干名再用 (8-1) 的最优分配精确评估。
    # 代理指标只用来排序，最终结论仍由精确目标给出——这一点必须说清楚。
    print()
    print("=" * 78)
    print("K = 2 的穷举核对（1023 种二分法：先按代理指标筛，再精确评估前列）")
    print("=" * 78)
    comps_list = list(range(ncomp))
    rows = []
    for mask in range(1, 1 << (ncomp - 1)):        # 固定第 0 个分量在 A 组
        A = [c for c in comps_list if (mask >> c) & 1]
        B = [c for c in comps_list if not ((mask >> c) & 1)]
        if not A or not B:
            continue
        need = {g: 0 for g in TYPES}
        nmin = 0
        ok = True
        for grp in (A, B):
            sub = [p1_stub[i]["stops"] for c in grp for i in by_comp[c]]
            try:
                n, nf, _ = Q4.min_fleet(W, sub)
            except Exception:
                ok = False
                break
            nmin += n
            for g in TYPES:
                need[g] += nf[g]
        if not ok:
            continue
        proxy = sum(max(0, need[g] - INV["drone"][g]) for g in TYPES)
        rows.append((proxy, nmin, tuple(sorted(A)), tuple(sorted(B)), need))
    rows.sort(key=lambda x: (x[0], x[1]))
    print(f"  筛完 {len(rows)} 种；代理缺口最小 {rows[0][0]}，"
          f"其中缺口为 0 的有 {sum(1 for r in rows if r[0] == 0)} 种")
    head = rows[:12]
    best_exact, best_g = None, None
    for proxy, nmin, A, B, need in head:
        r = evaluate([list(A), list(B)])
        if best_exact is None or r["score"] < best_exact["score"]:
            best_exact, best_g = r, [list(A), list(B)]
    print(f"  前 12 名精确评估的最优目标：{best_exact['score']}")
    print(f"    分区 {[[sorted(x for c in grp for x in comps[c]) for grp in best_g]]}")
    print(f"    → 与局部搜索得到的 K=2 最优"
          f"{'一致' if abs(best_exact['tmax'] - out[2]['detail']['tmax']) < 1e-6 else '**不一致，需复查**'}")
    out["K2_exhaustive"] = dict(n_evaluated=len(rows),
                                n_zero_proxy=sum(1 for r in rows if r[0] == 0),
                                best_score=list(best_exact["score"]),
                                best_groups=[sorted(x for c in grp for x in comps[c])
                                             for grp in best_g])

    print()
    print(f"排程缓存命中表大小 {len(cache)}")
    with open(os.path.join(HERE, "out_q4_partition.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print("已写出 out_q4_partition.json")


if __name__ == "__main__":
    main()
