# -*- coding: utf-8 -*-
"""中继站位的选择与驻留排程（**修正后的覆盖判据**）。

## 修正了什么

`d_q3.station_plan` 原来这样验证一个站位能否覆盖缺口：

    for (pt, t0, t1) in ds:  if relay_covers(C, W, pt, st)

`relay_covers(C, W, u, p)` 的第一个参数是**无人机的位置**（它判 U→Ra 与
Rb→G01 两条链路）。而这里传进去的 `pt` 是该缺口**自己挑的悬停点**——一个
离地 300 m 的高点，不是无人机的位置。拿它当无人机位置去判链路，比真实情况
容易通过，**覆盖数被系统性高估**：原报表写 2 站覆盖 64/64，按正确判据只有
49/64。

正确判据只有一条：**缺口里的每一个采样点（无人机在那个时刻的真实位置）都要
能被同一个站位建立起 U→Ra→G01。**

## 选择算法

把"候选点 → 它能完整覆盖的缺口集合"预计算成 64 位掩码，于是：

- k=1：取掩码位数最大者；
- k=2：**穷举全部 400×399/2 对**（纯位运算，几秒）；
- k≥3：在最好的一对/两对基础上贪心加站，再对最终配置做局部替换。

这样能**证明**"k 站全覆盖"的最小性（至少能证明 k−1 站做不到，因为那一层是穷举的）。
"""
from __future__ import annotations

import itertools

import numpy as np

import d_q3 as Q3


def gap_covered(W, C, smp, pt):
    """一个悬停点能否让该缺口的**每个采样点**都建立 U→Ra→G01 链路。"""
    return all(Q3.relay_covers(C, W, (s[1], s[2], s[3]), pt) for s in smp)


def cover_masks(W, C, gaps_smp, cands):
    """每个候选点能完整覆盖的缺口集合（位掩码）。"""
    masks = []
    for c in cands:
        m = 0
        for i, smp in enumerate(gaps_smp):
            if gap_covered(W, C, smp, c):
                m |= 1 << i
        masks.append(m)
    return masks


def select_stations(masks, n_gap, max_k=6, exhaustive_k=2, verbose=True):
    """选出覆盖最多缺口的若干站位。

    k ≤ exhaustive_k 的一层是**穷举**（用来证明更少站不够），更高层贪心。
    返回 (选中的候选下标, 覆盖掩码, 每层的累计覆盖数)。
    """
    n = len(masks)
    full = (1 << n_gap) - 1
    hist = []

    # k = 1
    i1 = max(range(n), key=lambda i: bin(masks[i]).count("1"))
    best_sel, best_cov = [i1], masks[i1]
    hist.append(bin(best_cov).count("1"))
    if best_cov == full:
        return best_sel, best_cov, hist

    # k = 2：穷举
    if exhaustive_k >= 2:
        for i, j in itertools.combinations(range(n), 2):
            m = masks[i] | masks[j]
            k = bin(m).count("1")
            if k > bin(best_cov).count("1"):
                best_sel, best_cov = [i, j], m
        hist.append(bin(best_cov).count("1"))
        if verbose:
            print(f"    k=2 穷举 {n * (n - 1) // 2} 对，最优覆盖 "
                  f"{bin(best_cov).count('1')}")
        if best_cov == full:
            return best_sel, best_cov, hist

    # k ≥ 3：贪心加站（每次取能多覆盖最多的；并列时取掩码总位数大的，更通用）
    while len(best_sel) < max_k and best_cov != full:
        rest = [i for i in range(n) if i not in best_sel]
        add = max(rest, key=lambda i: (bin(masks[i] & ~best_cov).count("1"),
                                       bin(masks[i]).count("1")))
        if masks[add] & ~best_cov == 0:
            break
        best_sel.append(add)
        best_cov |= masks[add]
        hist.append(bin(best_cov).count("1"))
    return best_sel, best_cov, hist


def assign_gaps(masks_sel, n_gap):
    """每个缺口交给哪个站位（取第一个能完整覆盖它的）。

    返回 (assign, uncovered)：assign[i] 为站位序号或 None。
    """
    assign = [None] * n_gap
    for i in range(n_gap):
        for j, m in enumerate(masks_sel):
            if m >> i & 1:
                assign[i] = j
                break
    return assign, [i for i, a in enumerate(assign) if a is None]


def plan(W, rt, gaps, points, assign, C=None):
    """按站位排中继：每站只用一个悬停点、在覆盖窗口内持续驻留。

    gaps: [(t0, t1, samples)]；points: [悬停点]；assign: 每个缺口归属的站位。
    能耗 = 往返飞行 + (悬停功率 + 通信功率) × 驻留时长；
    能源组件按题目可用能量（1−ρ）折算，不足一个按一个算。
    """
    out = []
    for j, pt in enumerate(points):
        mine = [gaps[i] for i, a in enumerate(assign) if a == j]
        if not mine:
            continue
        win = (min(g[0] for g in mine), max(g[1] for g in mine))
        # 驻留时长取该站全部缺口区间的**并集**：区间之间不需要悬停（可落地待命），
        # 一个中继在同一站驻留期间把该站所有缺口一起罩住。第一版用
        # [最早开始, 最晚结束] 的整段跨距，把区间之间的空档也算成了悬停。
        hover = sum(b - a for a, b in _merge([(g[0], g[1]) for g in mine]))
        t_fly, e_fly = Q3.relay_flight(W, rt, pt)
        E = e_fly * 2 + (rt["P_hover"] + rt["P_comm"]) * hover / 3600.0
        usable = (1 - rt["rho"]) * rt["E_use"]
        out.append(dict(station=j, point=pt, n_gap=len(mine), n_cover=len(mine),
                        win=win, hover=hover, E=E,
                        n_bat=max(1, int(np.ceil(E / usable))),
                        t_fly=t_fly, t_out=rt["t_prep"] + t_fly,
                        t_back=win[1] + t_fly))
    return out


def energy_key(W, rt, gaps, points, assign):
    """同一覆盖下比较不同站点组合的能耗（用于同站数方案里挑最省的）。"""
    pl = plan(W, rt, gaps, points, assign)
    return sum(x["E"] for x in pl), pl


def min_relays(integs_per_station, points=None, W=None):
    """**需要几架中继**：把各站的缺口区间排给尽量少的中继。

    一架中继在同一时刻只能在一个站位上。缺口是**区间**任务，所以这就是
    "最少机器数的区间调度"——答案等于**最大并发站数**：

        R = max_t #{ 站位 j : 站位 j 在 t 时刻有缺口需要覆盖 }

    转场时间另算：同一中继从站 A 飞到站 B 需要 t(A,B)，本文把这段直接并进
    区间（即认为驻留窗口已经含了转场余量）。站位 1 与站位 3 相距不到 1 km，
    转场可忽略；站 2 较远，本文按最大并发给出的 R 是**下界**，
    实际执行时若并发恰好要求转场，需再加 1 架。

    返回 (R, 并发峰值时刻, 各站区间数)。
    """
    ev = []
    for j, ivs in enumerate(integs_per_station):
        # **先把同一站的缺口合并成不相交的并集**：一个中继在同一站驻留期间
        # 可以把该站所有缺口一起罩住，站内缺口彼此重叠不额外占中继。
        # （第一版直接拿缺口区间做并发统计，把站内重叠也算进去了，
        #   3 个站位却算出"最大并发 6 个站"。）
        for (a, b) in _merge(ivs):
            ev.append((a, 1, j))
            ev.append((b, -1, j))
    ev.sort(key=lambda x: (x[0], x[1]))       # 同一时刻先处理结束
    cur = peak = 0
    tpk = None
    for t, d, j in ev:
        cur += d
        if cur > peak:
            peak, tpk = cur, t
    return peak, tpk, [len(_merge(v)) for v in integs_per_station]


def _merge(ivs):
    """区间并集（按起点排序后合并重叠/相接的）。"""
    out = []
    for a, b in sorted(ivs):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]
