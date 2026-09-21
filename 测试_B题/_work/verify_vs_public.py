"""对照公开结果复核脚本（2022 B 题）。

目的：把本文的两个关键指标换算到公开文献所用的口径上，
避免"用最大值比人家的最小值"这类不可比的对照。

公开口径（来自网络检索到的两篇论文）：
  · 问题 2   —— 统计锥形编队**相邻无人机间距（共 30 条边）**的标准差：
                调整前 4.8848 → 调整后 0.0018，极差 24.3218 → 0.0092，
                均值 50.4069 → 50.7856。
  · 问题 1(3)—— GitHub 公开实现：总体误差率由 5.21% 降至 3.29%
                （其 5.21% 与表 1 的**平均**相对偏差 5.78% 接近，疑为均值口径，
                 而本文报的是**最大**偏差，需换算后可比）。
"""
import sys

import numpy as np

import geo_core as G
from geo_core import R
from b_solve import (cone_ideal, shape_residual, solve_lsq, pair_index,
                     initial_positions, TABLE1)

np.set_printoptions(suppress=True)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def edges_at(ideal, target=50.0, tol=1e-6):
    return [(a, b) for a in range(10) for b in range(a + 1, 10)
            if abs(np.linalg.norm(ideal[a] - ideal[b]) - target) < tol]


def run_bootstrap(groups, rounds, step=0.5, seed=0, pert=8.0, ideal=None,
                  record_edges=False):
    """两组交替自举仿真。返回 (末位置, 形状残差曲线, 间距标准差曲线)。"""
    rr = np.random.default_rng(seed)
    pos = {i: ideal[i] + rr.normal(0, pert, 2) for i in range(10)}
    ed = edges_at(ideal)
    hs, sds = [shape_residual(pos, ideal)], [float(np.std(
        [np.linalg.norm(pos[a] - pos[b]) for a, b in ed]))]
    for rnd in range(rounds):
        tx = list(groups[rnd % len(groups)])
        mv = [i for i in range(10) if i not in tx]
        srcs = [pos[i] for i in tx]
        new = dict(pos)
        for j in mv:
            meas = G.pairwise_angles(pos[j], srcs)
            try:
                q = solve_lsq(meas, tx, pos[j], ideal)
            except Exception:
                continue
            new[j] = np.asarray(pos[j]) + step * (ideal[j] - q)
        pos = new
        hs.append(shape_residual(pos, ideal))
        if record_edges:
            sds.append(float(np.std([np.linalg.norm(pos[a] - pos[b])
                                     for a, b in ed])))
    return pos, hs, sds


print("=" * 74)
print("【0】锥形理想构型的相邻边数（核对公开文献的「共 30 条边」口径）")
print("=" * 74)
ideal = cone_ideal(50.0)
ed = edges_at(ideal)
print(f"  本文理想构型：4 行 1+2+3+4 的实心三角格点，间距 50 m")
print(f"  理想间距恰为 50 m 的点对数 = {len(ed)} 条")
print(f"  全部点对 C(10,2) = 45，其中理想间距 = {sorted({round(float(np.linalg.norm(ideal[a]-ideal[b])),2) for a in range(10) for b in range(a+1,10)})}")
print(f"  → 本文 {len(ed)} 条 ≠ 公开文献的 30 条，说明两文的理想锥形布局不同，")
print(f"    绝对数值不可直接比，须同时看**相对降幅**与**初始散布**。")

print()
print("=" * 74)
print("【1】问题 2 按公开口径重算（相邻间距标准差，扣除/不扣除刚体运动都看）")
print("=" * 74)
G12 = [[0, 6, 9], [1, 5, 8]]
for seed in (0, 1, 2):
    p40, h40, s40 = run_bootstrap(G12, 40, seed=seed, ideal=ideal, record_edges=True)
    p60, h60, s60 = run_bootstrap(G12, 60, seed=seed, ideal=ideal, record_edges=True)
    print(f"  seed={seed} 形状残差 {h60[0]:8.3f} → 40轮 {h40[-1]:8.3f} → 60轮 {h60[-1]:8.3f} m")
    print(f"           间距标准差 初始 {s60[0]:7.3f} → 40轮 {s40[-1]:7.3f} → 60轮 {s60[-1]:7.3f} m"
          f"   （降至初始的 {s60[-1]/s60[0]*100:.4f}%）")

print()
print("=" * 74)
print("【2】把初始散布标定到公开条件的 4.8848 m，再比调整后的绝对值")
print("=" * 74)
# 公开条件：调整前间距标准差 4.8848，调整后 0.0018（30 条边）
target0 = 4.8848


def init_spacing_sd(pert, seed=0):
    """纯初始构型的相邻间距标准差（不跑任何一轮）。"""
    rr = np.random.default_rng(seed)
    p0 = {i: ideal[i] + rr.normal(0, pert, 2) for i in range(10)}
    return float(np.std([np.linalg.norm(p0[a] - p0[b]) for a, b in ed]))


lo, hi = 0.5, 20.0
for _ in range(40):                      # 二分标定扰动强度，使初始间距标准差命中 4.8848
    mid = (lo + hi) / 2
    if init_spacing_sd(mid) < target0:
        lo = mid
    else:
        hi = mid
pert_cal = (lo + hi) / 2
sd0 = init_spacing_sd(pert_cal)
print(f"  标定：扰动强度 N(0,{pert_cal:.3f}^2) → 初始间距标准差 {sd0:.4f} m")
print(f"        （公开条件 4.8848 m；本文正文原用 N(0,8^2)，"
      f"初始散布 {init_spacing_sd(8.0):.3f} m 为公开的 {init_spacing_sd(8.0)/4.8848:.2f} 倍）")
for rounds in (40, 60, 120):
    p, h, s = run_bootstrap(G12, rounds, seed=0, pert=pert_cal, ideal=ideal,
                            record_edges=True)
    print(f"  {rounds:>4} 轮：形状残差 {h[-1]:8.4f} m   间距标准差 {s[-1]:8.4f} m"
          f"   （公开调整后 0.0018 m）")

print()
print("=" * 74)
print("【3】问题 1(3) 换算到公开口径（均值 vs 最大）")
print("=" * 74)
TABLE1_vals = TABLE1
dev = [abs(TABLE1_vals[i][0] - R) for i in range(1, 10)]
print(f"  表 1 初始径向偏差 |r−100|: {[round(d,2) for d in dev]}")
print(f"  最大 {max(dev):.2f} m（相对 {max(dev)/R*100:.2f}%）   "
      f"均值 {np.mean(dev):.2f} m（相对 {np.mean(dev)/R*100:.2f}%）")
print(f"  → 公开实现的 5.21% 与本文的**均值**口径 {np.mean(dev)/R*100:.2f}% 吻合，")
print(f"    说明 5.21% 是均值口径；本文正文报的 12.01% 是**最大**口径，两者不可直接比。")


def one_round(pos, tx, mv, step, ideal):
    srcs = [pos[i] for i in tx]
    new = dict(pos)
    for j in mv:
        meas = G.pairwise_angles(pos[j], srcs)
        try:
            q = solve_lsq(meas, tx, pos[j], ideal)
        except Exception:
            continue
        new[j] = np.asarray(pos[j]) + step * (ideal[j] - q)
    return new


ideal_c = {i: G.ideal_pos(i) for i in range(10)}
plans = [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
print()
print(f"  {'轮次':>6}{'最大偏差':>12}{'均值偏差':>12}{'最大相对':>12}{'均值相对':>12}")
pos = initial_positions()
for rnd in range(1, 41):
    plan = plans[(rnd - 1) % 3]
    pos = one_round(pos, [0] + plan, [i for i in range(1, 10) if i not in plan],
                    0.7, ideal_c)
    if rnd in (5, 10, 20, 40):
        d = [float(np.linalg.norm(pos[i] - G.ideal_pos(i))) for i in range(1, 10)]
        print(f"  {rnd:>6}{max(d):>12.4f}{np.mean(d):>12.4f}"
              f"{max(d)/R*100:>11.2f}%{np.mean(d)/R*100:>11.2f}%")
print(f"  → 同一方案（λ=0.7 + 三等分轮换 + 4 源最小二乘）在两种口径下的结果；")
print(f"    与公开实现的 3.29% 对比时须用同口径。")
