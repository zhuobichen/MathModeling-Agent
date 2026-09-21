# -*- coding: utf-8 -*-
"""问题 1、2 的分析与计算。

问题 1：给出交会定位区域直径的算法，并回答"以该直径为直径的圆能否覆盖
        定位区域"。核心事实：对平面点集 K，最小覆盖圆半径 r_min 满足
            D/2 ≤ r_min ≤ D/√3        （左端显然，右端是 Jung 定理）
        等号分别对应"直径对的中点即为圆心"与"等边三角形"。故以 D 为直径
        的圆**一般不能**覆盖，只在 r_min = D/2 时能。

问题 2：已知 S1 处测得的示向度，选择第二检测点 S2。用"期望定位区域直径"
        作为目标函数，在 (b, φ) 极坐标网格上做蒙特卡洛评估，给出候选区域。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import geo_core as G

HERE = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(20260921)


# ══════════════════════════════════════════════════════════
def region_metrics(points, thetas):
    """定位区域的直径、最小覆盖圆半径、二者之比。"""
    poly = G.localize(points, thetas)
    if len(poly) < 2:
        return None
    D, pair = G.polygon_diameter(poly)
    c, r = G.min_enclosing_circle(poly)
    return dict(poly=poly, D=D, r_min=r, ratio=(2 * r / D if D > 1e-12 else 1.0),
                pair=pair, area=poly_area(poly))


def poly_area(poly):
    p = np.asarray(poly, float)
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


# ── 问题 1 ────────────────────────────────────────────────
def obs_from_source(src, pts, errs=None):
    """给真源 src 与检测点 pts，生成含误差的示向度观测。

    注意形参不能叫 G：会遮住模块别名 `geo_core as G`。
    """
    src = np.asarray(src, float)
    th = []
    for k, S in enumerate(pts):
        e = 0.0 if errs is None else errs[k]
        th.append((G.bearing(S, src) + e) % 360.0)
    return th


def q1():
    print("=" * 78)
    print("问题 1  交会定位区域直径的算法，以及「以直径为直径的圆」能否覆盖")
    print("=" * 78)

    print("")
    print("① 直径算法自检（旋转卡壳 vs 逐对暴力）")
    rng = np.random.default_rng(3)
    n_ok = 0
    for k in range(8):
        Gsrc = rng.uniform(-900, 900, 2)
        pts = rng.uniform(-1400, 1400, (k + 2, 2))
        th = obs_from_source(Gsrc, pts, rng.uniform(-1, 1, k + 2))
        m = region_metrics(pts, th)
        if m is None or len(m["poly"]) < 3:
            continue
        P = G.convex_hull(m["poly"])
        brute = max(np.linalg.norm(P[i] - P[j])
                    for i in range(len(P)) for j in range(i + 1, len(P)))
        print(f"   n={k+2}  D(卡壳)={m['D']:.6f}  D(暴力)={brute:.6f}  "
              f"差={abs(m['D']-brute):.2e}")
        n_ok += 1
    print(f"   （有效算例 {n_ok} 个，其余因区域退化跳过）")

    print("")
    print("② 理论界：平面点集 K 的最小覆盖圆半径 r_min 满足")
    print("      D/2  ≤  r_min  ≤  D/√3       （右端为 Jung 定理）")
    print(f"      即「以 D 为直径的圆」（半径 D/2）只在 r_min = D/2 时能覆盖 K；")
    print(f"      最坏情形需要直径 {2/np.sqrt(3):.6f}·D，约放大 15.5%。")

    print("")
    print("③ 交会定位区域的实际检验（真源 G=(0,0)，检测点按几何布置）")
    cases = [
        ("两点·源处交会角 90°", [(-800, 0), (0, -800)]),
        ("两点·源处交会角 30°", [(-1200, 0), (-1200, 400)]),
        ("两点·源处交会角 150°", [(-800, 0), (0, -300)]),
        ("三点·绕源 120° 均布", [(800, 0), (-400, 692.8), (-400, -692.8)]),
        ("四点·绕源 90° 均布", [(800, 0), (0, 800), (-800, 0), (0, -800)]),
        ("三点·同侧集中", [(-900, 220), (-900, -220), (-1500, 0)]),
    ]
    rows = []
    for name, pts in cases:
        th = obs_from_source((0, 0), pts, [0.8, -0.9, 0.6, -0.7][:len(pts)])
        m = region_metrics(pts, th)
        if m is None or len(m["poly"]) < 3:
            print(f"   {name}: 区域退化")
            continue
        ok = m["ratio"] <= 1.0 + 1e-6
        print(f"   {name}: 顶点{len(m['poly'])}  D={m['D']:9.2f} m  "
              f"r_min={m['r_min']:9.2f}  2r_min/D={m['ratio']:.4f}  "
              f"覆盖={'能' if ok else '不能'}")
        rows.append(dict(name=name, nv=len(m["poly"]), D=m["D"],
                         r_min=m["r_min"], ratio=m["ratio"], cover=bool(ok)))

    print("")
    print("④ 最坏情形：正三角形区域")
    L = 1000.0
    tri = np.array([[0, 0], [L, 0], [L / 2, L * np.sqrt(3) / 2]])
    D, _ = G.polygon_diameter(tri)
    c, r = G.min_enclosing_circle(tri)
    print(f"   边长 {L:.0f} 正三角形：D={D:.2f}  r_min={r:.2f}  "
          f"2r_min/D={2*r/D:.6f}（理论 2/√3={2/np.sqrt(3):.6f}）")
    print("   → 以 D 为直径的圆不能覆盖，需放大到 1.1547 倍。")

    print("")
    print("⑤ 结论：算法给出 D 与 r_min 两个量后即可判定")
    print("   覆盖 ⟺ r_min ≤ D/2（等价于直径对的中点即为最小覆盖圆圆心）；")
    print("   否则需把圆的直径放大 2r_min/D 倍，该倍数不超过 2/√3。")
    return rows


# ── 问题 2 ────────────────────────────────────────────────
def q2(S1=(0.0, 0.0), theta1=0.0, rho=1500.0, n_mc=200,
       b_grid=None, phi_grid=None, seed=7):
    """在极坐标网格上评估第二检测点，返回 (b, phi) → 期望定位直径。"""
    rng = np.random.default_rng(seed)
    S1 = np.asarray(S1, float)
    # 源的先验：落在 W1 ∩ 目标区域 ∩ {|G−S1|≤ρ} 内，按面积均匀
    # 用拒绝采样，并偏向"较远"以体现难点（题中有效接收半径 1000~1500）
    Gpts = sample_source(S1, theta1, rho, n_mc, rng)

    if b_grid is None:
        b_grid = np.arange(200.0, 2601.0, 100.0)
    if phi_grid is None:
        phi_grid = np.arange(0.0, 360.0, 10.0)

    out = np.full((len(b_grid), len(phi_grid)), np.nan)
    for i, b in enumerate(b_grid):
        for j, phi in enumerate(phi_grid):
            S2 = S1 + b * G.unit(theta1 + phi)
            if np.linalg.norm(S2) > G.ARENA_R:
                continue
            ds = []
            for Gp in Gpts:
                # 第二点测量：真方位 + 误差（该点处误差固定）
                th2_true = G.bearing(S2, Gp)
                th2 = (th2_true + rng.uniform(-1, 1)) % 360.0
                m = region_metrics([S1, S2], [theta1, th2])
                if m is None or m["D"] <= 0:
                    continue
                ds.append(m["D"])
            if ds:
                out[i, j] = float(np.mean(ds))
    return b_grid, phi_grid, out, Gpts


def sample_source(S1, theta1, rho, n, rng):
    """源的先验样本：在 W1∩圆域 内，且距离 S1 不超过 ρ。"""
    pts = []
    while len(pts) < n:
        t = rng.uniform(50.0, rho)
        dth = rng.uniform(-G.DELTA_DEG, G.DELTA_DEG)
        p = S1 + t * G.unit(theta1 + dth)
        if np.linalg.norm(p) <= G.ARENA_R:
            pts.append(p)
    return np.array(pts)


def q2_report():
    print()
    print("=" * 78)
    print("问题 2  第二个检测点的选择策略")
    print("=" * 78)

    print("\n① 几何关系：设源在 S1 的示向度方向上、距离 t，第二点取在与该方向")
    print("   成 φ 角、距 S1 为 b 处。源处的交会角 γ 满足")
    print("      γ = ∠S1 G S2")
    print("   定位区域的横向误差 ∝ t·δ/sin γ，故希望 γ 尽量接近 90°。")
    print("   检测约束：第二点到源的距离 √(t²+b²−2bt·cosφ) ≤ ρ。")

    print("\n② 蒙特卡洛：期望定位区域直径 E[D] 随 (b, φ) 的变化")
    b_grid, phi_grid, out, Gpts = q2()
    finite = out[np.isfinite(out)]
    print(f"   ρ=1500 m，源先验为 W1∩圆域∩{{|G−S1|≤ρ}} 内均匀，"
          f"网格 {len(b_grid)}×{len(phi_grid)}，每格 200 次")
    print(f"   E[D] 范围 {finite.min():.1f} ~ {finite.max():.1f} m")

    # 按 b 汇总（对 φ 取最小）
    print("\n   按 b 汇总（对 φ 取最优）")
    print(f"   {'b/m':>7}{'最优 φ/°':>10}{'最小 E[D]/m':>14}")
    best = None
    for i, b in enumerate(b_grid):
        row = out[i]
        if not np.any(np.isfinite(row)):
            continue
        j = int(np.nanargmin(row))
        print(f"   {b:>7.0f}{phi_grid[j]:>10.0f}{row[j]:>14.1f}")
        if best is None or row[j] < best[0]:
            best = (row[j], b, phi_grid[j])

    print(f"\n   最优：b={best[1]:.0f} m，φ={best[2]:.0f}°（相对示向度方向），"
          f"E[D]={best[0]:.1f} m")

    # 候选区域：E[D] ≤ 1.2 × 最优 的格子
    thr = best[0] * 1.2
    cells = [(b_grid[i], phi_grid[j]) for i in range(len(b_grid))
             for j in range(len(phi_grid))
             if np.isfinite(out[i, j]) and out[i, j] <= thr]
    if cells:
        bs = [c[0] for c in cells]
        ps = sorted(set(c[1] for c in cells))
        print(f"\n③ 候选区域（E[D] ≤ 1.2×最优，共 {len(cells)} 格）")
        print(f"   b ∈ [{min(bs):.0f}, {max(bs):.0f}] m，φ 覆盖 {len(ps)} 个方向")

    # 垂直方向附近的细扫
    print("\n④ 固定 φ=90°（垂直于示向度）细扫")
    bf = np.arange(200.0, 2201.0, 100.0)
    _, _, outf, _ = q2(b_grid=bf, phi_grid=np.array([90.0]))
    print(f"   {'b/m':>7}{'E[D]/m':>12}")
    for i, b in enumerate(bf):
        v = outf[i, 0]
        if np.isfinite(v):
            print(f"   {b:>7.0f}{v:>12.1f}")

    return dict(b_grid=b_grid.tolist(), phi_grid=phi_grid.tolist(),
                E=out.tolist(), best=dict(E=best[0], b=best[1], phi=best[2]))


if __name__ == "__main__":
    q1()
    res = q2_report()
    with open(os.path.join(HERE, "out_q12.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)
    print("\n已保存 out_q12.json")
