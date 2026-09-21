# -*- coding: utf-8 -*-
"""论文插图。"""
from __future__ import annotations

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import geo_core as G
import robot
import simulator as S

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.abspath(os.path.join(HERE, "..", "_交付", "figures"))
os.makedirs(FIG, exist_ok=True)


def draw_ring(ax, R=1800, **kw):
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(R * np.cos(th), R * np.sin(th), **kw)


# ── 图1：交会定位区域、直径与最小覆盖圆 ──────────────────
def fig1():
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    src = np.array([0.0, 0.0])

    # (a) 两点交会
    pts = [np.array([-800, 0.0]), np.array([0.0, -800])]
    th = [(G.bearing(p, src) + e) % 360 for p, e in zip(pts, [0.8, -0.9])]
    poly = G.localize(pts, th)
    D, pair = G.polygon_diameter(poly)
    c, r = G.min_enclosing_circle(poly)
    for p, t in zip(pts, th):
        for s in (-1, 1):
            d = G.unit(t + s * 1.0)
            ax[0].plot([p[0], p[0] + 1400 * d[0]], [p[1], p[1] + 1400 * d[1]],
                       "b-", lw=0.9, alpha=.7)
        ax[0].plot(*p, "ks", ms=6)
    P = np.vstack([poly, poly[:1]])
    ax[0].fill(P[:, 0], P[:, 1], color="orange", alpha=.45, zorder=3)
    ax[0].plot(P[:, 0], P[:, 1], "r-", lw=1.6, zorder=4)
    ax[0].plot([pair[0][0], pair[1][0]], [pair[0][1], pair[1][1]], "r--", lw=1.8,
               label=f"直径 D={D:.1f} m")
    thc = np.linspace(0, 2 * np.pi, 300)
    ax[0].plot(c[0] + r * np.cos(thc), c[1] + r * np.sin(thc), "g-", lw=1.4,
               label=f"最小覆盖圆 r={r:.1f}")
    ax[0].plot(*src, "r*", ms=14, label="真实源")
    ax[0].legend(fontsize=8); ax[0].set_aspect("equal")
    ax[0].set_title(f"(a) 两点交会：2r/D = {2*r/D:.4f}（可覆盖）")
    ax[0].grid(alpha=.3)

    # (b) 三点绕源 120° → 反例
    pts2 = [np.array([800, 0.0]), np.array([-400, 692.8]), np.array([-400, -692.8])]
    th2 = [(G.bearing(p, src) + e) % 360 for p, e in zip(pts2, [0.8, -0.9, 0.6])]
    poly2 = G.localize(pts2, th2)
    D2, pair2 = G.polygon_diameter(poly2)
    c2, r2 = G.min_enclosing_circle(poly2)
    P2 = np.vstack([poly2, poly2[:1]])
    ax[1].fill(P2[:, 0], P2[:, 1], color="orange", alpha=.45, zorder=3)
    ax[1].plot(P2[:, 0], P2[:, 1], "r-", lw=1.6, zorder=4)
    ax[1].plot([pair2[0][0], pair2[1][0]], [pair2[0][1], pair2[1][1]], "r--",
               lw=1.8, label=f"直径 D={D2:.1f} m")
    ax[1].plot(c2[0] + r2 * np.cos(thc), c2[1] + r2 * np.sin(thc), "g-", lw=1.4,
               label=f"最小覆盖圆 r={r2:.1f}")
    ax[1].plot(*src, "r*", ms=14)
    for p in pts2:
        ax[1].plot(*p, "ks", ms=6)
    ax[1].legend(fontsize=8); ax[1].set_aspect("equal")
    ax[1].set_title(f"(b) 三点 120°：2r/D = {2*r2/D2:.4f}（覆盖不了）")
    ax[1].grid(alpha=.3)

    # (c) 2r/D 的分布
    rng = np.random.default_rng(11)
    vals = []
    for _ in range(400):
        n = int(rng.integers(2, 6))
        src2 = rng.uniform(-600, 600, 2)
        pp = rng.uniform(-1500, 1500, (n, 2))
        tt = [(G.bearing(p, src2) + rng.uniform(-1, 1)) % 360 for p in pp]
        pl = G.localize(pp, tt)
        if len(pl) < 3:
            continue
        Dj, _ = G.polygon_diameter(pl)
        if Dj < 1e-6:
            continue
        _, rj = G.min_enclosing_circle(pl)
        vals.append(2 * rj / Dj)
    vals = np.array(vals)
    ax[2].hist(vals, bins=40, color="steelblue", edgecolor="w")
    ax[2].axvline(1.0, color="k", ls=":", lw=1.4, label="下界 1")
    ax[2].axvline(2 / np.sqrt(3), color="r", ls="--", lw=1.5,
                  label=f"Jung 上界 {2/np.sqrt(3):.4f}")
    ax[2].set_xlabel("$2r_{\\min}/D$")
    ax[2].set_ylabel("频数")
    ax[2].set_title(f"(c) {len(vals)} 个交会区域的 $2r_{{\\min}}/D$ 分布")
    ax[2].grid(alpha=.3); ax[2].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_b1.png"), dpi=130)
    plt.close()
    print("fig_b1 完成，2r/D 超过 1 的比例 = %.1f%%"
          % ((vals > 1 + 1e-6).mean() * 100))


# ── 图2：问题 2 的第二检测点 ─────────────────────────────
def fig2():
    import q12
    b_grid, phi_grid, out, _ = q12.q2()
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))
    Ph, B = np.meshgrid(np.radians(phi_grid), b_grid)
    m = np.ma.masked_invalid(out)
    cs = ax[0].pcolormesh(Ph, B, m, shading="auto", cmap="viridis")
    plt.colorbar(cs, ax=ax[0], label="$E[D]$ / m")
    ib, ip = np.unravel_index(np.nanargmin(out), out.shape)
    ax[0].plot(np.radians(phi_grid[ip]), b_grid[ib], "r*", ms=16,
               label=f"最优 $b$={b_grid[ib]:.0f} m, $\\varphi$={phi_grid[ip]:.0f}°")
    ax[0].set_xlabel("$\\varphi$（相对示向度方向）/ rad")
    ax[0].set_ylabel("$b$ / m")
    ax[0].set_title("(a) 期望定位直径 $E[D]$ 的分布")
    ax[0].legend(fontsize=8)
    ax[1].plot(b_grid, [np.nanmin(r) for r in out], "o-", lw=1.6,
               label="对 $\\varphi$ 取最优")
    j90 = int(np.argmin(np.abs(phi_grid - 90)))
    ax[1].plot(b_grid, out[:, j90], "s--", lw=1.4, label="固定 $\\varphi=90°$")
    i0 = int(np.argmin([np.nanmin(r) for r in out]))
    ax[1].axvline(b_grid[i0], color="r", ls=":", lw=1.5,
                  label=f"最优 $b$={b_grid[i0]:.0f} m")
    ax[1].set_xlabel("第二检测点距离 $b$ / m")
    ax[1].set_ylabel("$E[D]$ / m")
    ax[1].set_title("(b) 随距离的变化")
    ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_b2.png"), dpi=130)
    plt.close()
    print("fig_b2 完成")


# ── 图3：问题 3 的一局轨迹 ───────────────────────────────
def fig3(seed=7003):
    pts = [np.asarray(p, float) for p in robot.survey_ring(6, margin=0.70)]
    srcs = S.sample_case(np.random.default_rng(seed), dir_frac=0.0)
    sim = S.Simulator(srcs, seed=seed)
    rb = robot.Robot(sim)
    st = rb.run(survey_pts=list(pts), interleave=False)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))
    draw_ring(ax[0], 1800, color="k", lw=1.3, label="目标区域 (R=1800)")
    for p in pts:
        ax[0].plot(*p, "b^", ms=9, label="侦察点" if p is pts[0] else None)
    for s in srcs:
        ax[0].plot(*s.pos, "r*", ms=13,
                   label="干扰源" if s is srcs[0] else None)
    P = np.array([r["p"] for r in rb.record])
    ax[0].plot(P[:, 0], P[:, 1], "g-", lw=1.0, alpha=.8, label="机器狗轨迹")
    ax[0].plot(0, 0, "ko", ms=7, label="起点/终点")
    ax[0].set_aspect("equal"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
    ax[0].set_title(f"(a) 一局轨迹（清除 {st['cleared']}/{st['total']}，"
                    f"总 {st['virtual_time_s']:.0f} s）")

    # 时间构成
    lab = ["移动", "检测", "换频道", "清除"]
    val = [st["t_move"], st["t_measure"], st["t_switch"], st["t_clear"]]
    ax[1].bar(lab, val, color=["C0", "C1", "C2", "C3"])
    for i, v in enumerate(val):
        ax[1].text(i, v, f"{v:.0f}s\n{v/sum(val)*100:.0f}%", ha="center",
                   va="bottom", fontsize=9)
    ax[1].set_ylabel("虚拟时间 / s")
    ax[1].set_title("(b) 定位清除总时间的构成")
    ax[1].grid(alpha=.3, axis="y")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_b3.png"), dpi=130)
    plt.close()
    print(f"fig_b3 完成（seed={seed}: 清除 {st['cleared']}/{st['total']}）")


# ── 图4：问题 4 —— 定向源与内外环 ────────────────────────
def fig4():
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.0))
    draw_ring(ax[0], 1800, color="k", lw=1.3, label="目标区域")
    inner = [np.asarray(p, float) for p in robot.survey_ring(6, margin=0.70)]
    outer = []
    for i in range(16):
        th = 2 * math.pi * i / 16 + math.pi / 16
        outer.append(np.array([1900 * math.cos(th), 1900 * math.sin(th)]))
    for p in inner:
        ax[0].plot(*p, "b^", ms=8, label="内环侦察点" if p is inner[0] else None)
    for p in outer:
        ax[0].plot(*p, "gv", ms=8, label="外环侦察点" if p is outer[0] else None)
    # 示例定向源：朝向场地外
    Gp = np.array([1400 * math.cos(math.radians(40)),
                   1400 * math.sin(math.radians(40))])
    d = math.radians(40)
    th = np.linspace(d - math.pi / 2, d + math.pi / 2, 100)
    ax[0].fill(np.concatenate([[Gp[0]], Gp[0] + 1500 * np.cos(th)]),
               np.concatenate([[Gp[1]], Gp[1] + 1500 * np.sin(th)]),
               color="red", alpha=.13, label="定向源覆盖半平面")
    ax[0].plot(*Gp, "r*", ms=15, label="定向源")
    ax[0].set_aspect("equal"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8,
                                                                 loc="lower left")
    ax[0].set_title("(a) 定向源只覆盖半平面，需从正面（场外）观察")

    # 清除率对比
    import json
    with open(os.path.join(HERE, "out_trials.json"), encoding="utf-8") as f:
        d0 = json.load(f)
    dfs = [0.3, 0.5, 0.7, 1.0]
    a = [d0["q4"][f"df={x}"]["ratio_mean"] * 100 for x in dfs]
    t = [d0["q4"][f"df={x}"]["avg_time_mean"] for x in dfs]
    a2 = [d0["q4_inner_only"][f"df={x}"]["ratio_mean"] * 100
          for x in (0.5, 1.0)]
    x = np.arange(len(dfs))
    ax[1].bar(x - 0.2, a, 0.4, label="内环 + 外环（23 点）", color="C0")
    ax[1].bar([1, 3], a2, 0.4, label="仅内环（7 点）", color="C3", alpha=.8)
    for i, (v, tt) in enumerate(zip(a, t)):
        ax[1].text(i - 0.2, v + 1, f"{v:.1f}%\n{tt:.0f}s", ha="center",
                   fontsize=8)
    for xx, v in zip([1, 3], a2):
        ax[1].text(xx, v + 1, f"{v:.1f}%", ha="center", fontsize=8)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([f"定向 {int(x*100)}%" for x in dfs])
    ax[1].set_ylabel("清除率 / %")
    ax[1].set_ylim(0, 118)
    ax[1].set_title("(b) 定向源占比对清除率的影响")
    ax[1].grid(alpha=.3, axis="y"); ax[1].legend(fontsize=8, loc="lower left")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_b4.png"), dpi=130)
    plt.close()
    print("fig_b4 完成")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    print("图已保存到", FIG, sorted(os.listdir(FIG)))
