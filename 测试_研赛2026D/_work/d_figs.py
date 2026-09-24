# -*- coding: utf-8 -*-
"""论文插图。"""
from __future__ import annotations

import json
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

import d_core as D
from d_q2 import World, SITES, TYPES, initial_groups, merge_pass, schedule, evaluate
import d_q3 as Q3

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.abspath(os.path.join(HERE, "..", "_交付", "figures"))
os.makedirs(FIG, exist_ok=True)


def fig1(W):
    """场景：DEM 高程 + 节点。"""
    g = W.geo
    fig, ax = plt.subplots(figsize=(9, 6))
    ext = [g.lon0, g.lon1, g.lat1, g.lat0]
    im = ax.imshow(g.dem, extent=ext, cmap="terrain", origin="upper")
    plt.colorbar(im, ax=ax, label="高程 / m", shrink=.85)
    ax.plot(W.nodes["O01"]["lon"], W.nodes["O01"]["lat"], "r*", ms=17,
            label="O01 调度中心")
    xs = [W.nodes[s]["lon"] for s in SITES]
    ys = [W.nodes[s]["lat"] for s in SITES]
    ax.plot(xs, ys, "ko", ms=5, label="服务区 S001–S015")
    for s, x, y in zip(SITES, xs, ys):
        ax.annotate(s[1:], (x, y), fontsize=6, xytext=(3, 3),
                    textcoords="offset points")
    ax.set_xlabel("经度 / °"); ax.set_ylabel("纬度 / °")
    ax.set_title("(a) 30 m DEM 与任务节点")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_d1.png"), dpi=130)
    plt.close()
    print("fig_d1 ok")


def fig2(W, p1, e1):
    """问题 2：架次甘特图 + 逐箱交付时刻。"""
    fig, ax = plt.subplots(2, 1, figsize=(11, 7),
                           gridspec_kw=dict(height_ratios=[2, 1]))
    dorder = sorted({p["drone"] for p in p1})
    ymap = {d: i for i, d in enumerate(dorder)}
    cols = {"A": "C0", "B": "C1", "C": "C2"}
    for p in sorted(p1, key=lambda x: x["start"]):
        ax[0].barh(ymap[p["drone"]], p["t"], left=p["start"], height=.6,
                   color=cols[p["g"]], alpha=.85)
    ax[0].set_yticks(range(len(dorder))); ax[0].set_yticklabels(dorder, fontsize=8)
    ax[0].axvline(3600, color="r", ls="--", lw=1.2, label="首批截止 3600 s")
    ax[0].set_xlabel("时刻 / s"); ax[0].set_ylabel("运输无人机")
    ax[0].set_title(f"(a) 运输架次甘特图（{e1['n_sorties']} 架次，"
                    f"完成 {e1['t_done']:.0f} s）")
    h = [plt.Rectangle((0, 0), 1, 1, color=cols[g]) for g in "ABC"]
    ax[0].legend(h + [plt.Line2D([], [], color="r", ls="--")],
                 ["A 型", "B 型", "C 型", "首批截止 3600 s"],
                 fontsize=8, loc="lower right")
    ax[0].grid(alpha=.3, axis="x")

    t = np.array([e1["deliv"][b["box"]] for b in W.boxes])
    ax[1].hist(t, bins=30, color="C0", alpha=.85)
    ax[1].axvline(3600, color="r", ls="--", lw=1.2)
    ax[1].set_xlabel("交付完成时刻 / s"); ax[1].set_ylabel("货箱数")
    ax[1].set_title("(b) 80 个货箱的交付时刻分布")
    ax[1].grid(alpha=.3, axis="y")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_d2.png"), dpi=130)
    plt.close()
    print("fig_d2 ok")


def fig3(W, C, p1, stations):
    """问题 3：直连链路的可用性沿航程的变化 + 中继站位。"""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for k, p in enumerate(p1[:14]):
        tr = Q3.sortie_track(W, p)
        tt = np.array([x[0] for x in tr])
        ok = np.array([1 if Q3.direct_ok(C, W, x[1], x[2], x[3]) else 0
                       for x in tr])
        ax[0].plot(tt, ok * (1 + 0.12 * (k % 5)), lw=.9, alpha=.8)
    ax[0].set_yticks([0, 1]); ax[0].set_yticklabels(["无直连", "直连"])
    ax[0].set_xlabel("时刻 / s"); ax[0].set_title("(a) 前 14 条架次的直连可用性")
    ax[0].grid(alpha=.3)

    geo = W.geo
    ext = [geo.lon0, geo.lon1, geo.lat1, geo.lat0]
    im = ax[1].imshow(geo.dem, extent=ext, cmap="gray", origin="upper",
                      alpha=.7)
    ax[1].plot(W.nodes["O01"]["lon"], W.nodes["O01"]["lat"], "r*", ms=15)
    ax[1].plot([W.nodes[s]["lon"] for s in SITES],
               [W.nodes[s]["lat"] for s in SITES], "b.", ms=5)
    for i, s in enumerate(stations, 1):
        ax[1].plot(s["point"][0], s["point"][1], "o", ms=11, mfc="none",
                   mec="C3", mew=2.2)
        ax[1].annotate(f"中继站 {i}\n(覆盖 {s['n_cover']} 个缺口)",
                       (s["point"][0], s["point"][1]), fontsize=8,
                       color="C3", xytext=(6, 6), textcoords="offset points")
    ax[1].set_xlabel("经度 / °"); ax[1].set_ylabel("纬度 / °")
    ax[1].set_title("(b) 两个中继站位（离地 300 m）")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_d3.png"), dpi=130)
    plt.close()
    print("fig_d3 ok")


def fig4(W, q4):
    """问题 4：两种分区的资源需求与缺口。"""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for k, K in enumerate(("2", "3")):
        sch = q4["schemes"][K]
        n = len(sch["groups"])
        ns = []
        from collections import Counter
        gid = {s: i for i, gp in enumerate(sch["groups"]) for s in gp}
        gg1 = merge_pass(W, initial_groups(W), allow_types=("A", "B"))
        for i in range(n):
            ns.append(sum(1 for st in gg1 if all(gid[x] == i for x, _ in st)))
        ax[0].bar(np.arange(n) + k * 0.4, ns, width=.38,
                  label=f"K={K}（极差 {max(ns)-min(ns)}）")
    ax[0].set_xlabel("任务组编号"); ax[0].set_ylabel("架次数")
    ax[0].set_title("(a) 各任务组的架次数（均衡性）")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3, axis="y")

    lab = ["运输机\n(B 型)", "中继无人机", "中继能源组件"]
    inv = [2, 2, 6]
    k2 = [q4["schemes"]["2"]["need_drone"]["B"],
          q4["schemes"]["2"]["need_relay"], 4]
    k3 = [q4["schemes"]["3"]["need_drone"]["B"],
          q4["schemes"]["3"]["need_relay"], 5]
    x = np.arange(3)
    ax[1].bar(x - 0.2, k2, .38, label="K=2 需求", color="C0")
    ax[1].bar(x + 0.2, k3, .38, label="K=3 需求", color="C1")
    ax[1].plot(x, inv, "r*", ms=14, label="现有库存")
    ax[1].set_xticks(x); ax[1].set_xticklabels(lab, fontsize=8)
    ax[1].set_ylabel("数量")
    ax[1].set_title("(b) 资源需求与库存（红星为库存）")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, axis="y")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_d4.png"), dpi=130)
    plt.close()
    print("fig_d4 ok")


if __name__ == "__main__":
    W = World()
    rt, W.relays, W.rbats = D.load_relays()
    C = D.Comms(W.geo, W.nodes, D.load_comm())
    g1 = merge_pass(W, initial_groups(W), allow_types=("A", "B"))
    p1 = schedule(W, g1)
    e1 = evaluate(W, p1)
    demands = []
    for p in p1:
        for (t0, t1, smp) in Q3.gaps_of(W, C, p):
            pt = Q3.pick_point(W, C, smp, Q3.candidate_points(W, C, 20), rt)
            if pt:
                demands.append((pt, t0, t1))
    centers = Q3.cluster_stations(W, demands, 2)
    st = Q3.station_plan(W, rt, demands, centers, C)
    q4 = json.load(open(os.path.join(HERE, "out_q4.json"), encoding="utf-8"))
    fig1(W); fig2(W, p1, e1); fig3(W, C, p1, st); fig4(W, q4)
    print("图已保存到", FIG, sorted(os.listdir(FIG)))
