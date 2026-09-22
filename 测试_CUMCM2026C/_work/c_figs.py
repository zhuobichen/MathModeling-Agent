# -*- coding: utf-8 -*-
"""论文插图。"""
from __future__ import annotations

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

import c_data as D
import c_solve as C
import mg_core as M

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.abspath(os.path.join(HERE, "..", "_交付", "figures"))
os.makedirs(FIG, exist_ok=True)


def hrs():
    return np.arange(M.T) * M.DT


def fig1():
    E, P, Q = D.load_att1()
    sol = M.solve_day(E, P, Q, 6000.0, cyclic=True)
    t = hrs()
    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    ax[0].plot(t, P, "k-", lw=1.6, label="小区负载")
    ax[0].plot(t, Q, "C2-", lw=1.6, label="光伏发电")
    ax[0].fill_between(t, 0, Q, color="C2", alpha=.15)
    ax[0].set_ylabel("功率 / kW"); ax[0].legend(fontsize=9); ax[0].grid(alpha=.3)
    ax[0].set_title("(a) 负荷与光伏（附件 1）")
    ax[1].plot(t, E, "C3-", lw=1.6)
    ax[1].set_ylabel("电价 / (元/kWh)"); ax[1].grid(alpha=.3)
    ax[1].set_title("(b) 分时电价")
    ax[2].plot(t, sol["buy"], "C0-", lw=1.5, label="计划购电")
    ax[2].plot(t, sol["dis"], "C1-", lw=1.3, label="放电")
    ax[2].plot(t, -sol["ch"], "C4-", lw=1.3, label="充电（负向）")
    ax2 = ax[2].twinx()
    ax2.plot(t, sol["soc"] / 1000, "k--", lw=1.3, label="储电量（右轴，MWh）")
    ax2.set_ylabel("储电量 / MWh")
    ax2.axhline(M.SOC_MAX / 1000, color="r", ls=":", lw=1)
    ax2.axhline(M.SOC_MIN / 1000, color="r", ls=":", lw=1)
    ax[2].set_xlabel("时刻 / h"); ax[2].set_ylabel("功率 / kW")
    ax[2].grid(alpha=.3)
    h1, l1 = ax[2].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax[2].legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    ax[2].set_title("(c) 最优购电、充放电与储电量")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_c1.png"), dpi=130)
    plt.close()
    print("fig_c1 ok")


def fig2():
    E, _, _ = D.load_att1()
    load, pv = D.load_att2()
    days = C.run_q2(E)
    sel = days[31:]
    x = np.arange(len(sel))
    cost = np.array([d["cost"] for d in sel])
    buy = np.array([float(d["sol"]["buy"].sum() * M.DT) for d in sel])
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    ax[0, 0].plot(x, buy, lw=.9, color="C0")
    ax[0, 0].set_ylabel("日购电量 / kWh"); ax[0, 0].grid(alpha=.3)
    ax[0, 0].set_title("(a) 逐日计划购电量（2025-02-01 起）")
    ax[0, 1].plot(x, cost, lw=.9, color="C3")
    ax[0, 1].set_ylabel("日购电费 / 元"); ax[0, 1].grid(alpha=.3)
    ax[0, 1].set_title("(b) 逐日购电费")
    # 月度汇总
    mon = np.zeros(11)
    for i, d in enumerate(sel):
        mon[D.day_label(d["d"])[0] - 2] += d["cost"]
    ax[1, 0].bar(range(2, 13), mon, color="C0", alpha=.85)
    ax[1, 0].set_xlabel("月份"); ax[1, 0].set_ylabel("月购电费 / 元")
    ax[1, 0].set_title("(c) 月度购电费"); ax[1, 0].grid(alpha=.3, axis="y")
    # 负荷 vs 光伏（全年日均）
    ax[1, 1].plot(np.arange(365), load.sum(1) * M.DT, lw=.9, label="日负荷")
    ax[1, 1].plot(np.arange(365), pv.sum(1) * M.DT, lw=.9, label="日光伏")
    ax[1, 1].set_xlabel("日序号"); ax[1, 1].set_ylabel("电量 / kWh")
    ax[1, 1].legend(fontsize=8); ax[1, 1].grid(alpha=.3)
    ax[1, 1].set_title("(d) 全年日负荷与日光伏")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_c2.png"), dpi=130)
    plt.close()
    print("fig_c2 ok")
    return days


def fig3():
    E, _, _ = D.load_att1()
    F = D.load_att3(); L, V = D.load_att2()
    cal = range(0, 31)
    bias = C.calibrate_bias(F, V, cal)
    sigma = C.calibrate_sigma(F, V, cal)
    # (a) 预报误差分布（按发布时刻）
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for iss, c in ((0, "C0"), (6, "C1"), (12, "C2")):
        errs = []
        for d in range(365):
            for k in range(1, 25):
                h = (iss + k - 1) % 24
                if h * 6 < iss * 6:
                    continue
                errs.append(V[d][h * 6:(h + 1) * 6].mean() - F[(d, iss)][k - 1])
        e = np.array(errs)
        ax[0].hist(e[e != 0], bins=60, histtype="step", lw=1.4, color=c,
                   label=f"{iss}:00 发布（均值 {e.mean():+.0f}，σ={e.std():.0f}）")
    ax[0].axvline(0, color="k", ls=":", lw=1)
    ax[0].set_xlabel("预报误差（实际−预报）/ kW"); ax[0].set_ylabel("频数")
    ax[0].set_title("(a) 光伏预报误差分布"); ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3)
    # (b) 偏差订正效果（按步长）
    ks = range(1, 25)
    b = [bias(0, k) for k in ks]
    s = [sigma(0, k) for k in ks]
    ax[1].plot(list(ks), b, "o-", ms=3, lw=1.4, label="偏差（实际−预报的均值）")
    ax[1].plot(list(ks), s, "s--", ms=3, lw=1.4, label="标准差 σ")
    ax[1].axhline(0, color="k", ls=":", lw=1)
    ax[1].set_xlabel("预报步长 k / 小时"); ax[1].set_ylabel("kW")
    ax[1].set_title("(b) 0:00 发布的偏差与标准差"); ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3)
    # (c) 余量扫描
    zs = [0.0, 0.25, 0.524, 0.75, 1.0]
    tot = []
    for z in zs:
        days = C.run_q3(E, bias=bias, sigma=sigma, z=z)
        sel = [d for d in days if d["d"] >= 31]
        tot.append(sum(d["cost"] for d in sel) / 1e4)
    ax[2].plot(zs, tot, "o-", lw=1.6)
    j = int(np.argmin(tot))
    ax[2].plot(zs[j], tot[j], "r*", ms=15,
               label=f"最优 z={zs[j]}，{tot[j]:.0f} 万元")
    ax[2].axhline(tot[0], color="k", ls=":", lw=1.2,
                  label=f"不加余量 {tot[0]:.0f} 万元")
    ax[2].set_xlabel("余量系数 z（余量 = z·σ）"); ax[2].set_ylabel("总费用 / 万元")
    ax[2].set_title("(c) 分位余量的扫描"); ax[2].legend(fontsize=8)
    ax[2].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_c3.png"), dpi=130)
    plt.close()
    print("fig_c3 ok", [f"{v:.1f}" for v in tot])


def fig4():
    E, _, _ = D.load_att1()
    A = D.load_att4()
    F = D.load_att3(); L, V = D.load_att2()
    bias = C.calibrate_bias(F, V, range(0, 31))
    sigma = C.calibrate_sigma(F, V, range(0, 31))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    ax[0].plot(np.arange(M.T) * M.DT, E, lw=1.6, label="附件1 分时电价")
    ax[0].plot(np.arange(M.T) * M.DT, A[80], lw=1.0, alpha=.85,
               label="附件4 实时电价（3-22）")
    ax[0].set_xlabel("时刻 / h"); ax[0].set_ylabel("电价 / (元/kWh)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[0].set_title("(a) 两种电价机制")
    ax[1].hist(A.ravel(), bins=80, color="C3", alpha=.8)
    ax[1].set_xlabel("电价 / (元/kWh)"); ax[1].set_ylabel("频数")
    ax[1].grid(alpha=.3)
    ax[1].set_title(f"(b) 附件4 电价分布（{A.min():.3f}~{A.max():.3f}）")
    # 四种情形的费用对比
    labels = ["问题2\n(固定电价\n完美信息)", "问题4-2\n(实时电价\n完美信息)",
              "问题3\n(固定电价\n预报+调整)", "问题4-3\n(实时电价\n预报+调整)"]
    vals = [1225.5, 1284.1, 1496.1, 1562.6]
    cols = ["C0", "C1", "C2", "C3"]
    ax[2].bar(range(4), vals, color=cols, alpha=.85)
    for i, v in enumerate(vals):
        ax[2].text(i, v + 15, f"{v:.0f}", ha="center", fontsize=9)
    ax[2].set_xticks(range(4)); ax[2].set_xticklabels(labels, fontsize=8)
    ax[2].set_ylabel("全年购电总费用 / 万元"); ax[2].grid(alpha=.3, axis="y")
    ax[2].set_title("(c) 四种情形的费用对比（2025-02-01 起 334 天）")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_c4.png"), dpi=130)
    plt.close()
    print("fig_c4 ok")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    print("图已保存到", FIG, sorted(os.listdir(FIG)))
