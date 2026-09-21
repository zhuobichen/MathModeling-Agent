# -*- coding: utf-8 -*-
"""生成论文插图。"""
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

import a_solve as A
import drying_core as dc
import a_export as EX

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "_交付", "figures")
os.makedirs(FIG, exist_ok=True)


def fields(res, radii):
    return dc.sample_at(res, radii)


def main():
    t_bc, T_bc, C_bc = A.load_bc()
    t_R, R_att = A.load_radius()
    bc = A.BC(t_bc, T_bc, C_bc)
    fR, _ = A.make_R_of_t(t_R, R_att)

    print("求解中…")
    s1, r1 = A.run_case("q1", dc.props_q1, dc.consts_q1, bc, 1800.0, A.dt_short,
                        out_times=list(np.arange(1.0, 1801.0, 1.0)))
    s2, r2 = A.run_case("q2", dc.props_q3, dc.consts_q3, bc, 10800.0, A.dt_short,
                        out_times=list(np.arange(1.0, 10801.0, 1.0)))
    s3, r3 = A.run_case("q3", dc.props_q3, dc.consts_q3, bc, 216000.0, A.dt_long,
                        record_every=60.0)
    td3, mx3 = A.drying_time(r3)
    s4, r4 = A.run_case("q4", dc.props_q4, dc.consts_q4, bc, 480000.0, A.dt_long,
                        R_of_t=fR, record_every=60.0, material_frame=True)
    td4, mx4 = A.drying_time(r4)

    # ── 图1：问题 1 ──────────────────────────────────────
    dev = np.array([0, 0.5, 1, 1.5, 2.0])
    T1, C1 = fields(r1, dev)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.0))
    for j, d in enumerate(dev):
        ax[0].plot(r1["t"][1:], T1[1:, j], lw=1.3, label=f"r={d} cm")
        ax[1].plot(r1["t"][1:], C1[1:, j], lw=1.3, label=f"r={d} cm")
    ax[0].plot(t_bc, T_bc, "k--", lw=1.2, label="烘房 $T_\\infty$")
    ax[1].plot(t_bc, C_bc, "k--", lw=1.2, label="烘房 $C_\\infty$")
    ax[0].set_xlabel("时间 / s"); ax[0].set_ylabel("温度 / °C")
    ax[0].set_title("(a) 药材温度（问题 1）"); ax[0].grid(alpha=.3)
    ax[0].legend(fontsize=7)
    ax[1].set_xlabel("时间 / s"); ax[1].set_ylabel("水分浓度 / (kg/kg)")
    ax[1].set_title("(b) 药材水分浓度（问题 1）"); ax[1].grid(alpha=.3)
    ax[1].legend(fontsize=7)
    # 空间剖面
    for tt, c in ((300, "C0"), (900, "C1"), (1800, "C2")):
        i = int(np.argmin(np.abs(r1["t"] - tt)))
        ax[2].plot(dev, C1[i], "o-", ms=3, color=c, label=f"t={tt} s")
    ax[2].set_xlabel("到药材中心距离 / cm"); ax[2].set_ylabel("水分浓度 / (kg/kg)")
    ax[2].set_title("(c) 水分径向剖面"); ax[2].grid(alpha=.3); ax[2].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_a1.png"), dpi=130)
    plt.close()

    # ── 图2：问题 2 ──────────────────────────────────────
    T2, C2 = fields(r2, dev)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for j, d in enumerate(dev):
        ax[0].plot(r2["t"][1:] / 3600, C2[1:, j], lw=1.3, label=f"r={d} cm")
        ax[1].plot(r2["t"][1:] / 3600, T2[1:, j], lw=1.3, label=f"r={d} cm")
    ax[0].plot(t_bc / 3600, C_bc, "k--", lw=1.2, label="烘房 $C_\\infty$")
    ax[1].plot(t_bc / 3600, T_bc, "k--", lw=1.2, label="烘房 $T_\\infty$")
    ax[0].set_xlabel("时间 / h"); ax[0].set_ylabel("水分浓度 / (kg/kg)")
    ax[0].set_title("(a) 水分浓度（问题 2）"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=7)
    ax[1].set_xlabel("时间 / h"); ax[1].set_ylabel("温度 / °C")
    ax[1].set_title("(b) 温度（问题 2）"); ax[1].grid(alpha=.3); ax[1].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_a2.png"), dpi=130)
    plt.close()

    # ── 图3：问题 3 烘干过程 ─────────────────────────────
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    tt = np.array([6, 12, 24, 36, 48]) * 3600
    for c, t_ in zip(plt.cm.viridis(np.linspace(0, .85, len(tt))), tt):
        i = int(np.argmin(np.abs(r3["t"] - t_)))
        Cv = dc.sample_at(r3, list(np.arange(0, 2.01, 0.1)))[1][i]
        ax[0].plot(np.arange(0, 2.01, 0.1), Cv, lw=1.4, color=c, label=f"{t_/3600:.0f} h")
    ax[0].axhline(0.15, color="r", ls="--", lw=1.4, label="判据 0.15")
    ax[0].set_xlabel("到药材中心距离 / cm"); ax[0].set_ylabel("水分浓度 / (kg/kg)")
    ax[0].set_title("(a) 水分径向剖面随时间的变化"); ax[0].grid(alpha=.3)
    ax[0].legend(fontsize=7)
    ax[1].semilogy(r3["t"][1:] / 3600, mx3[1:], lw=1.5, color="C0")
    ax[1].axhline(0.15, color="r", ls="--", lw=1.4, label="判据 0.15")
    ax[1].axvline(td3 / 3600, color="g", ls=":", lw=1.6,
                  label=f"烘干时长 {td3/3600:.2f} h")
    ax[1].set_xlabel("时间 / h"); ax[1].set_ylabel("全场最大水分浓度 / (kg/kg，对数轴)")
    ax[1].set_title("(b) 全场最大水分浓度的收敛"); ax[1].grid(alpha=.3, which="both")
    ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_a3.png"), dpi=130)
    plt.close()

    # ── 图4：问题 4 收缩与附件 2 对照 ────────────────────
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.1))
    tt = np.array([6, 24, 48, 72, 96]) * 3600
    for c, t_ in zip(plt.cm.plasma(np.linspace(0, .8, len(tt))), tt):
        i = int(np.argmin(np.abs(r4["t"] - t_)))
        Rc = r4["R"][i] * 100
        g = np.arange(0, Rc + 1e-9, 0.1)
        Cv = dc.sample_at(r4, list(g))[1][i]
        ax[0].plot(g, Cv, lw=1.4, color=c, label=f"{t_/3600:.0f} h (R={Rc:.2f} cm)")
    ax[0].axhline(0.15, color="r", ls="--", lw=1.4, label="判据 0.15")
    ax[0].set_xlabel("到药材中心距离 / cm"); ax[0].set_ylabel("水分浓度 / (kg/kg)")
    ax[0].set_title("(a) 收缩过程中水分剖面的变化"); ax[0].grid(alpha=.3)
    ax[0].legend(fontsize=6.5)
    # 半径
    ax[1].plot(t_R / 3600, R_att * 100, "o-", ms=3, lw=1.3, label="附件 2 半径")
    ax[1].plot(r4["t"] / 3600, r4["R"] * 100, lw=1.2, ls="--", label="模型所用 R(t)")
    ax[1].set_xlabel("时间 / h"); ax[1].set_ylabel("半径 / cm")
    ax[1].set_title("(b) 药材半径的收缩"); ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
    # 模型 vs 附件2 反推的含水率
    V0 = (1 + A.C_INIT) / (760 + 90 * A.C_INIT)

    def cbar_from_R(Rcm):
        VV = (Rcm / 2.0) ** 2
        k = VV * (1 + A.C_INIT) / (760 + 90 * A.C_INIT)
        return (760 * k - 1) / (1 - 90 * k)

    ax[2].plot(t_R / 3600, [cbar_from_R(v * 100) for v in R_att], "o-", ms=3,
               lw=1.3, label="由附件 2 半径反推")
    ax[2].plot(r4["t"] / 3600, mx4, lw=1.4, label="模型（附录 4）")
    for mult, ls in ((5.0, "--"),):
        def props(C, T, m=mult):
            rho, cp, k, D = dc.props_q4(C, T)
            return rho, cp, k, D * m
        _, rr = A.run_case("d5", props, dc.consts_q4, bc, 480000.0, A.dt_long,
                           R_of_t=fR, record_every=60.0, material_frame=True)
        td5, mx5 = A.drying_time(rr)
        ax[2].plot(rr["t"] / 3600, mx5, ls=ls, lw=1.4,
                   label=f"模型（D×{mult:.0f}，{td5/3600:.1f} h）")
    ax[2].axhline(0.15, color="r", ls=":", lw=1.4, label="判据 0.15")
    ax[2].set_xlim(0, 140); ax[2].set_ylim(-0.1, 2.6)
    ax[2].set_xlabel("时间 / h"); ax[2].set_ylabel("水分浓度 / (kg/kg)")
    ax[2].set_title("(c) 模型与附件 2 的对照"); ax[2].grid(alpha=.3)
    ax[2].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_a4.png"), dpi=130)
    plt.close()

    print("图已保存到", FIG, os.listdir(FIG))


if __name__ == "__main__":
    main()
