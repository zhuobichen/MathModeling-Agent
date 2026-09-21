# -*- coding: utf-8 -*-
"""标定与可辨识性插图。"""
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
from calibrate import cbar_from_radius, model_cbar

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.abspath(os.path.join(HERE, "..", "_交付", "figures"))
TMAX = 520000.0


def run(pD, phm=1.0):
    def props(C, T):
        rho, cp, k, D = dc.props_q4(C, T)
        return rho, cp, k, D * pD
    consts = lambda: dict(h=25.0, hm=8e-7 * phm)
    return A.run_case("fc", props, consts, BC, TMAX, A.dt_long,
                      R_of_t=FR, record_every=60.0, material_frame=True)


t_bc, T_bc, C_bc = A.load_bc()
t_R, R_att = A.load_radius()
BC = A.BC(t_bc, T_bc, C_bc)
FR, _ = A.make_R_of_t(t_R, R_att)

if __name__ == "__main__":
    C_obs = np.array([cbar_from_radius(v * 100.0) for v in R_att])

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    ax[0].plot(t_R / 3600, C_obs, "k.-", ms=4, lw=1.6, label="附件 2 反推（观测）")
    for pD, c in ((1.0, "C0"), (5.0, "C1"), (100.0, "C2")):
        s, res = run(pD)
        idx = np.array([int(np.argmin(np.abs(res["t"] - tt))) for tt in t_R])
        Cm = np.array([model_cbar(res, s, i) for i in idx])
        ax[0].plot(t_R / 3600, Cm, color=c, lw=1.4, label=f"模型 $D\\times${pD:.0f}")
    ax[0].axhline(0.15, color="r", ls="--", lw=1.2, label="判据 0.15")
    ax[0].set_xlabel("时间 / h")
    ax[0].set_ylabel("体平均含水率 $\\bar C$ / (kg/kg)")
    ax[0].set_title("(a) 附件 2 反推曲线与模型的对照")
    ax[0].set_ylim(-0.15, 2.75); ax[0].grid(alpha=.3); ax[0].legend(fontsize=7.5)

    phis = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500])
    rm = []
    for pD in phis:
        s, res = run(float(pD))
        idx = np.array([int(np.argmin(np.abs(res["t"] - tt))) for tt in t_R])
        Cm = np.array([model_cbar(res, s, i) for i in idx])
        rm.append(float(np.sqrt(np.mean((Cm - C_obs) ** 2))))
    ax[1].semilogx(phis, rm, "o-", color="C3", lw=1.6, ms=5,
                   label="单参数 $\\varphi_D$ 标定")
    ax[1].axhline(0.1577, color="k", ls=":", lw=1.4,
                  label="纯表面控制下界（$D\\to\\infty$）")
    ax[1].set_xlabel("$D$ 放大倍数 $\\varphi_D$（对数轴）")
    ax[1].set_ylabel("全曲线 RMSE$(\\bar C)$")
    ax[1].set_title("(b) 残差随 $\\varphi_D$ 单调下降、无内点极小")
    ax[1].set_ylim(0.1, 0.6); ax[1].grid(alpha=.3, which="both")
    ax[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig_a5.png"), dpi=130)
    print("saved", os.path.join(FIG, "fig_a5.png"))
