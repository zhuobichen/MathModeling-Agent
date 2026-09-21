# -*- coding: utf-8 -*-
"""用附件 2 的半径数据反演标定问题 4 的扩散系数。

思路：附件 2 给出 R(t)。由附录 4 的密度式把 R 换算成**体平均含水率** C̄(t)
（该换算在 t=0 处自洽给出 2.55），这给出一条独立于模型的观测曲线。把模型
的 D 乘一个待定因子 φ，求使模型 C̄(t; φ) 与该观测曲线残差最小的 φ。

这不是"调参数去凑答案"，而是标准的参数辨识：先给出可辨识性与残差分析，
再用标定后的模型给出烘干时长，并同时报告未标定的原值。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import a_solve as A
import drying_core as dc

HERE = os.path.dirname(os.path.abspath(__file__))
TMAX = 520000.0          # 需覆盖未标定(472337 s)与标定后的烘干时刻
RHO_S = 760.0            # 附录 4 的密度式 ρ = 760 + 90C
K_RHO = 90.0


def cbar_from_radius(R_cm, C0=2.55, R0=2.0):
    """由半径反推体平均含水率（附录 4 密度式，均匀 C 近似）。"""
    g = lambda C: (1.0 + C) / (RHO_S + K_RHO * C)
    g0 = g(C0)
    kk = (R_cm / R0) ** 2 * g0
    return (RHO_S * kk - 1.0) / (1.0 - K_RHO * kk)


def model_cbar(sol, s, idx):
    """模型在给定记录点上的体平均含水率（按当前体积加权）。"""
    V = s.V / s.V.sum()
    return float(np.sum(sol["C"][idx] * V))


def main():
    t_bc, T_bc, C_bc = A.load_bc()
    t_R, R_att = A.load_radius()
    bc = A.BC(t_bc, T_bc, C_bc)
    fR, _ = A.make_R_of_t(t_R, R_att)

    # 观测曲线
    C_obs = np.array([cbar_from_radius(v * 100.0) for v in R_att])
    print("=" * 78)
    print("附件 2 → 体平均含水率的观测曲线")
    print("=" * 78)
    print(f"  t=0 校验：由 R=2.000 cm 反推 C̄={C_obs[0]:.4f}（应等于初值 2.55）")
    print(f"  数据点 {len(t_R)} 个，覆盖 {t_R[-1]/3600:.1f} h；"
          f"C̄ 由 {C_obs[0]:.3f} 降到 {C_obs[-1]:.4f}")

    def run(phi):
        def props(C, T, p=phi):
            rho, cp, k, D = dc.props_q4(C, T)
            return rho, cp, k, D * p
        s, res = A.run_case("cal", props, dc.consts_q4, bc, TMAX, A.dt_long,
                            R_of_t=fR, record_every=60.0, material_frame=True)
        return s, res

    def misfit(phi, verbose=False):
        s, res = run(phi)
        rec = res["t"]
        idx = np.array([int(np.argmin(np.abs(rec - tt))) for tt in t_R])
        Cm = np.array([model_cbar(res, s, i) for i in idx])
        r = Cm - C_obs
        rmse = float(np.sqrt(np.mean(r ** 2)))
        td, _ = A.drying_time(res)
        if verbose:
            return rmse, td, Cm, idx
        return rmse

    print()
    print("=" * 78)
    print("φ = D 放大倍数 的残差扫描")
    print("=" * 78)
    print(f"  {'φ':>6}{'RMSE(C̄)':>12}{'烘干时长/h':>14}")
    scan = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0]
    rmses = []
    for phi in scan:
        r, td, _, _ = misfit(phi, verbose=True)
        rmses.append(r)
        print(f"  {phi:>6.1f}{r:>12.4f}{td/3600:>14.2f}")

    print()
    print("=" * 78)
    print("黄金分割细化")
    print("=" * 78)
    lo, hi = 1.0, 12.0
    gr = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = misfit(c), misfit(d)
    for _ in range(16):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = misfit(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = misfit(d)
    phi_opt = (a + b) / 2
    rmse_opt, td_opt, Cm_opt, idx = misfit(phi_opt, verbose=True)
    print(f"  最优 φ* = {phi_opt:.3f}   RMSE = {rmse_opt:.4f}")

    # 未标定的残差作对照
    rmse_raw, td_raw, Cm_raw, _ = misfit(1.0, verbose=True)
    print(f"  未标定 φ=1   RMSE = {rmse_raw:.4f}   烘干 {td_raw/3600:.2f} h")
    print(f"  标定后       RMSE = {rmse_opt:.4f}   烘干 {td_opt/3600:.2f} h"
          f"（残差降为原来的 {rmse_opt/rmse_raw*100:.1f}%）")

    print()
    print("=" * 78)
    print("标定拟合的逐点对照（每 6 h）")
    print("=" * 78)
    print(f"  {'t/h':>7}{'C̄ 观测':>10}{'C̄ 模型(未标定)':>16}{'C̄ 模型(标定)':>15}")
    for tt in np.arange(0, t_R[-1] + 1, 21600):
        j = int(np.argmin(np.abs(t_R - tt)))
        print(f"  {t_R[j]/3600:>7.1f}{C_obs[j]:>10.4f}"
              f"{Cm_raw[j]:>16.4f}{Cm_opt[j]:>15.4f}")

    out = dict(phi_opt=float(phi_opt),
               rmse_raw=float(rmse_raw), rmse_cal=float(rmse_opt),
               drying_h_raw=float(td_raw / 3600),
               drying_h_cal=float(td_opt / 3600),
               c_obs=C_obs.tolist(),
               c_model_cal=Cm_opt.tolist(),
               c_model_raw=Cm_raw.tolist(),
               t=t_R.tolist())
    with open(os.path.join(HERE, "out_calib.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    np.savez_compressed(os.path.join(HERE, "calib.npz"),
                        t=t_R, C_obs=C_obs, C_raw=Cm_raw, C_cal=Cm_opt)
    print("\n已保存 out_calib.json")


if __name__ == "__main__":
    main()
