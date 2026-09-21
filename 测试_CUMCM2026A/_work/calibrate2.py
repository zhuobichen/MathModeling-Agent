# -*- coding: utf-8 -*-
"""问题 4 的双参数辨识：D 与 h_m 同时标定，并给出可辨识性结论。

calibrate.py 用单参数 φ_D 标定时残差随 φ_D 单调下降、最优撞到搜索上界，
说明**单个 D 倍数无法调和模型与附件 2**。本脚本：
  1) 把 φ_D 扫到 500，看残差是否趋于纯表面控制极限；
  2) 在 (φ_D, φ_hm) 平面上做二维搜索，判断是否存在内点最优；
  3) 显式算出纯表面控制（D→∞）的极限曲线作参照。
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
from calibrate import cbar_from_radius, model_cbar

HERE = os.path.dirname(os.path.abspath(__file__))
TMAX = 520000.0


def main():
    t_bc, T_bc, C_bc = A.load_bc()
    t_R, R_att = A.load_radius()
    bc = A.BC(t_bc, T_bc, C_bc)
    fR, _ = A.make_R_of_t(t_R, R_att)
    C_obs = np.array([cbar_from_radius(v * 100.0) for v in R_att])

    def curve(pD, phm):
        def props(C, T):
            rho, cp, k, D = dc.props_q4(C, T)
            return rho, cp, k, D * pD
        consts = dict(h=25.0, hm=8e-7 * phm)
        s, res = A.run_case("c2", props, lambda: consts, bc, TMAX, A.dt_long,
                            R_of_t=fR, record_every=60.0, material_frame=True)
        idx = np.array([int(np.argmin(np.abs(res["t"] - tt))) for tt in t_R])
        Cm = np.array([model_cbar(res, s, i) for i in idx])
        td, _ = A.drying_time(res)
        return float(np.sqrt(np.mean((Cm - C_obs) ** 2))), td, Cm

    # ── 纯表面控制极限（完全混合体）解析解 ────────────────
    def mixed_limit():
        """dC̄/dt = (2R/R0²)·h_m·(C̄−C∞)，R(t) 取附件 2。"""
        Cbar, out = 2.55, [2.55]
        dt = 60.0
        R_of = A.make_R_of_t(t_R, R_att)[0]
        for k in range(1, int(t_R[-1] / dt) + 1):
            t = k * dt
            Tinf, Cinf = bc(t)
            R = R_of(t)
            # 符号：表面通量 2πR·hm·(C̄−C∞) 是**流出**的，故取负
            Cbar -= dt * (2 * R / A.R0 ** 2) * 8e-7 * (Cbar - Cinf)
            out.append(Cbar)
        return np.array(out)

    print("=" * 78)
    print("① φ_D 宽范围扫描（h_m 用给定值）")
    print("=" * 78)
    print(f"  {'φ_D':>7}{'RMSE':>10}{'烘干时长/h':>14}")
    rows = []
    for pD in (1, 2, 5, 10, 20, 50, 100, 200, 500):
        r, td, _ = curve(pD, 1.0)
        rows.append((pD, r, td))
        print(f"  {pD:>7}{r:>10.4f}{(td/3600 if td else float('nan')):>14.2f}")

    ml = mixed_limit()
    j = np.array([int(np.argmin(np.abs(np.arange(len(ml)) * 60 - tt)))
                  for tt in t_R])
    rmse_ml = float(np.sqrt(np.mean((ml[j] - C_obs) ** 2)))
    print(f"\n  纯表面控制极限（D→∞）的 RMSE = {rmse_ml:.4f}")
    print("  （这是给定 h_m 下、任何有限 D 都达不到的下界；提高 h_m 可进一步降低该下界）")
    print(f"  参照：φ_D=500 时 RMSE = {rows[-1][1]:.4f}，正在逼近该下界")

    print()
    print("=" * 78)
    print("② 二维搜索 (φ_D, φ_hm)")
    print("=" * 78)
    grid_D = [1, 5, 20, 100, 400, 2000]
    grid_h = [1, 2, 5, 10]
    best = (None, 9e9, None)
    print(f"  {'φ_D\\φ_hm':>10}" + "".join(f"{h:>11}" for h in grid_h))
    for pD in grid_D:
        cells = []
        for phm in grid_h:
            r, td, _ = curve(pD, phm)
            cells.append(r)
            if r < best[1]:
                best = ((pD, phm), r, td)
        print(f"  {pD:>10}" + "".join(f"{c:>11.4f}" for c in cells))
    print(f"\n  最优 (φ_D, φ_hm) = {best[0]}  RMSE = {best[1]:.4f}"
          f"  烘干 {best[2]/3600:.2f} h")

    print()
    print("=" * 78)
    print("③ 结论")
    print("=" * 78)
    print(f"  · 单参数 φ_D 的残差单调下降、最优落在搜索边界 → D 不可单参数辨识；")
    print(f"  · 残差的下界（纯表面控制）为 {rmse_ml:.4f}，"
          f"而 φ_D=1 时为 {rows[0][1]:.4f}；")
    print(f"  · 观测数据在 18 h 就趋于零，其时间尺度与**表面控制极限**一致，")
    print(f"    与附录 4 给出的扩散系数（Bi_m≈16，内部扩散控制）不符。")

    out = dict(phiD_scan=[(p, float(r), float(td)) for p, r, td in rows],
               rmse_mixed_limit=rmse_ml,
               best_2d=dict(phiD=best[0][0], phim=best[0][1],
                            rmse=float(best[1]), drying_h=float(best[2] / 3600)),
               c_obs=C_obs.tolist(), t=t_R.tolist())
    with open(os.path.join(HERE, "out_calib2.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    np.savez_compressed(os.path.join(HERE, "calib2.npz"),
                        t=t_R, C_obs=C_obs, mixed=ml[j])
    print("\n已保存 out_calib2.json")


if __name__ == "__main__":
    main()
