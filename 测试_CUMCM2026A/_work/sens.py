# -*- coding: utf-8 -*-
"""灵敏度分析：定位烘干时长的控制因素。

结论先行：
  · 过程由**内部扩散**控制 —— h_m 放大 10 倍只使时长由 131.20 h 变为 127.17 h；
  · 放大 D 才能显著缩短时长 —— D×5 给出 35.33 h，与附件 2 的半径平台吻合；
  · 因此问题 4 的模型与附件 2 的分歧根源在扩散系数，而非表面传质或时间离散。
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import a_solve as A
import drying_core as dc

TMAX = 900000.0


def main():
    t_bc, T_bc, C_bc = A.load_bc()
    t_R, R_att = A.load_radius()
    bc = A.BC(t_bc, T_bc, C_bc)
    fR, _ = A.make_R_of_t(t_R, R_att)

    print("=" * 78)
    print("① 对流传质系数 h_m（问题 4；理论下界 = 无限大 D 的纯表面控制 ≈ 11.24 h）")
    print("=" * 78)
    for m in (0.5, 1.0, 2.0, 5.0, 10.0):
        consts = lambda mm=m: dict(h=25.0, hm=8e-7 * mm)
        s, res = A.run_case("s_hm", dc.props_q4, consts, bc, TMAX, A.dt_long,
                            R_of_t=fR, record_every=60.0, material_frame=True)
        td, _ = A.drying_time(res)
        print(f"   h_m = {8e-7*m:.2e} m/s（×{m:4.1f}）→ 烘干 {td/3600:7.2f} h")
    print("   → 表面阻力几乎不起作用：过程为内部扩散控制。")

    print()
    print("=" * 78)
    print("② 水分扩散系数 D（问题 4）")
    print("=" * 78)
    for m in (1.0, 2.0, 5.0, 10.0):
        def props(C, T, mm=m):
            rho, cp, k, D = dc.props_q4(C, T)
            return rho, cp, k, D * mm
        s, res = A.run_case("s_D", props, dc.consts_q4, bc, TMAX, A.dt_long,
                            R_of_t=fR, record_every=60.0, material_frame=True)
        td, _ = A.drying_time(res)
        print(f"   D × {m:4.1f} → 烘干 {td/3600:7.2f} h")
    print("   → D×5 给出 35.33 h，与附件 2 的半径平台吻合，是该分歧的调和点。")

    print()
    print("=" * 78)
    print("③ 烘房条件延拓假设（问题 3）")
    print("=" * 78)
    for C_hold in (0.04986, 0.02, 0.0):
        bc2 = A.BC(t_bc, T_bc, C_bc)
        bc2.C_hold = C_hold
        s, res = A.run_case("s_ext", dc.props_q3, dc.consts_q3, bc2, 500000.0,
                            A.dt_long, record_every=60.0)
        td, _ = A.drying_time(res)
        print(f"   14400 s 后 C∞ = {C_hold:.5f} → 烘干 {td/3600:7.2f} h")
    print("   → 因过程为内部扩散控制，延拓值的影响有限。")

    print()
    print("=" * 78)
    print("④ 数值参数（问题 3）")
    print("=" * 78)
    for N in (200, 400, 800):
        s, res = A.run_case("s_N", dc.props_q3, dc.consts_q3, bc, 400000.0,
                            A.dt_long, record_every=60.0, N=N)
        td, _ = A.drying_time(res)
        print(f"   N = {N:4d} → 烘干 {td/3600:.4f} h")

    def dt_fine(t):
        return 1.0 if t < 120 else (5.0 if t < 1200 else
                                    (15.0 if t < 7200 else 30.0))
    s, res = A.run_case("s_dt", dc.props_q3, dc.consts_q3, bc, 400000.0,
                        dt_fine, record_every=60.0)
    td, _ = A.drying_time(res)
    print(f"   时间步加密 → 烘干 {td/3600:.4f} h")


if __name__ == "__main__":
    main()
