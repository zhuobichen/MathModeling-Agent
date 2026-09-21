# -*- coding: utf-8 -*-
"""二维轴对称解验证一维径向假设。

两步：
  ① 自检：把圆柱取得足够长（H 很大），二维解应退化到一维解——若两者
     不一致，说明二维实现本身有错；
  ② 实测：用题目几何（R=2 cm、L=25 cm，两端也参与对流）算问题 3，
     与一维结果比较，量化一维假设对烘干时长的偏差。
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import a_solve as A
import drying_core as dc
from drying2d import Cylinder2D

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    t_bc, T_bc, C_bc = A.load_bc()
    bc = A.BC(t_bc, T_bc, C_bc)

    print("=" * 78)
    print("① 自检：长圆柱（H=5 m）的二维解 应 退化为一维解")
    print("=" * 78)
    for Nr, Nz in ((40, 20), (60, 40)):
        s2 = Cylinder2D(Nr, Nz, dc.props_q3, dc.consts_q3, R=0.02, H=5.0)
        sol = s2.run(14400.0, A.dt_long, 28.0, 2.55, bc, record_every=3600.0)
        s1, r1 = A.run_case("v1", dc.props_q3, dc.consts_q3, bc, 14400.0,
                            A.dt_long, record_every=3600.0)
        # 比较轴向对称面上的径向剖面
        C2 = sol["C"][-1][:, 0]                     # j=0 即 z=0（中点）
        C1 = r1["C"][-1]
        rmid = 0.5 * (np.linspace(0, 0.02, Nr + 1)[:-1]
                      + np.linspace(0, 0.02, Nr + 1)[1:])
        C1i = np.interp(rmid, (np.arange(400) + 0.5) / 400 * 0.02, C1)
        err = float(np.max(np.abs(C2 - C1i)))
        print(f"   Nr={Nr:3d} Nz={Nz:3d}  4 h 后径向剖面最大差 = {err:.3e}"
              f"   {'OK' if err < 5e-3 else '**FAIL**'}")

    print()
    print("=" * 78)
    print("② 实测：题目几何（R=2 cm、L=25 cm，两端对流）")
    print("=" * 78)
    print(f"  {'Nr×Nz':>10}{'烘干时长/h':>14}{'与一维之差':>14}")
    res = {}
    for Nr, Nz in ((40, 40), (60, 60), (80, 80)):
        s2 = Cylinder2D(Nr, Nz, dc.props_q3, dc.consts_q3, R=0.02, H=0.125)
        sol = s2.run(220000.0, A.dt_long, 28.0, 2.55, bc, record_every=60.0)
        td, _ = s2.drying_time(sol)
        res[(Nr, Nz)] = td
        print(f"  {f'{Nr}×{Nz}':>10}{(td/3600 if td else float('nan')):>14.2f}")

    s1, r1 = A.run_case("v1b", dc.props_q3, dc.consts_q3, bc, 216000.0,
                        A.dt_long, record_every=60.0)
    td1, _ = A.drying_time(r1)
    print(f"  {'一维 400':>10}{td1/3600:>14.2f}")

    fine = res[(80, 80)]
    print()
    print(f"  二维（80×80）{fine/3600:.2f} h  vs  一维 {td1/3600:.2f} h"
          f"  → 一维高估 {(td1/fine-1)*100:.2f}%")
    print("  → 一维径向假设使烘干时长偏长（保守）；两端面对流提供了额外的干燥通道。")

    np.savez_compressed(os.path.join(HERE, "val2d.npz"),
                        keys=np.array([f"{a}x{b}" for a, b in res]),
                        vals=np.array([res[k] for k in res]),
                        td1=td1)
    print("\n已保存 val2d.npz")


if __name__ == "__main__":
    main()
