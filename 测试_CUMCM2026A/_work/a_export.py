# -*- coding: utf-8 -*-
"""把 a_solve 的场结果写成 result1~result4.xlsx，并打印论文表格 1~6。

模板结构（附件 3）：
  result1/2  两个工作表「温度」「水分浓度」；A 列=时间(s)；第 1 行=到中心距离(cm)
  result3    单表，水分浓度
  result4    单表，水分浓度，末列改为「药材表面」
所有数值保留四位小数。
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import a_solve as A
import drying_core as dc

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "_交付")
DEV = A.DEV                       # 0.0 ~ 2.0 cm，步长 0.1
R4 = 4


def r4(x):
    return np.round(np.asarray(x, float), R4)


def sample(sol, radii):
    """按给定半径采样；超出该时刻药材表面的位置返回 nan。"""
    return dc.sample_at(sol, radii)


def write_two_sheet(path, t, T, C):
    import openpyxl
    wb = openpyxl.Workbook()
    for name, arr in (("温度", T), ("水分浓度", C)):
        ws = wb.active if name == "温度" else wb.create_sheet()
        ws.title = name
        ws.cell(1, 1, "时间\\到药材中心的距离")
        for j, d in enumerate(DEV):
            ws.cell(1, j + 2, d)
        for i in range(len(t)):
            ws.cell(i + 2, 1, int(round(t[i])))
            for j in range(len(DEV)):
                v = arr[i, j]
                if np.isfinite(v):
                    ws.cell(i + 2, j + 2, float(round(v, R4)))
    wb.save(path)


def write_one_sheet(path, t, C, extra_surface=None):
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.cell(1, 1, "时间\\到药材中心的距离")
    for j, d in enumerate(DEV):
        ws.cell(1, j + 2, d)
    if extra_surface is not None:
        ws.cell(1, len(DEV) + 2, "药材表面")
    for i in range(len(t)):
        ws.cell(i + 2, 1, int(round(t[i])))
        for j in range(len(DEV)):
            v = C[i, j]
            if np.isfinite(v):
                ws.cell(i + 2, j + 2, float(round(v, R4)))
        if extra_surface is not None:
            ws.cell(i + 2, len(DEV) + 2, float(round(extra_surface[i], R4)))
    wb.save(path)


# ══════════════════════════════════════════════════════════
def table_block(res, radii_table, times_h_or_s, is_hour, kind, surface=False):
    """打印论文用表格。kind: 'T' 或 'C'。"""
    radii = np.array(radii_table, float)
    T, C = sample(res, radii)
    arr = T if kind == "T" else C
    head = "时间/s" if not is_hour else "时间/h"
    cols = "".join(f"{r:>10}" for r in radii_table)
    if surface:
        cols += f"{'药材表面':>10}"
    print(f"{head:>10}{cols}")
    for tt in times_h_or_s:
        i = int(np.argmin(np.abs(res["t"] - tt)))
        row = "".join(f"{arr[i, j]:>10.4f}" for j in range(len(radii_table)))
        if surface:
            row += f"{res['C'][i, -1]:>10.4f}"
        label = f"{tt/3600:.1f}" if is_hour else f"{int(tt)}"
        print(f"{label:>10}{row}")


def main():
    os.makedirs(OUT, exist_ok=True)
    t_bc, T_bc, C_bc = A.load_bc()
    t_R, R_att = A.load_radius()
    bc = A.BC(t_bc, T_bc, C_bc)
    fR, _ = A.make_R_of_t(t_R, R_att)

    print("重新求解四个问题（用于导出）…")
    out1 = list(np.arange(1.0, 1800.0 + 0.5, 1.0))
    s1, r1 = A.run_case("q1", dc.props_q1, dc.consts_q1, bc, 1800.0,
                        A.dt_short, out_times=out1)
    out2 = list(np.arange(1.0, 10800.0 + 0.5, 1.0))
    s2, r2 = A.run_case("q2", dc.props_q3, dc.consts_q3, bc, 10800.0,
                        A.dt_short, out_times=out2)
    # 问题3：算到烘干后再多留 1 h，导出到烘干时刻
    s3, r3 = A.run_case("q3", dc.props_q3, dc.consts_q3, bc, 216000.0,
                        A.dt_long, record_every=60.0)
    td3, _ = A.drying_time(r3)
    s4, r4 = A.run_case("q4", dc.props_q4, dc.consts_q4, bc, 480000.0,
                        A.dt_long, R_of_t=fR, record_every=60.0,
                        material_frame=True)
    td4, _ = A.drying_time(r4)

    # ── result1 ──────────────────────────────────────────
    T1, C1 = sample(r1, DEV)
    write_two_sheet(os.path.join(OUT, "result1.xlsx"), r1["t"][1:], T1[1:], C1[1:])
    print(f"result1.xlsx  {len(r1['t'])-1} 行 x {len(DEV)} 列 x 2 表")

    # ── result2 ──────────────────────────────────────────
    T2, C2 = sample(r2, DEV)
    write_two_sheet(os.path.join(OUT, "result2.xlsx"), r2["t"][1:], T2[1:], C2[1:])
    print(f"result2.xlsx  {len(r2['t'])-1} 行 x {len(DEV)} 列 x 2 表")

    # ── result3 / result4（截到烘干时刻）──────────────────
    i3 = int(np.argmin(np.abs(r3["t"] - td3))) + 1
    C3 = sample(r3, DEV)[1]
    write_one_sheet(os.path.join(OUT, "result3.xlsx"), r3["t"][1:i3], C3[1:i3])
    print(f"result3.xlsx  {i3-1} 行 x {len(DEV)} 列")

    i4 = int(np.argmin(np.abs(r4["t"] - td4))) + 1
    C4 = sample(r4, DEV)[1]
    write_one_sheet(os.path.join(OUT, "result4.xlsx"), r4["t"][1:i4], C4[1:i4],
                    extra_surface=r4["C"][1:i4, -1])
    print(f"result4.xlsx  {i4-1} 行 x {len(DEV)+1} 列（含药材表面）")

    # ══════════════════════════════════════════════════════
    print()
    print("=" * 78)
    print("表 1  30 分钟内药材的温度 / °C")
    print("=" * 78)
    table_block(r1, [0, 0.5, 1, 1.5, 2], [100, 300, 600, 900, 1200, 1500, 1800],
                False, "T")
    print()
    print("=" * 78)
    print("表 2  30 分钟内药材的水分浓度 / (kg/kg)")
    print("=" * 78)
    table_block(r1, [0, 0.5, 1, 1.5, 2], [100, 300, 600, 900, 1200, 1500, 1800],
                False, "C")
    print()
    print("=" * 78)
    print("表 3  3 小时内药材的温度 / °C")
    print("=" * 78)
    table_block(r2, [0, 0.5, 1, 1.5, 2], [1800, 3600, 5400, 7200, 9000, 10800],
                True, "T")
    print()
    print("=" * 78)
    print("表 4  3 小时内药材的水分浓度 / (kg/kg)")
    print("=" * 78)
    table_block(r2, [0, 0.5, 1, 1.5, 2], [1800, 3600, 5400, 7200, 9000, 10800],
                True, "C")

    print()
    print("=" * 78)
    print(f"表 5  药材烘干过程的水分浓度 / (kg/kg)   烘干结束时间 = {td3:.0f} s "
          f"= {td3/3600:.4f} h")
    print("=" * 78)
    t5 = [6 * 3600 * k for k in range(1, int(td3 // (6 * 3600)) + 1)] + [td3]
    table_block(r3, [0, 0.5, 1, 1.5, 2], t5, True, "C")

    print()
    print("=" * 78)
    print(f"表 6  考虑收缩的烘干过程水分浓度 / (kg/kg)   烘干结束时间 = {td4:.0f} s "
          f"= {td4/3600:.4f} h")
    print("=" * 78)
    t6 = [6 * 3600 * k for k in range(1, int(td4 // (6 * 3600)) + 1)] + [td4]
    table_block(r4, [0, 0.5, 1, 1.5, 2], t6, True, "C", surface=True)

    with open(os.path.join(HERE, "out_summary.json"), "w", encoding="utf-8") as f:
        import json
        json.dump(dict(q1=dict(T_center_1800=float(r1["T"][-1][0]),
                               T_surface_1800=float(r1["T"][-1][-1]),
                               C_center_1800=float(r1["C"][-1][0]),
                               C_surface_1800=float(r1["C"][-1][-1])),
                       q2=dict(T_center_3h=float(r2["T"][-1][0]),
                               T_surface_3h=float(r2["T"][-1][-1]),
                               C_center_3h=float(r2["C"][-1][0]),
                               C_surface_3h=float(r2["C"][-1][-1])),
                       q3=dict(drying_time_s=float(td3), drying_time_h=td3 / 3600),
                       q4=dict(drying_time_s=float(td4), drying_time_h=td4 / 3600)),
                  f, ensure_ascii=False, indent=1)
    print("\n已写入 _交付/result1~4.xlsx 与 out_summary.json")


if __name__ == "__main__":
    main()
