# -*- coding: utf-8 -*-
"""全年确认：对时修正后，偏差订正与分位对冲到底还要不要。"""
from __future__ import annotations

import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c_data as D
import c_solve as C

WARM = 31          # 正式窗口 2/1 起


def run(label, use_bias, z):
    load, pv_act = D.load_att2()
    fc = D.load_att3()
    E, _, _ = D.load_att1()
    bias = C.calibrate_bias(fc, pv_act, range(0, WARM)) if use_bias else None
    sigma = C.calibrate_sigma(fc, pv_act, range(0, WARM)) if z else None
    soc = 6000.0
    tot = em_kwh = sem = 0.0
    nd = 0
    for d in range(WARM, 365):
        r = C.solve_one_day_q3(d, E, load[d], pv_act[d], fc, soc,
                               bias=bias, hedge=0.0, sigma=sigma, z=z)
        tot += r["cost"]
        em_kwh += r["emerg_kwh"]
        sem += r["cost_em"]
        soc = r["soc1"]
        nd += 1
    print(f"  {label:<40} 合计 {tot:13.2f} 元  结算 {tot-sem:13.2f}  "
          f"紧急费 {sem:11.2f}  紧急 {em_kwh:9.1f} kWh  ({nd} 天)")
    return tot


if __name__ == "__main__":
    print("全年 2/1–12/31，334 天")
    run("新对时 + 不订正 + 不对冲", False, 0.0)
    run("新对时 + 不订正 + 对冲 z=0.5", False, 0.5)
    run("新对时 + 订正 + 不对冲", True, 0.0)
