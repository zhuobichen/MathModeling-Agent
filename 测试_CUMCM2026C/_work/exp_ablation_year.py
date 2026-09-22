# -*- coding: utf-8 -*-
"""全年消融，口径与 c_solve.run_q3 完全一致（SOC 自 1 月 1 日起滚动，
只统计 2/1–12/31 的 334 天），这样四行才可以横向比。

对照的「旧铺法」是本文初稿的口径：把 fc24[k−1] 铺在 [issue+k−1, issue+k)
上（比真身早一小时、且小时内取常数），配「小时均值口径」的偏差表。
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c_data as D
import c_solve as C
import mg_core as M

WARM = 31
NEW = D.forecast_to_grid


def f2g_old(fc24, issue_hour, beta=None, hedge=None, z=0.0):
    g = np.full(D.T, np.nan)
    for k in range(1, 25):
        h = (issue_hour + k - 1) % 24
        lo = h * 6
        if lo < issue_hour * 6:
            continue
        v = float(fc24[k - 1])
        if beta is not None:
            v += beta(issue_hour, k)
        if hedge is not None:
            v -= z * hedge(issue_hour, k)
        g[lo:lo + 6] = v
    return g


def cal_old(stat):
    def make(fc, pv_act, days):
        acc = {}
        for d in days:
            for iss in (0, 6, 12, 18):
                for k in range(1, 25):
                    h = (iss + k - 1) % 24
                    if h * 6 < iss * 6:
                        continue
                    acc.setdefault((iss, k), []).append(
                        float(pv_act[d][h * 6:(h + 1) * 6].mean()
                              - fc[(d, iss)][k - 1]))
        if stat == "mean":
            return lambda i, k: (float(np.mean(acc[(i, k)]))
                                 if acc.get((i, k)) else 0.0)
        return lambda i, k: (float(np.std(acc[(i, k)], ddof=1))
                             if acc.get((i, k)) and len(acc[(i, k)]) > 1 else 0.0)
    return make


def run(label, old, use_bias, z):
    load, pv_act = D.load_att2()
    fc = D.load_att3()
    E, _, _ = D.load_att1()
    D.forecast_to_grid = f2g_old if old else NEW
    if old:
        bias = cal_old("mean")(fc, pv_act, range(0, WARM)) if use_bias else None
        sigma = cal_old("std")(fc, pv_act, range(0, WARM)) if z else None
    else:
        bias = C.calibrate_bias(fc, pv_act, range(0, WARM)) if use_bias else None
        sigma = C.calibrate_sigma(fc, pv_act, range(0, WARM)) if z else None
    soc = 6000.0
    tot = em = 0.0
    for d in range(365):
        r = C.solve_one_day_q3(d, E, load[d], pv_act[d], fc, soc,
                               bias=bias, hedge=0.0, sigma=sigma, z=z)
        if d >= WARM:
            tot += r["cost"]
            em += r["emerg_kwh"]
        soc = r["soc1"]
    print(f"  {label:<42} {tot:13.2f} 元  紧急 {em:9.1f} kWh", flush=True)
    D.forecast_to_grid = NEW
    return tot, em


if __name__ == "__main__":
    print("全年 2/1–12/31，334 天，SOC 自 1/1 滚动（与正式结果同口径）")
    run("初稿口径：早一小时常数铺法 + 旧口径订正 + z=0.75", True, True, 0.75)
    run("对时修正 + 不订正 + 不对冲（正式）", False, False, 0.0)
    run("对时修正 + 不订正 + 对冲 z=0.5", False, False, 0.5)
    run("对时修正 + 订正 + 不对冲", False, True, 0.0)
