# -*- coding: utf-8 -*-
"""实验：按「整点瞬时值」的正确口径重建附件 3 的预报网格，并修正偏差标定。

发现（见 exp_fc_interp.py 与那次相关性检验）：附件 3 的第 k 小时预报
fc24[k−1] 是「发布后第 k 小时整点」**那一瞬间**的功率，不是该小时的平均。
判据是把预报与附件 2 实际值在几种对时方式下比 MAE：

    整点 X = issue+k       MAE 117.15   r 0.9957   ← 真身
    整点 X = issue+k−1     MAE 667.05   r 0.9401
    (h,h+1] 均值           MAE 308.80   r 0.9856

本文原先把 fc24[k−1] 铺在 [issue+k−1, issue+k) 上（比真身早一小时、且在
小时内取常数）；偏差标定又把预报与「小时均值」相减——两处对时都不对。

本实验把两处都改成同刻口径：
    1. 网格：以整点 τ_k = issue+k 为结点、值 fc24[k−1]，线性插值；
    2. 偏差/标准差：E[实际在 τ_k 的值 − fc24[k−1]]。
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

T = D.T


def actual_at(pv_day, tau):
    """附件 2 第 j 列 = 时刻 (j+1)/6 h 的值；返回整点 tau 处的实测。"""
    idx = int(round(tau * 6)) - 1
    if idx < 0:
        return 0.0
    if idx >= T:
        return float(pv_day[-1])
    return float(pv_day[idx])


def grid_knot(fc24, issue, beta=None, hedge=None, z=0.0):
    """按整点结点线性插值出 144 步网格（abs 小时 → 值）。"""
    ks = [(issue + k, float(fc24[k - 1])) for k in range(1, 25)]
    if beta is not None:
        ks = [(t, v + beta(issue, k)) for k, (t, v) in
              enumerate(ks, start=1)]
    if hedge is not None:
        ks = [(t, v - z * hedge(issue, k)) for k, (t, v) in
              enumerate(ks, start=1)]
    use = [(t, v) for t, v in ks if t <= 24]
    if not use:
        return np.full(T, np.nan)
    ts = np.array([t for t, _ in use], float)
    vs = np.array([v for _, v in use], float)
    g = np.full(T, np.nan)
    for j in range(T):
        t = (j + 1) / 6.0                 # 该步结束的时刻（小时）
        if t < issue:
            continue
        g[j] = float(np.interp(t, ts, vs))    # 早于首个结点时取平
    return g


def calibrate_same_instant(fc, pv_act, days, stat):
    """同刻口径的偏差/标准差：实际在整点 τ=issue+k 的值 − 预报。"""
    acc = {}
    for d in days:
        for iss in (0, 6, 12, 18):
            for k in range(1, 25):
                tau = iss + k
                if tau > 24:
                    continue
                f = float(fc[(d, iss)][k - 1])
                if iss > 0:                     # 与现口径一样只看当天时段
                    pass
                acc.setdefault((iss, k), []).append(
                    actual_at(pv_act[d], tau) - f)
    if stat == "mean":
        def f_(iss, k):
            v = acc.get((iss, k))
            return float(np.mean(v)) if v else 0.0
    else:
        def f_(iss, k):
            v = acc.get((iss, k))
            return float(np.std(v, ddof=1)) if v and len(v) > 1 else 0.0
    return f_


def run(fix, ndays=60, warm=60, z=0.75, tag=""):
    load, pv_act = D.load_att2()
    fc = D.load_att3()
    E, _, _ = D.load_att1()
    prices = E
    if fix:
        bias = calibrate_same_instant(fc, pv_act, range(0, 31), "mean")
        sigma = calibrate_same_instant(fc, pv_act, range(0, 31), "std")
    else:
        bias = C.calibrate_bias(fc, pv_act, range(0, 31))
        sigma = C.calibrate_sigma(fc, pv_act, range(0, 31))
    soc = 6000.0
    tot = em_kwh = 0.0
    for d in range(warm, warm + ndays):
        if fix:
            D.forecast_to_grid = (
                lambda f, iss, _d=d: grid_knot(f, iss, beta=bias,
                                               hedge=sigma, z=z))
        else:
            D.forecast_to_grid = D.__dict__["_orig_f2g"]
        r = C.solve_one_day_q3(d, prices, load[d], pv_act[d], fc, soc,
                               bias=bias, sigma=sigma, z=z)
        tot += r["cost"]
        em_kwh += r["emerg_kwh"]
        soc = r["soc1"]
    print(f"  {tag:<40} 费用 {tot:14.2f} 元   紧急 {em_kwh:10.1f} kWh")
    return tot, em_kwh


def main():
    D._orig_f2g = D.forecast_to_grid
    print("60 天窗口（3 月 1 日 – 4 月 29 日）")
    run(False, tag="现状：早一小时的常数铺法 + 均值口径偏差")
    run(True, tag="修正：整点插值 + 同刻口径偏差/标准差")
    # 修正铺法但不带订正/对冲，看基准
    D.forecast_to_grid = D.__dict__["_orig_f2g"]


if __name__ == "__main__":
    main()
