# -*- coding: utf-8 -*-
"""实验：附件 3 的逐小时预报怎么铺到 10 分钟网格，对问题 3 的影响。

与网上公开解答对照时发现，别的解法把整点预报**线性插值**到 10 分钟
（而本文原先把每个小时的值在小时内取常数）。这两种铺法在早晚爬坡段差别
最大：常数铺法认为 6:00-7:00 整小时都是同一个功率，插值铺法则让功率从
日出前的 0 平滑爬上去。本文问题 3 的紧急购电恰好集中在早晨 6:00-9:30，
所以值得实测一下。

三种铺法：
  A 分段常数（现状）
  B 整点值为结点线性插值
  C 在 B 的基础上，把发布时刻那一点换成刚结束的 10 分钟实测值
    （发布时刻的实测已经拿到，用它当插值起点是合规的）
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


def grid_step(fc24, issue_hour):
    """A：分段常数（现行实现）。"""
    g = np.full(T, np.nan)
    for k in range(1, 25):
        h = (issue_hour + k - 1) % 24
        lo = h * 6
        if lo < issue_hour * 6:
            continue
        g[lo:lo + 6] = fc24[k - 1]
    return g


def _knots(fc24, issue_hour):
    """把 24 个预报值摆到各自的整点上，返回 {绝对小时: 值}。"""
    kn = {}
    for k in range(1, 25):
        h = (issue_hour + k - 1) % 24
        if h < issue_hour and issue_hour > 0:
            continue
        kn[h] = float(fc24[k - 1])
    return kn


def grid_linear(fc24, issue_hour, first_value=None):
    """B/C：整点值为结点做线性插值；first_value 给定时用它替换发布时刻的值。"""
    kn = _knots(fc24, issue_hour)
    hs = sorted(kn)
    if not hs:
        return np.full(T, np.nan)
    lo, hi = hs[0], hs[-1]
    ys = [kn[h] for h in hs]
    if first_value is not None:
        ys[0] = float(first_value)
    g = np.full(T, np.nan)
    # 覆盖到最后一个结点所在小时的末尾（np.interp 末端自动取平），
    # 否则 23:50-24:00 那一步落在最后一个结点之外，会留成 nan。
    for j in range(min((hi + 1) * 6, T) - lo * 6):
        t = lo + (j + 0.5) / 6.0            # 该 10 分钟步的中点（小时）
        g[lo * 6 + j] = float(np.interp(t, hs, ys))
    # 发布时刻之前的那几步（若 lo > issue_hour）保持 nan
    g[:issue_hour * 6] = np.nan
    return g


def run(mode, bias, sigma, z=0.75, ndays=60, warm=60):
    """跑问题 3 的一段窗口，返回（结算+紧急）总费用与紧急电量。"""
    load, pv_act = D.load_att2()
    fc = D.load_att3()
    soc = 6000.0
    tot = em_kwh = 0.0
    from datetime import date
    E, _, _ = D.load_att1()
    prices = E
    for d in range(warm, warm + ndays):
        def grid(f, iss, _d=d):
            if mode == "A":
                return grid_step(f, iss)
            if mode == "B":
                return grid_linear(f, iss)
            # C：发布时刻用"刚结束的 10 分钟实测"作插值起点
            k = iss * 6
            if k == 0:
                fv = 0.0
            else:
                fv = float(pv_act[_d][k - 1])
            return grid_linear(f, iss, first_value=fv)
        D.forecast_to_grid = grid
        r = C.solve_one_day_q3(d, prices, load[d], pv_act[d], fc, soc,
                               bias=bias, sigma=sigma, z=z)
        tot += r["cost"]
        em_kwh += r["emerg_kwh"]
        soc = r["soc1"]
    return tot, em_kwh


def main():
    fc_all = D.load_att3()
    _, pv_all = D.load_att2()
    bias = C.calibrate_bias(fc_all, pv_all, range(0, 31))
    sigma = C.calibrate_sigma(fc_all, pv_all, range(0, 31))
    orig = D.forecast_to_grid
    print(f"{'铺法':<34}{'60 天费用/元':>16}{'紧急电量/kWh':>16}")
    for mode, name in (("A", "A 分段常数（本文现状）"),
                       ("B", "B 整点线性插值"),
                       ("C", "C 线性插值 + 实测起点")):
        D.forecast_to_grid = orig
        try:
            tot, em = run(mode, bias, sigma)
        except Exception as e:            # noqa: BLE001
            print(f"{name:<34}  失败：{type(e).__name__}: {e}")
            continue
        print(f"{name:<34}{tot:>16.2f}{em:>16.1f}")
    D.forecast_to_grid = orig


if __name__ == "__main__":
    main()
