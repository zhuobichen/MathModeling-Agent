# -*- coding: utf-8 -*-
"""消融：对时修正 / 偏差订正 / 分位对冲 各自值多少钱（60 天窗口）。

对时修正与偏差订正必须一起做——原来的「系统性偏差 −666 kW」本身就是
拿整点预报去比小时均值比出来的假象，两处对时一改，偏差表就塌下去了。
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c_data as D
import c_solve as C


def run(label, align, use_bias, z, ndays=60, warm=60):
    load, pv_act = D.load_att2()
    fc = D.load_att3()
    E, _, _ = D.load_att1()
    bias = C.calibrate_bias(fc, pv_act, range(0, 31)) if use_bias else None
    sigma = C.calibrate_sigma(fc, pv_act, range(0, 31))
    soc = 6000.0
    tot = em = 0.0
    for d in range(warm, warm + ndays):
        if align:
            D.forecast_to_grid = D.__dict__["_f2g_new"]
        else:
            D.forecast_to_grid = D.__dict__["_f2g_old"]
            # 旧铺法配旧口径的偏差表
            b_old = C.__dict__["_bias_old"](fc, pv_act, range(0, 31))
            s_old = C.__dict__["_sigma_old"](fc, pv_act, range(0, 31))
            r = C.solve_one_day_q3(d, E, load[d], pv_act[d], fc, soc,
                                   bias=b_old, hedge=0.0, sigma=s_old, z=z)
            tot += r["cost"]; em += r["emerg_kwh"]; soc = r["soc1"]
            continue
        r = C.solve_one_day_q3(d, E, load[d], pv_act[d], fc, soc,
                               bias=bias, hedge=0.0,
                               sigma=(sigma if z else None), z=z)
        tot += r["cost"]; em += r["emerg_kwh"]; soc = r["soc1"]
    print(f"  {label:<44} {tot:14.2f} 元   紧急 {em:9.1f} kWh")
    return tot


def main():
    import c_data, c_solve
    c_data._f2g_new = c_data.forecast_to_grid
    c_solve._bias_old = c_solve.calibrate_bias
    c_solve._sigma_old = c_solve.calibrate_sigma
    # 旧的铺法与旧口径：从 git 历史取回不便，这里就地重建
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
    c_data._f2g_old = f2g_old

    def cal_old(pool_stat):
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
            if pool_stat == "mean":
                f = lambda i, k: float(np.mean(acc[(i, k)])) if acc.get((i, k)) else 0.0
            else:
                f = lambda i, k: (float(np.std(acc[(i, k)], ddof=1))
                                  if acc.get((i, k)) and len(acc[(i, k)]) > 1 else 0.0)
            return f
        return make
    c_solve._bias_old = cal_old("mean")
    c_solve._sigma_old = cal_old("std")

    print("60 天窗口（3/1–4/29）")
    run("旧：早一小时的常数铺法 + 旧口径订正 + 对冲 z=0.75", False, True, 0.0)
    run("新对时 + 不订正 + 不对冲", True, False, 0.0)
    run("新对时 + 不订正 + 对冲 z=0.5", True, False, 0.5)
    run("新对时 + 不订正 + 对冲 z=0.75", True, False, 0.75)
    run("新对时 + 不订正 + 对冲 z=1.0", True, False, 1.0)
    run("新对时 + 订正 + 对冲 z=0.75", True, True, 0.75)


if __name__ == "__main__":
    main()
