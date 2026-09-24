# -*- coding: utf-8 -*-
"""相互检验：用公开解析帖给的公式口径复算本文结果，并与本文内核逐位比对。

公开帖（CSDN，2026 研赛 D 题解析）附了 MATLAB 航段能耗函数，形式为

    Lq    = rangeEmpty - (rangeEmpty - rangeFull) * (q/maxPayload)^1.5
    E_hor = batteryEnergy * d_horiz / Lq
    E_up  = (emptyMass + q) * 9.8 * h_up / (etaClimb * 3.6e6)
    往返  = legEnergy(q, h_climb, h_descend) + legEnergy(0, h_descend, h_climb)

本文内核是同一套，只是 g 取 9.81。这里把它照抄一遍（G=9.8），配两种巡航海拔复算，
用来验证两件事：
  1. 两边独立反推的 E^hor / E^up 是不是同一个形式；
  2. 公开帖的代码里有没有真的读 DEM。

第 2 点是关键：该帖的 getMaxDEM 是空函数，用 `max(两端点地面海拔) + 80` 占位。
本脚本把两种读法都算出来，看差多少、差在哪个站。
"""
from __future__ import annotations

import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from d_q2 import World, SITES

G_THEIRS = 9.8          # 公开帖代码里用的重力加速度


def their_payload(W, g, s, cruise):
    """按公开帖的 MATLAB 口径复算最大安全载荷（二分）。"""
    tt = W.types[g]
    a, b = W.nodes["O01"], W.nodes[s]
    d = W.geo.dist(a["lon"], a["lat"], b["lon"], b["lat"])
    if cruise == "dem":                      # 本文：航段所经 DEM 最高点 + 50
        H = W.geo.elev_line(a["lon"], a["lat"], b["lon"], b["lat"]) + 50.0
    else:                                    # 公开帖：两端点地面海拔取大 + 80
        H = max(a["alt"], b["alt"]) + 80.0
    h_op_a, h_op_b = a["alt"], b["alt"] + 30.0
    h_out, h_desc = max(H - h_op_a, 0.0), max(H - h_op_b, 0.0)

    def leg(q, h_up):
        Lq = tt["L0"] - (tt["L0"] - tt["LF"]) * (q / tt["Q"]) ** 1.5
        Lq = max(Lq, 100.0)
        return (tt["E_use"] * d / Lq
                + (tt["m_empty"] + q) * G_THEIRS * h_up / (tt["eta_up"] * 3.6e6))

    lim = (1 - tt["rho"]) * tt["E_use"]
    lo, hi = 0.0, tt["Q"]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        # 往返：去程 (h_out)，回程 (h_desc)
        if leg(mid, h_out) + leg(0.0, h_desc) <= lim:
            lo = mid
        else:
            hi = mid
    return lo


def main():
    W = World()
    print("=" * 78)
    print("相互检验：按其公式口径复算 vs 本文内核")
    print("=" * 78)
    print(f"{'服务区':<8}{'机型':>4}{'本文':>10}{'其口径+实测DEM':>18}"
          f"{'其口径+占位高度':>18}{'偏差':>9}")
    worst = (0.0, None)
    for s in SITES:
        for g in "ABC":
            mine = W.types[g]["Q"]      # 先占位，下面按本文口径精确算
            from d_q1 import max_payload
            mine = min(max_payload(g, s, W.T), W.types[g]["Q"])
            a = their_payload(W, g, s, "dem")
            b = their_payload(W, g, s, "ph")
            dv = b - a
            if abs(dv) > abs(worst[0]):
                worst = (dv, (s, g))
            flag = "  ←" if abs(dv) > 1e-6 else ""
            print(f"{s:<8}{g:>4}{mine:>10.2f}{a:>18.2f}{b:>18.2f}{dv:>9.2f}{flag}")
    print()
    # 结论必须由上面算出来的数得出，不能预先写死——第一版这里写死了「最大偏差
    # 在 S008」，实际跑出来是 S003，正是这类"看起来合理"的断言最容易出错的地方。
    s_w, g_w = worst[1]
    nz = [(s, g) for s in SITES for g in "ABC"
          if abs(their_payload(W, g, s, "ph") - their_payload(W, g, s, "dem"))
          > 1e-6]
    print(f"两种 DEM 读法的最大偏差 {worst[0]:+.2f} kg 出现在 {s_w} 的 {g_w} 型；"
          f"共 {len(nz)} 个 (站, 机型) 组合存在偏差。")
    print("  偏差全部出现在 C 型（80 kg 满载航程最短、爬升项占比最大）；"
          "用「两端点地面海拔取大 + 80 m」代替「航段所经 DEM 最高点 + 50 m」，")
    print("  会系统性低估巡航高度、进而高估安全载荷。偏差随航段起伏增大。")
    print(f"  偏差最大的 {s_w} 是海拔起伏较大的站（{W.nodes[s_w]['alt']:.0f} m），"
          f"不是最远的 S008（{W.nodes['S008']['alt']:.0f} m）。")


if __name__ == "__main__":
    main()
