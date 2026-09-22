# -*- coding: utf-8 -*-
"""独立验证：只读「已写出的结果文件」+ 原始附件，反算各项指标并与报告值比对。

刻意不复用求解过程中的内存对象——这样能查出"算出来了但没写对"这类错误
（本次就出过：模板多两列、dict.get 默认值被无条件求值导致 KeyError）。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import openpyxl

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c_data as D
import mg_core as M

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "_交付"))
ATT5 = os.path.abspath(os.path.join(HERE, "..", "附件", "附件5"))


def read_buy_sheet(path, sheet, nrow):
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb[sheet]
    P = np.zeros((nrow, M.T))
    for i in range(nrow):
        for j in range(M.T):
            v = ws.cell(i + 2, j + 2).value
            P[i, j] = float(v) if v is not None else 0.0
    kwh = np.array([float(ws.cell(i + 2, M.T + 2).value or 0) for i in range(nrow)])
    cost = np.array([float(ws.cell(i + 2, M.T + 3).value or 0) for i in range(nrow)])
    return P, kwh, cost


def check_result1():
    print("=" * 76)
    print("【1】问题 1：由 result1.xlsx 反算")
    print("=" * 76)
    E, P, Q = D.load_att1()
    wb = openpyxl.load_workbook(os.path.join(OUT, "result1.xlsx"),
                                data_only=True)
    ws = wb["计划购电量"]
    buy = np.array([float(ws.cell(i + 2, 2).value) for i in range(M.T)])
    # 表2
    ws2 = wb["充放电量"]
    ch = np.zeros(M.T); dis = np.zeros(M.T)
    # 表2 给的是时段内的电量（kWh），换算成功率：kW = kWh / 小时数
    for i, (a, b) in enumerate(D.PERIODS):
        ch[a * 6:b * 6] = float(ws2.cell(i + 2, 2).value) / (b - a)
        dis[a * 6:b * 6] = float(ws2.cell(i + 2, 3).value) / (b - a)
    soc0 = float(ws2.cell(2, 5).value); soc1 = float(ws2.cell(3, 5).value)
    # 反算
    buy_kwh = buy.sum()
    cost = float((E * buy).sum())
    # 能量（净负荷 = 购电 + 放 − 充）
    net = buy + dis * M.DT - ch * M.DT
    print(f"  购电量 {buy_kwh:.4f} kWh    购电费 {cost:.4f} 元")
    print(f"  0:00/24:00 储电量 {soc0:.4f} / {soc1:.4f} kWh  （应相等）")
    print(f"  充放电量：充 {ch.sum()*M.DT:.4f}  放 {dis.sum()*M.DT:.4f} kWh")
    # 从表2 的时段值重算 SOC（只有 0:00 与 24:00 有值，中间不可验；改为看总量）
    # 全局能量守恒：Σ购电 = Σ负荷 − Σ光伏 + (充−放)   （弃光为 0）
    lhs = buy_kwh
    rhs = (P.sum() - Q.sum()) * M.DT + (ch.sum() - dis.sum()) * M.DT
    print(f"  能量守恒：购电 {lhs:.4f}  应等于 (负荷−光伏)+(充−放) {rhs:.4f}"
          f"   差 {abs(lhs-rhs):.2e}")
    print(f"  无储能基线 {(E*np.maximum(P-Q,0)*M.DT).sum():.4f} 元")
    # 容差按**相对**给：结果文件保留 4 位小数，144 步累计舍入可达 ~1e-2 kWh，
    # 用绝对 1e-6 会把自己的舍入当成求解错误。
    rel = abs(lhs - rhs) / max(abs(lhs), 1e-9)
    print(f"  相对差 {rel:.2e}（容差 1e-6）")
    return rel < 1e-6


def check_daywise(fname, prices, start=31, sample=6):
    print()
    print("=" * 76)
    print(f"【】{fname}：逐日抽查")
    print("=" * 76)
    load, pv = D.load_att2()
    P, kwh, cost = read_buy_sheet(os.path.join(OUT, fname), "计划购电量",
                                  D.DAYS - start)
    n = P.shape[0]
    ok = True
    idxs = np.linspace(0, n - 1, sample).astype(int)
    print(f"  {'日序号':>7}{'购电量/kWh':>14}{'表内全天值':>14}{'表内电费':>14}{'重算电费':>14}")
    for i in idxs:
        d = start + i
        pr = prices if np.asarray(prices).ndim == 1 else prices[d]
        rec = float((pr * P[i]).sum())
        print(f"  {d:>7}{P[i].sum():>14.4f}{kwh[i]:>14.4f}{cost[i]:>14.4f}{rec:>14.4f}")
        ok &= abs(cost[i] - rec) < 1e-3
    return ok


def check_totals():
    print()
    print("=" * 76)
    print("【汇总】五个结果文件的整体指标")
    print("=" * 76)
    E, _, _ = D.load_att1()
    _, pv = D.load_att2()
    for f, pr in (("result2.xlsx", E), ("result4-2.xlsx", D.load_att4())):
        P, kwh, cost = read_buy_sheet(os.path.join(OUT, f), "计划购电量",
                                      D.DAYS - 31)
        print(f"  {f:<16} 购电量 {kwh.sum():>14.1f} kWh  购电费 {cost.sum():>14.2f} 元")
    for f, pr in (("result3.xlsx", E), ("result4-3.xlsx", D.load_att4())):
        P, kwh, cost = read_buy_sheet(os.path.join(OUT, f), "计划购电量",
                                      D.DAYS - 31)
        A, akwh, acost = read_buy_sheet(os.path.join(OUT, f), "调整购电量",
                                        D.DAYS - 31)
        print(f"  {f:<16} 计划购电量 {kwh.sum():>14.1f}  调整后 {akwh.sum():>14.1f} kWh"
              f"  结算 {acost.sum():>14.2f} 元")
        # 紧急购电
        wb = openpyxl.load_workbook(os.path.join(OUT, f), data_only=True)
        ws = wb["紧急购电量"]
        tot = sum(float(ws.cell(i, 3).value or 0)
                  for i in range(2, ws.max_row + 1))
        print(f"  {'':<16} 紧急购电量 {tot:>14.1f} kWh")


if __name__ == "__main__":
    a = check_result1()
    b = check_daywise("result2.xlsx", D.load_att1()[0])
    c = check_daywise("result4-2.xlsx", D.load_att4())
    check_totals()
    print()
    print("总体:", "通过" if (a and b and c) else "**有失败项**")
