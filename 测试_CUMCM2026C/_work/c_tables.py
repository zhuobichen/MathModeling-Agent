# -*- coding: utf-8 -*-
"""生成论文表 1、表 2、表 3 所需的数值。

表1  指定时间段的购电量 + 全天购电量 + 全天购电费
表2  储能设备在 6 个 4 小时时段的充放电量 + 0:00 / 24:00 储电量
表3  指定日期的紧急购电量（按时间段）
"""
from __future__ import annotations

import json
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

# 表1 的六个时段：10:00-10:10 等 → 步号 = 小时*6
T1_HOURS = [10, 12, 14, 16, 18, 20]
SPEC_DATES = [(3, 20), (6, 21), (9, 23), (12, 21)]


def _step(hour):
    return hour * 6


def table12_from_buy(buy_kwh_144, ch_kw, dis_kw, price, soc0, soc1):
    """从 144 步的购电量(kWh)、充放电功率(kW) 生成表1、表2 的行。"""
    t1 = {}
    for h in T1_HOURS:
        t1[f"{h}:00-{h}:10"] = float(buy_kwh_144[_step(h)])
    day_kwh = float(buy_kwh_144.sum())
    day_cost = float((price * buy_kwh_144).sum())
    t2 = []
    for (a, b) in D.PERIODS:
        t2.append((f"{a}:00-{b}:00",
                   float(ch_kw[a * 6:b * 6].sum() * M.DT),
                   float(dis_kw[a * 6:b * 6].sum() * M.DT)))
    return t1, day_kwh, day_cost, t2, float(soc0), float(soc1)


def load_solution_days(fname, start=31):
    """读回某结果文件：逐日 144 步购电量、充放电、SOC、紧急量。"""
    wb = openpyxl.load_workbook(os.path.join(OUT, fname), data_only=True)
    ws = wb["计划购电量"]
    n = ws.max_row - 1
    P = np.zeros((n, M.T))
    for i in range(n):
        for j in range(M.T):
            P[i, j] = float(ws.cell(i + 2, j + 2).value or 0)
    w3 = wb["充放电量"]
    ch = np.zeros((n, M.T)); dis = np.zeros((n, M.T))
    soc0 = np.zeros(n); soc1 = np.zeros(n)
    for i in range(n):
        r = 2 + i * 6
        for k, (a, b) in enumerate(D.PERIODS):
            ch[i, a * 6:b * 6] = float(w3.cell(r + k, 3).value or 0) / (b - a)
            dis[i, a * 6:b * 6] = float(w3.cell(r + k, 4).value or 0) / (b - a)
        soc0[i] = float(w3.cell(r, 6).value or 0)
        soc1[i] = float(w3.cell(r + 1, 6).value or 0)
    w4 = wb["紧急购电量"]
    em = {}
    for i in range(2, w4.max_row + 1):
        dt = str(w4.cell(i, 1).value or "")
        seg = w4.cell(i, 2).value
        val = float(w4.cell(i, 3).value or 0)
        if not dt:
            continue
        em.setdefault(dt, []).append((seg, val))
    return dict(P=P, ch=ch, dis=dis, soc0=soc0, soc1=soc1, em=em, start=start)


def main():
    E, P1, Q1 = D.load_att1()
    A4 = D.load_att4()
    res = {}

    # ── 问题 1 ──────────────────────────────────────────
    wb = openpyxl.load_workbook(os.path.join(OUT, "result1.xlsx"),
                                data_only=True)
    ws = wb["计划购电量"]
    buy = np.array([float(ws.cell(i + 2, 2).value) for i in range(M.T)])
    w2 = wb["充放电量"]
    ch = np.zeros(M.T); dis = np.zeros(M.T)
    for k, (a, b) in enumerate(D.PERIODS):
        ch[a * 6:b * 6] = float(w2.cell(k + 2, 2).value) / (b - a)
        dis[a * 6:b * 6] = float(w2.cell(k + 2, 3).value) / (b - a)
    res["q1"] = table12_from_buy(buy, ch, dis, E, 6000.0,
                                 float(w2.cell(3, 5).value))

    # ── 问题 2 / 4-2：表1、表2 取 2025-03-20（表3 指定日期之一）──
    for key, f, pr in (("q2", "result2.xlsx", E),
                       ("q42", "result4-2.xlsx", A4)):
        sol = load_solution_days(f)
        i = D.day_index(*SPEC_DATES[0]) - sol["start"]
        price = E if np.asarray(pr).ndim == 1 else pr[D.day_index(*SPEC_DATES[0])]
        res[key] = table12_from_buy(sol["P"][i], sol["ch"][i], sol["dis"][i],
                                    price, sol["soc0"][i], sol["soc1"][i])

    # ── 问题 3 / 4-3 ────────────────────────────────────
    for key, f, pr in (("q3", "result3.xlsx", E),
                       ("q43", "result4-3.xlsx", A4)):
        sol = load_solution_days(f)
        i = D.day_index(*SPEC_DATES[0]) - sol["start"]
        price = E if np.asarray(pr).ndim == 1 else pr[D.day_index(*SPEC_DATES[0])]
        res[key] = table12_from_buy(sol["P"][i], sol["ch"][i], sol["dis"][i],
                                    price, sol["soc0"][i], sol["soc1"][i])
        # 表3：四个指定日期的紧急购电
        spec = {}
        for (mm, dd) in SPEC_DATES:
            lbl = f"2025-{mm:02d}-{dd:02d}"
            spec[lbl] = sol["em"].get(lbl, [])
        res[key] = res[key] + (spec,)

    # ── 打印 ────────────────────────────────────────────
    def show(name, r):
        t1, tot, cost, t2, s0, s1 = r[:6]
        print("=" * 76)
        print(name)
        print("=" * 76)
        print("  表1  购电量")
        for k, v in t1.items():
            print(f"    {k:<14}{v:>14.4f} kWh")
        print(f"    全天购电量   {tot:>14.4f} kWh")
        print(f"    全天购电费   {cost:>14.4f} 元")
        print("  表2  充放电量")
        for lb, c, d in t2:
            print(f"    {lb:<14}充 {c:>12.4f}   放 {d:>12.4f} kWh")
        print(f"    0:00 储电量 {s0:.4f}    24:00 储电量 {s1:.4f} kWh")
        if len(r) > 6:
            print("  表3  紧急购电量（指定日期）")
            for dt, segs in r[6].items():
                if not segs:
                    print(f"    {dt}: 无")
                else:
                    tt = sum(x[1] for x in segs)
                    print(f"    {dt}: 共 {tt:.4f} kWh，{len(segs)} 个时段"
                          f"（{segs[0][0]} 起）")
        print()

    show("问题 1", res["q1"])
    for k, nm in (("q2", "问题 2（2025-03-20 为例）"),
                  ("q3", "问题 3（2025-03-20 为例）"),
                  ("q42", "问题 4 对应问题 2（2025-03-20 为例）"),
                  ("q43", "问题 4 对应问题 3（2025-03-20 为例）")):
        show(nm, res[k])

    with open(os.path.join(HERE, "out_tables.json"), "w",
              encoding="utf-8") as fh:
        json.dump({k: (v[:6] if len(v) > 6 else v) for k, v in res.items()},
                  fh, ensure_ascii=False, indent=1, default=float)
    print("已保存 out_tables.json")


if __name__ == "__main__":
    main()
