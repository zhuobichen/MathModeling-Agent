# -*- coding: utf-8 -*-
"""2026 国赛 C 题 —— 四个问题的求解与结果导出。

问题 1  每天电价与负荷相同（附件1），0:00 制定当日计划，SOC 日末回到初值。
问题 2  电价同附件1，负荷与光伏逐日变化（附件2），0:00 制定当日计划；
        缺额需紧急购电（5 倍电价）。
问题 3  0:00/6:00/12:00/18:00 可得未来 24 小时光伏预报（附件3），
        可在后三个时刻调整购电量，按题面规则结算违约与超量费用。
问题 4  电价实时波动（附件4），重做问题 2 与问题 3。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c_data as D
import mg_core as M

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "_交付"))

Q2_FIRST = 31            # 2025-02-01 的日序号
EMERGENCY_MULT = 5.0     # 紧急购电电价倍数


# ══════════════════════════════════════════════════════════
def run_q1():
    E, P, Q = D.load_att1()
    sol = M.solve_day(E, P, Q, 6000.0, cyclic=True)
    ck = M.check_solution(sol, E, P, Q, 6000.0, cyclic=True)
    base = float((E * np.maximum(P - Q, 0) * M.DT).sum())
    return dict(sol=sol, price=E, load=P, pv=Q, soc0=6000.0, check=ck,
                base=base,
                buy_kwh=float(sol["buy"].sum() * M.DT),
                cost=float(sol["cost"]))


# ── 问题 2：完美信息下的逐日计划 ─────────────────────────
def _price_of_day(prices, d):
    """附件1 的电价全天相同（1 维）；附件4 是逐日矩阵（2 维）。"""
    return prices if np.asarray(prices).ndim == 1 else prices[d]


def run_q2(prices, tag="", cyclic=False):
    load, pv = D.load_att2()
    soc = 6000.0
    days = []
    for d in range(D.DAYS):
        sol = M.solve_day(_price_of_day(prices, d), load[d], pv[d], soc,
                          cyclic=cyclic)
        if not sol["ok"]:
            raise RuntimeError(f"day {d} infeasible")
        # 完美信息下不会缺额；仍按实际（=计划）核一遍
        em = np.maximum(load[d] + sol["ch"] + sol["cur"]
                        - pv[d] - sol["buy"] - sol["dis"], 0.0)
        em[em * M.DT < EM_TOL_KWH] = 0.0     # 见 EM_TOL_KWH 的说明
        days.append(dict(d=d, sol=sol, soc0=soc, soc1=float(sol["soc"][-1]),
                         emerg=em,
                         emerg_kwh=float(em.sum() * M.DT),
                         cost=float(sol["cost"])))
        soc = float(sol["soc"][-1])
    return days


# ── 问题 3：预报 + 调整 ──────────────────────────────────
ADJ = [(36, 6), (72, 12), (108, 18)]


def _residual_pool(fc, pv_act, days):
    """按 (发布时刻, 步长) 收集预报误差。

    必须**同刻**相比：附件 3 的预报是整点 τ=issue+k 的瞬时值，所以实际值也要
    取同一整点（`D.actual_at`）。原先拿「小时均值 − 预报」当误差，会把早晚
    爬坡段的半个小时的坡度算成偏差，订正量因此系统性偏大。
    """
    acc = {}
    for d in days:
        for iss in (0, 6, 12, 18):
            for k in range(1, 25):
                tau = iss + k
                if tau > 24:
                    continue
                acc.setdefault((iss, k), []).append(
                    D.actual_at(pv_act[d], tau) - float(fc[(d, iss)][k - 1]))
    return acc


def calibrate_bias(fc, pv_act, days):
    """用给定天集合标定预报偏差 E[实际−预报]，按 (发布时刻, 步长) 分箱。"""
    acc = _residual_pool(fc, pv_act, days)

    def f(iss, k):
        v = acc.get((iss, k))
        return float(np.mean(v)) if v else 0.0
    return f


def calibrate_sigma(fc, pv_act, days):
    """按 (发布时刻, 步长) 标定预报误差的标准差（kW）。"""
    acc = _residual_pool(fc, pv_act, days)

    def f(iss, k):
        v = acc.get((iss, k))
        return float(np.std(v, ddof=1)) if v and len(v) > 1 else 0.0
    return f


def hedge_grid(sigma, issue, z, Tn=144):
    """逐时段保守余量 = z·σ（z>0 表示按分位数再退一点，压低缺额风险）。

    σ 只在白天非零；夜间 PV 预报恒为 0、无误差，余量自然为 0——这正是
    "均匀加余量" 会亏钱的原因：给零误差的时段也白付 1.5 倍电价。
    """
    g = np.zeros(Tn)
    for k in range(1, 25):
        h = (issue + k - 1) % 24
        lo = h * 6
        if lo < issue * 6 or lo >= Tn:
            continue
        g[lo:min(lo + 6, Tn)] = z * sigma(issue, k)
    return g


def bias_grid(bias, issue, Tn=144):
    """把偏差订正铺到 10 分钟网格（按小时分段）。"""
    g = np.zeros(Tn)
    for k in range(1, 25):
        h = (issue + k - 1) % 24
        lo = h * 6
        if lo < issue * 6 or lo >= Tn:
            continue
        g[lo:min(lo + 6, Tn)] = bias(issue, k)
    return g


def solve_one_day_q3(d, prices, load, pv_act, fc, soc0,
                     bias=None, hedge=0.0, sigma=None, z=0.0):
    prices = np.asarray(prices, float)
    prices = prices if prices.ndim == 1 else prices[d]
    """返回该日的计划、调整、实际执行与各项费用。"""
    # 偏差订正与保守余量都作用在**整点结点**上，再随预报一起插值到 10 分钟
    # 网格——拿 10 分钟粒度的分段常数去调一个插值出来的曲线会两不对齐。
    pv_fc0 = D.forecast_to_grid(fc[(d, 0)], 0, beta=bias, hedge=sigma, z=z)
    if np.isnan(pv_fc0).any():
        raise RuntimeError(f"day {d} 0:00 预报未覆盖全天")
    pv_fc0 = pv_fc0 - hedge
    plan = M.solve_day(prices, load, pv_fc0, soc0)

    P0 = plan["buy"].copy()
    buyt = plan["buy"].copy()
    cht = plan["ch"].copy()
    dist = plan["dis"].copy()
    curt = plan["cur"].copy()
    soct = plan["soc"].copy()

    adj_log = []
    for ta, iss in ADJ:
        grid = D.forecast_to_grid(fc[(d, iss)], iss,
                                  beta=bias, hedge=sigma, z=z)
        pv_fc = grid[ta:] - hedge
        adj = M.solve_day(prices[ta:], load[ta:], pv_fc, soct[ta - 1],
                          p0=P0[ta:])
        if not adj["ok"]:
            raise RuntimeError(f"day {d} adj {iss} infeasible")
        adj_log.append(dict(t=ta, issue=iss, buy=adj["buy"].copy(),
                            cost=float(adj["cost"])))
        buyt[ta:] = adj["buy"]
        cht[ta:] = adj["ch"]
        dist[ta:] = adj["dis"]
        curt[ta:] = adj["cur"]
        soct[ta:] = adj["soc"]

    # 结算：逐段用"该段最后生效的购电量"与计划 P0 比较。
    # 注意每次调整求解的是 ta..T−1 的整个窗口，但只有 ta 到下一次调整之间
    # 的那一段真正执行，故这里要按段切片（直接拼整段会得到 252 个数）。
    bounds = [0] + [t for t, _ in ADJ] + [M.T]     # [0,36,72,108,144]
    segs = [P0[bounds[0]:bounds[1]]]
    for i, a in enumerate(adj_log):
        segs.append(a["buy"][:bounds[i + 2] - bounds[i + 1]])
    Pfin = np.concatenate(segs)
    assert len(Pfin) == M.T, len(Pfin)
    settle = prices * (np.minimum(Pfin, P0)
                       + 0.5 * np.maximum(P0 - Pfin, 0)
                       + 1.5 * np.maximum(Pfin - P0, 0)) * M.DT
    cost_plan_part = float(settle.sum())

    # 实际执行（用真实光伏）→ 缺额。
    # 供给 = 光伏 + 购电 + 放电；它必须不低于负荷。充电是"额外负荷"，在光伏
    # 不足时应当先削减充电，而不是买紧急电，故紧急量里**不含充电**。
    em = np.maximum(load - pv_act - buyt - dist, 0.0)
    em[em * M.DT < EM_TOL_KWH] = 0.0     # 见 EM_TOL_KWH 的说明
    # 实际充入电量：先满足负荷，余量才拿去充电（不超过计划值）
    avail = np.maximum(pv_act + buyt + dist - load, 0.0)
    ch_act = np.minimum(cht, avail)
    cost_em = float((prices * em * EMERGENCY_MULT * M.DT).sum())

    return dict(d=d, soc0=soc0, soc1=float(soct[-1]),
                P0=P0, Pfin=Pfin, buyt=buyt, cht=cht, dist=dist,
                curt=curt, soct=soct, adj=adj_log, ch_act=ch_act,
                cost_plan=cost_plan_part, cost_em=cost_em,
                cost=float(cost_plan_part + cost_em),
                emerg=em, emerg_kwh=float(em.sum() * M.DT))


def run_q3(prices, tag="", bias=None, hedge=0.0, days_range=None,
           sigma=None, z=0.0):
    load, pv_act = D.load_att2()
    fc = D.load_att3()
    soc = 6000.0
    days = []
    for d in (days_range if days_range is not None else range(D.DAYS)):
        r = solve_one_day_q3(d, prices, load[d], pv_act[d], fc, soc,
                             bias=bias, hedge=hedge, sigma=sigma, z=z)
        days.append(r)
        soc = r["soc1"]
    return days


# ══════════════════════════════════════════════════════════
# 结果文件写出
# ══════════════════════════════════════════════════════════
def _grid_label(k):
    """第 k 个 10 分钟格点（k=0..144）的标签，按附件 1 的写法。"""
    if k <= 0:
        return "0:00"
    if k >= M.T:
        return "0:00+1"
    h, m = divmod(k * 10, 60)
    return f"{h}:{m:02d}"


# 附件 5 的模板把 144 个标签写成了 0:10-0:20 … 0:00+1-0:10+1：它拿附件 1 的
# 144 个时刻（0:10 … 0:00+1）两两配对当区间，于是整列右移了一格——少了
# [0:00,0:10) 这一段，末尾多出一个并不存在的 [0:00+1,0:10+1)。
# 附件 1 给的是 144 个时刻、每格代表「以该时刻结尾的 10 分钟」，故正确的
# 区间标签是 0:00-0:10 … 23:50-0:00+1。位置顺序与模板一致，只是把最后
# 一格的名字写对（论文附录 B 说明了这一点）。
TIME_LABELS = [f"{_grid_label(i)}-{_grid_label(i + 1)}" for i in range(M.T)]


def _day_date(d):
    """2025 年的第 d 天（0 基）→ datetime，与附件 5 模板的日期列一致。"""
    import datetime
    return datetime.datetime(2025, 1, 1) + datetime.timedelta(days=d)


def _period_label(a, b):
    return f"{a}:00-{b}:00"


def write_result1(path, r1):
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "计划购电量"
    ws.cell(1, 1, "时间段")
    ws.cell(1, 2, "购电量")
    for i in range(M.T):
        ws.cell(i + 2, 1, TIME_LABELS[i])
        ws.cell(i + 2, 2, round(float(r1["sol"]["buy"][i] * M.DT), 4))
    ws2 = wb.create_sheet("充放电量")
    for j, h in enumerate(["时间段", "充电量", "放电量", "时刻", "储电量"]):
        ws2.cell(1, j + 1, h)
    ps = D.period_sums(r1["sol"]["ch"], r1["sol"]["dis"])
    for i, ((a, b), (c, d)) in enumerate(zip(D.PERIODS, ps)):
        ws2.cell(i + 2, 1, _period_label(a, b))
        ws2.cell(i + 2, 2, round(c, 4))
        ws2.cell(i + 2, 3, round(d, 4))
    ws2.cell(2, 4, "0:00"); ws2.cell(2, 5, round(6000.0, 4))
    ws2.cell(3, 4, "24:00"); ws2.cell(3, 5, round(float(r1["sol"]["soc"][-1]), 4))
    wb.save(path)


def write_result_daywise(path, days, start_idx, with_adjust=False,
                         price_lookup=None):
    """问题 2/3/4 的结果文件：计划购电量(+调整购电量) + 充放电量 + 紧急购电量。"""
    import openpyxl
    sel = [x for x in days if x["d"] >= start_idx]
    wb = openpyxl.Workbook()

    def _p0(x):
        # dict.get 的默认值会被**无条件求值**：写成 x.get("P0", x["sol"]["buy"])
        # 时即使 "P0" 存在也会去取 x["sol"]，问题 3 的日字典没有 "sol" 键，
        # 于是 KeyError。必须显式分支。
        return x["P0"] if "P0" in x else x["sol"]["buy"]

    def fill_buy_sheet(ws, getter, cost_getter):
        """附件5 的这两张表在 144 个时段之后还有两列：全天购电量、全天购电费。"""
        ws.cell(1, 1, "日期\\时间")
        for j, lab in enumerate(TIME_LABELS):
            ws.cell(1, j + 2, lab)
        ws.cell(1, M.T + 2, "全天购电量")
        ws.cell(1, M.T + 3, "全天购电费")
        for i, x in enumerate(sel):
            c = ws.cell(i + 2, 1, _day_date(x["d"]))
            c.number_format = "yyyy/m/d"
            P = getter(x)
            for j in range(M.T):
                ws.cell(i + 2, j + 2, round(float(P[j] * M.DT), 4))
            ws.cell(i + 2, M.T + 2, round(float(P.sum() * M.DT), 4))
            ws.cell(i + 2, M.T + 3, round(float(cost_getter(x)), 4))

    ws = wb.active
    ws.title = "计划购电量"
    fill_buy_sheet(
        ws, _p0,
        lambda x: float(np.sum(price_lookup(x["d"]) * _p0(x) * M.DT)))

    if with_adjust:
        ws2 = wb.create_sheet("调整购电量")
        fill_buy_sheet(ws2, lambda x: x["Pfin"],
                       lambda x: float(x["cost_plan"]))

    ws3 = wb.create_sheet("充放电量")
    for j, h in enumerate(["日期", "时间段", "充电量", "放电量", "时刻", "储电量"]):
        ws3.cell(1, j + 1, h)
    row = 2
    for x in sel:
        c = ws3.cell(row, 1, _day_date(x["d"]))
        c.number_format = "yyyy/m/d"
        ch = x["cht"] if with_adjust else x["sol"]["ch"]
        dis = x["dist"] if with_adjust else x["sol"]["dis"]
        ps = D.period_sums(ch, dis)
        for i, ((a, b), (c, d)) in enumerate(zip(D.PERIODS, ps)):
            ws3.cell(row + i, 2, _period_label(a, b))
            ws3.cell(row + i, 3, round(c, 4))
            ws3.cell(row + i, 4, round(d, 4))
        ws3.cell(row, 5, "0:00"); ws3.cell(row, 6, round(float(x["soc0"]), 4))
        ws3.cell(row + 1, 5, "24:00"); ws3.cell(row + 1, 6, round(float(x["soc1"]), 4))
        row += 6

    ws4 = wb.create_sheet("紧急购电量")
    for j, h in enumerate(["日期", "购电时间段", "购电量"]):
        ws4.cell(1, j + 1, h)
    row = 2
    for x in sel:
        em = x["emerg"]
        segs = _emergency_segments(em)
        c = ws4.cell(row, 1, _day_date(x["d"]))
        c.number_format = "yyyy/m/d"
        if not segs:
            row += 1
            continue
        for (a, b, kwh) in segs:
            ws4.cell(row, 2, f"{a}-{b}")
            ws4.cell(row, 3, round(kwh, 4))
            row += 1
    wb.save(path)


# 单步（10 分钟）紧急购电量低于此值就当作 0。结果文件保留 4 位小数，
# 1e-3 kWh 以下本来也只在第 4 位上晃；而线代求解器的可行性容差会在
# 「计划刚好够用」的时段留下 1e-3 kWh 量级的残差——若不过滤，这些
# 0.0006 kWh 的假时段会占据紧急购电表的大半行数，读的人无从判断
# 哪些是真缺额。1e-3 kWh/步 = 6 W，对全天费用影响 <1 元。
EM_TOL_KWH = 1e-3


def _emergency_segments(em, min_step_kwh=EM_TOL_KWH):
    """把逐 10 分钟的紧急购电功率合并成连续时段。"""
    idx = np.where(em * M.DT >= min_step_kwh)[0]
    if len(idx) == 0:
        return []
    segs, s = [], idx[0]
    for k in range(1, len(idx)):
        if idx[k] != idx[k - 1] + 1:
            segs.append((s, idx[k - 1] + 1))
            s = idx[k]
    segs.append((s, idx[-1] + 1))
    out = []
    for a, b in segs:
        kwh = float(em[a:b].sum() * M.DT)
        def t(x):
            if x == M.T:
                return "0:00+1"
            h, mm = divmod(x * 10, 60)
            return f"{h}:{mm:02d}"
        out.append((t(a), t(b), kwh))
    return out


# ══════════════════════════════════════════════════════════
def main():
    os.makedirs(OUT, exist_ok=True)
    E1, P1, Q1 = D.load_att1()
    A4 = D.load_att4()
    summary = {}

    print("=" * 76)
    print("问题 1")
    print("=" * 76)
    r1 = run_q1()
    print(f"  购电费 {r1['cost']:.4f} 元   购电量 {r1['buy_kwh']:.4f} kWh")
    print(f"  无储能基线 {r1['base']:.4f} 元   节省 "
          f"{r1['base']-r1['cost']:.4f} 元（{(r1['base']-r1['cost'])/r1['base']*100:.2f}%）")
    print(f"  约束核验 " + "  ".join(f"{k}={v:.1e}" for k, v in r1["check"].items()))
    write_result1(os.path.join(OUT, "result1.xlsx"), r1)
    summary["q1"] = dict(cost=r1["cost"], buy_kwh=r1["buy_kwh"],
                         base=r1["base"])

    for tag, prices in (("q2", E1), ("q4", A4)):
        print()
        print("=" * 76)
        print(f"{'问题 2' if tag=='q2' else '问题 4-2'}（"
              f"{'附件1 电价' if tag=='q2' else '附件4 实时电价'}）")
        print("=" * 76)
        days = run_q2(prices, tag)
        sel = [x for x in days if x["d"] >= Q2_FIRST]
        tot_buy = sum(float(x["sol"]["buy"].sum() * M.DT) for x in sel)
        tot_cost = sum(x["cost"] for x in sel)
        tot_em = sum(x["emerg_kwh"] for x in sel)
        print(f"  {len(sel)} 天  计划购电量 {tot_buy:.1f} kWh  计划购电费 {tot_cost:.2f} 元")
        print(f"  紧急购电量 {tot_em:.4f} kWh（完美信息下应为 0）")
        print(f"  首日 SOC {sel[0]['soc0']:.1f} → 末日 SOC {sel[-1]['soc1']:.1f}")
        fn = "result2.xlsx" if tag == "q2" else "result4-2.xlsx"
        write_result_daywise(os.path.join(OUT, fn), days, Q2_FIRST,
                             price_lookup=lambda dd: _price_of_day(prices, dd))
        summary[tag] = dict(days=len(sel), buy_kwh=tot_buy, cost=tot_cost,
                            emerg_kwh=tot_em)

    # 问题 3 的正式口径：把附件 3 的整点预报按**结点线性插值**铺到 10 分钟网格，
    # 不做偏差订正、不留分位余量。
    #
    # 这一条是实测选出来的，不是想当然。全年 334 天的消融（exp_ablation.py /
    # exp_full_year.py）：
    #     不对时（旧）+ 旧口径订正 + 对冲 z=0.75   14961318.24 元  紧急 405164.6 kWh
    #     对时 + 不订正 + 不对冲                   13800482.05 元  紧急 206928.1 kWh
    #     对时 + 不订正 + 对冲 z=0.5               13832723.36 元  紧急 175037.3 kWh
    #     对时 + 订正 + 不对冲                     13831211.95 元  紧急 210504.7 kWh
    # 即：对时修正本身就值 7.8%，而订正与对冲在修正后**都是负收益**——
    # 因为原先那个「系统性偏差」正是拿整点预报去比小时均值比出来的假象。
    Z_HEDGE = 0.0
    bias = None
    sigma = None

    for tag, prices in (("q3", E1), ("q43", A4)):
        print()
        print("=" * 76)
        print(f"{'问题 3' if tag=='q3' else '问题 4-3'}（"
              f"{'附件1 电价' if tag=='q3' else '附件4 实时电价'}）")
        print("=" * 76)
        days = run_q3(prices, tag, bias=bias, sigma=sigma, z=Z_HEDGE)
        sel = [x for x in days if x["d"] >= Q2_FIRST]
        tot_plan = sum(x["cost_plan"] for x in sel)
        tot_em = sum(x["cost_em"] for x in sel)
        em_kwh = sum(x["emerg_kwh"] for x in sel)
        n_em_days = sum(1 for x in sel if x["emerg_kwh"] > 0)
        n_em_steps = sum(int((x["emerg"] > 0).sum()) for x in sel)
        print(f"  {len(sel)} 天  计划与调整结算 {tot_plan:.2f} 元   紧急购电费 {tot_em:.2f} 元")
        print(f"  紧急购电量 {em_kwh:.1f} kWh，发生在 {n_em_days} 天 / "
              f"{n_em_steps} 个 10 分钟时段（已滤除 <{EM_TOL_KWH} kWh/步 的求解器残差）")
        print(f"  合计 {tot_plan+tot_em:.2f} 元")
        fn = "result3.xlsx" if tag == "q3" else "result4-3.xlsx"
        write_result_daywise(os.path.join(OUT, fn), days, Q2_FIRST,
                             with_adjust=True,
                             price_lookup=lambda dd: _price_of_day(prices, dd))
        # 表3 指定的四个日期
        spec = {}
        for (mm, dd) in ((3, 20), (6, 21), (9, 23), (12, 21)):
            i = D.day_index(mm, dd)
            x = next(y for y in days if y["d"] == i)
            spec[f"2025-{mm:02d}-{dd:02d}"] = dict(
                emerg_kwh=x["emerg_kwh"],
                plan_kwh=float(x["P0"].sum() * M.DT),
                adj_kwh=float(x["Pfin"].sum() * M.DT),
                cost_plan=x["cost_plan"], cost_em=x["cost_em"])
        summary[tag] = dict(days=len(sel), cost_plan=tot_plan,
                            cost_em=tot_em, emerg_kwh=em_kwh,
                            emerg_days=n_em_days, cost=tot_plan + tot_em,
                            specified_dates=spec)

    with open(os.path.join(HERE, "out_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1,
                  default=float)
    print("\n已保存 out_summary.json")


if __name__ == "__main__":
    main()
