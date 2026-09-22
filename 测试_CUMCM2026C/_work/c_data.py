# -*- coding: utf-8 -*-
"""数据装载。所有数组形状约定为 (天数, 144)，10 分钟一步。"""
from __future__ import annotations

import os

import numpy as np
import openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
ATT = os.path.join(HERE, "..", "附件")

DT = 1.0 / 6.0
T = 144

# 附件2 的日期范围：2025-01-01 ~ 2025-12-31，共 365 天
DAYS = 365
MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def day_index(month, day):
    """(月, 日) → 0 起的日序号。"""
    return sum(MONTH_DAYS[:month - 1]) + (day - 1)


def day_label(i):
    m = 1
    d = i + 1
    for ml in MONTH_DAYS:
        if d <= ml:
            return m, d
        d -= ml
        m += 1
    raise ValueError(i)


def load_att1():
    """附件1：某天的电价、小区负载、光伏发电预测功率（144 步）。"""
    wb = openpyxl.load_workbook(os.path.join(ATT, "附件1.xlsx"), data_only=True)
    rows = [r for r in wb.active.iter_rows(min_row=2, values_only=True)]
    price = np.array([float(r[1]) for r in rows])
    load = np.array([float(r[2]) for r in rows])
    pv = np.array([float(r[3]) for r in rows])
    assert len(price) == T
    return price, load, pv


def _load_matrix(path, sheet):
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb[sheet] if sheet in wb.sheetnames else wb.active
    rows = [r for r in ws.iter_rows(min_row=2, values_only=True)]
    rows = [r for r in rows if r[0] is not None]
    M = np.array([[float(x) for x in r[1:1 + T]] for r in rows])
    return M


def load_att2():
    """附件2：全年小区负载与光伏实际功率，各 (365,144)。"""
    p = os.path.join(ATT, "附件2.xlsx")
    load = _load_matrix(p, "小区负载")
    pv = _load_matrix(p, "光伏发电实际功率")
    assert load.shape == (DAYS, T), load.shape
    return load, pv


def load_att3():
    """附件3：预报。返回 dict[(day, issue_hour)] = (24,) 的逐小时预报。

    issue_hour ∈ {0, 6, 12, 18}；预报 k 小时 = 自发布时刻起第 k 小时的光伏功率。
    """
    wb = openpyxl.load_workbook(os.path.join(ATT, "附件3.xlsx"), data_only=True)
    ws = wb.active
    rows = [r for r in ws.iter_rows(min_row=2, values_only=True)]
    out = {}
    day = -1
    for r in rows:
        if r[0]:
            day += 1
        if r[1] is None:
            continue
        hh = int(str(r[1]).split(":")[0])
        out[(day, hh)] = np.array([float(x) for x in r[2:26]])
    assert day == DAYS - 1, day
    return out


def actual_at(pv_day, tau):
    """附件 2 第 j 列是时刻 (j+1)/6 小时的值，故「整点 tau」= 第 tau*6−1 列。"""
    idx = int(round(tau * 6)) - 1
    if idx < 0:
        return 0.0
    if idx >= T:
        return float(pv_day[-1])
    return float(pv_day[idx])


def forecast_to_grid(fc24, issue_hour, beta=None, hedge=None, z=0.0):
    """把「自 issue_hour 发布的 24 个逐小时预报」铺到 144 步的 10 分钟网格。

    附件 3 的第 k 小时预报 fc24[k−1] 是**发布后第 k 小时整点那一瞬间**的功率，
    不是该小时的平均值。判据是把预报与附件 2 实测在几种对时方式下比 MAE
    （全样本 21900 个）：

        整点 X = issue+k        MAE 117.15   r 0.9957   ← 真身
        整点 X = issue+k−1      MAE 667.05   r 0.9401
        (h, h+1] 小时均值       MAE 308.80   r 0.9856

    所以正确用法是：以整点 τ_k = issue+k 为结点、取 fc24[k−1]，在结点间
    **线性插值**到 10 分钟网格。原先的做法是把 fc24[k−1] 铺在
    [issue+k−1, issue+k) 上（早了一小时，且小时内取常数），早晨爬坡段
    因此系统性错位。

    beta / hedge 为偏差与标准差的查表函数 f(issue, k)，同样作用在结点上
    （也必须同刻，不能拿「小时均值口径」的偏差去订正「整点瞬时」的预报）。
    当天得不到预报的步返回 nan。
    """
    ts, vs = [], []
    for k in range(1, 25):
        tau = issue_hour + k
        if tau > 24:                 # 超出当天 → 属于次日，本次不用
            continue
        v = float(fc24[k - 1])
        if beta is not None:
            v += beta(issue_hour, k)
        if hedge is not None:
            v -= z * hedge(issue_hour, k)
        ts.append(tau)
        vs.append(v)
    g = np.full(T, np.nan)
    if not ts:
        return g
    ts = np.array(ts, float)
    vs = np.array(vs, float)
    for j in range(T):
        t = (j + 1) / 6.0            # 该步结束的时刻（小时）
        if t < issue_hour:
            continue
        g[j] = float(np.interp(t, ts, vs))   # 早于首个结点时取平
    return g


def load_att4():
    """附件4：全年逐 10 分钟电价 (365,144)。"""
    return _load_matrix(os.path.join(ATT, "附件4.xlsx"), "Sheet1")


# ── 4 小时时段划分（用于表2 的充放电量）──────────────────
PERIODS = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24)]


def period_sums(sol_ch, sol_dis):
    """把 144 步的充/放电功率聚合到 6 个 4 小时时段，返回 kWh。"""
    out = []
    for a, b in PERIODS:
        lo, hi = a * 6, b * 6
        out.append((float(sol_ch[lo:hi].sum() * DT),
                    float(sol_dis[lo:hi].sum() * DT)))
    return out
