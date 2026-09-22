# -*- coding: utf-8 -*-
"""2026 国赛 C 题 —— 微网购电策略的线性规划核心。

**当日调度模型**（时间步 Δt = 1/6 h，一天 T=144 步）

决策变量（每步 t=0..T-1）
    buy_t ≥ 0      向外部电网的购电功率 [kW]
    ch_t ∈ [0,5000]  储能充电功率 [kW]（电网侧取电）
    dis_t ∈ [0,5000] 储能放电功率 [kW]（负荷侧供电）
    cur_t ≥ 0      光伏弃电量 [kW]
    soc_t          该步结束时的储电量 [kWh]

等式约束
    功率平衡  PV_t + buy_t + dis_t = load_t + ch_t + cur_t
    电量演化  soc_t = soc_{t-1} + η·ch_t·Δt − dis_t·Δt/η,   η = 0.9
    区间约束  1200 ≤ soc_t ≤ 10800,  soc_{-1} = soc_0(给定)
    周期约束  soc_{T-1} = soc_0（问题 1 要求；其余问题按日结转）

目标
    min Σ price_t · buy_t · Δt

## 问题 3 的调整结算

在 00:00 定下计划购电量 P0_t；在 06:00/12:00/18:00 可把**其后**各步的购电量
改为 P_t。按题面：

    计划购电量高于调整购电量的部分，违约电价是交易时刻电价的 50%；
    调整购电量高于计划购电量的部分，超出部分的电价是交易时刻电价的 1.5 倍。

即单步结算费用为

    g(P; P0) = price·[ min(P, P0) + 0.5·(P0−P)⁺ + 1.5·(P−P0)⁺ ]
             = price·[ 0.5·P0 + 0.5·P + (P−P0)⁺ ]

（第二个等号是本文件用到的化简形式：把 (P0−P)⁺ 用 (P−P0)⁺ 表示。
 它关于 P 是凸的分段线性函数，故仍可用 LP，只需为每步加一个铰链变量。）
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix

DT = 1.0 / 6.0            # 小时
T = 144                   # 每天 144 个 10 分钟区间
ETA = 0.9                 # 充放电效率
P_MAX = 5000.0            # 最大充放电功率 kW
SOC_MIN, SOC_MAX = 1200.0, 10800.0
SOC_CAP = 12000.0

I_BUY, I_CH, I_DIS, I_CUR, I_SOC = 0, 1, 2, 3, 4
NV = 5                    # 每步 5 个变量


def solve_day(price, load, pv, soc0, p0=None, cyclic=False,
              method="highs"):
    """求解一段时间的调度 LP。

    p0 = None  → 目标 min Σ price·buy·Δt                        （计划阶段）
    p0 给定     → 目标按题面调整结算规则：
                  Σ price·[ min(P,P0) + 0.5·(P0−P)⁺ + 1.5·(P−P0)⁺ ]·Δt
                化简为 Σ price·[ 0.5·P0 + 0.5·P + (P−P0)⁺ ]·Δt
                （0.5·P0 项是常数，不影响最优解，但计入返回的 cost）
    """
    Tn = len(price)
    n = Tn * NV
    idx = lambda t, k: t * NV + k
    hinged = p0 is not None

    # ---- 目标 ----
    n2 = n + (Tn if hinged else 0)
    c = np.zeros(n2)
    for t in range(Tn):
        c[idx(t, I_BUY)] = price[t] * DT * (0.5 if hinged else 1.0)
        if hinged:
            c[n + t] = price[t] * DT          # 铰链变量 e_t ≥ buy_t − P0_t

    # ---- 等式约束 ----
    rows, cols, vals, beq = [], [], [], []
    r = 0
    for t in range(Tn):                        # 功率平衡
        for k, v in ((I_BUY, 1.0), (I_DIS, 1.0), (I_CH, -1.0), (I_CUR, -1.0)):
            rows.append(r); cols.append(idx(t, k)); vals.append(v)
        beq.append(load[t] - pv[t]); r += 1
    for t in range(Tn):                        # 电量演化
        rows.append(r); cols.append(idx(t, I_SOC)); vals.append(1.0)
        if t > 0:
            rows.append(r); cols.append(idx(t - 1, I_SOC)); vals.append(-1.0)
        rows.append(r); cols.append(idx(t, I_CH)); vals.append(-ETA * DT)
        rows.append(r); cols.append(idx(t, I_DIS)); vals.append(DT / ETA)
        beq.append(soc0 if t == 0 else 0.0); r += 1
    if hinged:                                 # e_t − buy_t ≥ −P0_t
        for t in range(Tn):
            rows.append(r); cols.append(n + t); vals.append(1.0)
            rows.append(r); cols.append(idx(t, I_BUY)); vals.append(-1.0)
            beq.append(-p0[t]); r += 1

    A_eq = coo_matrix((vals, (rows, cols)), shape=(r, n2)).tocsr()
    b_eq = np.array(beq)

    # ---- 上下界 ----
    lb = np.zeros(n2)
    ub = np.full(n2, np.inf)
    for t in range(Tn):
        ub[idx(t, I_CH)] = P_MAX
        ub[idx(t, I_DIS)] = P_MAX
        lb[idx(t, I_SOC)] = SOC_MIN
        ub[idx(t, I_SOC)] = SOC_MAX

    if cyclic:
        A2 = coo_matrix(([1.0], ([0], [idx(Tn - 1, I_SOC)])),
                        shape=(1, n2)).tocsr()
        A_eq = _vstack(A_eq, A2)
        b_eq = np.append(b_eq, soc0)

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=list(zip(lb, ub)),
                  method=method)
    if not res.success:
        return dict(ok=False, msg=res.message)
    x = res.x
    out = dict(ok=True, obj=float(res.fun), n=n2)
    for k, nm in ((I_BUY, "buy"), (I_CH, "ch"), (I_DIS, "dis"),
                  (I_CUR, "cur"), (I_SOC, "soc")):
        out[nm] = np.array([x[idx(t, k)] for t in range(Tn)])
    if hinged:
        out["e"] = x[n:]
        out["cost"] = float(np.sum(price * (0.5 * p0 + 0.5 * out["buy"]) * DT)
                            + np.sum(price * out["e"] * DT))
    else:
        out["cost"] = float(np.sum(price * out["buy"] * DT))
    return out


def _vstack(A, B):
    from scipy.sparse import vstack
    return vstack([A, B]).tocsr()


# ══════════════════════════════════════════════════════════
def check_solution(sol, price, load, pv, soc0, cyclic=False):
    """把解代回原约束逐条核验，返回最大违反量（用于自检）。"""
    bal = np.max(np.abs(pv + sol["buy"] + sol["dis"]
                        - load - sol["ch"] - sol["cur"]))
    soc = sol["soc"]
    prev = np.concatenate([[soc0], soc[:-1]])
    dyn = np.max(np.abs(soc - (prev + ETA * sol["ch"] * DT
                               - sol["dis"] * DT / ETA)))
    lim = max(np.max(np.maximum(sol["ch"] - P_MAX, 0)),
              np.max(np.maximum(sol["dis"] - P_MAX, 0)),
              np.max(np.maximum(SOC_MIN - soc, 0)),
              np.max(np.maximum(soc - SOC_MAX, 0)),
              np.max(np.maximum(-sol["buy"], 0)),
              np.max(np.maximum(-sol["cur"], 0)))
    cyc = abs(soc[-1] - soc0) if cyclic else 0.0
    return dict(balance=float(bal), dynamics=float(dyn), bounds=float(lim),
                cyclic=float(cyc))
