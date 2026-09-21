# -*- coding: utf-8 -*-
"""2026 高教社杯 A 题「药材的烘干问题」——四个子问题的求解与结果导出。

问题 1：附录 2 常物性，R = 2 cm 固定，边界取附件 1 的烘房温湿度，前 1800 s。
问题 2：附录 3 变物性，同上，前 3 h。
问题 3：附录 3 变物性，续算到药材各处水分浓度 < 0.15 kg/kg。
问题 4：附录 4 变物性 + 药材收缩（半径由附件 2 给出），同问题 3 的判据。

结果写入 result1.xlsx ~ result4.xlsx（模板结构见附件 3）。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.interpolate import PchipInterpolator

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import drying_core as dc

HERE = os.path.dirname(os.path.abspath(__file__))
ATT = os.path.join(HERE, "..", "附件")
OUT = os.path.join(HERE, "..", "_交付")

R0 = 0.02          # 初始半径 [m]
T_INIT = 28.0      # 初始温度 [°C]
C_INIT = 2.55      # 初始干基含水率 [kg/kg]
C_TARGET = 0.15    # 干燥判据 [kg/kg]
N_CELL = 400       # 径向单元数

DEV = [round(0.1 * k, 1) for k in range(21)]   # 0.0 ~ 2.0 cm


# ══════════════════════════════════════════════════════════
# 数据装载
# ══════════════════════════════════════════════════════════
def load_bc():
    """附件 1：烘房温度与水分浓度（60 s 间隔，0~14400 s）。"""
    import openpyxl
    wb = openpyxl.load_workbook(os.path.join(ATT, "附件1.xlsx"), data_only=True)
    rows = [r for r in wb.active.iter_rows(values_only=True)][1:]
    rows = [r for r in rows if r[0] is not None]
    t = np.array([float(r[0]) for r in rows])
    T = np.array([float(r[1]) for r in rows])
    C = np.array([float(r[2]) for r in rows])
    return t, T, C


def load_radius():
    """附件 2：药材半径随时间（1800 s 间隔，0~259200 s）。"""
    import openpyxl
    wb = openpyxl.load_workbook(os.path.join(ATT, "附件2.xlsx"), data_only=True)
    rows = [r for r in wb.active.iter_rows(values_only=True)][1:]
    rows = [r for r in rows if r[0] is not None]
    t = np.array([float(r[0]) for r in rows])
    R = np.array([float(r[1]) for r in rows]) / 100.0     # cm → m
    return t, R


# ══════════════════════════════════════════════════════════
class BC:
    """烘房条件的时间插值（含 14400 s 之后的延拓假设）。"""

    def __init__(self, t, T, C, t_hold=None):
        self.t = t
        self.fT = PchipInterpolator(t, T)
        self.fC = PchipInterpolator(t, C)
        # 默认延拓点 = 数据的最后一个时刻。注意 t_hold 必须是**标量**：
        # 写成 self.t_hold = t（数组）会让 argmin(|t - t_hold|) 变成
        # 数组减自己的全零，永远选到第 0 点 —— 于是 14400 s 之后误用
        # 初始的 28 °C / 0.0196 而不是末态的 50 °C / 0.05，长时程结果
        # 整体偏慢数倍，而且因为数值"看起来合理"很不容易发现。
        self.t_hold = float(t[-1]) if t_hold is None else float(t_hold)
        i = int(np.argmin(np.abs(t - self.t_hold)))
        self.T_hold = float(T[i])
        self.C_hold = float(C[i])

    def __call__(self, tt):
        tt = np.asarray(tt, float)
        T = np.where(tt <= self.t[-1], self.fT(np.clip(tt, self.t[0], self.t[-1])),
                     self.T_hold)
        C = np.where(tt <= self.t[-1], self.fC(np.clip(tt, self.t[0], self.t[-1])),
                     self.C_hold)
        return float(T), float(C)


def make_R_of_t(t, R, t_end_hold=None):
    f = PchipInterpolator(t, R)
    tax = t[-1] if t_end_hold is None else t_end_hold
    Rhold = float(R[-1])

    def fR(tt):
        return float(f(tt)) if tt <= tax else Rhold
    return fR, f


# ══════════════════════════════════════════════════════════
# 时间步调度
# ══════════════════════════════════════════════════════════
def dt_short(t):
    """问题 1、2：全程 1 s（要按 1 s 输出）。"""
    return 1.0


def dt_long(t):
    """问题 3、4：早期细分，后期放大。尺度由 D 与 R² 定。"""
    if t < 60:
        return 1.0
    if t < 600:
        return 5.0
    if t < 3600:
        return 15.0
    return 60.0


# ══════════════════════════════════════════════════════════
def run_case(key, props_fn, consts_fn, bc, t_end, dt_fn, R_of_t=None,
             out_times=None, record_every=None, N=N_CELL, theta=0.5,
             material_frame=False):
    s = dc.CylinderSolver(N, props_fn, consts_fn(), R0=R0,
                          material_frame=material_frame)
    res = s.run(t_end=t_end, dt_fn=dt_fn, T_init=T_INIT, C_init=C_INIT,
                bc=bc, R_of_t=R_of_t, out_times=out_times,
                record_every=record_every, theta=theta)
    return s, res


def balance_check(tag, s, res):
    """能量/质量守恒闭合：蓄积增量 vs 累计边界通量。"""
    Rmet2 = (s.R0 if s.material_frame else res["R"][-1]) ** 2
    V = 2 * np.pi * s.V * Rmet2
    rho, cp, k, D = s.props(res["C"][-1], res["T"][-1])
    dE = float(np.sum(rho * cp * (res["T"][-1] - T_INIT) * V))
    dW = float(np.sum((res["C"][-1] - C_INIT) * V))
    qh, qm = res["bal"]["qh"], res["bal"]["qm"]
    eh = abs(dE - qh) / max(abs(qh), 1e-30)
    em = abs(dW + qm) / max(abs(qm), 1e-30)
    print(f"  [{tag}] 守恒：能量 {dE:14.4f} vs {qh:14.4f} (相对差 {eh:.2e})"
          f"；水分 {dW:+.8f} vs {-qm:+.8f} (相对差 {em:.2e})")
    return eh, em


def drying_time(res, target=C_TARGET):
    """首次出现全场 C < target 的时刻 [s]。"""
    mx = res["C"].max(axis=1)
    idx = np.where(mx < target)[0]
    if len(idx) == 0:
        return None, mx
    i = idx[0]
    # 在两级记录之间线性插值，得到更精确的跨越时刻
    if i == 0:
        return float(res["t"][0]), mx
    t0, t1 = res["t"][i - 1], res["t"][i]
    m0, m1 = mx[i - 1], mx[i]
    if m0 == m1:
        return float(t0), mx
    return float(t0 + (target - m0) * (t1 - t0) / (m1 - m0)), mx


# ══════════════════════════════════════════════════════════
def main():
    t_bc, T_bc, C_bc = load_bc()
    t_R, R_att = load_radius()
    bc = BC(t_bc, T_bc, C_bc)
    print("=" * 74)
    print("2026 A 题  药材的烘干问题")
    print("=" * 74)
    print(f"附件1 烘房：{len(t_bc)} 点 {t_bc[0]:.0f}~{t_bc[-1]:.0f} s；"
          f"14400 s 处 T={bc.T_hold:.3f} °C, C={bc.C_hold:.5f}")
    print(f"附件2 半径：{len(t_R)} 点 {t_R[0]:.0f}~{t_R[-1]:.0f} s；"
          f"R: {R_att[0]*100:.3f} → {R_att[-1]*100:.3f} cm")
    print(f"网格 {N_CELL} 单元，时间格式 Crank–Nicolson（θ=0.5）")

    # ── 问题 1 ────────────────────────────────────────────
    print("\n" + "=" * 74)
    print("问题 1  预热平衡阶段（附录 2 常物性，R=2 cm 固定，0~1800 s）")
    print("=" * 74)
    out1 = list(np.arange(1.0, 1800.0 + 0.5, 1.0))
    s1, r1 = run_case("q1", dc.props_q1, dc.consts_q1, bc, 1800.0,
                      dt_short, out_times=out1)
    T1, C1 = dc.sample_at(r1, DEV)
    print(f"  末态：中心 T={r1['T'][-1][0]:.4f} °C  表面 T={r1['T'][-1][-1]:.4f} °C")
    print(f"        中心 C={r1['C'][-1][0]:.4f}     表面 C={r1['C'][-1][-1]:.4f}")
    print(f"  平衡校验：表面 C 相对烘房 C_inf={bc(1800.0)[1]:.5f} 的偏差 "
          f"{r1['C'][-1][-1] - bc(1800.0)[1]:+.5f}")

    # ── 问题 2 ────────────────────────────────────────────
    print("\n" + "=" * 74)
    print("问题 2  全过程前 3 h（附录 3 变物性，R=2 cm 固定）")
    print("=" * 74)
    out2 = list(np.arange(1.0, 10800.0 + 0.5, 1.0))
    s2, r2 = run_case("q2", dc.props_q3, dc.consts_q3, bc, 10800.0,
                      dt_short, out_times=out2)
    T2, C2 = dc.sample_at(r2, DEV)
    print(f"  3 h 末态：中心 T={r2['T'][-1][0]:.4f} °C  表面 T={r2['T'][-1][-1]:.4f} °C")
    print(f"           中心 C={r2['C'][-1][0]:.4f}     表面 C={r2['C'][-1][-1]:.4f}")

    # ── 问题 3 ────────────────────────────────────────────
    print("\n" + "=" * 74)
    print("问题 3  确定烘干所需时间（附录 3 变物性，判据 C<0.15 处处成立）")
    print("=" * 74)
    tmax3 = 600000.0     # 上限查 6.9 天，足够覆盖
    s3, r3 = run_case("q3", dc.props_q3, dc.consts_q3, bc, tmax3,
                      dt_long, record_every=60.0)
    td3, mx3 = drying_time(r3)
    print(f"  烘干时长 = {td3:.0f} s = {td3/3600:.3f} h")
    print(f"  （记录末态全场最大 C = {mx3[-1]:.5f}）")

    # ── 问题 4 ────────────────────────────────────────────
    print("\n" + "=" * 74)
    print("问题 4  考虑收缩（附录 4 变物性 + 附件 2 半径）")
    print("=" * 74)
    fR, fR_raw = make_R_of_t(t_R, R_att)
    s4, r4 = run_case("q4", dc.props_q4, dc.consts_q4, bc, tmax3,
                      dt_long, R_of_t=fR, record_every=60.0,
                      material_frame=True)
    td4, mx4 = drying_time(r4)
    print(f"  烘干时长 = {td4:.0f} s = {td4/3600:.3f} h")
    print(f"  末态半径 = {r4['R'][-1]*100:.4f} cm "
          f"（附件 2 在 {t_R[-1]:.0f} s 后为 {R_att[-1]*100:.3f} cm）")

    # ── 守恒校验 ─────────────────────────────────────────
    print("")
    print("=" * 74)
    print("守恒校验（蓄积增量 vs 累计边界通量）")
    print("=" * 74)
    balance_check("问题1", s1, r1)
    balance_check("问题2", s2, r2)
    balance_check("问题3", s3, r3)
    balance_check("问题4", s4, r4)

    # ── 汇总存盘 ─────────────────────────────────────────
    os.makedirs(OUT, exist_ok=True)
    summary = dict(
        q1=dict(T_end_center=float(r1["T"][-1][0]),
                T_end_surface=float(r1["T"][-1][-1]),
                C_end_center=float(r1["C"][-1][0]),
                C_end_surface=float(r1["C"][-1][-1])),
        q2=dict(T_end_center=float(r2["T"][-1][0]),
                T_end_surface=float(r2["T"][-1][-1]),
                C_end_center=float(r2["C"][-1][0]),
                C_end_surface=float(r2["C"][-1][-1])),
        q3=dict(drying_time_s=td3, drying_time_h=td3 / 3600),
        q4=dict(drying_time_s=td4, drying_time_h=td4 / 3600,
                R_end_cm=float(r4["R"][-1] * 100)),
    )
    with open(os.path.join(HERE, "out_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)

    np.savez_compressed(
        os.path.join(HERE, "state.npz"),
        dev=np.array(DEV),
        q1_t=r1["t"], q1_T=r1["T"], q1_C=r1["C"],
        q2_t=r2["t"], q2_T=r2["T"], q2_C=r2["C"],
        q3_t=r3["t"], q3_C=r3["C"], q3_R=r3["R"],
        q4_t=r4["t"], q4_C=r4["C"], q4_R=r4["R"],
    )
    print("\n已保存 state.npz 与 out_summary.json")


if __name__ == "__main__":
    main()
