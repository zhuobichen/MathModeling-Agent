# -*- coding: utf-8 -*-
"""对 drying_core 的数值验证。

强验证：常物性 + 定值边界时无限长圆柱有分离变量级数解，可直接比对。
        （问题 1 的温度场在物性上恰好是常物性，故这不是造出来的特例。）

     theta = Σ A_n J_0(λ_n ξ) exp(-λ_n² Fo)
     A_n = 2 J_1(λ_n) / (λ_n (J_0(λ_n)² + J_1(λ_n)²))
     λ_n 满足  Bi J_0(λ_n) = λ_n J_1(λ_n),   Fo = α t / R²,  Bi = hR/k

另做：网格收敛、时间步收敛、能量/质量守恒闭合。
"""
import sys

import numpy as np
from scipy.optimize import brentq
from scipy.special import j0, j1

sys.stdout.reconfigure(encoding="utf-8")

import drying_core as dc


# ── 解析解 ────────────────────────────────────────────────
def cylinder_roots(Bi, n=200):
    """求 λ J_1(λ) = Bi J_0(λ) 的前 n 个正根。"""
    f = lambda L: L * j1(L) - Bi * j0(L)
    roots, L, step = [], 1e-6, 0.02
    prev = f(L + 1e-9)
    while len(roots) < n and L < 900:
        Ln = L + step
        cur = f(Ln)
        if prev == 0:
            roots.append(L)
        elif prev * cur < 0:
            roots.append(brentq(f, L, Ln, xtol=1e-14))
        prev, L = cur, Ln
    return np.array(roots[:n])


def cylinder_theta(xi, Fo, Bi, n=200):
    """无限长圆柱，初始均匀、表面对流、环境定值。"""
    L = cylinder_roots(Bi, n)
    A = 2 * j1(L) / (L * (j0(L) ** 2 + j1(L) ** 2))
    xi = np.atleast_1d(np.asarray(xi, float))
    out = np.zeros_like(xi)
    for ln, an in zip(L, A):
        out += an * j0(ln * xi) * np.exp(-ln ** 2 * Fo)
    return out


# ── 无物性依赖的常系数物性（用于对拍）────────────────────
def make_const_props(rho, cp, k, D):
    def f(C, T):
        n = np.shape(C)
        return (np.full(n, rho), np.full(n, cp), np.full(n, k),
                np.full(n, D))
    return f


RHO, CP, K, R0, H, HM = 820.0, 2600.0, 0.36, 0.02, 25.0, 8e-7
ALPHA = K / (RHO * CP)
D_CONST = 7e-9 * np.exp(-0.89 / 2.55)
BI = H * R0 / K
BI_M = HM * R0 / D_CONST


def run_case(N, dt, t_end, k=K, D=D_CONST, h=H, hm=HM, theta=0.5,
             T_inf=50.0, C_inf=0.05):
    s = dc.CylinderSolver(N, make_const_props(RHO, CP, k, D),
                          dict(h=h, hm=hm), R0=R0)
    res = s.run(t_end=t_end, dt_fn=lambda t: dt,
                T_init=28.0, C_init=2.55,
                bc=lambda t: (T_inf, C_inf), out_times=[t_end], theta=theta)
    return s, res


def main():
    ok = True
    print("=" * 72)
    print("【1】温度场 vs 无限长圆柱解析级数解（常物性、定值边界）")
    print(f"    alpha={ALPHA:.6e} m²/s   Bi={BI:.5f}   N=400  dt=1 s")
    print("=" * 72)
    for t in (60.0, 300.0, 600.0, 1800.0):
        s, res = run_case(400, 1.0, t)
        Fo = ALPHA * t / R0 ** 2
        xi = (np.arange(400) + 0.5) / 400
        num = (res["T"][-1] - 50.0) / (28.0 - 50.0)
        ana = cylinder_theta(xi, Fo, BI)
        err = np.max(np.abs(num - ana))
        print(f"   t={t:6.0f}s Fo={Fo:8.5f}  最大|Δθ|={err:.3e}  "
              f"{'OK' if err < 1e-4 else '**FAIL**'}")
        ok &= err < 1e-4

    print()
    print("=" * 72)
    print("【2】水分场 vs 解析级数解（常 D、定值边界）")
    print(f"    D={D_CONST:.6e} m²/s   Bi_m={BI_M:.6f}")
    print("=" * 72)
    for t in (300.0, 900.0, 1800.0):
        s, res = run_case(400, 1.0, t)
        Fo = D_CONST * t / R0 ** 2
        xi = (np.arange(400) + 0.5) / 400
        num = (res["C"][-1] - 0.05) / (2.55 - 0.05)
        ana = cylinder_theta(xi, Fo, BI_M)
        err = np.max(np.abs(num - ana))
        print(f"   t={t:6.0f}s Fo={Fo:8.5f}  最大|Δθ|={err:.3e}  "
              f"{'OK' if err < 1e-5 else '**FAIL**'}")
        ok &= err < 1e-5

    print()
    print("=" * 72)
    print("【3】网格收敛（t=1800 s，dt=1 s）")
    print("=" * 72)
    for N in (50, 100, 200, 400, 800):
        s, res = run_case(N, 1.0, 1800.0)
        print(f"   N={N:5d}  T(0)={res['T'][-1][0]:.8f}  "
              f"T(R)={res['T'][-1][-1]:.8f}  C(0)={res['C'][-1][0]:.8f}  "
              f"C(R)={res['C'][-1][-1]:.8f}")

    print()
    print("=" * 72)
    print("【4】时间步收敛（t=600 s，N=400；后向欧拉 vs Crank–Nicolson）")
    print("=" * 72)
    for th, nm in ((1.0, "后向欧拉(1阶)"), (0.5, "C-N(2阶)   ")):
        print(f"   {nm}:")
        prev = None
        for dt in (4.0, 2.0, 1.0, 0.5):
            s, res = run_case(400, dt, 600.0, theta=th)
            v = res["T"][-1][0]
            print(f"      dt={dt:5.2f}s  T(0)={v:.9f}"
                  + (f"   Δ={v - prev:+.3e}" if prev is not None else ""))
            prev = v

    print()
    print("=" * 72)
    print("【5】能量与质量守恒闭合（t=1800 s，N=400，dt=1 s，C-N）")
    print("=" * 72)
    s, res = run_case(400, 1.0, 1800.0)
    V = 2 * np.pi * s.V * R0 ** 2
    dE = RHO * CP * np.sum((res["T"][-1] - 28.0) * V)
    print(f"   能量：蓄热增量 {dE:14.6f} J/m   累计边界热流 "
          f"{res['bal']['qh']:14.6f} J/m   相对差 "
          f"{abs(dE - res['bal']['qh']) / abs(res['bal']['qh']):.3e}")
    # 注意：水分方程 dC/dt = (1/r)∂/∂r(Dr∂C/∂r) 里的 C 是干基含水率，
    # 方程本身**不含密度**，所以这里不能乘 RHO。乘了会让守恒差恰好
    # 等于 ρ=820（实测 819），是个很容易误判成"求解器漏水"的自造陷阱。
    dM = np.sum((res["C"][-1] - 2.55) * V)
    print(f"   水分：含水量增量 {dM:12.6f} kg/m  累计边界湿流 "
          f"{-res['bal']['qm']:12.6f} kg/m  相对差 "
          f"{abs(dM + res['bal']['qm']) / abs(res['bal']['qm']):.3e}")

    print()
    print("=" * 72)
    print("总体:", "全部通过" if ok else "**有失败项**")
    print("=" * 72)


if __name__ == "__main__":
    main()
