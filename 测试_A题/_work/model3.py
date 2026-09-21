"""2022 A题 · 问题3、4：浮子垂荡+纵摇，振子沿轴滑动（4 自由度）

广义坐标 q = [z, θ, s, φ]
  z  浮子垂荡位移 (m)
  θ  浮子纵摇角位移 (rad)
  s  振子沿中轴相对平衡位置的滑动位移 (m)
  φ  中轴相对浮子的转角 (rad)

小角度线性化下的动能（已含附加质量/附加转动惯量）：
  T = ½(m1+μ)ż² + ½J θ̇² + ½m2[(ż+ṡ)² + (r0+s)²(θ̇+φ̇)²]
  其中 J = J_buoy + J_add（浮子自身纵摇转动惯量 + 纵摇附加转动惯量）
势能：
  V = ½k_h z² + ½k_θ θ² + ½k_spring s² + ½k_torsion φ²
耗散函数：
  D = ½ν ż² + ½ν_θ θ̇² + ½c ṡ² + ½c_θ φ̇²
广义力：Q_z = f cos ωt，Q_θ = L cos ωt，Q_s = Q_φ = 0

拉格朗日方程给出：
 (m1+m2+μ)z̈ + m2 s̈ + ν ż + k_h z = f cos ωt
 m2(z̈ + s̈) − m2(r0+s)(θ̇+φ̇)² + k_spring s + c ṡ = 0
 J θ̈ + m2[(r0+s)²(θ̈+φ̈) + 2(r0+s)ṡ(θ̇+φ̇)] + k_θ θ + ν_θ θ̇ = L cos ωt
 m2[(r0+s)²(θ̈+φ̈) + 2(r0+s)ṡ(θ̇+φ̇)] + k_torsion φ + c_θ φ̇ = 0

输出功率：P = c ṡ² + c_θ φ̇²
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root, minimize
import model_core as M

# ── 几何：浮子绕横轴（过重心）的转动惯量，按"质量均匀分布于壳体表面"积分 ──
def buoy_inertia():
    """对圆柱壳体 + 圆锥壳体积分求绕横轴的转动惯量（过整体重心）。"""
    m1, R, hc, hk, rho_ = M.m1, M.R_buoy, M.h_cyl, M.h_cone, M.rho

    def shell_area_cylinder():     # 圆柱侧面积
        return 2 * np.pi * R * hc

    def cone_slant():              # 圆锥母线长
        return np.hypot(R, hk)

    def shell_area_cone():         # 圆锥侧面积
        return np.pi * R * cone_slant()

    A_cyl, A_cone = shell_area_cylinder(), shell_area_cone()
    A_tot = A_cyl + A_cone
    sig = m1 / A_tot                # 面密度 kg/m2

    # 坐标：圆锥底面在 y=0，向上为圆锥（高 hk，半径由 R 线性缩到 0），再向上为圆柱
    # 用数值积分求 质量、质心、绕横轴(过原点)的转动惯量 ∫(y²+ (x²+z²)) 中横轴取 x 轴
    # 绕 x 轴的转动惯量 = ∫(y²+z²)dm；轴对称下 ∫z²dm = ½∫(x²+z²)dm = ½∫r²dm
    N = 4000
    m_tot = 0.0; my = 0.0; Ix = 0.0
    # 圆锥：y∈[0,hk]，半径 ρ(y)=R(1-y/hk)，侧面积微元 dA = 2πρ ds
    ys = np.linspace(0, hk, N)
    for y in ys:
        rad = R * (1 - y / hk)
        ds = cone_slant() / hk * (1.0)   # dy→ds 比例
        dA = 2 * np.pi * rad * (cone_slant() / hk) * (hk / N)
        dm = sig * dA
        m_tot += dm; my += dm * y
        Ix += dm * (y**2 + 0.5 * rad**2)   # z² 平均为 r²/2
    # 圆柱：y∈[hk, hk+hc]，半径 R
    ys2 = np.linspace(hk, hk + hc, N)
    for y in ys2:
        dA = 2 * np.pi * R * (hc / N)
        dm = sig * dA
        m_tot += dm; my += dm * y
        Ix += dm * (y**2 + 0.5 * R**2)
    ycm = my / m_tot
    # 平行轴：绕质心的转动惯量
    I_cm = Ix - m_tot * ycm**2
    return dict(I_about_origin=Ix, I_cm=I_cm, ycm=ycm,
                A_tot=A_tot, sigma=sig, m_check=m_tot)

BUOY_I = buoy_inertia()

# 振子（实心圆柱）绕自身质心横轴的转动惯量，若需要随中轴转动可计入
I_osc = M.m2 * (3 * M.R_osc**2 + M.h_osc**2) / 12

R0 = 0.5   # 振子平衡位置到铰链的距离（取弹簧原长）


def make_rhs3(prob, c=10000.0, c_theta=1000.0, J_buoy=None):
    h = M.HYDRO[prob]
    mu, nu, nu_th, f, L, w = h["mu"], h["nu"], h["nu_th"], h["f"], h["L"], h["omega"]
    Jadd = h["J"]
    J = (J_buoy if J_buoy is not None else BUOY_I["I_cm"]) + Jadd
    k_h = M.k_h; k_th = M.k_theta_hydro
    ks = M.k_spring; kt = M.k_torsion
    m2 = M.m2; mT = M.m1 + M.m2 + mu

    def rhs(t, y):
        z, vz, th, wth, s, vs, ph, wph = y
        r = R0 + s
        W = wth + wph                     # 轴相对惯性系的总角速度
        Fc = f * np.cos(w * t); Mc = L * np.cos(w * t)
        # 线性方程组 A·[az, ath, as, aph] = b
        A = np.array([
            [mT,    0.0,        m2,    0.0],
            [m2,    0.0,        m2,    0.0],
            [0.0,   J + m2*r*r, 0.0,   m2*r*r],
            [0.0,   m2*r*r,     0.0,   m2*r*r],
        ])
        cent = m2 * r * W * W
        b = np.array([
            Fc - nu * vz - k_h * z,
            cent - ks * s - c * vs,
            Mc - k_th * th - nu_th * wth - 2 * m2 * r * vs * W,
            -kt * ph - c_theta * wph - 2 * m2 * r * vs * W,
        ])
        acc = np.linalg.solve(A, b)
        return (vz, acc[0], wth, acc[1], vs, acc[2], wph, acc[3])
    return rhs


def solve_p3(prob=3, c=10000.0, c_theta=1000.0, n_periods=40, dt=0.2,
             J_buoy=None, rtol=1e-9, atol=1e-11):
    h = M.HYDRO[prob]; T = 2 * np.pi / h["omega"]
    t_end = n_periods * T
    t_eval = np.arange(0.0, t_end + 1e-9, dt)
    sol = solve_ivp(make_rhs3(prob, c, c_theta, J_buoy), (0.0, t_end),
                    [0.0] * 8, t_eval=t_eval, method="RK45", rtol=rtol, atol=atol)
    assert sol.success, sol.message
    return t_eval, sol.y     # [z,vz,th,wth,s,vs,ph,wph]


# ── 周期打靶（8 维，用于问题4 的稳态寻优）──
def periodic_orbit3(prob, c=10000.0, c_theta=1000.0, J_buoy=None,
                    x0=None, n_dense=800, rtol=1e-10, atol=1e-12):
    h = M.HYDRO[prob]; T = 2 * np.pi / h["omega"]
    rhs = make_rhs3(prob, c, c_theta, J_buoy)

    def one_period(X):
        s = solve_ivp(rhs, (0.0, T), X, method="RK45", rtol=rtol, atol=atol)
        return s.y[:, -1]

    guess = x0 if x0 is not None else np.zeros(8)
    r = root(lambda X: one_period(X) - X, guess, method="hybr", tol=1e-11)
    t_eval = np.linspace(0.0, T, n_dense)
    s = solve_ivp(rhs, (0.0, T), r.x, t_eval=t_eval, method="RK45", rtol=rtol, atol=atol)
    return s.t, s.y, r.x, r.success


def avg_power3(prob, c=10000.0, c_theta=1000.0, J_buoy=None, x0=None):
    """稳态平均输出功率 = 线性阻尼功率 + 旋转阻尼功率。"""
    t, y, Xs, ok = periodic_orbit3(prob, c, c_theta, J_buoy, x0)
    P = c * y[5] ** 2 + c_theta * y[7] ** 2
    return float(np.trapezoid(P, t) / (t[-1] - t[0])), Xs, ok
