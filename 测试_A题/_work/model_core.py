"""2022 A题 波浪能装置 · 核心模型（问题1、2：仅垂荡）"""
import numpy as np
from scipy.integrate import solve_ivp

# ── 附件4 物理几何参数 ──
m1 = 4866.0      # 浮子质量 kg
R_buoy = 1.0     # 浮子底半径 m
h_cyl = 3.0      # 圆柱部分高度 m
h_cone = 0.8     # 圆锥部分高度 m
m2 = 2433.0      # 振子质量 kg
R_osc = 0.5      # 振子半径 m
h_osc = 0.5      # 振子高度 m
rho = 1025.0     # 海水密度 kg/m3
g = 9.8          # 重力加速度
k_spring = 80000.0   # 弹簧刚度 N/m
L0 = 0.5             # 弹簧原长 m
k_torsion = 250000.0 # 扭转弹簧刚度 N·m
k_theta_hydro = 8890.7  # 静水恢复力矩系数 N·m

# ── 导出量 ──
DISPLACED_VOL = (m1 + m2) / rho          # 平衡所需排水体积 m3
S_waterplane = np.pi * R_buoy**2         # 水线面面积 m2
DRAFT = DISPLACED_VOL / S_waterplane     # 平衡吃水 m
k_h = rho * g * S_waterplane             # 静水恢复力系数 N/m
V_cyl = np.pi * R_buoy**2 * h_cyl

# ── 附件3 各问水动力系数 ──
HYDRO = {
    1: dict(omega=1.4005, mu=1335.535,  J=6779.315, nu=656.3616, nu_th=151.4388, f=6250.0, L=1230.0),
    2: dict(omega=2.2143, mu=1165.992,  J=7131.290, nu=167.8395, nu_th=2992.724, f=4890.0, L=2560.0),
    3: dict(omega=1.7152, mu=1028.876,  J=7001.914, nu=683.4558, nu_th=654.3383,  f=3640.0, L=1690.0),
    4: dict(omega=1.9806, mu=1091.099,  J=7142.493, nu=528.5018, nu_th=1655.909,  f=1760.0, L=2140.0),
}


def damping_force(v, c, alpha=None, c0=None):
    """PTO 阻尼力。alpha=None → 线性阻尼 c*v；否则幂律 c0*|v|^alpha*v。"""
    if alpha is None:
        return c * v
    return c0 * (abs(v) ** alpha) * v


def make_rhs(h, c=10000.0, alpha=None, c0=None):
    """构造仅垂荡的二自由度运动方程右端。"""
    mu, nu, f, omega = h["mu"], h["nu"], h["f"], h["omega"]

    def rhs(t, y):
        x1, v1, x2, v2 = y
        u = x2 - x1
        v = v2 - v1
        Fd = damping_force(v, c, alpha, c0)
        a1 = (f * np.cos(omega * t) - nu * v1 - k_h * x1 + k_spring * u + Fd) / (m1 + mu)
        a2 = (-k_spring * u - Fd) / m2
        return (v1, a1, v2, a2)
    return rhs


def solve_heave(prob, c=10000.0, alpha=None, c0=None, n_periods=40.0, dt=0.2,
                rtol=1e-9, atol=1e-11):
    """积分前 n_periods 个波浪周期，返回固定步长的结果。"""
    h = HYDRO[prob]
    omega = h["omega"]
    T = 2 * np.pi / omega
    t_end = n_periods * T
    t_eval = np.arange(0.0, t_end + 1e-9, dt)
    sol = solve_ivp(make_rhs(h, c, alpha, c0), (0.0, t_end), [0.0, 0.0, 0.0, 0.0],
                    t_eval=t_eval, method="RK45", rtol=rtol, atol=atol)
    assert sol.success, sol.message
    return t_eval, sol.y  # y = [x1, v1, x2, v2]


def avg_power_heave(prob, c=10000.0, alpha=None, c0=None, warmup=20, total=40,
                    rtol=1e-8, atol=1e-10, n_dense=4000):
    """稳态平均输出功率（后 (total-warmup) 个周期）。"""
    h = HYDRO[prob]
    omega = h["omega"]
    T = 2 * np.pi / omega
    t0, t1 = warmup * T, total * T
    t_eval = np.linspace(t0, t1, n_dense)
    sol = solve_ivp(make_rhs(h, c, alpha, c0), (0.0, t1), [0.0, 0.0, 0.0, 0.0],
                    t_eval=t_eval, method="RK45", rtol=rtol, atol=atol)
    assert sol.success, sol.message
    v = sol.y[3] - sol.y[1]          # 相对速度
    P = damping_force(v, c, alpha, c0) * v
    return float(np.trapezoid(P, sol.t) / (t1 - t0))


# ── 线性情形的解析稳态解 ──
def steady_state_linear(prob, c):
    """线性阻尼下的稳态复振幅。

    方程：
      -(m1+mu)w^2 X1 = F - i w nu X1 - k_h X1 + K (X2 - X1)
      -m2 w^2 X2     = -K (X2 - X1),          K = k + i w c
    解得：
      X2 = K X1 / (K - m2 w^2),  u = m2 w^2 X1 / (K - m2 w^2)
      X1 = F / { -(m1+mu)w^2 + i w nu + k_h - K m2 w^2 / (K - m2 w^2) }
    返回 (X1, X2, u)；平均功率 = 0.5 * c * w^2 * |u|^2。
    """
    h = HYDRO[prob]
    w, mu, nu, F = h["omega"], h["mu"], h["nu"], h["f"]
    m_eff = m1 + mu
    K = k_spring + 1j * w * c
    denom_couple = K * m2 * w**2 / (K - m2 * w**2)
    X1 = F / (-m_eff * w**2 + 1j * w * nu + k_h - denom_couple)
    X2 = K * X1 / (K - m2 * w**2)
    u = X2 - X1
    return X1, X2, u


def avg_power_linear_exact(prob, c):
    """线性阻尼下稳态平均输出功率的精确值。"""
    h = HYDRO[prob]
    _, _, u = steady_state_linear(prob, c)
    return 0.5 * c * h["omega"]**2 * abs(u) ** 2


def avg_power_heave_long(prob, c=10000.0, alpha=None, c0=None,
                         n_periods=300, avg_periods=20,
                         n_dense=6000, rtol=1e-9, atol=1e-11):
    """长时间积分到稳态，取最后 avg_periods 个周期求平均（用于非线性情形）。"""
    h = HYDRO[prob]
    w = h["omega"]
    T = 2 * np.pi / w
    t1 = n_periods * T
    t0 = t1 - avg_periods * T
    t_eval = np.linspace(t0, t1, n_dense)
    sol = solve_ivp(make_rhs(h, c, alpha, c0), (0.0, t1), [0.0, 0.0, 0.0, 0.0],
                    t_eval=t_eval, method="RK45", rtol=rtol, atol=atol)
    assert sol.success, sol.message
    v = sol.y[3] - sol.y[1]
    P = damping_force(v, c, alpha, c0) * v
    return float(np.trapezoid(P, sol.t) / (t1 - t0))


# ── 周期打靶法：直接求稳态周期解 ──
def periodic_orbit(prob, c=10000.0, alpha=None, c0=None, x0=None,
                   n_dense=2000, rtol=1e-10, atol=1e-12, x_prev=None):
    """求受迫振动的稳态周期解（Poincaré 映射不动点）。

    相比"积分几百个周期等瞬态衰减"，打靶法直接求解 X(T)=X(0)，
    与该系统的阻尼强弱无关——轻阻尼下（如本问 ν=167.8，τ≈72 s）快一个量级。
    """
    from scipy.optimize import root
    h = HYDRO[prob]
    w = h["omega"]; T = 2 * np.pi / w
    rhs = make_rhs(h, c, alpha, c0)

    def one_period(X):
        s = solve_ivp(rhs, (0.0, T), X, method="RK45", rtol=rtol, atol=atol)
        return s.y[:, -1]

    guess = x0 if x0 is not None else (
        np.asarray(x_prev, dtype=float) if x_prev is not None else np.zeros(4))
    sol_root = root(lambda X: one_period(X) - X, guess, method="hybr", tol=1e-11)
    Xs = sol_root.x
    t_eval = np.linspace(0.0, T, n_dense)
    s = solve_ivp(rhs, (0.0, T), Xs, t_eval=t_eval, method="RK45", rtol=rtol, atol=atol)
    return s.t, s.y, Xs, sol_root.success


def avg_power_periodic(prob, c=10000.0, alpha=None, c0=None, **kw):
    """稳态周期解上的平均输出功率（精确到周期解精度）。"""
    t, y, Xs, ok = periodic_orbit(prob, c, alpha, c0, **kw)
    v = y[3] - y[1]
    P = damping_force(v, c, alpha, c0) * v
    return float(np.trapezoid(P, t) / (t[-1] - t[0])), Xs, ok
