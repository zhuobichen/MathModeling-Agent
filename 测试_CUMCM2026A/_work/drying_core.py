# -*- coding: utf-8 -*-
"""2026 高教社杯 A 题「药材的烘干问题」——一维轴对称耦合传热传质求解器。

模型（圆柱，轴向中点处取一维径向）：

    热量：  rho*cp * dT/dt = (1/r) * d/dr ( k*r*dT/dr )
    水分：  dC/dt         = (1/r) * d/dr ( D*r*dC/dr )

边界：  r=0   对称      dT/dr = 0,  dC/dr = 0
        r=R   对流换热  -k*dT/dr = h*(T_inf - T_s)
              对流传质  -D*dC/dr = hm*(C_s - C_inf)

数值：有限体积 + Crank–Nicolson（θ=0.5，二阶且 L-稳定）+ Picard 迭代
      处理物性对 C、T 的依赖。外边界用「壁面扩散 + 对流」串联阻力。

坐标：问题 4 的药材会收缩，半径 R(t) 由附件 2 给出。改用**物质坐标**
      ξ = r/R_met（ξ∈[0,1] 固定），内部输运的度规锚在初始半径 R0 上，
      当前半径 R(t) 只进入边界条件：
          dC/dt = (1/(ξ R0²)) * d/dξ ( D ξ dC/dξ )
          -(D/R(t)) dC/dξ|_1 = hm*(C_s - C_inf)
      理由：干基含水率是「单位干物质的水量」，干物质守恒，水只能经表面
      离开；R 缩小只减小蒸发面积、不缩短物质路径。若内部也改用 R(t)，
      ∫C dV 会随收缩凭空减少（那部分水没走表面），质量不守恒。
      对 R≡const（问题 1~3）两式相同。
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded

# ── 物理常数 ──────────────────────────────────────────────
T0_K = 273.15


def _safe(C, lo=1e-6):
    return np.maximum(C, lo)


# ── 物性：问题 1（附录 2，常数）────────────────────────────
def props_q1(C, T_C):
    C = _safe(C)
    n = np.shape(C)
    rho = np.full(n, 820.0)
    cp = np.full(n, 2600.0)
    k = np.full(n, 0.36)
    D = 7e-9 * np.exp(-0.89 / C)
    return rho, cp, k, D


def consts_q1():
    return dict(h=25.0, hm=8e-7)


# ── 物性：问题 2、3（附录 3）──────────────────────────────
def props_q3(C, T_C):
    C = _safe(C)
    Tk = T_C + T0_K
    rho = 650.0 + 128.0 * C
    cp = 1450.0 + 2736.0 * C / (C + 1.0)
    k = 0.21 + 0.38 * C / (C + 1.0)
    D = 2.4e-3 * np.exp(-0.45 / C) * np.exp(-3850.0 / Tk)
    return rho, cp, k, D


def consts_q3():
    return dict(h=25.0, hm=8e-7)


# ── 物性：问题 4（附录 4）─────────────────────────────────
def props_q4(C, T_C):
    C = _safe(C)
    Tk = T_C + T0_K
    rho = 760.0 + 90.0 * C
    cp = 1850.0 + 2150.0 * C / (C + 1.0)
    k = 0.12 + 0.20 * C / (C + 1.0)
    D = 4.2e-4 * np.exp(-0.30 / C) * np.exp(-3850.0 / Tk)
    return rho, cp, k, D


def consts_q4():
    return dict(h=25.0, hm=8e-7)


# ══════════════════════════════════════════════════════════
class CylinderSolver:
    """一维轴对称有限体积求解器（变半径物质坐标）。

    Parameters
    ----------
    N    : 网格单元数（径向均匀）
    props: f(C, T_C) -> (rho, cp, k, D)，逐点数组
    R0   : 初始半径 [m]
    """

    def __init__(self, N, props, consts, R0=0.02, material_frame=False):
        self.N = N
        self.props = props
        self.R0 = R0
        # material_frame=True：药材收缩时用**物质坐标**度规（锚定初始半径
        # R0），当前半径 R(t) 只出现在边界条件里。见 _conductances 的说明。
        self.material_frame = material_frame
        self.R_met = R0
        self.h = consts["h"]
        self.hm = consts["hm"]
        self._gamma = self.h   # 当前求解场的对流系数（热 h / 湿 hm）
        # 物质坐标 ξ 上的固定网格（单元中心）
        d = 1.0 / N
        self.dxi = d
        self.xi_f = np.linspace(0.0, 1.0, N + 1)          # 面
        self.xi_c = 0.5 * (self.xi_f[:-1] + self.xi_f[1:])  # 心
        # 度规：V_i = ∫ ξ dξ（每 2π R² 因子），A_f = ξ_f
        self.V = 0.5 * (self.xi_f[1:] ** 2 - self.xi_f[:-1] ** 2)
        self.Af = self.xi_f

    # ---------- 组装并求解一个时间步 ----------
    def _conductances(self, coeff_k, R):
        """面导 g_in / g_out 与外边界面导 G。R 为该时刻的实际半径。

        度规半径 R_met 与边界半径 R 的区分（问题 4 收缩时二者不同）：

        · 内部扩散用 R_met。干基含水率 C 是「单位干物质的水量」，干物质
          守恒，水只能从表面离开。沿物质坐标 ξ 做水量衡算：
              rho_d0 R0^2 * d(C)/dt = (1/ξ) d/dξ ( D ξ dC/dξ )
          度规锚在**初始**半径 R0 上 —— 收缩不会改变内部的物质路径长度。
        · 表面通量用当前半径：总面积 = 2πR(t)·hm·(C_s−C_inf)，R 缩小则
          蒸发面积变小。

        若内部也改用当前半径（常见的"网格随半径缩放"写法），收缩会让
        ∫C dV 凭空减少 —— 那部分水没有从表面流走，质量不守恒。实测按
        那种写法 Q4 的质量闭合差随 R 缩小而单调增大。故收缩问题必须
        用物质坐标度规。
        """
        N, dxi = self.N, self.dxi
        Af = self.Af
        R_met = self.R0 if self.material_frame else R
        Rm2 = R_met * R_met
        kf_in = np.empty(N)
        kf_out = np.empty(N)
        kf_in[0] = 0.0
        kf_in[1:] = _harm(coeff_k[:-1], coeff_k[1:])
        kf_out[:-1] = kf_in[1:]
        kf_out[-1] = 0.0
        g_in = kf_in * Af[:-1] / dxi / Rm2
        g_out = kf_out * Af[1:] / dxi / Rm2
        R_tot = dxi * R / (2.0 * coeff_k[-1]) + 1.0 / self._gamma
        G = Af[-1] * R / (R_tot * Rm2)
        return g_in, g_out, G

    def _solve_field(self, phi_old, phi_it, R, dt, coeff_a, coeff_k,
                     phi_inf, theta):
        """一个时间步：θ 格式（θ=1 后向欧拉，θ=0.5 Crank–Nicolson）。

        在物质坐标 ξ 上，方程除以 2πR² 后为

            a·V_i·dφ_i/dt = (1/R²)·[ G_{i+1/2}(φ_{i+1}-φ_i)
                                   - G_{i-1/2}(φ_i-φ_{i-1}) ]

        其中 G_f = κ_f·ξ_f/Δξ。**1/R² 这个度规因子是必须的**：
        物理距离 = R·Δξ，ξ 坐标下的等效扩散系数要除以 R²。漏掉它会让
        扩散项比真实值小 R² 倍（R=0.02 时小 2500 倍），表现为内部几乎
        不响应、能量守恒差若干量级。

        theta 格式：a·V/dt·(φ^{n+1}-φ^n) = θ·L(φ^{n+1}) + (1-θ)·L(φ^n)。

        取 θ=0.5 得二阶精度。一阶后向欧拉在 dt=1 s 时残差约 2e-4（相对），
        落在四位小数的第 4 位上，故不能用。系数按当前迭代值冻结，
        非线性只通过 Picard 迭代进入（φ_it 仅用于物性求值）。

        矩阵 = a·V/dt·I − θ·L（L 在系数冻结后是线性的），
        RHS  = a·V/dt·φ^n + (1−θ)·L(φ^n)。
        """
        N, dxi = self.N, self.dxi
        V = self.V
        g_in, g_out, G = self._conductances(coeff_k, R)
        aV = coeff_a * V

        # 显式部分 L(φ^n)，用旧场（系数按当前迭代值冻结，标准半隐式处理）
        Lold = np.zeros(N)
        Lold[:-1] += g_out[:-1] * (phi_old[1:] - phi_old[:-1])
        Lold[1:] += g_in[1:] * (phi_old[:-1] - phi_old[1:])
        Lold[-1] += G * (phi_inf - phi_old[-1])

        lower = -theta * g_in[1:]
        upper = -theta * g_out[:-1]
        diag = aV / dt + theta * (g_in + g_out)
        diag[-1] += theta * G
        ab = np.zeros((3, N))
        ab[0, 1:] = upper
        ab[1, :] = diag
        ab[2, :-1] = lower

        # 隐式部分已进入矩阵（θ·L 对冻结系数是线性的），RHS 只保留显式部分，
        # 外加从 LHS 移过来的外边界驱动项 θ·G·φ_inf。
        #   漏掉 θ·G·φ_inf：外边界只剩 (1-θ)·G·(φ_inf-φ^n)，热流减半甚至反向，
        #   实测温度从 28 降到 27.3（冷却），与"烘房 50°C"完全相反。
        #   而把 θ·L(φ_it) 整个加进来则是重复计入，外边界变成 φ_new≈(φ_inf-φ_it)，
        #   迭代在两值间振荡不收敛。
        #   **只加到最后一个单元**：G 与 phi_inf 都是标量，写成
        #   `rhs = … + theta*G*phi_inf` 会广播到全部 N 个单元，而 diag 只在
        #   末单元加了 θG，于是内部每个单元凭空多出 θG·phi_inf 的驱动力
        #   （实测一步就把中心从 28 冲到 626°C）。
        rhs = aV / dt * phi_old + (1.0 - theta) * Lold
        rhs[-1] += theta * G * phi_inf
        return solve_banded((1, 1), ab, rhs)

    # ---------- 一个时间步 ----------
    def step(self, T, C, dt, R, T_inf, C_inf, n_picard=4, theta=0.5):
        """推进一步，返回 (T_new, C_new) [°C, kg/kg]。

        每次 Picard 迭代都要**重新**设定 self._gamma：热量用 h、水分用 hm。
        漏掉这一步（或只在循环外设一次）会让第 2~n 次迭代的热求解用上 hm，
        此时外边界热阻 1/hm = 1.25e6，热量几乎进不去 —— 表现为温度场
        完全不动而水分场正常演化，很容易被误当成"温度的时间尺度慢"。
        """
        Tn, Cn = T.copy(), C.copy()
        for _ in range(n_picard):
            rho, cp, k, D = self.props(Cn, Tn)
            self._gamma = self.h
            Tn = self._solve_field(T, Tn, R, dt, rho * cp, k, T_inf, theta)
            self._gamma = self.hm
            Cn = self._solve_field(C, Cn, R, dt, np.ones_like(C), D,
                                   C_inf, theta)
            np.maximum(Cn, 0.0, out=Cn)
        return Tn, Cn

    # ---------- 积分 ----------
    def run(self, t_end, dt_fn, T_init, C_init, bc, R_of_t=None,
            out_times=None, record_every=None, theta=0.5, n_picard=4):
        """积分到 t_end。

        dt_fn(t)      : 当前允许的最大步长
        bc(t)         : (T_inf, C_inf)
        R_of_t(t)     : 半径 [m]，None 表示不收缩
        out_times     : 需要精确记录的采样时刻（升序）
        record_every  : 或按固定间隔记录
        返回 dict(t=..., T=..., C=..., R=..., flux_h=..., flux_m=...)
        """
        R0 = self.R0
        t = 0.0
        T = np.full(self.N, float(T_init))
        C = np.full(self.N, float(C_init))
        R = R_of_t(0.0) if R_of_t else R0

        rec_t, rec_T, rec_C, rec_R = [0.0], [T.copy()], [C.copy()], [R]
        accum = dict(qh=0.0, qm=0.0, t_prev=0.0)

        if out_times is None:
            out_times = []
            if record_every:
                out_times = list(np.arange(record_every, t_end + 1e-9, record_every))
        out_times = list(out_times)
        oi = 0

        while t < t_end - 1e-12:
            t_next_out = out_times[oi] if oi < len(out_times) else t_end
            dt = min(dt_fn(t), t_next_out - t, t_end - t)
            if dt <= 1e-12:
                if oi < len(out_times):
                    oi += 1
                    continue
                break
            T_inf, C_inf = bc(t + 0.5 * dt)   # 半步取值，二阶更好
            # 半径取半步值：度规因子 1/R² 在时间上以中点值参与，保持二阶
            R_mid = R_of_t(t + 0.5 * dt) if R_of_t else R0
            T, C = self.step(T, C, dt, R_mid, T_inf, C_inf,
                            n_picard=n_picard, theta=theta)
            t += dt
            R = R_of_t(t) if R_of_t else R0
            self._accumulate(accum, T, C, dt, R, T_inf, C_inf)

            while oi < len(out_times) and t >= out_times[oi] - 1e-9:
                rec_t.append(out_times[oi])
                rec_T.append(T.copy())
                rec_C.append(C.copy())
                rec_R.append(R)
                oi += 1

        return dict(t=np.array(rec_t), T=np.array(rec_T), C=np.array(rec_C),
                    R=np.array(rec_R), bal=accum)

    def _accumulate(self, accum, T, C, dt, R, T_inf, C_inf):
        """积分边界累计热流/湿流，用于能量与质量守恒校验。

        单位长度上的总量 = 2πR × 单位面积通量，单位面积串联阻力为
        Δξ·R/(2κ) + 1/Γ。
        """
        dxi = self.dxi
        rho, cp, k, D = self.props(C, T)
        # 热量：由烘房进入药材（J/m）
        Rh = dxi * R / (2 * k[-1]) + 1.0 / self.h
        accum["qh"] += (T_inf - T[-1]) / Rh * 2 * np.pi * R * dt
        # 水分：由药材表面进入烘房。注意 C 是干基含水率，衡算以**干物质**
        # 为基准，故累计量记在 2πR0² 的度规下（对应"每单位长度药材的
        # C·干物质量"），而不是当前体积 2πR²。
        R_met = self.R0 if self.material_frame else R
        Rm = dxi * R / (2 * D[-1]) + 1.0 / self.hm
        accum["qm"] += (C[-1] - C_inf) / Rm * 2 * np.pi * R * dt

    # ---------- 体积平均（每单位长度，含 2π）----------
    def total_moisture(self, C, R):
        rho_ = None
        return None


def _harm(a, b):
    """调和平均；任一侧为 0 时返回 0。"""
    out = np.zeros_like(a)
    m = (a > 0) & (b > 0)
    out[m] = 2.0 * a[m] * b[m] / (a[m] + b[m])
    return out


# ── 输出采样：把单元中心值插到指定半径 ─────────────────────
def sample_at(sol, radii_cm):
    """在给定时刻序列上取指定半径处的场值。

    sol : run() 的返回；radii_cm 为「到药材中心距离」[cm]
    返回 (T_out, C_out)，形状 (n_t, n_r)；超出该时刻药材表面的位置填 nan。
    """
    N = sol["T"].shape[1]
    xi_c = (np.arange(N) + 0.5) / N
    out_T, out_C = [], []
    for k in range(len(sol["t"])):
        R = sol["R"][k]                       # [m]
        R_cm = R * 100.0
        xq = np.asarray(radii_cm, float) / R_cm   # 物理半径 → ξ
        inside = xq <= 1.0 + 1e-12
        xq_c = np.clip(xq, xi_c[0], xi_c[-1])
        out_T.append(np.where(inside, np.interp(xq_c, xi_c, sol["T"][k]), np.nan))
        out_C.append(np.where(inside, np.interp(xq_c, xi_c, sol["C"][k]), np.nan))
    return np.array(out_T), np.array(out_C)
