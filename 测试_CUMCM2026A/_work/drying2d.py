# -*- coding: utf-8 -*-
"""二维轴对称（有限长圆柱）求解器——用于验证一维径向假设。

几何：半径 R=2 cm、长 L=25 cm，两端面也对流换热传质。
控制方程（柱坐标，轴对称）：
    a·∂φ/∂t = (1/r)∂_r(κ r ∂_rφ) + ∂_z(κ ∂_zφ)
边界：r=0 与 z=L/2 对称；r=R 与 z=0,L 为对流。

离散：与一维版同一套有限体积 + Crank–Nicolson + Picard，
      二维网格用五点格式，隐式步解稀疏线性方程组。
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import splu

T0_K = 273.15


def _harm(a, b):
    out = np.zeros_like(a)
    m = (a > 0) & (b > 0)
    out[m] = 2.0 * a[m] * b[m] / (a[m] + b[m])
    return out


class Cylinder2D:
    """有限长圆柱（半径 R，半长 H=L/2，取 z∈[0,H] 的对称半域）。"""

    def __init__(self, Nr, Nz, props, consts, R=0.02, H=0.125):
        self.Nr, self.Nz = Nr, Nz
        self.props, self.R, self.H = props, R, H
        consts = consts() if callable(consts) else consts
        self.h, self.hm = consts["h"], consts["hm"]
        self._gamma = self.h

        # r 方向：单元中心，含 r=0 的对称面
        rf = np.linspace(0.0, R, Nr + 1)
        self.rc = 0.5 * (rf[:-1] + rf[1:])
        self.dr = R / Nr
        # z 方向：z=0 为对称面（圆柱中点），z=H 为端面
        zf = np.linspace(0.0, H, Nz + 1)
        self.zc = 0.5 * (zf[:-1] + zf[1:])
        self.dz = H / Nz

        # 单元体积（含 2π 因子：V = 2π r dr dz → 取 2π=1）
        self.Vr = 0.5 * (rf[1:] ** 2 - rf[:-1] ** 2)   # ∫r dr
        self.Ar = rf[1:]                                # r 面面积因子
        self.Vz = np.full(Nz, self.dz)
        # 二维单元体积 V_ij = Vr_i * dz
        self.V = np.outer(self.Vr, self.Vz)

        # 索引
        self.idx = np.arange(Nr * Nz).reshape(Nr, Nz)

    # ---------- 组装 ----------
    def _assemble(self, phi_old, R, dt, coeff_a, coeff_k, phi_inf, theta):
        Nr, Nz = self.Nr, self.Nz
        dr, dz = self.dr, self.dz
        V = self.V
        aV = (coeff_a * V).ravel()
        idg = self.idx

        # 面上扩散系数
        kr = np.empty((Nr, Nz)); kr[0] = 0.0
        kr[1:] = _harm(coeff_k[:-1], coeff_k[1:])
        kz = np.empty((Nr, Nz)); kz[:, 0] = 0.0
        kz[:, 1:] = _harm(coeff_k[:, :-1], coeff_k[:, 1:])

        # r 面导与 z 面导都**必须带各自的面积因子**（已把 2π 约掉）：
        #   r 面 (i+1/2)：面积 = r_face·dz  → g = κ·r_face·dz/Δr
        #   z 面 (j+1/2)：面积 = ∫r dr = Vr → g = κ·Vr/Δz
        # 漏掉它们只会让两个方向的相对权重错掉（r 向少 dz、z 向少 Vr），
        # 而各向同性的算例仍能"看起来收敛"，是很容易混过去的错。
        rface = self.Ar                     # r_{i+1/2}，长度 Nr
        gr = kr * rface[:, None] * dz / dr
        gz = kz * self.Vr[:, None] / dz
        # 内部面导（用于矩阵）
        g_in_r = np.zeros((Nr, Nz)); g_in_r[1:] = gr[1:]
        g_out_r = np.zeros((Nr, Nz)); g_out_r[:-1] = gr[1:]
        g_in_z = np.zeros((Nr, Nz)); g_in_z[:, 1:] = gz[:, 1:]
        g_out_z = np.zeros((Nr, Nz)); g_out_z[:, :-1] = gz[:, 1:]

        # 边界：r=R（i=Nr-1）与 z=H（j=Nz-1）为对流，串联阻力
        Gr = np.zeros((Nr, Nz)); Gz = np.zeros((Nr, Nz))
        Rr = dr / (2.0 * coeff_k[-1, :]) + 1.0 / self._gamma
        Gr[-1, :] = rface[-1] * dz / Rr            # 面积 R·dz
        Rz = dz / (2.0 * coeff_k[:, -1]) + 1.0 / self._gamma
        Gz[:, -1] = self.Vr / Rz                   # 面积 Vr

        # 显式部分 L(φ^n)
        def L(phi):
            out = np.zeros((Nr, Nz))
            out[:-1, :] += g_out_r[:-1, :] * (phi[1:, :] - phi[:-1, :])
            out[1:, :] += g_in_r[1:, :] * (phi[:-1, :] - phi[1:, :])
            out[:, :-1] += g_out_z[:, :-1] * (phi[:, 1:] - phi[:, :-1])
            out[:, 1:] += g_in_z[:, 1:] * (phi[:, :-1] - phi[:, 1:])
            out[-1, :] += Gr[-1, :] * (phi_inf - phi[-1, :])
            out[:, -1] += Gz[:, -1] * (phi_inf - phi[:, -1])
            return out

        Lold = L(phi_old)

        # 隐式矩阵
        diag = aV / dt + theta * (g_in_r + g_out_r + g_in_z + g_out_z).ravel()
        diag[idg[-1, :]] += theta * Gr[-1, :]
        diag[idg[:, -1]] += theta * Gz[:, -1]

        rows, cols, vals = [], [], []
        n = Nr * Nz
        rows.append(np.arange(n)); cols.append(np.arange(n)); vals.append(diag)
        # r 方向耦合
        m = (slice(1, Nr), slice(None))
        rows.append(idg[m].ravel()); cols.append(idg[0:Nr - 1, :].ravel())
        vals.append((-theta * g_in_r[1:, :]).ravel())
        rows.append(idg[0:Nr - 1, :].ravel()); cols.append(idg[1:Nr, :].ravel())
        vals.append((-theta * g_out_r[:-1, :]).ravel())
        # z 方向耦合
        rows.append(idg[:, 1:].ravel()); cols.append(idg[:, 0:Nz - 1].ravel())
        vals.append((-theta * g_in_z[:, 1:]).ravel())
        rows.append(idg[:, 0:Nz - 1].ravel()); cols.append(idg[:, 1:].ravel())
        vals.append((-theta * g_out_z[:, :-1]).ravel())

        A = coo_matrix((np.concatenate(vals),
                        (np.concatenate(rows), np.concatenate(cols))),
                       shape=(n, n)).tocsr()

        rhs = (aV / dt * phi_old.ravel() + (1.0 - theta) * Lold.ravel())
        rhs[idg[-1, :]] += theta * Gr[-1, :] * phi_inf
        rhs[idg[:, -1]] += theta * Gz[:, -1] * phi_inf
        return A, rhs

    def step(self, T, C, dt, T_inf, C_inf, n_picard=3, theta=0.5):
        Tn, Cn = T.copy(), C.copy()
        for _ in range(n_picard):
            rho, cp, k, D = self.props(Cn, Tn)
            self._gamma = self.h
            A, rhs = self._assemble(T, self.R, dt, rho * cp, k, T_inf, theta)
            Tn = splu(A.tocsc()).solve(rhs).reshape(self.Nr, self.Nz)
            self._gamma = self.hm
            A, rhs = self._assemble(C, self.R, dt, np.ones_like(C), D,
                                    C_inf, theta)
            Cn = splu(A.tocsc()).solve(rhs).reshape(self.Nr, self.Nz)
            np.maximum(Cn, 0.0, out=Cn)
        return Tn, Cn

    def run(self, t_end, dt_fn, T_init, C_init, bc, record_every=None,
            theta=0.5, n_picard=3):
        T = np.full((self.Nr, self.Nz), float(T_init))
        C = np.full((self.Nr, self.Nz), float(C_init))
        rec = {"t": [0.0], "T": [T.copy()], "C": [C.copy()]}
        t = 0.0
        nxt = record_every if record_every else t_end
        while t < t_end - 1e-9:
            dt = min(dt_fn(t), nxt - t, t_end - t)
            if dt <= 1e-12:
                break
            T_inf, C_inf = bc(t + 0.5 * dt)
            T, C = self.step(T, C, dt, T_inf, C_inf,
                             n_picard=n_picard, theta=theta)
            t += dt
            if record_every and t >= nxt - 1e-9:
                rec["t"].append(t); rec["T"].append(T.copy())
                rec["C"].append(C.copy())
                nxt += record_every
        return {k: (np.array(v) if k == "t" else np.array(v))
                for k, v in rec.items()}

    def drying_time(self, sol, target=0.15):
        mx = sol["C"].reshape(len(sol["t"]), -1).max(axis=1)
        idx = np.where(mx < target)[0]
        if len(idx) == 0:
            return None, mx
        i = idx[0]
        if i == 0:
            return float(sol["t"][0]), mx
        t0, t1 = sol["t"][i - 1], sol["t"][i]
        m0, m1 = mx[i - 1], mx[i]
        return float(t0 + (target - m0) * (t1 - t0) / (m1 - m0)), mx
