# -*- coding: utf-8 -*-
"""本地模拟器：按附件 1、附件 2 的说明复现无线电干扰源环境。

**这不是官方模拟器。** 官方模拟器需用参赛队号在线登录，且其下载链接已失效、
启动测试的截止时间（2026-09-13 17:30）已过，无法运行。本文件按附件 2 的
协议与物理规则逐条实现，用于验证策略与算法；其统计结果不能冒充正式测试成绩。

覆盖的规则（附件 2 章节号）：
  1.1 目标区域半径 1800、坐标与角度约定
  2.1 有效接收半径 ρ ∈ [1000,1500]
  2.2 全向 / 定向（定向方向两侧各 90°）
  2.3 示向度误差 ∈ [−1°,1°]，**同一位置固定**
  2.4 近距离阈值 5 m；清除半径 20 m
  4.x 虚拟时间：移动 |Δp|/5；换频道 1 s；检测 5 s；清除 成功 5 s / 未发现 3 s
  —   /clear 不换频道、不产生切换耗时
"""
from __future__ import annotations

import math

import numpy as np

ARENA_R = 1800.0
SPEED = 5.0
T_MEASURE = 5.0
T_SWITCH = 1.0
T_CLEAR_OK = 5.0
T_CLEAR_MISS = 3.0
NEAR_R = 5.0
CLEAR_R = 20.0
N_CHANNEL = 20
V_MAX = 360000.0


def wrap360(a):
    return float(a % 360.0)


class Source:
    __slots__ = ("ch", "pos", "rho", "kind", "dir_deg", "cleared")

    def __init__(self, ch, pos, rho, kind="omni", dir_deg=None):
        self.ch = ch
        self.pos = np.asarray(pos, float)
        self.rho = float(rho)
        self.kind = kind                 # "omni" | "dir"
        self.dir_deg = None if dir_deg is None else float(dir_deg)
        self.cleared = False

    def in_coverage(self, p):
        """检测点 p 是否落在该源的有效覆盖角度范围内。"""
        if self.kind == "omni":
            return True
        d = np.asarray(p, float) - self.pos
        if np.linalg.norm(d) < 1e-12:
            return True
        # 源指向 dir_deg 方向；覆盖范围是该方向两侧各 90°
        ang = math.degrees(math.atan2(d[1], d[0])) % 360.0
        diff = abs((ang - self.dir_deg + 180.0) % 360.0 - 180.0)
        return diff <= 90.0 + 1e-9


class Simulator:
    """一局测试。所有时间单位秒，长度单位米。"""

    def __init__(self, sources, seed=0, err_scale=1.0):
        self.sources = list(sources)
        self.by_ch = {s.ch: s for s in sources}
        self.rng = np.random.default_rng(seed)
        self.err_scale = err_scale
        # 示向度误差：位置的固定函数（平滑伪随机场），全局落在 [-1,1]
        k = self.rng.uniform(1e-4, 1.2e-3, size=(6, 2))
        ph = self.rng.uniform(0, 2 * np.pi, size=6)
        self._ek = k
        self._ep = ph

        self.virtual_time = 0.0
        self.pos = np.array([0.0, 0.0])
        self.channel = 1
        self.entered = False
        self.n_requests = 0
        self.log = []
        # 时间分解（用于分析策略的瓶颈）
        self.t_move = 0.0
        self.t_switch = 0.0
        self.t_measure = 0.0
        self.t_clear = 0.0

    # ── 示向度误差：同一位置固定 ──────────────────────────
    def bearing_error(self, p):
        v = sum(math.sin(float(self._ek[i] @ p) + self._ep[i]) for i in range(6))
        return self.err_scale * math.tanh(v / 2.0)      # ∈ (−1,1)

    # ── 内部：真正推进时间的动作 ──────────────────────────
    def _move_time(self, p):
        return float(np.linalg.norm(np.asarray(p, float) - self.pos)) / SPEED

    def measure(self, p, channel):
        """返回 (result, svd_deg or None, dt_dict)。"""
        p = np.asarray(p, float)
        if not (np.all(np.isfinite(p)) and np.max(np.abs(p)) <= 2e6):
            raise ValueError("bad position")
        if not (isinstance(channel, (int, np.integer)) and 1 <= channel <= 20):
            raise ValueError("bad channel")
        dt_move = self._move_time(p)
        dt_switch = T_SWITCH if channel != self.channel else 0.0
        dt = dt_move + dt_switch + T_MEASURE
        self.pos = p.copy()
        self.channel = int(channel)
        self.virtual_time += dt
        self.t_move += dt_move
        self.t_switch += dt_switch
        self.t_measure += T_MEASURE
        self.n_requests += 1
        self.virtual_time = min(self.virtual_time, V_MAX)

        s = self.by_ch.get(int(channel))
        if s is None or s.cleared:
            res, svd = "no_signal", None
        else:
            d = float(np.linalg.norm(p - s.pos))
            if d > s.rho or not s.in_coverage(p):
                res, svd = "no_signal", None
            elif d <= NEAR_R:
                res, svd = "near", None
            else:
                svd = wrap360(math.degrees(math.atan2(s.pos[1] - p[1],
                                                       s.pos[0] - p[0]))
                              + self.bearing_error(p))
                res = "direction"
        self.log.append(("measure", p.tolist(), int(channel), res, svd, dt))
        return res, svd, dict(move=dt_move, switch=dt_switch, act=T_MEASURE, dt=dt)

    def clear(self, p, channel):
        p = np.asarray(p, float)
        if not (np.all(np.isfinite(p)) and np.max(np.abs(p)) <= 2e6):
            raise ValueError("bad position")
        if not (isinstance(channel, (int, np.integer)) and 1 <= channel <= 20):
            raise ValueError("bad channel")
        dt_move = self._move_time(p)
        self.pos = p.copy()
        self.n_requests += 1

        s = self.by_ch.get(int(channel))
        if s is not None and not s.cleared and \
                float(np.linalg.norm(p - s.pos)) <= CLEAR_R:
            s.cleared = True
            dt = dt_move + T_CLEAR_OK
            res = "success"
        else:
            dt = dt_move + T_CLEAR_MISS
            res = "no_target_in_range"
        self.virtual_time += dt
        self.t_move += dt_move
        self.t_clear += dt - dt_move
        self.log.append(("clear", p.tolist(), int(channel), res, None, dt))
        return res, dict(move=dt_move, act=dt - dt_move, dt=dt)

    # ── 统计 ─────────────────────────────────────────────
    def stats(self, real_time_s):
        n_all = len(self.sources)
        n_clr = sum(1 for s in self.sources if s.cleared)
        return dict(total=n_all, cleared=n_clr,
                    ratio=(n_clr / n_all if n_all else 0.0),
                    virtual_time_s=self.virtual_time,
                    avg_time_s=(self.virtual_time / n_clr if n_clr else float("nan")),
                    real_time_s=real_time_s, n_requests=self.n_requests,
                    t_move=self.t_move, t_switch=self.t_switch,
                    t_measure=self.t_measure, t_clear=self.t_clear)


# ══════════════════════════════════════════════════════════
def sample_case(rng, n_min=10, n_max=16, dir_frac=0.0, rho_min=1000.0,
                rho_max=1500.0, r_min=0.0):
    """随机生成一局案例。dir_frac>0 时按比例生成定向干扰源。"""
    n = int(rng.integers(n_min, n_max + 1))
    chs = rng.choice(np.arange(1, N_CHANNEL + 1), size=n, replace=False)
    srcs = []
    for c in chs:
        # 区域内均匀（按面积）
        r = ARENA_R * math.sqrt(rng.uniform(0, 1))
        a = rng.uniform(0, 2 * math.pi)
        pos = np.array([r * math.cos(a), r * math.sin(a)])
        rho = rng.uniform(rho_min, rho_max)
        if rng.uniform() < dir_frac:
            srcs.append(Source(int(c), pos, rho, "dir", rng.uniform(0, 360)))
        else:
            srcs.append(Source(int(c), pos, rho, "omni"))
    return srcs
