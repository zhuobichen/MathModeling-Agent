# -*- coding: utf-8 -*-
"""机器狗策略：覆盖式侦察 → 交会定位 → 逼近清除 → 复查。

设计要点
--------
1) 侦察点必须保证「场地内任一点到最近侦察点的距离 ≤ ρ_min = 1000 m」，
   否则 ρ 取到下限的全向源可能一直探测不到。对半径 1800 的圆域，用
   「中心 + 半径 a 的 k 点环」，环上最坏点距离为 1800·sin(π/k)，
   取 k=6 时为 900 m，留有余量。
2) 定向源只在小半个平面辐射，故同一区域需从**不同方位**的侦察点看两次，
   环状布置天然满足。
3) 交会定位用问题 1 的算法：把每个「(检测点, 示向度)」变成张角 2° 的角楔，
   取交即为定位区域；其直径为不确定度。
4) 逼近清除：在估计位置附近反复测量，把新方位线并入交会，逐步收紧；
   出现 near 时直接清除。清除半径 20 m，故不确定度降到约 20 m 即可动手。
5) 频道切换：同一侦察点上按当前频道起沿最短路径扫完 20 个频道；
   已清除的频道记为空，后续不再扫。
"""
from __future__ import annotations

import math

import numpy as np

import geo_core as G
from simulator import (ARENA_R, CLEAR_R, N_CHANNEL, NEAR_R, SPEED, T_MEASURE,
                       T_SWITCH, V_MAX)

ALL_CH = list(range(1, N_CHANNEL + 1))
MAX_MEASURE = 20000          # 保险丝，防止死循环


# ══════════════════════════════════════════════════════════
def survey_ring(k=6, margin=1.0):
    """给出覆盖用的侦察点集：中心 + 半径 a 的 k 点环。

    a 取最小化最坏距离的 1800·cos(π/k)。
    """
    a = ARENA_R * math.cos(math.pi / k) * margin
    pts = [np.array([0.0, 0.0])]
    for i in range(k):
        th = 2 * math.pi * i / k
        pts.append(np.array([a * math.cos(th), a * math.sin(th)]))
    return pts


def worst_cover_radius(pts, n=4000):
    """数值估计：场地内任一点到最近侦察点的最大距离。"""
    rng = np.random.default_rng(0)
    r = ARENA_R * np.sqrt(rng.uniform(0, 1, n))
    a = rng.uniform(0, 2 * math.pi, n)
    P = np.stack([r * np.cos(a), r * np.sin(a)], axis=1)
    P = np.vstack([P, [[0, 0]]])
    d = np.min(np.linalg.norm(P[:, None, :] - np.array(pts)[None, :, :], axis=2),
               axis=1)
    return float(d.max())


def route_nn(start, targets):
    """最近邻 + 2-opt 的巡回顺序（不含回到起点）。"""
    pts = [np.asarray(t, float) for t in targets]
    if not pts:
        return []
    order = []
    cur = np.asarray(start, float)
    rem = list(range(len(pts)))
    while rem:
        j = min(rem, key=lambda i: np.linalg.norm(pts[i] - cur))
        order.append(j)
        cur = pts[j]
        rem.remove(j)

    def path_len(seq, s):
        L, c = 0.0, np.asarray(s, float)
        for i in seq:
            L += float(np.linalg.norm(pts[i] - c))
            c = pts[i]
        return L

    improved = True
    while improved:
        improved = False
        for i in range(len(order) - 1):
            for j in range(i + 1, len(order)):
                cand = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                if path_len(cand, start) < path_len(order, start) - 1e-9:
                    order = cand
                    improved = True
    return order


# ══════════════════════════════════════════════════════════
class Robot:
    def __init__(self, sim, log=None):
        self.sim = sim
        self.lines = {c: [] for c in ALL_CH}      # ch -> [(point, theta)]
        self.est = {}                              # ch -> (pos, unc)
        self.empty = set()                         # 已确认无源的频道
        self.cleared = []
        self.n_measure = 0
        self.tries = {}                            # 频道→逼近失败次数
        self.trace = []
        self.record = log if log is not None else []

    # ---------- 基本动作 ----------
    def do_measure(self, p, ch):
        if self.n_measure > MAX_MEASURE:
            raise RuntimeError("测量次数超限")
        res, svd, _ = self.sim.measure(p, ch)
        self.n_measure += 1
        self.record.append(dict(op="measure", p=np.asarray(p, float).tolist(),
                                ch=int(ch), res=res, svd=svd,
                                t=self.sim.virtual_time))
        if res == "direction":
            self.lines[ch].append((np.asarray(p, float), float(svd)))
            self._reestimate(ch)
        return res, svd

    def do_clear(self, p, ch):
        res, _ = self.sim.clear(p, ch)
        self.record.append(dict(op="clear", p=np.asarray(p, float).tolist(),
                                ch=int(ch), res=res, t=self.sim.virtual_time))
        if res == "success":
            self.cleared.append(ch)
            self.empty.add(ch)
            self.est.pop(ch, None)
        return res

    def _reestimate(self, ch):
        ls = self.lines[ch]
        if len(ls) < 2:
            self.est[ch] = (ls[0][0] + 500 * G.unit(ls[0][1]), 1e9)
            return
        pts = [p for p, _ in ls]
        th = [t for _, t in ls]
        poly = G.localize(pts, th)
        if len(poly) < 2:
            # 交会为空：多因误差把真值排除在外，退回用最近两条线
            pts2, th2 = pts[-2:], th[-2:]
            poly = G.localize(pts2, th2)
        if len(poly) < 2:
            return
        D, _ = G.polygon_diameter(poly)
        self.est[ch] = (poly.mean(axis=0), float(D))

    # ---------- 扫描 ----------
    def sweep(self, p, channels=None):
        """在 p 处依次检测给定频道；按当前频道起的最短扫序。"""
        chs = list(channels if channels is not None else
                   [c for c in ALL_CH if c not in self.empty])
        if not chs:
            return {}
        # 从当前频道出发的最近邻顺序（1 维，等价于按距离排序）
        cur = self.sim.channel
        rem = sorted(chs, key=lambda c: abs(c - cur))
        out = {}
        for c in rem:
            res, svd = self.do_measure(p, c)
            out[c] = (res, svd)
        return out

    # ---------- 逼近清除 ----------
    def approach_and_clear(self, ch, max_iter=10):
        """先直取，失败再走「已知可见点」。全向源直取即可、省时间；
        定向源常需后者（见 _via_seen）。"""
        if self._direct(ch):
            return True
        return self._via_seen(ch, max_iter=max_iter)

    def _direct(self, ch, max_iter=6):
        """从当前位置直奔估计位置：对全向源足够，且行程最短。"""
        if ch not in self.est:
            return False
        pos, unc = self.est[ch]
        cur = pos
        for it in range(max_iter):
            res, svd = self.do_measure(cur, ch)
            if res == "near":
                return self.do_clear(cur, ch) == "success"
            if res == "direction":
                if ch in self.est:
                    cur = 0.5 * (cur + self.est[ch][0])
                else:
                    cur = cur + 25.0 * G.unit(svd)
            else:
                return False
        res, _ = self.do_measure(cur, ch)
        if res == "near":
            return self.do_clear(cur, ch) == "success"
        return self.do_clear(cur, ch) == "success"

    def _via_seen(self, ch, max_iter=10):
        """从「曾看到过该源」的检测点出发沿连线逼近。

        **关键性质**：若检测点 S 落在定向源的覆盖半平面内，则从 S 沿直线
        走向源，全程仍在覆盖范围内（半平面是凸集，且 (p(t)−G)·u 对 t 线性）。
        因此接近必须**从见过它的点出发**，否则可能绕到源的背面而一直
        no_signal —— 实测这正是定向源清除率只有 34% 的原因。
        """
        if ch not in self.est and not self.lines[ch]:
            return False
        seen = [p for p, _ in self.lines[ch]]
        if not seen:
            return False
        sim = self.sim
        # ① 走到最近的「曾看到过」的检测点，重新拿一次方位
        order = sorted(seen, key=lambda p: np.linalg.norm(p - sim.pos))
        base, res, svd = None, None, None
        for p in order[:4]:
            r, s_ = self.do_measure(p, ch)
            if r in ("direction", "near"):
                base, res, svd = p, r, s_
                break
        if base is None:
            return False
        if res == "near":
            return self.do_clear(base, ch) == "success"

        # ② 沿方位推进，步长按剩余距离收缩
        cur = np.asarray(base, float)
        for it in range(max_iter):
            d_est = float(np.linalg.norm(self.est[ch][0] - cur))                 if ch in self.est else 200.0
            step = float(np.clip(d_est * (0.75 ** (it // 2)), 8.0, 600.0))
            nxt = cur + step * G.unit(svd)
            r, s2 = self.do_measure(nxt, ch)
            if r == "near":
                return self.do_clear(nxt, ch) == "success"
            if r == "direction":
                cur, svd = nxt, s2
                if step <= 9.0:
                    return self.do_clear(nxt, ch) == "success"
            else:
                # 步长过大越过了源：折半回退
                cur = cur + (step * 0.5) * G.unit(svd)                     if False else cur
                if step <= 9.0:
                    return self.do_clear(cur, ch) == "success"
        return self.do_clear(cur, ch) == "success"

    # ---------- 主流程 ----------
    def run(self, survey_pts=None, interleave=True, verbose=False):
        sim = self.sim
        sim.entered = True
        if survey_pts is None:
            survey_pts = survey_ring(6)
        pending = list(survey_pts)

        # 先就近扫一圈（含中心）
        cur = np.array([0.0, 0.0])
        while pending:
            j = min(range(len(pending)),
                    key=lambda i: np.linalg.norm(pending[i] - cur))
            p = pending.pop(j)
            self.sweep(p)
            cur = p
            if verbose:
                print(f"   侦察点 ({p[0]:.0f},{p[1]:.0f}) 后 "
                      f"t={sim.virtual_time:.0f}s 已定位 "
                      f"{sum(1 for c in self.est if self.est[c][1] < 1e8)}")
            if interleave:
                # 顺路清除已定位且不确定度已足够小的源
                ready = [c for c in self.est
                         if self.est[c][1] <= 120.0 and c not in self.empty]
                for c in sorted(ready,
                                key=lambda c: np.linalg.norm(self.est[c][0] - cur)):
                    if c in self.empty:
                        continue
                    self.approach_and_clear(c)

        # 收尾：把剩下的源按最近邻路线逐個清除
        while True:
            todo = [c for c in self.est if c not in self.empty]
            if not todo:
                break
            order = route_nn(sim.pos, [self.est[c][0] for c in todo])
            progressed = False
            for i in order:
                c = todo[i]
                if c in self.empty:
                    continue
                if self.approach_and_clear(c):
                    progressed = True
            if not progressed:
                break
        return sim.stats(0.0)


# ══════════════════════════════════════════════════════════
def run_sector(self, unc_thr=250.0, verbose=False):
    """分区推进：沿环按角度顺序扫描，每个环点后清除本扇区已定位的源。

    动机：移动占总耗时 80% 以上。把「侦察」与「清除」放在同一条环状
    路线上，避免先绕环一周再折返清源造成的重复行程。
    """
    ring = survey_ring(6)[1:]
    ring = sorted(ring, key=lambda p: math.atan2(p[1], p[0]))
    start = min(range(6), key=lambda i: np.linalg.norm(ring[i]))
    order = [ring[(start + i) % 6] for i in range(6)]
    self.sweep(np.array([0.0, 0.0]))

    def flush(cur):
        """清除已定位且精度足够的源（最近邻顺序），返回清除数。"""
        got = 0
        while True:
            cand = [c for c in self.est
                    if c not in self.empty and self.est[c][1] <= unc_thr]
            if not cand:
                break
            cand.sort(key=lambda c: np.linalg.norm(self.est[c][0] - cur))
            c = cand[0]
            if self.approach_and_clear(c):
                got += 1
            else:
                self.empty.add(c)          # 放弃，避免死循环
            cur = self.sim.pos
        return got

    for p in order:
        self.sweep(p)
        flush(p)
        if verbose:
            print(f"   环点 ({p[0]:.0f},{p[1]:.0f}) t={self.sim.virtual_time:.0f}s "
                  f"已清 {len(self.cleared)}")
    # 收尾：放宽阈值再试一轮
    while True:
        todo = [c for c in self.est if c not in self.empty]
        if not todo:
            break
        order2 = route_nn(self.sim.pos, [self.est[c][0] for c in todo])
        prog = False
        for i in order2:
            c = todo[i]
            if c in self.empty:
                continue
            if self.approach_and_clear(c):
                prog = True
        if not prog:
            break
    return self.sim.stats(0.0)


Robot.run_sector = run_sector


# ══════════════════════════════════════════════════════════
def run_tour(self, verbose=False):
    """合成路线：把「剩余侦察点」与「已定位源」放进同一条巡回里一次走完。

    动机：移动占 80% 以上耗时，而"绕环侦察"与"折返清除"是两段互相重叠
    的长途行程。合为一条巡回后，去清源的路上就顺带完成了侦察。
    """
    ring = survey_ring(6)[1:]
    self.sweep(np.array([0.0, 0.0]))                 # 中心先扫
    ring = sorted(ring, key=lambda p: math.atan2(p[1], p[0]))
    self.sweep(ring[0])                              # 就近再扫一个环点
    pending = [np.asarray(p, float) for p in ring[1:]]

    def plan():
        """当前待办：剩余侦察点（视为必须经过）+ 已定位源。

        侦察点用**下标**标识：numpy 数组不能用 list.remove（== 是逐元素比较）。
        """
        jobs = [("sweep", i, pending[i]) for i in range(len(pending))]
        jobs += [("clear", c, self.est[c][0]) for c in self.est
                 if c not in self.empty]
        return jobs

    while True:
        jobs = plan()
        if not jobs:
            break
        cur = self.sim.pos
        # 逐次挑选最近的待办（在线最近邻：每次只决定下一个去哪）
        j = min(range(len(jobs)),
                key=lambda i: np.linalg.norm(jobs[i][2] - cur))
        kind, ch, pos = jobs[j]
        if kind == "sweep":
            self.sweep(pos)
            del pending[ch]
        else:
            if ch not in self.empty:
                self.approach_and_clear(ch)
            if ch not in self.empty:
                self.empty.add(ch)       # 屡次失败则放弃，避免死循环
        if verbose:
            print(f"   {kind} ({pos[0]:.0f},{pos[1]:.0f}) "
                  f"t={self.sim.virtual_time:.0f}s 已清 {len(self.cleared)}")
    return self.sim.stats(0.0)


Robot.run_tour = run_tour


# ══════════════════════════════════════════════════════════
def run_rho(self, survey_pts=None, verbose=False):
    """滚动时域重优化（RHO）巡线。

    每个决策时刻把「未访问侦察点 ∪ 已定位未清除源」当作一个在线 TSP，
    用最近邻 + 2-opt 重解，**只执行第一条腿**，拿到新观测后立即重规划。
    与 run() 的差别在于：侦察与清除不再分成两段长途，而是合成一条巡线，
    从而消除"先绕环一周、再折返清源"的重复行程。
    """
    sim = self.sim
    sim.entered = True
    pts = [np.asarray(p, float) for p in
           (survey_pts if survey_pts is not None else survey_ring(6, margin=0.70))]
    pending = list(pts)
    guard = 0
    while True:
        guard += 1
        if guard > 20000:
            break
        jobs = [("sweep", i, pending[i]) for i in range(len(pending))]
        jobs += [("clear", c, self.est[c][0]) for c in self.est
                 if c not in self.empty]
        if not jobs:
            break
        order = route_nn(sim.pos, [j[2] for j in jobs])
        kind, ref, pos = jobs[order[0]]
        if kind == "sweep":
            self.sweep(pos)
            del pending[ref]
        else:
            ch = ref
            if ch not in self.empty:
                self.approach_and_clear(ch)
            if ch not in self.empty:
                # 不能一次失败就放弃：定向源从背面逼近会 no_signal，
                # 多试几次（每次都会换一个"曾看到过"的起点）再放弃。
                self.tries[ch] = self.tries.get(ch, 0) + 1
                if self.tries[ch] >= 3:
                    self.empty.add(ch)
        if verbose:
            print(f"   {kind} ({pos[0]:.0f},{pos[1]:.0f}) "
                  f"t={sim.virtual_time:.0f}s 已清 {len(self.cleared)}")
    return sim.stats(0.0)


Robot.run_rho = run_rho


# ══════════════════════════════════════════════════════════
def angular_order(pts, start_from):
    """按极角排序成一圈，并从离 start_from 最近的点起绕。"""
    P = [np.asarray(p, float) for p in pts]
    P = sorted(P, key=lambda p: math.atan2(p[1], p[0]))
    if not P:
        return P
    i = min(range(len(P)), key=lambda k: np.linalg.norm(P[k] - start_from))
    return P[i:] + P[:i]


def run_ring2(self, survey_pts=None, verbose=False):
    """侦察按「外环一圈 + 径向连接 + 内环一圈」走，避免内外环之间来回穿插。

    动机：混合半径的"就近跳"巡线会在内外环之间反复穿插，把行程放大近一倍。
    按极角各自绕一圈只需两条环的周长加两次径向连接。
    """
    sim = self.sim
    sim.entered = True
    pts = [np.asarray(p, float) for p in
           (survey_pts if survey_pts is not None else survey_ring(6, margin=0.70))]
    ctr = np.array([0.0, 0.0])
    inner = [p for p in pts if np.linalg.norm(p - ctr) < 1500]
    outer = [p for p in pts if np.linalg.norm(p - ctr) >= 1500]
    # 半径大的环放最后走会更绕，故先走外环、再进内环，最后在内环附近清源
    if np.linalg.norm(sim.pos) < 1500 and inner:
        seq = angular_order(inner, sim.pos) + angular_order(outer, ctr)
    else:
        seq = angular_order(outer, sim.pos) + angular_order(inner, ctr)
    for p in seq:
        self.sweep(p)
        if verbose:
            print(f"   侦察 ({p[0]:.0f},{p[1]:.0f}) t={sim.virtual_time:.0f}s")
    # 清除：最近邻 + 2-opt
    while True:
        todo = [c for c in self.est if c not in self.empty]
        if not todo:
            break
        order = route_nn(sim.pos, [self.est[c][0] for c in todo])
        prog = False
        for i in order:
            c = todo[i]
            if c in self.empty:
                continue
            if self.approach_and_clear(c):
                prog = True
        if not prog:
            break
    return sim.stats(0.0)


Robot.run_ring2 = run_ring2
