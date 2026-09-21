# -*- coding: utf-8 -*-
"""2026 国赛 B 题 —— 交会定位的几何基础。

约定（与附件 2 一致）：x 轴正东、y 轴正北、逆时针为正，坐标单位米，角度单位度。

核心对象：给定若干检测点 S_i 与其示向度 θ_i（含 ±1° 误差），
    第 i 个检测点给出的可能区域 = 以 S_i 为顶点、张角 2° 的角楔
        W_i = { P : |wrap(angle(S_i→P) − θ_i)| ≤ δ },  δ = 1°
    定位区域 R = ⋂ W_i ∩ 目标区域（半径 1800 的圆域）。
角楔是凸集（两个半平面之交），故 R 是凸多边形。
"""
from __future__ import annotations

import numpy as np

ARENA_R = 1800.0
DELTA_DEG = 1.0          # 示向度误差上界


def wrap180(a):
    """把角度归一到 (−180, 180]。"""
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def unit(deg):
    """方位角（度，逆时针自 x 轴）→ 单位向量。"""
    r = np.radians(deg)
    return np.array([np.cos(r), np.sin(r)])


def bearing(p_from, p_to):
    """从 p_from 指向 p_to 的方位角（度，[0,360)）。"""
    d = np.asarray(p_to, float) - np.asarray(p_from, float)
    return float(np.degrees(np.arctan2(d[1], d[0])) % 360.0)


# ══════════════════════════════════════════════════════════
# 半平面裁剪（Sutherland–Hodgman）
# ══════════════════════════════════════════════════════════
def clip_halfplane(poly, n, c):
    """保留满足 n·p ≤ c 的部分。poly 为 (m,2) 顶点序列（逆时针）。"""
    if len(poly) == 0:
        return poly
    out = []
    m = len(poly)
    for i in range(m):
        a = poly[i]
        b = poly[(i + 1) % m]
        da = n @ a - c
        db = n @ b - c
        if da <= 0:
            out.append(a)
        if (da < 0 < db) or (db < 0 < da):
            t = da / (da - db)
            out.append(a + t * (b - a))
    return np.array(out) if out else np.empty((0, 2))


def clip_circle(poly, center=(0.0, 0.0), radius=ARENA_R, nseg=720):
    """用圆（凸）裁剪多边形：以正 nseg 边形近似，逐边裁剪。"""
    c = np.asarray(center, float)
    ang = np.linspace(0, 2 * np.pi, nseg, endpoint=False)
    for i in range(nseg):
        # 外法向：边上两点的中垂方向
        p1 = c + radius * np.array([np.cos(ang[i]), np.sin(ang[i])])
        p2 = c + radius * np.array([np.cos(ang[(i + 1) % nseg]),
                                    np.sin(ang[(i + 1) % nseg])])
        mid = 0.5 * (p1 + p2)
        nvec = mid - c
        nn = np.linalg.norm(nvec)
        if nn < 1e-12:
            continue
        nvec = nvec / nn
        # 弦所在直线：过 mid、法向 nvec；保留 nvec·p ≤ nvec·mid
        poly = clip_halfplane(poly, nvec, float(nvec @ mid))
        if len(poly) == 0:
            return poly
    return poly


def sector_region(S, theta_deg, delta=DELTA_DEG, box=1e7):
    """单个检测点的可能区域：以 S 为顶点、张角 2δ 的角楔（先按大框截断）。"""
    S = np.asarray(S, float)
    big = np.array([[-box, -box], [box, -box], [box, box], [-box, box]])
    # 角楔 = 两条边界射线的内侧之交。用两条过 S 的直线半平面表示：
    #   方向 d 的射线“内侧”不是半平面，但 2δ 角楔 = { n1·p ≤ c1 } ∩ { n2·p ≤ c2 }
    #   其中 n 为两条边界线的外法向。
    dn = unit(theta_deg - delta)      # 下边界方向
    dp = unit(theta_deg + delta)      # 上边界方向
    # 下边界的外法向 = dp（指向楔内的一侧取反）
    n1 = np.array([dp[1], -dp[0]])    # 与 dp 垂直
    n2 = np.array([-dn[1], dn[0]])
    # 选取使楔内含 (θ) 方向的外法向
    dd = unit(theta_deg)
    if n1 @ dd > 0:
        n1 = -n1
    if n2 @ dd > 0:
        n2 = -n2
    poly = clip_halfplane(big, n1, float(n1 @ S))
    poly = clip_halfplane(poly, n2, float(n2 @ S))
    return poly


def localize(points, thetas, delta=DELTA_DEG, arena=True, nseg=180):
    """交会定位区域。points: (n,2)；thetas: (n,) 示向度（度）。

    返回凸多边形顶点（逆时针，已去重）；若区域为空返回 (0,2)。
    """
    pts = np.asarray(points, float).reshape(-1, 2)
    th = np.asarray(thetas, float).ravel()
    assert len(pts) == len(th)
    poly = None
    for S, t in zip(pts, th):
        w = sector_region(S, float(t), delta=delta)
        poly = w if poly is None else intersect_convex(poly, w)
        if len(poly) == 0:
            return np.empty((0, 2))
    if arena and len(poly) and np.max(np.linalg.norm(poly, axis=1)) > ARENA_R:
        # 只有多边形真的越出场地时才裁圆：多数调用都整块在场内，
        # 无条件裁剪会白跑 nseg 次半平面裁剪，是最大的性能瓶颈。
        poly = clip_circle(poly, (0, 0), ARENA_R, nseg=nseg)
    return clean_poly(poly)


def intersect_convex(P, Q):
    """两个逆时针凸多边形的交（用 P 的每条边裁剪 Q 不合适；改为双向裁剪）。"""
    def clip_by(poly, clipper):
        # 用 clipper 的每条边（逆时针，内侧为左）裁剪 poly
        out = np.asarray(poly, float)
        m = len(clipper)
        cc = centroid(clipper)
        for i in range(m):
            a = clipper[i]
            b = clipper[(i + 1) % m]
            e = b - a
            nvec = np.array([-e[1], e[0]])      # 逆时针时内侧法向
            c = float(nvec @ a)
            if nvec @ cc > c:                   # 保证指向内部
                nvec, c = -nvec, -c
            out = clip_halfplane(out, nvec, c)
            if len(out) == 0:
                return out
        return out
    return clean_poly(clip_by(P, Q))


def centroid(poly):
    poly = np.asarray(poly, float)
    if len(poly) == 0:
        return np.zeros(2)
    if len(poly) == 1:
        return poly[0]
    if len(poly) == 2:
        return poly.mean(axis=0)
    # 用扇形重心（对凸多边形足够）
    return poly.mean(axis=0)


def clean_poly(poly, tol=1e-9):
    """去掉重合点，并按逆时针排序（质心角排序）。"""
    if poly is None or len(poly) == 0:
        return np.empty((0, 2))
    p = np.asarray(poly, float)
    keep = [0]
    for i in range(1, len(p)):
        if np.linalg.norm(p[i] - p[keep[-1]]) > tol:
            keep.append(i)
    p = p[keep]
    if len(p) > 1 and np.linalg.norm(p[0] - p[-1]) <= tol:
        p = p[:-1]
    if len(p) < 3:
        return p
    c = p.mean(axis=0)
    ang = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])
    return p[np.argsort(ang)]


# ══════════════════════════════════════════════════════════
# 多边形直径（旋转卡壳）与最小覆盖圆（Welzl）
# ══════════════════════════════════════════════════════════
def cross2(a, b):
    """二维叉积（np.cross 只接受三维向量，二维会直接报错）。"""
    return float(a[0] * b[1] - a[1] * b[0])


def convex_hull(pts):
    pts = np.unique(np.asarray(pts, float).round(12), axis=0)
    if len(pts) <= 2:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    def half(ps):
        h = []
        for p in ps:
            while len(h) >= 2 and cross2(h[-1] - h[-2], p - h[-2]) <= 0:
                h.pop()
            h.append(p)
        return h
    lo = half(pts)
    up = half(pts[::-1])
    return np.array(lo[:-1] + up[:-1])


def polygon_diameter(poly):
    """凸多边形直径（最大两点距离）。返回 (直径, 端点对)。"""
    P = convex_hull(poly)
    n = len(P)
    if n == 0:
        return 0.0, (None, None)
    if n == 1:
        return 0.0, (P[0], P[0])
    if n == 2:
        return float(np.linalg.norm(P[1] - P[0])), (P[0], P[1])
    best, pair = -1.0, (P[0], P[1])
    j = 1
    for i in range(n):
        ni = (i + 1) % n
        while True:
            nj = (j + 1) % n
            cur = abs(cross2(P[ni] - P[i], P[j] - P[i]))
            nxt = abs(cross2(P[ni] - P[i], P[nj] - P[i]))
            if nxt > cur:
                j = nj
            else:
                break
        for k in (i, ni):
            d = float(np.linalg.norm(P[k] - P[j]))
            if d > best:
                best, pair = d, (P[k], P[j])
    return best, pair


def min_enclosing_circle(poly):
    """Welzl 最小覆盖圆。返回 (圆心, 半径)。"""
    P = [tuple(p) for p in np.asarray(poly, float)]
    if not P:
        return np.zeros(2), 0.0
    rng = np.random.default_rng(0)
    rng.shuffle(P)

    def circ2(a, b):
        c = 0.5 * (a + b)
        return c, 0.5 * np.linalg.norm(b - a)

    def circ3(a, b, c):
        ax, ay = a; bx, by = b; cx, cy = c
        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-14:
            # 共线：取最远两点
            pts = [a, b, c]
            best = (0, 1); bd = -1
            for i in range(3):
                for k in range(i + 1, 3):
                    dd = (pts[i][0] - pts[k][0]) ** 2 + (pts[i][1] - pts[k][1]) ** 2
                    if dd > bd:
                        bd, best = dd, (i, k)
            return circ2(np.array(pts[best[0]]), np.array(pts[best[1]]))
        ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay)
              + (cx**2 + cy**2) * (ay - by)) / d
        uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx)
              + (cx**2 + cy**2) * (bx - ax)) / d
        ctr = np.array([ux, uy])
        return ctr, float(np.linalg.norm(ctr - np.array(a)))

    def inside(p, c, r):
        return np.linalg.norm(np.array(p) - c) <= r + 1e-9

    c, r = np.array(P[0], float), 0.0
    for i in range(1, len(P)):
        if inside(P[i], c, r):
            continue
        c, r = np.array(P[i], float), 0.0
        for j in range(i):
            if inside(P[j], c, r):
                continue
            c, r = circ2(np.array(P[i]), np.array(P[j]))
            for k in range(j):
                if inside(P[k], c, r):
                    continue
                c, r = circ3(np.array(P[i]), np.array(P[j]), np.array(P[k]))
    return c, float(r)
