"""2022 B题 · 纯方位无源定位的核心几何工具。

坐标约定：平面直角坐标系，原点为圆心 FY00，x 轴指向 FY01 的理想方向。
理想圆形编队：FY00 在原点，FY0k (k=1..9) 在半径 R=100 m 的圆周上，
极角 theta_k = (k-1)*40 度。

方位观测量的约定：接收机 P 测到的"方向信息"是 P 到任意两个发射源的
连线之间的夹角（0~180 度）。等价地，P 测到了一组方位角，整体旋转不可观测，
因此 n 个发射源只提供 n-1 个独立约束。
"""

import numpy as np

R = 100.0            # 理想圆周半径 m
N_CIRCLE = 9         # 圆周上的无人机数
THETA0 = 0.0         # FY01 的极角


def ideal_pos(i: int) -> np.ndarray:
    """编号 i (0..9) 的理想位置。i=0 为圆心。"""
    if i == 0:
        return np.array([0.0, 0.0])
    th = np.radians(THETA0 + (i - 1) * 360.0 / N_CIRCLE)
    return R * np.array([np.cos(th), np.sin(th)])


def ideal_polar(i: int):
    """理想位置的极坐标 (r, theta_deg)。"""
    p = ideal_pos(i)
    return float(np.hypot(*p)), float(np.degrees(np.arctan2(p[1], p[0])) % 360)


def angle_at(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """在点 p 处，向量 p->a 与 p->b 的夹角，返回 [0,180] 度（无符号）。"""
    u, v = np.asarray(a, float) - p, np.asarray(b, float) - p
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return np.nan
    c = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def bearings(p: np.ndarray, srcs) -> np.ndarray:
    """P 到各发射源的方位角（弧度），按 srcs 顺序。"""
    return np.array([np.arctan2(s[1] - p[1], s[0] - p[0]) for s in srcs])


def pairwise_angles(p: np.ndarray, srcs) -> np.ndarray:
    """P 处测到的所有两两夹角（度），按 (0,1),(0,2),... 顺序。"""
    out = []
    for a in range(len(srcs)):
        for b in range(a + 1, len(srcs)):
            out.append(angle_at(p, srcs[a], srcs[b]))
    return np.array(out)


# ── 后方交会：三点法（圆周角轨迹交点）──
def _circle_through(a, b, gamma_deg):
    """过点 a、b，且弦 ab 所对圆周角为 gamma 的圆。返回 (圆心, 半径)。

    圆心在 ab 的垂直平分线上，到弦的距离 d = |ab| / (2 tan gamma)。
    返回两个候选圆（P 在弦的两侧），本函数只给出 |d| 对应的圆心，
    符号由调用方按需选择。
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    mid = (a + b) / 2
    ab = b - a
    L = np.linalg.norm(ab)
    g = np.radians(gamma_deg)
    if abs(np.sin(g)) < 1e-12:
        return None
    d = L / (2 * np.tan(g))          # 有符号：g<90 为正，g>90 为负
    n = np.array([-ab[1], ab[0]]) / L
    return mid + d * n, abs(L / (2 * np.sin(g)))


def _circle_intersections(c1, r1, c2, r2):
    """两圆交点（0/1/2 个）。"""
    c1, c2 = np.asarray(c1, float), np.asarray(c2, float)
    d = np.linalg.norm(c2 - c1)
    if d < 1e-12:
        return []
    if d > r1 + r2 + 1e-9 or d < abs(r1 - r2) - 1e-9:
        return []
    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h2 = r1**2 - a**2
    h = np.sqrt(max(h2, 0.0))
    base = c1 + a * (c2 - c1) / d
    perp = np.array([-(c2 - c1)[1], (c2 - c1)[0]]) / d
    if h < 1e-9:
        return [base]
    return [base + h * perp, base - h * perp]


def resection(sa, sb, sc, gab_deg, gac_deg, gbc_deg, tol=1e-7):
    """三点后方交会：已知发射源 sa,sb,sc，已知 P 处三夹角，求 P。

    用弦 ab（对应角 gab）与弦 ac（对应角 gac）的两条圆周角轨迹求交，
    交点有两个（含已知点 a），取非 a 的那个。
    返回 (P, 候选数)。
    """
    sa, sb, sc = map(lambda x: np.asarray(x, float), (sa, sb, sc))
    cands = []
    for sgn_b in (1, -1):
        cb = _circle_through(sa, sb, gab_deg)
        if cb is None:
            continue
        c1 = (cb[0] if sgn_b > 0 else
              2 * (sa + sb) / 2 - cb[0])          # 关于弦镜像
        for sgn_c in (1, -1):
            cc = _circle_through(sa, sc, gac_deg)
            if cc is None:
                continue
            c2 = (cc[0] if sgn_c > 0 else 2 * (sa + sc) / 2 - cc[0])
            for p in _circle_intersections(c1, cb[1], c2, cc[1]):
                if np.linalg.norm(p - sa) < 1e-6:
                    continue
                # 必须校验**全部三个**夹角，不能只校验未参与构造的那一个。
                # 镜像候选点会落在同一条"过 sb、sc 的圆周角轨迹"上，
                # 于是 ∠(sb,sc) 天然与观测量相同，用它根本区分不出真假解；
                # 真正能区分的是参与构造的两个夹角——假解在它们上面差 20 度以上。
                ok = (abs(angle_at(p, sa, sb) - gab_deg) < tol and
                      abs(angle_at(p, sa, sc) - gac_deg) < tol and
                      abs(angle_at(p, sb, sc) - gbc_deg) < tol)
                if ok:
                    cands.append(p)
    # 去重
    uniq = []
    for p in cands:
        if not any(np.linalg.norm(p - q) < 1e-3 for q in uniq):
            uniq.append(p)
    if not uniq:
        return None, 0
    if len(uniq) == 1:
        return uniq[0], 1
    return uniq, len(uniq)
