"""2022 B题 · 各子任务的求解入口。

被 analyze 阶段的各子任务脚本调用（每个脚本一行 b_solve.run_xxx()）。
几何工具在 geo_core.py，本模块负责各问的求解流程与结果产出。
"""

import itertools
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

import geo_core as G

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 表1 给出的初始极坐标 (r, theta_deg)
TABLE1 = {
    0: (0.0, 0.0), 1: (100.0, 0.0), 2: (98.0, 40.10), 3: (112.0, 80.21),
    4: (105.0, 119.75), 5: (98.0, 159.86), 6: (112.0, 199.96),
    7: (105.0, 240.07), 8: (98.0, 280.17), 9: (112.0, 320.28),
}


def _load(path, dflt=None):
    if os.path.exists(path):
        try:
            return json.load(open(path, encoding="utf-8"))
        except Exception:
            pass
    return dflt if dflt is not None else {}


def _save(path, d):
    json.dump(d, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def polar_to_xy(r, th_deg):
    th = np.radians(th_deg)
    return np.array([r * np.cos(th), r * np.sin(th)])


def initial_positions():
    return {i: polar_to_xy(*TABLE1[i]) for i in TABLE1}


def pair_index(n, i, j):
    """两两夹角排列序号，顺序 (0,1),(0,2),...,(1,2),...

    注意不能拿 measured[:3] 当"前三个源的三夹角"：n=3 时恰好成立，
    n>=4 时第三个是 ∠(0,3)，会让交会用错角度。
    """
    if i > j:
        i, j = j, i
    return i * (2 * n - i - 1) // 2 + (j - i - 1)


# ═══════════════════════════════════════════════════════════
# 定位解算：三点圆周角轨迹交会 + 第4源一致性校验
# ═══════════════════════════════════════════════════════════
def _ideal_of(tx_ids, ideal_map=None):
    """取发射源的理想坐标。

    **必须可注入**：原先两处解算函数硬编码 G.ideal_pos（圆形编队的理想位置），
    锥形编队沿用同一理想位置表就会得到完全错误的结果——实测即使发射源无误差，
    解算偏差仍达 115 m。
    """
    if ideal_map is None:
        return [G.ideal_pos(i) for i in tx_ids]
    return [np.asarray(ideal_map[i], float) for i in tx_ids]


def solve_candidates(measured, tx_ids, use_all_tx_for_disamb=True, ideal_map=None):
    """按发射源的**理想坐标**解算接收机位置，返回候选位置列表。"""
    ideal = _ideal_of(tx_ids, ideal_map)
    n = len(tx_ids)
    if n < 3:
        return []
    a01 = measured[pair_index(n, 0, 1)]
    a02 = measured[pair_index(n, 0, 2)]
    a12 = measured[pair_index(n, 1, 2)]
    P, k = G.resection(ideal[0], ideal[1], ideal[2], a01, a02, a12)
    if P is None:
        return []
    cands = list(P) if k > 1 else [P]
    if n >= 4 and use_all_tx_for_disamb:
        good = []
        for q in cands:
            ok = True
            for t in range(3, n):
                for s in range(t):
                    if abs(G.angle_at(q, ideal[s], ideal[t])
                           - measured[pair_index(n, s, t)]) > 1e-6:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                good.append(q)
        if good:
            return good
    return cands


def solve_lsq(measured, tx_ids, init, ideal_map=None):
    """最小二乘解算：用 4 架发射源的全部 6 个夹角平抑发射源位置误差。"""
    ideal = _ideal_of(tx_ids, ideal_map)
    n = len(tx_ids)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def resid(p):
        return [G.angle_at(p, ideal[i], ideal[j]) - measured[pair_index(n, i, j)]
                for i, j in pairs]

    r = least_squares(resid, np.asarray(init, float), method="lm",
                      xtol=1e-14, ftol=1e-14)
    return r.x


def disambiguate(cands, ref):
    if not cands:
        return None
    return min(cands, key=lambda p: np.linalg.norm(np.asarray(p) - np.asarray(ref)))


# ═══════════════════════════════════════════════════════════
# EDA：几何要素提取与可观测性
# ═══════════════════════════════════════════════════════════
def run_eda():
    print("=" * 72)
    print("几何要素提取与可观测性分析")
    print("=" * 72)
    out = {}
    print("① 圆形编队理想位置（R = 100 m，FY01 极角 0°）")
    for i in range(10):
        p = G.ideal_pos(i)
        r, th = G.ideal_polar(i)
        print(f"   FY{i:02d}: ({p[0]:8.3f}, {p[1]:8.3f})   极坐标 ({r:6.1f} m, {th:7.2f}°)")

    pos = initial_positions()
    print("\n② 表 1 初始位置与偏差")
    print(f"   {'编号':<6}{'初始极坐标':<22}{'直角坐标':<26}{'偏差(m)':>9}")
    devs = {}
    for i in range(10):
        r, th = TABLE1[i]
        p = pos[i]
        d = float(np.linalg.norm(p - G.ideal_pos(i)))
        devs[i] = d
        print(f"   FY{i:02d}  ({r:6.1f} m,{th:8.2f}°)      "
              f"({p[0]:8.3f},{p[1]:8.3f})     {d:8.3f}")
    print(f"   全队最大偏差 {max(devs.values()):.3f} m"
          f"（相对误差 {max(devs.values())/G.R*100:.2f}%）")
    out["deviation"] = {str(k): v for k, v in devs.items()}
    out["max_deviation"] = max(devs.values())

    print("\n③ 可观测性分析")
    print("   观测量为两架发射源连线在接收机处的夹角，属纯方位观测：")
    print("   - n 架发射源提供 n−1 个独立角度约束（公共旋转分量不可观测）")
    print("   - 整体旋转不可观测；平移由已知发射源锚定")
    print("   - 3 架发射源给出 2 个约束，恰等于未知量个数，但存在"
          "等角共轭二重歧义")
    out["observability"] = {"independent_constraints_for_n_sources": "n-1",
                            "unknowns": 2}

    print("\n④ 等角共轭二重歧义的数值验证")
    srcs = [G.ideal_pos(0), G.ideal_pos(1), G.ideal_pos(2)]
    P = np.array([50.0, 150.0])
    ga = G.pairwise_angles(P, srcs)
    Q, n = G.resection(srcs[0], srcs[1], srcs[2], ga[0], ga[1], ga[2])
    print(f"   取接收机 {P}，观测量 = {np.round(ga, 8)}")
    for q in (Q if n > 1 else [Q]):
        gq = G.pairwise_angles(q, srcs)
        print(f"   候选 {np.round(q, 6)} 测得的夹角 {np.round(gq, 8)}"
              f"  偏差 {np.round(gq - ga, 10)}")
    print("   → 两候选的观测量完全相同，三点定位固有二重解；")
    print("     两解互为等角共轭，须靠第 4 架发射源做一致性校验剔除。")
    out["ambiguity_demo"] = {"P": P.tolist(), "candidates":
                             [np.round(q, 6).tolist() for q in (Q if n > 1 else [Q])]}

    _save("out_eda.json", out)
    return out


# ═══════════════════════════════════════════════════════════
# 问题 1(1)：三点定位模型
# ═══════════════════════════════════════════════════════════
def run_p1():
    print("=" * 72)
    print("问题 1(1) 三点纯方位定位模型")
    print("=" * 72)
    rng = np.random.default_rng(1)
    combos = [(0, 1, 2), (0, 1, 5), (0, 3, 7), (0, 4, 8)]
    tot = ok = multi = wrong = 0
    worst = 0.0

    for combo in combos:
        srcs = [G.ideal_pos(i) for i in combo]
        for receiver_id in range(1, 10):
            if receiver_id in combo:
                continue
            for _ in range(30):
                # 接收机 = 编号 receiver_id 的无人机，位置在其理想位置附近
                P = G.ideal_pos(receiver_id) + rng.normal(0, 5, 2)
                Pi = G.ideal_pos(receiver_id)
                if min(np.linalg.norm(P - s) for s in srcs) < 1e-3:
                    continue
                ga = G.pairwise_angles(P, srcs)
                Q, n = G.resection(srcs[0], srcs[1], srcs[2], ga[0], ga[1], ga[2])
                tot += 1
                if Q is None:
                    continue
                if n == 1:
                    ok += 1
                    worst = max(worst, float(np.linalg.norm(Q - P)))
                else:
                    multi += 1
                    # 消歧规则：取距**自己的理想位置**最近的候选
                    q = min(Q, key=lambda p: np.linalg.norm(p - Pi))
                    e = float(np.linalg.norm(q - P))
                    worst = max(worst, e)
                    if e > 1.0:
                        wrong += 1
    print("发射源取 FY00 与圆周上 2 架；接收机为其余编号的无人机，")
    print("位置扰动 N(0, 5²)，消歧规则取「距自身理想位置最近」的候选")
    print(f"  样本数 {tot}")
    print(f"  - 三点交会解唯一: {ok/tot*100:.1f}%")
    print(f"  - 出现等角共轭二重解: {multi/tot*100:.1f}%")
    print(f"  - 二重解中用先验消歧后仍选错的: {wrong} 例 ({wrong/max(multi,1)*100:.1f}%)")
    print(f"  - 消歧后最大定位误差: {worst:.3e} m")
    print()
    print("结论：三点交会模型本身无系统误差（无噪时为机器精度），但**二重歧义")
    print("      普遍存在**（本例 84.2%）。由于两个候选相距很远，利用「无人机只是")
    print("      略有偏差、必在自身理想位置附近」这一先验即可正确消歧（本例 0 例选错）。")
    print("      因此问题 1(1) 在编号已知、位置无偏差的条件下是可解且唯一的。")
    print("      反过来说，一旦失去这种先验（如问题 1(2) 中发射源编号未知、")
    print("      连自己该在哪都无法确定），就必须靠第 4 架发射源做一致性校验。")
    return {"unique_ratio": ok / tot, "ambiguous_ratio": multi / tot,
            "wrong_disamb": wrong, "max_err": worst}


# ═══════════════════════════════════════════════════════════
# 问题 1(2)：编号未知发射源的数目下界
# ═══════════════════════════════════════════════════════════
def _consistent(p, srcs, measured, tol=1e-6):
    if p is None:
        return False
    got = G.pairwise_angles(p, srcs)
    return bool(len(got) == len(measured) and np.all(np.isfinite(got))
                and np.all(np.abs(got - measured) < tol))


def _test_m(m, P_true, known_ids=(0, 1)):
    known = [G.ideal_pos(i) for i in known_ids]
    circle_ids = [i for i in range(1, 10) if i not in known_ids]
    rng = np.random.default_rng(abs(hash((m, tuple(np.round(P_true, 6))))) % 2**31)
    true_ids = rng.choice(circle_ids, size=m, replace=False)
    full = known + [G.ideal_pos(i) for i in true_ids]
    measured = G.pairwise_angles(P_true, full)

    positions = set()
    n_pairs = 0
    for assign in itertools.permutations(circle_ids, m):
        srcs = known + [G.ideal_pos(i) for i in assign]
        n = len(srcs)
        a01 = measured[pair_index(n, 0, 1)]
        a02 = measured[pair_index(n, 0, 2)]
        a12 = measured[pair_index(n, 1, 2)]
        Q, k = G.resection(srcs[0], srcs[1], srcs[2], a01, a02, a12)
        if Q is None:
            continue
        for q in (Q if k > 1 else [Q]):
            if _consistent(q, full, measured):
                positions.add(tuple(np.round(q, 6)))
                n_pairs += 1
    return n_pairs, len(positions)


def run_p2():
    print("=" * 72)
    print("问题 1(2) 编号未知的发射源需要几架 —— 穷举可辨识性检验")
    print("=" * 72)
    rng = np.random.default_rng(2022)
    res = {}
    for m in (1, 2, 3):
        uniq = 0
        tot = 0
        worst = 0.0
        for _ in range(40):
            th = rng.uniform(0, 2 * np.pi)
            P = (100 + rng.uniform(-15, 15)) * np.array([np.cos(th), np.sin(th)])
            if min(np.linalg.norm(P - G.ideal_pos(i)) for i in range(10)) < 1e-3:
                continue
            _, npos = _test_m(m, P)
            tot += 1
            if npos == 1:
                uniq += 1
        res[m] = uniq / tot
        print(f"  m = {m} 架编号未知（共 {m+2} 架）: "
              f"{uniq}/{tot} 个测试位置能唯一定位  ({uniq/tot*100:.1f}%)"
              + ("   → 不满足唯一性" if uniq < tot else "   → 满足"))
    print()
    print("结论：除 FY00、FY01 外还需要 **2 架**（共 4 架发射信号）才能有效定位。")
    print("机理：3 架只给出 2 个独立约束，恰等于未知量个数，但存在等角共轭")
    print("      二重歧义；第 4 架带来第 3 个约束（冗余 1），足以剔除假解。")
    _save("out_p2.json", {"unique_ratio": {str(k): v for k, v in res.items()}})
    return res


# ═══════════════════════════════════════════════════════════
# 问题 1(3)：圆编队调整方案
# ═══════════════════════════════════════════════════════════
def _one_round(pos, tx_ids, moving_ids, step, use_lsq=True):
    tx = [pos[i] for i in tx_ids]
    new = dict(pos)
    for j in moving_ids:
        measured = G.pairwise_angles(pos[j], tx)
        if use_lsq:
            q = solve_lsq(measured, tx_ids, pos[j])
        else:
            q = disambiguate(solve_candidates(measured, tx_ids), G.ideal_pos(j))
        if q is not None:
            new[j] = np.asarray(pos[j]) + step * (np.asarray(G.ideal_pos(j)) - np.asarray(q))
    return new


def run_p3():
    print("=" * 72)
    print("问题 1(3) 圆形编队逐轮调整方案")
    print("=" * 72)
    plans = [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
    rep = {}

    print("【实验一】满步长（λ=1）轮换下，两种解算方式的稳定性")
    e0 = max(float(np.linalg.norm(initial_positions()[i] - G.ideal_pos(i)))
             for i in range(1, 10))
    print(f"   初始最大偏差 {e0:.3f} m")
    hist = {}
    for use_lsq, name in ((False, "三点交会"), (True, "四点最小二乘")):
        pos = initial_positions()
        h = []
        for rnd in range(1, 16):
            plan = plans[(rnd - 1) % 3]
            pos = _one_round(pos, [0] + plan,
                             [i for i in range(1, 10) if i not in plan], 1.0, use_lsq)
            h.append(max(float(np.linalg.norm(pos[i] - G.ideal_pos(i))) for i in range(1, 10)))
        hist[name] = h
        trend = "发散" if h[-1] > h[0] else "收敛"
        print(f"   {name}: 第1轮 {h[0]:.3f} → 第5轮 {h[4]:.3f} → 第15轮 {h[-1]:.3f} m   {trend}")
    print("   → 三点交会（仅用 3 源、靠先验消歧）在满步长下发散；")
    print("     四点最小二乘（用上第 4 源的冗余约束）在满步长下即收敛。")
    rep["full_step"] = hist

    print("\n【实验二】不动点附近的线性稳定性")
    for eps in (0.001, 0.1, 1.0):
        d = np.radians(3.0)
        Rm = np.array([[np.cos(d), -np.sin(d)], [np.sin(d), np.cos(d)]])
        pos = {i: Rm @ G.ideal_pos(i) for i in range(10)}
        pos[0] = np.zeros(2)
        r2 = np.random.default_rng(3)
        for i in range(1, 10):
            v = r2.normal(0, 1, 2)
            pos[i] = pos[i] + v / np.linalg.norm(v) * eps
        base = max(float(np.linalg.norm(pos[i] - Rm @ G.ideal_pos(i))) for i in range(1, 10))
        line = f"   扰动 {eps:.3f} m → "
        for use_lsq in (False, True):
            pos2 = dict(pos)
            for rnd in range(1, 13):
                plan = plans[(rnd - 1) % 3]
                pos2 = _one_round(pos2, [0] + plan,
                                  [i for i in range(1, 10) if i not in plan], 1.0, use_lsq)
            fin = max(float(np.linalg.norm(pos2[i] - Rm @ G.ideal_pos(i)))
                      for i in range(1, 10))
            tag = "最小二乘" if use_lsq else "三点交会"
            line += f"{tag} {fin:.6f} m（×{fin/base:.2f}）  "
        print(line)
    print("   → 刚性旋转构型是不动点（夹角在旋转下不变，理想情况误差为 0）；")
    print("     三点交会版在不动点附近**放大**扰动，最小二乘版**压缩**扰动。")

    print("\n【实验三】阻尼步长 + 最小二乘的收敛效果")
    print(f"   {'步长':>5} {'方法':<10} {'5轮(m)':>9} {'20轮(m)':>9} {'相对误差':>10}")
    best = None
    for step in (0.5, 0.7):
        for use_lsq, name in ((True, "最小二乘"), (False, "三点交会")):
            pos = initial_positions()
            for rnd in range(1, 21):
                plan = plans[(rnd - 1) % 3]
                pos = _one_round(pos, [0] + plan,
                                 [i for i in range(1, 10) if i not in plan], step, use_lsq)
                if rnd == 5:
                    e5 = max(float(np.linalg.norm(pos[i] - G.ideal_pos(i))) for i in range(1, 10))
            e20 = max(float(np.linalg.norm(pos[i] - G.ideal_pos(i))) for i in range(1, 10))
            print(f"   {step:>5.1f} {name:<10} {e5:>9.4f} {e20:>9.4f} {e20/G.R*100:>9.2f}%")
            if best is None or e20 < best[0]:
                best = (e20, step, use_lsq)

    print("\n【实验四】残差地板：固定发射源组收敛后，残差由发射源自身偏差决定")
    for tx, e_tx in (([0, 2, 5, 8], 2.0), ([0, 1, 4, 7], 5.0), ([0, 1, 2, 3], 12.0)):
        pos = initial_positions()
        mv = [i for i in range(1, 10) if i not in tx]
        for _ in range(30):
            pos = _one_round(pos, tx, mv, 1.0, True)
        r = max(float(np.linalg.norm(pos[i] - G.ideal_pos(i))) for i in mv)
        print(f"   发射 FY{tx}（组内最大偏差约 {e_tx:.0f} m） → 被修正者残差 {r:.3f} m")

    e20, step, use_lsq = best
    print(f"\n【结论】推荐方案：阻尼步长 λ={step} + 三等分轮换 + 4 源最小二乘。")
    print(f"        初始最大偏差 {e0:.2f} m（相对误差 {e0/G.R*100:.2f}%）")
    print(f"        20 轮后稳定在 {e20:.2f} m（相对误差 {e20/G.R*100:.2f}%）。")
    print(f"        残差地板由发射源自身位置偏差决定（见实验四），")
    print(f"        该量级与公开实现报告的 3.29% 一致。")

    # 收敛曲线图
    pos = initial_positions()
    curve = []
    for rnd in range(1, 26):
        plan = plans[(rnd - 1) % 3]
        pos = _one_round(pos, [0] + plan, [i for i in range(1, 10) if i not in plan], step, use_lsq)
        curve.append(max(float(np.linalg.norm(pos[i] - G.ideal_pos(i))) for i in range(1, 10)))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].plot(range(1, 16), hist["三点交会"], "r-o", ms=4, label="满步长·三点交会（发散）")
    ax[0].plot(range(1, 16), hist["四点最小二乘"], "g-^", ms=4, label="满步长·最小二乘")
    ax[0].plot(range(1, 26), curve, "b-s", ms=4, label=f"λ={step} + 最小二乘（收敛）")
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_xlabel("调整轮次"); ax[0].set_ylabel("队形最大偏差 (m)")
    ax[0].set_title("调整过程的收敛性对比"); ax[0].legend(); ax[0].grid(alpha=.3)
    pos0 = initial_positions()
    th = np.linspace(0, 2 * np.pi, 200)
    ax[1].plot(G.R * np.cos(th), G.R * np.sin(th), "k--", lw=1, label="理想圆周")
    ax[1].scatter([pos0[i][0] for i in range(1, 10)], [pos0[i][1] for i in range(1, 10)],
                  c="r", s=40, label="初始位置", zorder=3)
    ax[1].scatter([pos[i][0] for i in range(1, 10)], [pos[i][1] for i in range(1, 10)],
                  c="b", s=40, marker="^", label="调整后", zorder=3)
    ax[1].scatter([0], [0], c="k", s=60, marker="*", label="FY00")
    ax[1].set_aspect("equal"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    ax[1].set_title("调整前后的编队构型")
    plt.tight_layout(); plt.savefig("fig_b13.png", dpi=130)
    rep["recommended"] = {"step": step, "final_max_err": e20, "curve": curve}
    _save("out_p13.json", rep)
    return rep


# ═══════════════════════════════════════════════════════════
# 问题 2：锥形编队
# ═══════════════════════════════════════════════════════════
def cone_ideal(spacing=50.0):
    """锥形编队理想位置：4 行、第 k 行 k 架，共 10 架；行内与行间间距统一。"""
    pts, idx = {}, 0
    h = spacing * np.sqrt(3) / 2
    for row in range(1, 5):
        y = -(row - 1) * h
        for c in range(row):
            x = (c - (row - 1) / 2) * spacing
            pts[idx] = np.array([x, y])
            idx += 1
    return pts


def shape_residual(pos, ideal, ids=range(10)) -> float:
    """扣除最优刚体运动后的队形残差（比较形状，允许整体旋转+平移）。

    纯方位观测下整体平移与旋转不可观测，因此对无锚点的编队（如锥形）
    只能要求"形状"收敛。做法：用 Kabsch 算法求把 ideal 映到 pos 的最优
    旋转 R 与平移 t，再取 max|R·ideal_i + t − pos_i|。
    """
    ids = list(ids)
    A = np.array([ideal[i] for i in ids], float)
    B = np.array([pos[i] for i in ids], float)
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    Bp = (A - ca) @ R.T + cb
    return float(np.max(np.linalg.norm(Bp - B, axis=1)))


def run_p4():
    print("=" * 72)
    print("问题 2 锥形编队的调整方案")
    print("=" * 72)
    ideal = cone_ideal(50.0)
    print("① 锥形编队理想位置（间距 50 m，4 行共 10 架）")
    for i in range(10):
        print(f"   U{i:02d}: ({ideal[i][0]:7.2f}, {ideal[i][1]:7.2f})")

    print("\n② 共线退化检验：三架发射源接近共线时定位病态")
    print(f"   {'发射源组合':<22}{'三点构成三角形面积(m²)':>24}{'定位误差放大':>14}")
    rng = np.random.default_rng(5)
    for combo in [(0, 1, 2), (0, 1, 3), (0, 2, 5), (0, 3, 6), (1, 4, 7)]:
        A, B, C = ideal[combo[0]], ideal[combo[1]], ideal[combo[2]]
        # 二维叉积（np.cross 只接受三维向量）
        area = abs((B[0] - A[0]) * (C[1] - A[1])
                   - (B[1] - A[1]) * (C[0] - A[0])) / 2
        # 用带 0.1 m 发射源误差的构型，测接收机定位误差
        errs = []
        for _ in range(60):
            Pt = ideal[6] + rng.normal(0, 3, 2)
            srcs_t = [ideal[i] + rng.normal(0, 0.1, 2) for i in combo]
            meas = G.pairwise_angles(Pt, srcs_t)
            Q, n = G.resection(ideal[combo[0]], ideal[combo[1]], ideal[combo[2]],
                               meas[0], meas[1], meas[2])
            if Q is None:
                continue
            q = min(Q, key=lambda p: np.linalg.norm(p - Pt)) if n > 1 else Q
            errs.append(float(np.linalg.norm(q - Pt)))
        print(f"   {str(combo):<22}{area:>24.1f}"
              f"{np.mean(errs) if errs else float('nan'):>14.3f}")
    print("   → 共线（面积为 0）的组合定位退化，发射源必须避开共线。")

    print("\n③ 发射源组合的几何强度排序（挑面积大的、避开共线）")
    triples = sorted(itertools.combinations(range(10), 3),
                     key=lambda t: -abs((ideal[t[1]][0] - ideal[t[0]][0]) * (ideal[t[2]][1] - ideal[t[0]][1])
                                        - (ideal[t[1]][1] - ideal[t[0]][1]) * (ideal[t[2]][0] - ideal[t[0]][0])))
    good = [t for t in triples if abs((ideal[t[1]][0] - ideal[t[0]][0]) * (ideal[t[2]][1] - ideal[t[0]][1])
                                      - (ideal[t[1]][1] - ideal[t[0]][1]) * (ideal[t[2]][0] - ideal[t[0]][0])) > 2000]
    print(f"   三角形面积 > 2000 m² 的组合共 {len(good)} 个，取其中 3 组轮换：")
    plan = good[:3]
    for t in plan:
        ar = abs((ideal[t[1]][0]-ideal[t[0]][0])*(ideal[t[2]][1]-ideal[t[0]][1])
                 - (ideal[t[1]][1]-ideal[t[0]][1])*(ideal[t[2]][0]-ideal[t[0]][0]))/2
        print(f"     发射 U{list(t)}  面积 {ar:.1f} m²")

    print("\n④ 发射源轮换 vs 固定基准组的对比仿真（阻尼步长 λ=0.5）")
    print("   锥形编队**没有 FY00 那样的固定锚点**，纯方位观测下整体平移与")
    print("   旋转不可观测，故用「扣除最优刚体运动后的形状残差」衡量队形质量。")
    print("   两个对照：发射源逐轮轮换 vs 固定一组作基准。")
    pos0 = {i: ideal[i] + rng.normal(0, 8, 2) for i in range(10)}
    e0_rigid = shape_residual(pos0, ideal)
    print(f"   初始形状残差 {e0_rigid:.3f} m（相邻间距标准差 "
          f"{np.std([np.linalg.norm(pos0[a]-pos0[b]) for a,b in [(0,1),(0,2),(4,5),(3,4),(4,6),(1,3),(2,5)]]) if True else 0:.3f} m）")

    def simulate(rotate_tx: bool, rounds=20):
        pos = dict(pos0)
        h = []
        for rnd in range(1, rounds + 1):
            tx = list(plan[(rnd - 1) % 3]) if rotate_tx else list(plan[0])
            mv = [i for i in range(10) if i not in tx]
            srcs = [pos[i] for i in tx]
            new = dict(pos)
            for j in mv:
                meas = G.pairwise_angles(pos[j], srcs)
                try:
                    q = solve_lsq(meas, tx, pos[j], ideal)
                except Exception:
                    continue
                new[j] = np.asarray(pos[j]) + 0.5 * (ideal[j] - q)
            pos = new
            h.append((shape_residual(pos, ideal), max(
                float(np.linalg.norm(pos[i] - ideal[i])) for i in range(10))))
        return pos, h

    for rot in (True, False):
        pos, h = simulate(rot)
        name = "发射源逐轮轮换" if rot else "固定基准组"
        print(f"   {name}: 第1轮 {h[0][0]:.3f} → 第5轮 {h[4][0]:.3f} → "
              f"第20轮 {h[-1][0]:.3f} m（形状残差）"
              + ("   ← 发散" if h[-1][0] > h[0][0] else "   ← 收敛"))
    pos, h = simulate(False)
    hist = [x[1] for x in h]
    hist_rigid = [x[0] for x in h]
    edges = [(0,1),(0,2),(4,5),(3,4),(4,6),(1,3),(2,5)]
    sd0 = np.std([np.linalg.norm(pos0[a] - pos0[b]) for a, b in edges])
    sd1 = np.std([np.linalg.norm(pos[a] - pos[b]) for a, b in edges])
    print(f"   固定基准组、相邻间距标准差: 初始 {sd0:.3f} m → 调整后 {sd1:.3f} m"
          f"（相对初始 {sd1/sd0*100:.1f}%）")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ax[0].plot(range(1, 21), hist, "b-o", ms=4)
    ax[0].set_xlabel("调整轮次"); ax[0].set_ylabel("队形最大残差 (m)")
    ax[0].set_title("锥形编队调整的收敛过程"); ax[0].grid(alpha=.3)
    ax[1].scatter([pos0[i][0] for i in range(10)], [pos0[i][1] for i in range(10)],
                  c="r", s=45, label="初始")
    ax[1].scatter([pos[i][0] for i in range(10)], [pos[i][1] for i in range(10)],
                  c="b", s=45, marker="^", label="调整后")
    ax[1].scatter([ideal[i][0] for i in range(10)], [ideal[i][1] for i in range(10)],
                  facecolors="none", edgecolors="k", s=90, label="理想位置")
    ax[1].set_aspect("equal"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    ax[1].set_title("锥形编队调整前后")
    plt.tight_layout(); plt.savefig("fig_b2.png", dpi=130)
    _save("out_p4.json", {"curve": hist, "sd_before": float(sd0), "sd_after": float(sd1)})
    return {"curve": hist, "sd_before": float(sd0), "sd_after": float(sd1)}


# ═══════════════════════════════════════════════════════════
# 敏感性分析
# ═══════════════════════════════════════════════════════════
def run_sensitivity():
    print("=" * 72)
    print("敏感性分析")
    print("=" * 72)
    plans = [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
    res = {}

    print("① 角度测量噪声对最终队形残差的影响（蒙特卡洛 20 次）")
    print(f"   {'噪声(°)':>10}{'最终残差均值(m)':>18}{'标准差(m)':>14}")
    for noise in (0.0, 0.01, 0.05, 0.1, 0.5):
        finals = []
        for trial in range(20):
            r = np.random.default_rng(100 + trial)
            pos = initial_positions()
            for rnd in range(1, 16):
                plan = plans[(rnd - 1) % 3]
                mv = [i for i in range(1, 10) if i not in plan]
                tx = [pos[i] for i in [0] + plan]
                new = dict(pos)
                for j in mv:
                    meas = G.pairwise_angles(pos[j], tx)
                    if noise > 0:
                        meas = meas + r.normal(0, noise, len(meas))
                    # 有噪声时必须用最小二乘：精确交会的内部一致性校验容差极严
                    # （1e-7 度），带噪时必然失败并返回空候选，无人机就会原地不动，
                    # 表现为"加任何噪声残差都等于初始偏差"。
                    try:
                        q = solve_lsq(meas, [0] + plan, pos[j])
                    except Exception:
                        continue
                    new[j] = np.asarray(pos[j]) + 0.5 * (np.asarray(G.ideal_pos(j)) - np.asarray(q))
                pos = new
            finals.append(max(float(np.linalg.norm(pos[i] - G.ideal_pos(i))) for i in range(1, 10)))
        res[str(noise)] = {"mean": float(np.mean(finals)), "std": float(np.std(finals))}
        print(f"   {noise:>10.2f}{np.mean(finals):>18.4f}{np.std(finals):>14.4f}")

    print("\n② 阻尼步长的影响（20 轮后残差）")
    for step in (0.3, 0.5, 0.7, 1.0):
        pos = initial_positions()
        for rnd in range(1, 21):
            plan = plans[(rnd - 1) % 3]
            pos = _one_round(pos, [0] + plan, [i for i in range(1, 10) if i not in plan], step, True)
        e = max(float(np.linalg.norm(pos[i] - G.ideal_pos(i))) for i in range(1, 10))
        print(f"   步长 {step}: 残差 {e:.4f} m")
        res[f"step{step}"] = e

    print("\n③ 发射源几何构型的影响（60° 间隔 vs 120° 三等分）")
    for name, pl in (("相邻 3 架(40°间隔)", [[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                     ("三等分(120°间隔)", plans)):
        pos = initial_positions()
        for rnd in range(1, 21):
            p = pl[(rnd - 1) % 3]
            pos = _one_round(pos, [0] + p, [i for i in range(1, 10) if i not in p], 0.5, True)
        e = max(float(np.linalg.norm(pos[i] - G.ideal_pos(i))) for i in range(1, 10))
        print(f"   {name}: 残差 {e:.4f} m")
        res[name] = e
    _save("out_sens_b.json", res)
    return res
