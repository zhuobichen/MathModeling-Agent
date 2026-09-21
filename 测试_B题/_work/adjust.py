"""问题 1(3)：圆编队的逐轮位置调整方案与仿真。

观测模型：接收机只能得到"到各发射源的连线两两夹角"，整体旋转与尺度不可观测；
但 FY00 位于圆心（恒为坐标原点），编队为已知的正九边形，因此接收机可以用
**发射源的理想坐标**解算自身在理想坐标系中的位置，再朝自己的理想位置移动。

难点：发射源自身也有偏差。用带偏差的发射源去解算，会得到有偏的位置。
方案的关键在于**轮换发射源并迭代**：每一轮让一部分无人机充当（近似）基准，
其余无人机被修正；轮换若干轮后全队收敛。

本文件实现该方案并做数值仿真，输出每轮各机的偏差与整体队形误差。
"""

import numpy as np

import geo_core as G

# 表1 给出的初始极坐标 (r, theta_deg)
TABLE1 = {
    0: (0.0, 0.0),
    1: (100.0, 0.0),
    2: (98.0, 40.10),
    3: (112.0, 80.21),
    4: (105.0, 119.75),
    5: (98.0, 159.86),
    6: (112.0, 199.96),
    7: (105.0, 240.07),
    8: (98.0, 280.17),
    9: (112.0, 320.28),
}


def polar_to_xy(r: float, th_deg: float) -> np.ndarray:
    th = np.radians(th_deg)
    return np.array([r * np.cos(th), r * np.sin(th)])


def initial_positions() -> dict[int, np.ndarray]:
    return {i: polar_to_xy(*TABLE1[i]) for i in TABLE1}


def solve_receiver(meas_srcs, measured, tx_ids):
    """接收机解算：按发射源的**理想坐标**做后方交会，返回候选位置列表。

    解算一律用理想坐标——这正是"接收机认为发射源应在理想位置"的建模假设。

    **关键**：三点交会存在等角共轭二重歧义（见 geo_core 说明）。若只用 3 个
    发射源，两个候选都满足观测量，只能靠"离理想位置近"这类先验去猜——而队形
    一旦整体旋转，"离理想位置近"就会挑错，迭代随即发散。
    因此第 4 个发射源必须用于**一致性校验**：把两个候选分别代回全部发射源重算
    夹角，只有几何上真正自洽的那个能通过。这正是问题 1(2) 的机理。
    """
    ideal = [G.ideal_pos(i) for i in tx_ids]
    n = len(tx_ids)

    def pidx(i, j):
        if i > j:
            i, j = j, i
        return i * (2 * n - i - 1) // 2 + (j - i - 1)

    if n < 3:
        return []
    a01 = measured[pidx(0, 1)]
    a02 = measured[pidx(0, 2)]
    a12 = measured[pidx(1, 2)]
    P, k = G.resection(ideal[0], ideal[1], ideal[2], a01, a02, a12)
    if P is None:
        return []
    cands = list(P) if k > 1 else [P]

    if n >= 4:                      # 用第 4 个源消歧
        good = []
        for q in cands:
            ok = True
            for t in range(3, n):
                for s in range(t):
                    want = measured[pidx(s, t)]
                    if abs(G.angle_at(q, ideal[s], ideal[t]) - want) > 1e-6:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                good.append(q)
        if good:
            return good
    return cands


def disambiguate(cands, my_ideal, prev_est=None):
    """消歧：取离"自己的理想位置"最近的候选。

    依据是任务背景——每架无人机只是"位置略有偏差"，
    对自身大致方位有先验，取不出真解的可能性很低。
    若无理想位置（如问题2 的锥形编队），则退回用上一轮估计。
    """
    if not cands:
        return None
    ref = my_ideal if my_ideal is not None else prev_est
    if ref is None:
        return cands[0]
    return min(cands, key=lambda p: np.linalg.norm(np.asarray(p) - np.asarray(ref)))


def one_round(pos, tx_ids, moving_ids, step=1.0):
    """执行一轮：tx_ids 发射，moving_ids 接收并按解算结果移动。

    **关键**：无人机只能按"解算出的位置"来估计自己该走多少：
        位移量 = 理想位置 − 解算位置
        真实落点 = 当前真实位置 + 位移量
    若写成"直接落到理想位置"，则残差恒为 0，仿真是假的——
    误差恰恰来自"解算位置 ≠ 真实位置"这一点。

    step=1.0 表示一次走完全部估计位移。
    返回 (新位置字典, 本轮各机的解算位置)。
    """
    tx_srcs = [pos[i] for i in tx_ids]
    solved = {}
    new = dict(pos)
    for j in moving_ids:
        measured = G.pairwise_angles(pos[j], tx_srcs)
        cands = solve_receiver(tx_srcs, measured, tx_ids)
        q = disambiguate(cands, G.ideal_pos(j))
        solved[j] = q
        if q is not None:
            move = np.asarray(G.ideal_pos(j)) - np.asarray(q)
            new[j] = np.asarray(pos[j]) + step * move
    return new, solved


def formation_error(pos, ids=range(1, 10)) -> float:
    """队形误差：各机到理想位置的最大偏差（仅用于评估，不参与解算）。"""
    return max(float(np.linalg.norm(pos[i] - G.ideal_pos(i))) for i in ids)


if __name__ == "__main__":
    pos = initial_positions()
    print("=" * 78)
    print("问题 1(3)：逐轮调整仿真")
    print("=" * 78)
    print(f"初始队形最大偏差: {formation_error(pos):.3f} m")
    for i in range(1, 10):
        d = np.linalg.norm(pos[i] - G.ideal_pos(i))
        print(f"  FY{i:02d}: 初始 {np.round(pos[i],3)}  偏差 {d:7.3f} m")
    print()
    # 每轮：FY00 + 3 架（尽量分散，约 120 度间隔）发射
    plan = [5, 8, 2]                    # 轮换的发射源
    for rnd in range(1, 5):
        tx = [0] + plan
        moving = [i for i in range(1, 10) if i not in plan]
        pos, solved = one_round(pos, tx, moving)
        print(f"第 {rnd} 轮: 发射 FY{plan}  →  修正 FY{moving}")
        print(f"  队形最大偏差: {formation_error(pos):.3f} m")
        for i in moving:
            e = np.linalg.norm(pos[i] - G.ideal_pos(i))
            print(f"    FY{i:02d} 解算={np.round(solved[i],3) if solved[i] is not None else None}"
                  f"  残差 {e:.4f} m")
        plan = plan[1:] + plan[:1]      # 轮换
        print()
