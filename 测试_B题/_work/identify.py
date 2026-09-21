"""问题 1(2)：编号未知的发射源需要几架才能有效定位。

设置：接收机 P 听到 FY00、FY01（编号已知、位置无偏差）以及 m 架**编号未知**的
发射源。由于编队为 9 架均匀分布，任一未知发射源必位于 FY02~FY09 这 8 个已知
位置之一，只是接收机不知道该信号来自哪一个。

判据：穷举 m 架信号到 8 个候选位置的所有单射指派；对每个指派解后方交会，
再把解出的位置代回**全部**观测量核验。若最终只有一组 (指派, 位置) 通过核验，
则认为可实现有效定位。

注意三点：
  1. 三点后方交会本身有**等角共轭**二重歧义（见 geo_core 的说明），
     所以即使指派正确也会给出两个候选位置，二者都需保留参与核验；
  2. 观测量只有"两两连线夹角"，整体旋转不可观测，因此核验必须比较
     **夹角集合**而非绝对方位；
  3. 只有 FY00、FY01 两架已知时，可利用的夹角只有 1 个，显然不足以定位。
"""

import itertools

import numpy as np

import geo_core as G


def all_pairwise(p, srcs):
    return G.pairwise_angles(p, srcs)


def pair_index(n: int, i: int, j: int) -> int:
    """两两夹角的排列序号：顺序为 (0,1),(0,2),…,(0,n-1),(1,2),…,(n-2,n-1)。

    注意**不能**用 measured[:3] 当"前三个源的三夹角"——那对 n=3 恰好成立，
    n≥4 时第三个是 ∠(0,3) 而不是 ∠(1,2)，会让交会解算用错角度、
    连正确指派都通不过核验（表现为"发射源越多反而越不可能定位"）。
    """
    if i > j:
        i, j = j, i
    return i * (2 * n - i - 1) // 2 + (j - i - 1)


def consistent(p, srcs, measured, tol=1e-6):
    """把候选位置 p 在完整发射源集合下重算夹角，与观测量比较。"""
    if p is None:
        return False
    got = all_pairwise(p, srcs)
    if len(got) != len(measured):
        return False
    if np.any(~np.isfinite(got)):
        return False
    return bool(np.all(np.abs(got - measured) < tol))


def solve_assignment(known, cand_pos, measured_all, full_srcs):
    """给定一组指派（候选位置），解出候选接收机位置并核验全部观测量。

    measured_all: 接收机测到的、按 full_srcs 顺序排列的两两夹角。
    返回通过核验的位置列表。
    """
    srcs = known + list(cand_pos)
    # 用前三个源构造交会（known[0], known[1], 第一个未知源）
    if len(srcs) < 3:
        return []
    n_src = len(full_srcs)
    a01 = measured_all[pair_index(n_src, 0, 1)]
    a02 = measured_all[pair_index(n_src, 0, 2)]
    a12 = measured_all[pair_index(n_src, 1, 2)]
    P, n = G.resection(srcs[0], srcs[1], srcs[2], a01, a02, a12)
    if P is None:
        return []
    cands = list(P) if n > 1 else [P]
    out = []
    for q in cands:
        if consistent(q, full_srcs, measured_all):
            out.append(q)
    return out


def test(m_unknown: int, P_true, known_ids=(0, 1), exclude_known_from_circle=True,
         tol=1e-6, verbose=False):
    """返回 (通过的 (指派, 位置) 组数, 解的个数)。"""
    known = [G.ideal_pos(i) for i in known_ids]
    circle_ids = [i for i in range(1, 10) if i not in known_ids]
    rng = np.random.default_rng(abs(hash((m_unknown, tuple(np.round(P_true, 6))))) % 2**31)

    true_ids = rng.choice(circle_ids, size=m_unknown, replace=False)
    unknown_pos = [G.ideal_pos(i) for i in true_ids]
    full_srcs = known + unknown_pos
    measured = all_pairwise(P_true, full_srcs)

    pass_count = 0
    solutions = []
    for assign in itertools.permutations(circle_ids, m_unknown):
        cand = [G.ideal_pos(i) for i in assign]
        for q in solve_assignment(known, cand, measured, full_srcs):
            pass_count += 1
            solutions.append((assign, tuple(np.round(q, 6))))

    if verbose:
        print(f"  m={m_unknown}: 真指派={tuple(true_ids)}, "
              f"通过核验的组合数={pass_count}, 不同位置数={len(set(s[1] for s in solutions))}")
    return pass_count, solutions


if __name__ == "__main__":
    print("=" * 74)
    print("问题 1(2)：编号未知的发射源需要几架 —— 穷举可辨识性检验")
    print("=" * 74)
    print("(接收机取圆周上略有偏差的位置，真值随机)")
    print()
    rng = np.random.default_rng(7)
    for m in (1, 2, 3, 4):
        amb = 0
        trials = 24
        for _ in range(trials):
            th = rng.uniform(0, 2 * np.pi)
            P = (100 + rng.uniform(-12, 12)) * np.array([np.cos(th), np.sin(th)])
            pc, _ = test(m, P)
            if pc != 1:
                amb += 1
        print(f"  m = {m} 架未知发射源（共 {m+2} 架）: "
              f"{trials-amb}/{trials} 个测试位置能唯一定位"
              + ("   ← 不满足" if amb else "   ← 满足"))
