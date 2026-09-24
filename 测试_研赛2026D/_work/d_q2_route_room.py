# -*- coding: utf-8 -*-
"""问题二：**路线到底还有多少空间**？把 7161 s 那条下界的差额拆开。

## 先把两个差分开

- **能耗**差 15.3%（90.58 对 78.54）：来自爬升势能与载货导致的等效航程下降，
  这部分是物理，消不掉。
- **时间**差 14.3%（8188 对 7161）：**这是另一回事**。7161 = Σ_r min_g t_r(g) / 8，
  它允许"每条架次都用自己最快的机型"——**忽略了机型数量约束**。所以它不是一个
  能达成的目标，而是一个松弛。

真正该问的是两件事：

  ① **机型分配的账**：在机型数量约束下，把每条架次分给可行机型、最小化
     max_g(工作量/n_g)，能到多少？（d_q2_type_ls.py 已做：8187 → 7930）
  ② **路线的账**：把两条架次并成一条多站路线，总时长是变短还是变长？
     若对**所有**两两组合都不变短，说明"逐站最少架次装箱"给出的路线集合
     在时长意义上没有可并的余地。

本脚本算 ②：枚举全部两两组合，对每种组合取**两边各自最优机型**与
**合并后最优机型**，比较"分开飞的总时长"与"并起来飞的时长"。
"""
from __future__ import annotations

import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

from d_q2 import World, TYPES, build_groups


def main():
    W = World()
    g1, p1, e1 = build_groups(W, allow_types=("A", "B"))
    n = len(g1)
    print(f"架次 {n} 条；当前完成 {e1['t_done']:.0f} s")

    def best_t(st):
        """一条架次用最快可行机型的时长。"""
        ts = W.ok_types(st)
        return min(W.route(g, st)[0] for g in ts), ts

    single = [best_t(st)[0] for st in g1]
    print(f"逐条最快时长之和 = {sum(single):.0f} s；/8 = {sum(single) / 8:.0f} s"
          f"（这就是 7161 那条松弛）")

    # **关键**：合并必须在**同一个容量档内**才叫"路线优化"。若并起来只有 C 型
    # 飞得动，那是在用稀缺机型换航程——那是表 7 已经研究过的容量档权衡
    # （按 C 容量组批：23 架次、9596 s），不是路线本身的空间。
    # 本方案按 B 型容量组批，所以只统计"合并后仍 A 或 B 飞得动"的组合。
    def merge_gain(a, b, restrict):
        sa, sb = g1[a], g1[b]
        cand = []
        for order in ((sa, sb), (sb, sa)):
            merged = [x for st in order for x in st]
            ts = [g for g in W.ok_types(merged) if g in restrict] if restrict                 else W.ok_types(merged)
            if ts:
                cand.append(min(W.route(g, merged)[0] for g in ts))
        if not cand:
            return None
        return single[a] + single[b] - min(cand)

    for label, restrict in (("同容量档内（A/B 飞得动）", ("A", "B")),
                            ("不限机型（可能落到 C 型）", None)):
        gg = []
        for a, b in itertools.combinations(range(n), 2):
            g2 = merge_gain(a, b, restrict)
            if g2 is not None:
                gg.append((g2, a, b))
        gg.sort(key=lambda x: -x[0])
        posg = [x for x in gg if x[0] > 1e-9]
        print(f"  【{label}】可合并 {len(gg)} 对，其中总时长更短 {len(posg)} 对"
              f"，最大增益 {gg[0][0]:.0f} s" if gg else f"  【{label}】无可合并")
        if posg:
            print(f"    最好的 3 个：" + "；".join(
                f"{gn:.0f} s ({'→'.join(x for x, _ in g1[a])}＋"
                f"{'→'.join(x for x, _ in g1[b])})" for gn, a, b in posg[:3]))
        if restrict is None:
            gains_any = gg
    # 反向：把多站架次拆开会不会更好？本方案是多站架次已经为 0
    multi = [k for k, st in enumerate(g1) if len(st) > 1]
    print(f"  多站架次 {len(multi)} 条（本方案为 0，故无可拆）")

    print()
    print("=" * 78)
    print("时间差的分解")
    print("=" * 78)
    print(f"  松弛下界（每条都用最快机型，忽略机型数量）  {sum(single) / 8:>8.0f} s")
    print(f"  分配账（受机型数量约束的最优分配，见 6.3.5）  {7930:>8.0f} s")
    print(f"  实际排程                                    {e1['t_done']:>8.0f} s")
    print(f"  → 松弛与实际之间差 {e1['t_done'] - sum(single) / 8:.0f} s："
          f"其中 **{7930 - sum(single) / 8:.0f} s 是机型数量约束**（不是路线）；")
    print(f"    剩下的 {e1['t_done'] - 7930:.0f} s（{(e1['t_done'] / 7930 - 1) * 100:.1f}%）"
          f"才是排程损耗（充电等待与时限）。")


if __name__ == "__main__":
    main()
