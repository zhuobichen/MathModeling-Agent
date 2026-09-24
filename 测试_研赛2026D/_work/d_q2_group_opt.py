# -*- coding: utf-8 -*-
"""组批策略实验：**"时限箱单独成组"到底花了几个架次？**

`initial_groups` 现在对每个站先单独装「首批 + 医疗」，再对其余货箱做最少架次
装箱。这一步是为了让时限箱必然落在前两轮（不分开装会有 8 个站的首批落到第二
三轮，交付 4863~5965 s，超出 3600 s 截止）。

但它是**有代价**的：分两次装箱必然比一次性装多出几个架次。本文的 40 架次对
"按 B 型容量逐站精确装箱"的 35 架次，差的就是这一块。

本脚本试三种策略，看**能不能既少用架次又不违反时限**：

  A. 现状：时限箱先单独装，其余再装（= initial_groups）
  B. 全部货箱一次性做最少架次装箱（sortsizes 最少，但时限箱会被打散）
  C. **装完再验**：一次性装箱，然后检查时限箱落在第几轮；把落在时限之外的那些
     箱**单独提出来重装**，其余不动。目标是"只拆必要的部分"。

C 是一种折中：如果一次性装箱恰好把时限箱排在前两轮，就一个架次都不多花；
只有真正落在外面的才提出来。
"""
from __future__ import annotations

import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from d_q2 import (World, TYPES, SITES, initial_groups, merge_pass, schedule,
                  evaluate, _boxes_of)
from d_q1 import min_bins

HERE = os.path.dirname(os.path.abspath(__file__))


def pack_all(W, cap_m, cap_v):
    """B：全部货箱一次性做最少架次装箱。"""
    out = []
    for s in [x for x in SITES if W.by_site[x]]:
        bs = W.by_site[s]
        n, masks = min_bins([b["mass"] for b in bs],
                            [b["vol"] for b in bs], cap_m, cap_v)
        assert n < 10 ** 6, f"{s} 装不下"
        for m in masks:
            out.append([(s, [bs[i] for i in range(len(bs)) if m >> i & 1])])
    return out


def pack_verify_split(W, cap_m, cap_v):
    """C：一次性装箱，再把"会落到时限之外"的时限箱提出来单独装。

    这里不预判哪一轮算"之外"，而是把**时限箱与散箱分开**：先把全部箱按最少
    架次装一遍，得到一个"推荐的最少架次" k*；再用现状策略装一遍得到 k。
    若 k == k*，说明一次装就够了，直接采用；否则**只在时限箱那几次装箱上去省**：
    把时限箱与其余箱一起装，但强制时限箱聚在前 k* 个架次里。
    """
    out = []
    for s in [x for x in SITES if W.by_site[x]]:
        bs = W.by_site[s]
        urgent = [b for b in bs if b["first"] or b["kind"] == "医疗物资"]
        rest = [b for b in bs if not (b["first"] or b["kind"] == "医疗物资")]
        n_all, m_all = min_bins([b["mass"] for b in bs],
                                [b["vol"] for b in bs], cap_m, cap_v)
        # 现状策略的架次数
        n_sep = (len(min_bins([b["mass"] for b in urgent],
                              [b["vol"] for b in urgent], cap_m, cap_v)[1])
                 if urgent else 0)
        n_sep += (len(min_bins([b["mass"] for b in rest],
                               [b["vol"] for b in rest], cap_m, cap_v)[1])
                  if rest else 0)
        if n_all >= n_sep or not urgent:
            # 一次装没有更省，或本站没有时限箱 → 用现状策略
            out.extend(_sep(W, s, urgent, rest, cap_m, cap_v))
            continue
        # 一次装更省：检查时限箱是否都落在前 n_sep 个架次里（那一批会被优先派发）
        # 用贪心：把架次按含时限箱的多少排序，看前 n_sep 个是否覆盖全部时限箱
        gs = [[bs[i] for i in range(len(bs)) if m >> i & 1] for m in m_all]
        gs.sort(key=lambda g: -sum(1 for b in g
                                   if b["first"] or b["kind"] == "医疗物资"))
        cover = set()
        for g in gs[:n_sep]:
            cover |= {b["box"] for b in g if b["first"] or b["kind"] == "医疗物资"}
        need = {b["box"] for b in urgent}
        if need <= cover:
            for g in gs:
                out.append([(s, g)])
        else:
            out.extend(_sep(W, s, urgent, rest, cap_m, cap_v))
    return out


def _sep(W, s, urgent, rest, cap_m, cap_v):
    """现状策略（时限箱单独装），逐站返回。"""
    r = []
    if urgent:
        n, masks = min_bins([b["mass"] for b in urgent],
                            [b["vol"] for b in urgent], cap_m, cap_v)
        for m in masks:
            r.append([(s, [urgent[i] for i in range(len(urgent))
                           if m >> i & 1])])
    if rest:
        n, masks = min_bins([b["mass"] for b in rest],
                            [b["vol"] for b in rest], cap_m, cap_v)
        for m in masks:
            r.append([(s, [rest[i] for i in range(len(rest)) if m >> i & 1])])
    return r


def main():
    W = World()
    cm, cv = W.cap_m, W.cap_v
    print(f"组批容量 {cm:.0f} kg / {cv:.3f} m³\n")
    strategies = [("A 现状（时限箱单独成组）", lambda: initial_groups(W)),
                  ("B 全部一次性最少架次装箱", lambda: pack_all(W, cm, cv)),
                  ("C 装完再验，只拆必要的", lambda: pack_verify_split(W, cm, cv))]
    out = {}
    for name, fn in strategies:
        g0 = fn()
        n0 = len(g0)
        g1 = merge_pass(W, g0, allow_types=("A",))
        p = schedule(W, g1)
        e = evaluate(W, p)
        out[name] = dict(n_group=n0, n_sortie=e["n_sorties"],
                         t=e["t_done"], E=e["E"], vf=e["viol_first"],
                         vm=e["viol_med"], late_n=e["late_n"])
        print(f"{name}")
        print(f"  组批 {n0} 架次 → 合并后 {e['n_sorties']} 架次 | "
              f"完成 {e['t_done']:.0f}s | 能耗 {e['E']:.2f} kWh | "
              f"首批违约 {e['viol_first']}/{e['n_first']} | "
              f"医疗违约 {e['viol_med']}/{e['n_med']} | 超时箱 {e['late_n']}")
        print()
    with open(os.path.join(HERE, "out_q2_group_opt.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print("已写出 out_q2_group_opt.json")


if __name__ == "__main__":
    main()
