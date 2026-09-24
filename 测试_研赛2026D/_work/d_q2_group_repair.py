# -*- coding: utf-8 -*-
"""组批：**先按最少架次装，只拆真正会违约的站**（论文 6.2 节的组批策略）。

## 为什么

`initial_groups` 无条件对**每个**站先把「首批 + 医疗」单独装一箱。这是为了让
时限箱必然落在前两轮，但它对**所有站**都付了代价：40 架次对"按 B 型容量逐站
精确装箱"的 35 架次，多出 5 个架次、完成时间多 1333 s、能耗多 11.9 kWh
（见 `d_q2_group_opt.py` 的策略 B）。

但逐个诊断发现：**真正会违约的站只有个别几个**。于是正确做法不是"全体预防"，
而是"先按最少架次装 → 跑一遍排程 → 只把真正违约的站拆开 → 再跑一遍"。

这是一个**贪心修复**：每轮只修当前违约的站，直到零违约。最坏情况退化成现状
策略（每站都拆），所以**不会比现状差**；而实测只拆了极少数站。

## 判据

违约的判定与 `evaluate` 完全一致（首批看 `t_first`、医疗看 `t_due`），
不另立一套口径——这一点很关键：组批策略与排程器必须用同一个时限定义。
"""
from __future__ import annotations

import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from d_q2 import (World, TYPES, SITES, initial_groups, merge_pass, schedule,
                  evaluate)
from d_q1 import min_bins
from d_q2_group_opt import pack_all


def site_of_sorts(groups, site):
    return [g for g in groups if any(s == site for s, _ in g)]


def split_site(W, groups, site, cap_m, cap_v):
    """把某站的架次全部换成"时限箱单独成组"的装法。"""
    bs = W.by_site[site]
    urgent = [b for b in bs if b["first"] or b["kind"] == "医疗物资"]
    rest = [b for b in bs if not (b["first"] or b["kind"] == "医疗物资")]
    new = [g for g in groups if not any(s == site for s, _ in g)]
    for blk in (urgent, rest):
        if not blk:
            continue
        n, masks = min_bins([b["mass"] for b in blk], [b["vol"] for b in blk],
                            cap_m, cap_v)
        for m in masks:
            new.append([(site, [blk[i] for i in range(len(blk))
                                if m >> i & 1])])
    return new


def repair(W, allow_types=("A",), verbose=True):
    """最少架次装箱 → 贪心修复违约站。返回 (groups, plan, ev, 拆过的站)。"""
    cm, cv = W.cap_m, W.cap_v
    groups = pack_all(W, cm, cv)
    split = set()
    hist = []
    for it in range(20):
        g = merge_pass(W, groups, allow_types=allow_types)
        p = schedule(W, g)
        e = evaluate(W, p)
        hist.append(dict(iter=it, n_sortie=e["n_sorties"], t=e["t_done"],
                         E=e["E"], vf=e["viol_first"], vm=e["viol_med"]))
        if e["viol_first"] == 0 and e["viol_med"] == 0:
            if verbose:
                print(f"  第 {it} 轮：{e['n_sorties']} 架次 / 完成 "
                      f"{e['t_done']:.0f}s / 能耗 {e['E']:.2f} kWh | "
                      f"首批 0、医疗 0 违反 → 收敛")
            return groups, p, e, split, hist
        # 找出违约的站
        bad = set()
        for b in W.boxes:
            t = e["deliv"][b["box"]]
            if b["first"] and t > b["t_first"] + 1e-9:
                bad.add(b["site"])
            elif b["kind"] == "医疗物资" and t > b["t_due"] + 1e-9:
                bad.add(b["site"])
        bad -= split
        if verbose:
            print(f"  第 {it} 轮：{e['n_sorties']} 架次 / 完成 {e['t_done']:.0f}s"
                  f" | 首批违约 {e['viol_first']}、医疗违约 {e['viol_med']}"
                  f" → 拆开 {sorted(bad)}")
        if not bad:
            return groups, p, e, split, hist
        for s in bad:
            groups = split_site(W, groups, s, cm, cv)
        split |= bad
    raise RuntimeError("修复不收敛")


def main():
    W = World()
    print("=" * 78)
    print("组批策略：先按最少架次装，只拆真正违约的站")
    print("=" * 78)
    groups, p, e, split, hist = repair(W)

    # 对照：现状（每站都拆）
    ga = merge_pass(W, initial_groups(W), allow_types=("A",))
    pa = schedule(W, ga)
    ea = evaluate(W, pa)
    # 对照：全部一次性装箱（不修复）
    gb = merge_pass(W, pack_all(W, W.cap_m, W.cap_v), allow_types=("A",))
    pb = schedule(W, gb)
    eb = evaluate(W, pb)

    print()
    print(f"{'策略':<34}{'架次':>6}{'完成/s':>9}{'能耗/kWh':>10}"
          f"{'首批违约':>9}{'医疗违约':>9}")
    for nm, ev in (("A 每站都拆（现状）", ea), ("B 全部一次性装箱（不修复）", eb),
                   ("C 只拆真正违约的站（本方案）", e)):
        print(f"{nm:<34}{ev['n_sorties']:>6}{ev['t_done']:>9.0f}{ev['E']:>10.2f}"
              f"{ev['viol_first']:>9}{ev['viol_med']:>9}")
    print()
    print(f"被拆开的站：{sorted(split) if split else '（一个都不需要）'}"
          f"，共 {len(split)} 个")

    out = dict(repair=[dict(k, **{}) for k in hist],
               best=dict(n_sortie=e["n_sorties"], t=e["t_done"], E=e["E"],
                         vf=e["viol_first"], vm=e["viol_med"],
                         by_type=e["by_type"]),
               split=sorted(split),
               base_A=dict(n_sortie=ea["n_sorties"], t=ea["t_done"], E=ea["E"]),
               base_B=dict(n_sortie=eb["n_sorties"], t=eb["t_done"], E=eb["E"],
                           vf=eb["viol_first"]))
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "out_q2_group_repair.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print("\n已写出 out_q2_group_repair.json")


if __name__ == "__main__":
    main()
