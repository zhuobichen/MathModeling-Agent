# -*- coding: utf-8 -*-
"""用**修正后的评估**（内层目标与外层一致）精算四个候选分区。

不跑搜索，只算数：表 17 的四列 + K=2 的全局最优。
"""
from __future__ import annotations
import json, os, sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import d_core as D
from d_q2 import World, TYPES, build_groups
import d_q3 as Q3
import d_q4 as Q4
import d_relay

HERE = os.path.dirname(os.path.abspath(__file__))
INV = Q4.INV

CASES = {
    "K=2 地理均衡型": [["S001","S002","S004","S005","S008","S009","S010","S011","S012","S013","S014"],
                     ["S003","S006","S007","S015"]],
    "K=2 资源缺口型（全局最优）": [
        ["S001","S004","S006","S010","S011","S012","S013","S014"],
        ["S002","S003","S005","S007","S008","S009","S015"]],
    "K=3 地理均衡型": [["S001","S002","S005","S009","S010","S011","S012","S013","S014"],
                     ["S003","S006","S007","S015"], ["S004","S008"]],
    "K=3 资源缺口型": [["S001","S005","S008","S009","S010","S011","S012","S014"],
                     ["S002","S003","S007","S015"], ["S004","S006","S013"]],
}

def main():
    W = World()
    rt, W.relays, W.rbats = D.load_relays()
    C = D.Comms(W.geo, W.nodes, D.load_comm())
    g1, p1, e1 = build_groups(W, allow_types=("A", "B"))
    q3 = json.load(open(os.path.join(HERE, "out_q3.json"), encoding="utf-8"))
    st = q3["stations"]
    print(f"运输 {len(p1)} 架次 / 完成 {e1['t_done']:.0f} s（不分区基准）\n")
    out = {}
    for name, groups in CASES.items():
        gid = {s: i for i, gp in enumerate(groups) for s in gp}
        idxs = [[k for k, x in enumerate(p1) if gid[x["stops"][0][0]] == i]
                for i in range(len(groups))]
        subs = [[p1[k]["stops"] for k in gg] for gg in idxs]
        # 跨组架次（应为 0）
        cross = sum(1 for x in p1 if len({gid[s] for s, _ in x["stops"]}) > 1)
        alloc = Q4.best_allocation(W, subs, t_shift=0.0)
        nd = {g: sum(alloc["per_group"][j]["cap"][g] for j in range(len(subs))) for g in TYPES}
        gap_d = sum(max(0, nd[g] - INV["drone"][g]) for g in TYPES)
        # 中继：各组需要哪些站位
        smask = {}
        for i, x in enumerate(p1):
            m = 0
            for (t0, t1, smp) in Q3.gaps_of(W, C, x):
                for j, s in enumerate(st):
                    if d_relay.gap_covered(W, C, smp, tuple(s["point"])):
                        m |= 1 << j; break
            smask[i] = m
        rneed = []
        for gg in idxs:
            m = 0
            for k in gg:
                m |= smask[k]
            rneed.append(bin(m).count("1"))
        rel = sum(rneed)
        ns = [len(s) for s in subs]
        print(f"{name}")
        print(f"  分区 {[len(g) for g in groups]} 站，架次 {ns}（极差 {max(ns)-min(ns)}），跨组 {cross}")
        print(f"  机型 {nd} 缺口 {gap_d} | 电池 {alloc['bat_tot']} 缺口 {alloc['gap_bat']}")
        print(f"  各组中继 {rneed} 合计 {rel} 架（库存 2）缺口 {max(0, rel-2)}")
        print(f"  违反 {alloc['viol']:.0f} | 各组完成 "
              f"{[round(x['t']) for x in alloc['per_group']]}")
        print(f"  目标 (违反, 机型缺口, 电池缺口, 中继缺口, 极差, 完成) = "
              f"({int(alloc['viol'])}, {gap_d}, {sum(alloc['gap_bat'].values())}, "
              f"{max(0, rel-2)}, {max(ns)-min(ns)}, {alloc['tmax']:.0f})\n")
        out[name] = dict(ns=ns, spread=max(ns)-min(ns), cross=cross,
                         gap_d=gap_d, bat=alloc["bat_tot"],
                         gap_bat=alloc["gap_bat"], rneed=rneed, relay=rel,
                         viol=int(alloc["viol"]), tmax=alloc["tmax"],
                         per=[round(x["t"]) for x in alloc["per_group"]],
                         cap=[alloc["per_group"][j]["cap"] for j in range(len(subs))])
    with open(os.path.join(HERE, "out_q4_eval4.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print("已写出 out_q4_eval4.json")

if __name__ == "__main__":
    main()
