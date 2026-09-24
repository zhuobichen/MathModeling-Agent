# -*- coding: utf-8 -*-
"""三个下界：架次数、能耗、完成时间。

下界比单点结果更能说明方案质量，但**下界必须真的成立**。本文第一版给过一个
"完成时间下界 = 总航程 / 最大巡航速度"，它算出 13496 s，比本文方案自己的
10973 s 还大——下界大于可行解，说明这个式子错了：它把 8 架无人机的飞行时间
当成串行累加，而实际上是并行执行。这里改为两个都能写出来的下界：

  ① 机队工作量下界（全局）：每个服务区至少被访问一次，故总飞行时间不少于
     Σ_i 2·d(O01,S_i)/v_max，除以机队规模即得 T 的下界。弱但绝对成立。
  ② 给定架次集合的排程下界（紧）：**固定本文的路线集合**后，任何满足
     "一架无人机同一时刻只执行一个架次" 的排程都有
         T ≥ max_g ( Σ_{r∈g} t_r / n_g )
     因为同型机的架次只能由同型机执行（不同机型的载荷/体积/能耗约束不同，
     架次在构造时就已绑定机型）。它不算全局下界，但能回答"路线定了之后，
     排程还有多少余地"。
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import d_core as D
from d_q2 import World, TYPES, SITES, initial_groups, merge_pass, schedule, evaluate

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    W = World()
    g0 = initial_groups(W)
    g1 = merge_pass(W, g0, allow_types=("A",))
    plan = schedule(W, g1)
    ev = evaluate(W, plan)

    print("=" * 78)
    print("【下界 1】架次数：容量下限与精确装箱")
    print("=" * 78)
    import math
    from d_q1 import min_bins
    for g in TYPES:
        Q, V = W.types[g]["Q"], W.types[g]["vol"]
        relax = 0
        for s in SITES:
            m = sum(b["mass"] for b in W.by_site[s])
            v = sum(b["vol"] for b in W.by_site[s])
            # 逐站的二维连续松弛：至少 max(ceil(m/Q), ceil(v/V)) 个架次
            relax += max(math.ceil(m / Q - 1e-9), math.ceil(v / V - 1e-9))
        print(f"  全用 {g} 型（{Q:.0f} kg / {V:.3f} m³）：逐站二维松弛下界 {relax} 架次")
    nb30 = sum(min_bins([b["mass"] for b in W.by_site[s]],
                        [b["vol"] for b in W.by_site[s]],
                        W.types["B"]["Q"], W.types["B"]["vol"])[0]
               for s in SITES)
    nb80 = sum(min_bins([b["mass"] for b in W.by_site[s]],
                        [b["vol"] for b in W.by_site[s]],
                        W.types["C"]["Q"], W.types["C"]["vol"])[0]
               for s in SITES)
    print(f"  按 B 型容量装箱（精确 DP，逐站求和）：{nb30} 架次")
    print(f"  按 C 型容量装箱（精确 DP，逐站求和）：{nb80} 架次")
    print(f"  本文正式方案：{ev['n_sorties']} 架次")
    print("  说明：架次数不是本文的紧下界——40 架次是被机队规模与流程时间顶上去的，")
    print("        容量上按 C 型容量只要 %d 架次，但那会让完成时间恶化 41%%。" % nb80)

    print()
    print("=" * 78)
    print("【下界 2】能耗：纯水平能耗（不计爬升、不计载货导致的等效航程下降）")
    print("=" * 78)
    Dg = defaultdict(float)
    for p in plan:
        node, q = "O01", sum(b["mass"] for _, gp in p["stops"] for b in gp)
        for s, gp in p["stops"]:
            Dg[p["g"]] += W.T.seg(p["g"], node, s, q)["d"]
            q -= sum(b["mass"] for b in gp)
            node = s
        Dg[p["g"]] += W.T.seg(p["g"], node, "O01", 0.0)["d"]
    Emin = sum(Dg[g] / W.types[g]["L0"] * W.types[g]["E_use"] for g in TYPES)
    for g in TYPES:
        print(f"  {g} 型：总水平航程 {Dg[g] / 1000:>7.1f} km，"
              f"空载标准航程 {W.types[g]['L0'] / 1000:.1f} km，"
              f"该项下界 {Dg[g] / W.types[g]['L0'] * W.types[g]['E_use']:>7.2f} kWh")
    print(f"  合计下界 E_min = {Emin:.2f} kWh；本文实算 {ev['E']:.2f} kWh，"
          f"高出 {(ev['E'] / Emin - 1) * 100:.1f}%")
    print("  差额来自爬升势能、以及载货使等效航程下降，两部分都无法避免。")

    print()
    print("=" * 78)
    print("【下界 3】完成时间")
    print("=" * 78)
    # ① 全局工作量下界
    vmax = max(W.types[g]["vc"] for g in TYPES)
    Wmin = sum(2 * W.geo.dist(W.nodes["O01"]["lon"], W.nodes["O01"]["lat"],
                             W.nodes[s]["lon"], W.nodes[s]["lat"]) / vmax
               for s in SITES)
    nd = len(W.drones)
    print(f"  ① 全局：每个服务区至少被访问一次 → 总飞行时间 ≥ "
          f"Σ 2d(O01,S_i)/v_max = {Wmin:.0f} s")
    print(f"     机队 {nd} 架 → T ≥ {Wmin / nd:.0f} s（弱）")
    # ② 给定架次集合的排程下界
    work = defaultdict(float)
    for p in plan:
        work[p["g"]] += p["t"]
    nf = {g: sum(1 for d in W.drones if d["type"] == g) for g in TYPES}
    print("  ② 给定本文架次集合：同型机只能执行同型架次 → T ≥ max_g(Σt_r/n_g)")
    for g in TYPES:
        print(f"     {g} 型：{int(sum(1 for p in plan if p['g'] == g)):>2} 架次，"
              f"工作量 {work[g]:>7.0f} s，{nf[g]} 架 → {work[g] / nf[g]:>7.0f} s")
    Tlb = max(work[g] / nf[g] for g in TYPES)
    print(f"     → T ≥ {Tlb:.0f} s；本文 {ev['t_done']:.0f} s，"
          f"高出 {(ev['t_done'] / Tlb - 1) * 100:.1f}%")

    print()
    print("=" * 78)
    print("机队利用率")
    print("=" * 78)
    busy = defaultdict(float)
    for p in plan:
        busy[p["drone"]] += p["t"]
    for d in W.drones:
        b = busy.get(d["id"], 0.0)
        print(f"  {d['id']}（{d['type']}）：占用 {b:>6.0f} s / 完成时刻 {ev['t_done']:.0f} s"
              f"  →  {b / ev['t_done'] * 100:>5.1f}%")
    tot = sum(busy.values())
    print(f"  全队合计 {tot:.0f} s / ({nd}×{ev['t_done']:.0f} s) → "
          f"{tot / (nd * ev['t_done']) * 100:.1f}%")

    out = dict(E_min=Emin, E=ev["E"], T_lb_fleet=Tlb, T_lb_global=Wmin / nd,
               T=ev["t_done"], n_sorties=ev["n_sorties"], n_min_B=nb30,
               n_min_C=nb80, work_by_type=dict(work),
               dist_by_type={g: Dg[g] for g in TYPES},
               util={d["id"]: busy.get(d["id"], 0.0) / ev["t_done"]
                     for d in W.drones})
    with open(os.path.join(HERE, "out_bounds.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print("\n已写出 out_bounds.json")


def max_payload_lim(W, g, s):
    from d_q1 import max_payload
    return max_payload(g, s, W.T)


if __name__ == "__main__":
    main()
