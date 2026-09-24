# -*- coding: utf-8 -*-
"""问题二：异构无人机多点多架次运输调度。

一架次可访问多个服务区；约束有货箱不可拆、载质量/装载体积、返航安全余量、
实体无人机与共享电池数量、充电周转、时限。硬约束只有两条（题目明说）：
  ① 首批保障货箱 ≤ 首批截止时间（3600 s）
  ② 医疗物资     ≤ 期望送达时间（3600 s）
其余物资的期望送达时间只用于衡量「配送及时性」（软指标）。

**关键设计**：先把每组货箱按「全部机型都装得下」的最小容量（A 型的 25 kg /
0.06 m³）分组，于是每个架次的机型可变；到排程阶段再按「哪个（无人机，电池）
最早可用」定机型。若一开始就按 C 型的 80 kg 组批，两架 C 机会被压成串行瓶颈，
首批与医疗时限全部违约——这是本文第一版踩过的坑。
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import d_core as D
from d_q1 import min_bins, max_payload

HERE = os.path.dirname(os.path.abspath(__file__))
SITES = [f"S{i:03d}" for i in range(1, 16)]
TYPES = ("A", "B", "C")


class World:
    def __init__(self):
        self.geo = D.Geo()
        self.nodes = D.load_nodes()
        self.geo.set_origin(self.nodes["O01"]["lon"], self.nodes["O01"]["lat"])
        self.types, self.drones, self.bats = D.load_drones()
        self.T = D.Transport(self.geo, self.nodes, self.types)
        self.boxes = D.load_boxes()
        self.by_site = defaultdict(list)
        for b in self.boxes:
            self.by_site[b["site"]].append(b)
        # 全机型统一容量（A 型的最小容量），保证机型可换
        self.cap_m = min(self.types[g]["Q"] for g in TYPES)
        self.cap_v = min(self.types[g]["vol"] for g in TYPES)

    def route(self, g, stops, k=None):
        tt = self.types[g]
        cur, t, E = "O01", 0.0, 0.0
        q = sum(b["mass"] for _, bs in stops for b in bs)
        for s, bs in stops:
            sg = self.T.seg(g, cur, s, q)
            t += sg["t"]
            E += sg["E"]
            q -= sum(b["mass"] for b in bs)
            cur = s
        sg = self.T.seg(g, cur, "O01", 0.0)
        t += sg["t"]
        E += sg["E"]
        k = sum(len(bs) for _, bs in stops) if k is None else k
        t += (tt["t_prep"] + k * tt["t_load"]
              + sum(tt["t_hand0"] + len(bs) * tt["t_hand1"] for _, bs in stops))
        return t, E

    def feasible(self, g, stops):
        tt = self.types[g]
        if sum(b["mass"] for _, bs in stops for b in bs) > tt["Q"] + 1e-9:
            return False
        if sum(b["vol"] for _, bs in stops for b in bs) > tt["vol"] + 1e-9:
            return False
        _, E = self.route(g, stops)
        return E <= (1 - tt["rho"]) * tt["E_use"] + 1e-9

    def ok_types(self, stops):
        return [g for g in TYPES if self.feasible(g, stops)]

    def urgency(self, stops):
        """排序键：首批优先、医疗次之、再按到期时间，同紧迫度时少箱优先。"""
        bs = _boxes_of(stops)
        return (0 if any(b["first"] for b in bs) else 1,
                0 if any(b["kind"] == "医疗物资" for b in bs) else 1,
                min(b["t_due"] for b in bs), len(bs))


def _boxes_of(stops):
    return [b for _, gp in stops for b in gp]


# ══════════════════════════════════════════════════════════
def initial_groups(W):
    """逐站组批：**先把「时限箱」（首批 + 医疗）单独装一箱**，再装其余。

    这一步不能省。若直接对全部货箱做最少架次装箱，首批箱会被打散到后面的
    架次里——实测 15 个服务区里有 8 个的首批箱落到了第二、三轮，交付时刻
    4863~5965 s，硬约束全部违约。时限箱单独成组后它们必然落在第一、二轮。
    """
    out = []
    for s in [x for x in SITES if W.by_site[x]]:
        bs = W.by_site[s]
        urgent = [b for b in bs if b["first"] or b["kind"] == "医疗物资"]
        rest = [b for b in bs if not (b["first"] or b["kind"] == "医疗物资")]
        if urgent:
            if sum(b["mass"] for b in urgent) > W.cap_m + 1e-9 or \
                    sum(b["vol"] for b in urgent) > W.cap_v + 1e-9:
                n, masks = min_bins([b["mass"] for b in urgent],
                                    [b["vol"] for b in urgent],
                                    W.cap_m, W.cap_v)
                for m in masks:
                    out.append([(s, [urgent[i] for i in range(len(urgent))
                                     if m >> i & 1])])
            else:
                out.append([(s, urgent)])
        if rest:
            n, masks = min_bins([b["mass"] for b in rest],
                                [b["vol"] for b in rest], W.cap_m, W.cap_v)
            assert n < 10 ** 6, f"{s} 在最小容量下装不下"
            for m in masks:
                out.append([(s, [rest[i] for i in range(len(rest))
                                 if m >> i & 1])])
    return out


def merge_pass(W, stops_list, max_stops=3, allow_types=TYPES):
    """把两个架次并成一条多站路线：省一次往返就并，且任意机型可行。"""
    changed = True
    while changed:
        changed = False
        best = None
        for i in range(len(stops_list)):
            for j in range(len(stops_list)):
                if i == j:
                    continue
                a, b = stops_list[i], stops_list[j]
                if len(a) + len(b) > max_stops:
                    continue
                if set(s for s, _ in a) & set(s for s, _ in b):
                    continue
                # 含时限箱的架次不合并：合并后航线变长，容易把时限顶破
                if any(b2["first"] or b2["kind"] == "医疗物资"
                       for b2 in _boxes_of(a) + _boxes_of(b)):
                    continue
                cand = a + b
                ts = [g for g in W.ok_types(cand) if g in allow_types]
                if not ts:
                    continue
                g = min(ts, key=lambda x: W.route(x, cand)[1])
                ta, _ = W.route(g, a)
                tb, _ = W.route(g, b)
                tc, _ = W.route(g, cand)
                gain = (ta + tb) - tc
                if gain <= 0:
                    continue
                if best is None or gain > best[0]:
                    best = (gain, i, j, cand)
        if best:
            gain, i, j, cand = best
            stops_list = [x for k, x in enumerate(stops_list) if k not in (i, j)]
            stops_list.append(cand)
            changed = True
    return stops_list


# ══════════════════════════════════════════════════════════
def schedule(W, stops_list, deadline_aware=True):
    """按（首批 → 医疗 → 到期时间）排序，逐个派给最早可用的（无人机，电池）。

    deadline_aware：有时限的架次只在「能按时送达」的候选里挑；挑不到才退回
    最早可用的那个。没有这一条时，调度会把 3 个站合并的 C 机长航线堆在
    0 时刻，而全队只有 2 架 C 机——A/B 机闲着，首批却全部超时。
    """
    pool = {g: [dict(id=f"{g}{i+1:02d}", free=0.0) for i in range(W.bats[g]["n"])]
            for g in TYPES}
    drones = {d["id"]: dict(id=d["id"], type=d["type"], free=0.0)
              for d in W.drones}

    order = sorted(range(len(stops_list)),
                   key=lambda i: (W.urgency(stops_list[i]),
                                  sum(b["mass"] for _, g in stops_list[i]
                                      for b in g)))
    plan = [None] * len(stops_list)
    for i in order:
        st = stops_list[i]
        due = min(b["t_due"] for b in _boxes_of(st))
        hard = any(b["first"] or b["kind"] == "医疗物资" for b in _boxes_of(st))
        cand = []
        for g in W.ok_types(st):
            t, E = W.route(g, st)
            for did, d in drones.items():
                if d["type"] != g:
                    continue
                for bt in pool[g]:
                    start = max(d["free"], bt["free"])
                    cand.append((start, E, did, bt, g, t))
        if not cand:
            raise RuntimeError(f"架次 {st} 无可用机型/无人机")
        if deadline_aware and hard:
            # 交付不晚于时限：末站时刻 ≈ start + t（保守）
            good = [c for c in cand if c[0] + c[5] <= due + 1e-9]
            if good:
                cand = good
        start, E, did, bt, g, t = min(cand, key=lambda x: (x[0], x[1]))
        end = start + t
        soc = max(1.0 - E / W.types[g]["E_use"], 0.0)
        chg = D.charge_time(soc, W.bats[g]["T_full"])
        # 逐站交付时刻
        deliv, node = {}, "O01"
        cur = start + W.types[g]["t_prep"] + sum(len(b2) for _, g2 in st
                                                for b2 in g2) * W.types[g]["t_load"]
        q = sum(b2["mass"] for _, g2 in st for b2 in g2)
        for s, gp in st:
            cur += W.T.seg(g, node, s, q)["t"]
            cur += W.types[g]["t_hand0"] + len(gp) * W.types[g]["t_hand1"]
            for b2 in gp:
                deliv[b2["box"]] = cur
            q -= sum(b2["mass"] for b2 in gp)
            node = s
        plan[i] = dict(stops=st, g=g, drone=did, battery=bt["id"], start=start,
                       end=end, t=t, E=E, soc=soc, chg=chg,
                       boxes=[b2["box"] for _, gp in st for b2 in gp],
                       deliv=deliv)
        drones[did]["free"] = end
        bt["free"] = end + chg
    return plan


def evaluate(W, plan):
    deliv, E = {}, 0.0
    for p in plan:
        deliv.update(p["deliv"])
        E += p["E"]
    vf = vm = nf = nm = 0
    late = 0.0
    late_n = 0
    for b in W.boxes:
        t = deliv.get(b["box"])
        if t is None:
            raise RuntimeError(f"货箱 {b['box']} 未交付")
        if b["first"]:
            nf += 1
            vf += t > b["t_first"] + 1e-9
        if b["kind"] == "医疗物资":
            nm += 1
            vm += t > b["t_due"] + 1e-9
        if t > b["t_due"] + 1e-9:
            late += t - b["t_due"]
            late_n += 1
    return dict(n_sorties=len(plan),
                t_done=max(p["end"] for p in plan),
                t_last_deliv=max(deliv.values()),
                E=E, late=late, late_n=late_n,
                viol_first=vf, n_first=nf, viol_med=vm, n_med=nm,
                deliv=deliv, by_type={g: sum(1 for p in plan if p["g"] == g)
                                      for g in TYPES})


def main():
    W = World()
    print("=" * 78)
    print("第一步：逐站按统一最小容量（%g kg / %g m³）组批"
          % (W.cap_m, W.cap_v))
    print("=" * 78)
    g0 = initial_groups(W)
    p0 = schedule(W, g0)
    e0 = evaluate(W, p0)
    print(f"  {e0['n_sorties']} 架次 | 完成 {e0['t_done']:.0f}s | "
          f"能耗 {e0['E']:.2f} kWh | 首批违约 {e0['viol_first']}/{e0['n_first']}"
          f" | 医疗违约 {e0['viol_med']}/{e0['n_med']}")

    print()
    print("=" * 78)
    print("第二步：跨站合并（按允许的机型分档，见权衡表）")
    print("=" * 78)
    variants = {}
    for name, allow in (("不合并", ()), ("只并 A 型可行的", ("A",)),
                        ("并 A/B 可行的", ("A", "B")), ("不限机型", TYPES)):
        gg = g0 if not allow else merge_pass(W, g0, allow_types=allow)
        pp = schedule(W, gg)
        ee = evaluate(W, pp)
        variants[name] = (gg, pp, ee)
        print(f"  {name:<14} {ee['n_sorties']:>3} 架次 | "
              f"完成 {ee['t_done']:>6.0f}s | 能耗 {ee['E']:>7.2f} kWh | "
              f"首批违约 {ee['viol_first']} | 医疗违约 {ee['viol_med']} | "
              f"超时箱 {ee['late_n']}")
    # 正式方案：硬约束全过的前提下，先比完成时间、再比架次数
    okv = {k: v for k, v in variants.items()
           if v[2]["viol_first"] == 0 and v[2]["viol_med"] == 0}
    pick = min(okv, key=lambda k: (okv[k][2]["t_done"], okv[k][2]["n_sorties"]))
    print(f"  → 正式方案取「{pick}」")
    g1, p1, e1 = variants[pick]

    print()
    print("=" * 78)
    print("第三步：排程明细")
    print("=" * 78)
    print(f"{'#':>3}{'无人机':>7}{'型':>3}{'电池':>6}{'开始':>7}{'返回':>7}"
          f"{'时长':>7}{'能耗':>7}{'末箱交付':>10}  路线")
    for i, p in enumerate(sorted(p1, key=lambda x: (x["start"], x["drone"])), 1):
        sites = "→".join(["O01"] + [s for s, _ in p["stops"]] + ["O01"])
        print(f"{i:>3}{p['drone']:>7}{p['g']:>3}{p['battery']:>6}"
              f"{p['start']:>7.0f}{p['end']:>7.0f}{p['t']:>7.0f}{p['E']:>7.3f}"
              f"{max(p['deliv'].values()):>10.0f}  {sites}")
    print(f"\n  机型分布 {e1['by_type']}")

    res = dict(summary={k: v for k, v in e1.items()
                        if k not in ("deliv",)},
               base={k: v for k, v in e0.items() if k not in ("deliv",)},
               plan=[dict(drone=p["drone"], g=p["g"], battery=p["battery"],
                          start=p["start"], end=p["end"], t=p["t"], E=p["E"],
                          sites=[s for s, _ in p["stops"]],
                          boxes=p["boxes"], deliv=p["deliv"]) for p in p1],
               deliv=e1["deliv"])
    with open(os.path.join(HERE, "out_q2.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    print("已写出 out_q2.json")
    return res


if __name__ == "__main__":
    main()
