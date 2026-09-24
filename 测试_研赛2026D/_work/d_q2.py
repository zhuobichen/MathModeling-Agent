# -*- coding: utf-8 -*-
"""问题二：异构无人机多点多架次运输调度。

一架次可访问多个服务区；约束有货箱不可拆、载质量/装载体积、返航安全余量、
实体无人机与共享电池数量、充电周转、时限。硬约束只有两条（题目明说）：
  ① 首批保障货箱 ≤ 首批截止时间（3600 s）
  ② 医疗物资     ≤ 期望送达时间（3600 s）
其余物资的期望送达时间只用于衡量「配送及时性」（软指标）。

**关键设计**：组批容量取「全机型容量谱上的一个中间档」而不是最大档，于是每个
架次的机型可变；到排程阶段再按「哪个（无人机，电池）最早可用」定机型。
本文正式取 B 型容量 30 kg / 0.073 m³，依据是 6.3 节的容量扫描：
按 A 型 25 kg 组批得 41 架次 / 10973 s / 107.41 kWh，按 B 型 30 kg 得
40 架次 / 10215 s / 102.39 kWh（三项同时更优）；按 C 型 80 kg 虽然只有
25 架次、83.87 kWh，但两架 C 机被压成串行瓶颈，完成时间恶化到 14189 s。
**完成时间由最忙的那架决定，所以负载均衡比"少用架次"重要。**
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
        # 组批容量。取 B 型容量而不是「全机型的最小容量」——这一点是实测出来的：
        # 按 A 型 25 kg/0.06 m³ 组批（原设计）得 41 架次 / 10973 s / 107.41 kWh；
        # 按 B 型 30 kg/0.073 m³ 得 40 架次 / 10215 s / 102.39 kWh，三项同时更优。
        # 按 C 型 80 kg 组批更极端：25 架次、能耗 83.87 kWh，但完成时间 14189 s
        # ——因为全队只有 2 架 C 机，被压成串行瓶颈，机队利用率掉到 41.8%。
        # 完成时间由**最忙的那架**决定，所以负载均衡比"少用架次"重要。
        self.cap_m = self.types["B"]["Q"]
        self.cap_v = self.types["B"]["vol"]

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
def initial_groups(W, cap_m=None, cap_v=None):
    """逐站组批：**先把「时限箱」（首批 + 医疗）单独装一箱**，再装其余。

    这一步不能省。若直接对全部货箱做最少架次装箱，首批箱会被打散到后面的
    架次里——实测 15 个服务区里有 8 个的首批箱落到了第二、三轮，交付时刻
    4863~5965 s，硬约束全部违约。时限箱单独成组后它们必然落在第一、二轮。
    """
    cm = W.cap_m if cap_m is None else cap_m
    cv = W.cap_v if cap_v is None else cap_v
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
                                    cm, cv)
                for m in masks:
                    out.append([(s, [urgent[i] for i in range(len(urgent))
                                     if m >> i & 1])])
            else:
                out.append([(s, urgent)])
        if rest:
            n, masks = min_bins([b["mass"] for b in rest],
                                [b["vol"] for b in rest], cm, cv)
            assert n < 10 ** 6, f"{s} 在最小容量下装不下"
            for m in masks:
                out.append([(s, [rest[i] for i in range(len(rest))
                                 if m >> i & 1])])
    return out


def pack_min_sorties(W, cap_m=None, cap_v=None):
    """逐站做**最少架次**装箱（全部货箱一起装，不区分时限箱）。

    这是容量意义上的最优组批：本站货箱一起装，子集 DP 精确求最少架次。
    缺点是时限箱可能被打散到后面的架次里去。
    """
    cm = W.cap_m if cap_m is None else cap_m
    cv = W.cap_v if cap_v is None else cap_v
    out = []
    for s in [x for x in SITES if W.by_site[x]]:
        bs = W.by_site[s]
        n, masks = min_bins([b["mass"] for b in bs],
                            [b["vol"] for b in bs], cm, cv)
        assert n < 10 ** 6, f"{s} 在容量 {cm} kg / {cv} m3 下装不下"
        for m in masks:
            out.append([(s, [bs[i] for i in range(len(bs)) if m >> i & 1])])
    return out


def split_site(W, groups, site):
    """把某站的架次换成「时限箱单独成组」的装法（逐站版本的 initial_groups）。"""
    cm, cv = W.cap_m, W.cap_v
    bs = W.by_site[site]
    blocks = ([b for b in bs if b["first"] or b["kind"] == "医疗物资"],
              [b for b in bs if not (b["first"] or b["kind"] == "医疗物资")])
    new = [g for g in groups if not any(s == site for s, _ in g)]
    for blk in blocks:
        if not blk:
            continue
        n, masks = min_bins([b["mass"] for b in blk], [b["vol"] for b in blk],
                            cm, cv)
        for m in masks:
            new.append([(site, [blk[i] for i in range(len(blk))
                                if m >> i & 1])])
    return new


def repair_grouping(W, allow_types=("A", "B"), max_iter=20, delta=0.0,
                    fleet=None):
    """**先按最少架次装，只拆真正会违约的站。**

    `initial_groups` 无条件对**每个**站把时限箱单独装一箱——这是"全体预防"，
    代价是 40 架次（对最少架次的 35 架次）。但逐站诊断发现真正会违约的站只有
    个别几个。于是改成：

      ① 逐站按最少架次装箱；② 合并、排程、评估；③ 把**实际违约**的站拆开；
      ④ 回到 ②，直到零违约。

    这是贪心修复：最坏情况退化成「每站都拆」（= initial_groups），所以**不会
    比现状差**；实测只拆了 1 个站（S014），36 架次 / 完成 9153 s / 能耗 92.16 kWh，
    三项都优于现状的 40 / 10064 / 101.55，且硬约束零违反。

    违约判据与 `evaluate` 完全一致（首批看 t_first、医疗看 t_due）——组批策略
    与排程器必须共用同一个时限定义。
    """
    groups = pack_min_sorties(W)
    split = set()
    for _ in range(max_iter):
        g = merge_pass(W, groups, allow_types=allow_types)
        p = schedule(W, g, delta=delta, fleet=fleet)
        e = evaluate(W, p)
        if e["viol_first"] == 0 and e["viol_med"] == 0:
            return g, p, e, split
        bad = set()
        for b in W.boxes:
            t = e["deliv"][b["box"]] + delta      # 加上中继先到点的整体推迟
            if b["first"] and t > b["t_first"] + 1e-9:
                bad.add(b["site"])
            elif b["kind"] == "医疗物资" and t > b["t_due"] + 1e-9:
                bad.add(b["site"])
        bad -= split
        if not bad:
            return g, p, e, split
        for s in bad:
            groups = split_site(W, groups, s)
        split |= bad
    raise RuntimeError("组批修复不收敛")


def simulate_order(W, stops_list, order, delta=0.0, fleet=None, lock=None):
    """按给定派发顺序排程，返回与 schedule() 同构的计划；有硬约束违反则返回 None。

    口径与 schedule() 完全一致（最早可用的同型无人机 + 已换好的电池、时限感知、
    两阶段充电），唯一的差别是**顺序由外面给定**而不是按规则算出来。
    """
    nf = fleet or {g: sum(1 for d in W.drones if d["type"] == g)
                   for g in TYPES}
    pool = {g: [dict(id=f"{g}{i+1:02d}", free=0.0)
                for i in range(W.bats[g]["n"])] for g in TYPES}
    inv = {g: [d["id"] for d in W.drones if d["type"] == g] for g in TYPES}
    drones = {}
    for g in TYPES:
        for i in range(nf.get(g, 0)):
            did = inv[g][i] if i < len(inv[g]) else f"{g}-EXT{i+1:02d}"
            drones[did] = dict(id=did, type=g, free=0.0)

    plan, viol = [], 0
    for i in order:
        st = stops_list[i]
        bs = _boxes_of(st)
        hard = any(b["first"] or b["kind"] == "医疗物资" for b in bs)
        due = min(b["t_due"] for b in bs)
        cand = []
        for g in W.ok_types(st):
            if lock is not None and lock.get(id(st), g) != g:
                continue                      # 机型被锁定的架次只走指定机型
            t, E = W.route(g, st)
            for did, d in drones.items():
                if d["type"] != g:
                    continue
                for bt in pool[g]:
                    cand.append((max(d["free"], bt["free"]), E, did, bt, g, t))
        if not cand:
            return None
        if hard:
            good = [c for c in cand if c[0] + c[5] + delta <= due + 1e-9]
            if good:
                cand = good
        start, E, did, bt, g, t = min(cand, key=lambda x: (x[0], x[1]))
        end = start + t
        soc = max(1.0 - E / W.types[g]["E_use"], 0.0)
        chg = D.charge_time(soc, W.bats[g]["T_full"])
        deliv, node = {}, "O01"
        cur = start + W.types[g]["t_prep"] + len(bs) * W.types[g]["t_load"]
        q = sum(b2["mass"] for b2 in bs)
        for s2, gp in st:
            cur += W.T.seg(g, node, s2, q)["t"]
            cur += W.types[g]["t_hand0"] + len(gp) * W.types[g]["t_hand1"]
            for b2 in gp:
                deliv[b2["box"]] = cur
            q -= sum(b2["mass"] for b2 in gp)
            node = s2
        plan.append(dict(stops=st, g=g, drone=did, battery=bt["id"],
                         start=start, end=end, t=t, E=E, soc=soc, chg=chg,
                         boxes=[b2["box"] for _, gp in st for b2 in gp],
                         deliv=deliv))
        drones[did]["free"] = end
        bt["free"] = end + chg
        if hard and end > due + 1e-9:
            viol += 1
    return None if viol else plan


def optimise_order(W, stops_list, delta=0.0, max_rounds=40, verbose=False,
                   fleet=None, lock=None):
    """**在派发顺序空间里做局部搜索**：交换任意两个架次，接受能缩短完成时间的。

    为什么需要：结果只取决于派发顺序，而前面只试过 7 个**固定规则**（紧迫度、
    LPT、EDD……），最好的是 9153 s。搜索之后是 **8188 s**——比最好的固定规则
    又短 10.5%，而且**恰好等于按新机型分配算出的机队工作量下界 8187 s**。

    邻域 = 交换顺序中的任意两对（36 个架次时是 630 个邻居），做最好改进；
    接受条件里带"零违反"，所以搜索**不会**用违约换时间。
    """
    def key(i):
        bs = _boxes_of(stops_list[i])
        if any(b["first"] or b["kind"] == "医疗物资" for b in bs):
            return (0, min(b["t_due"] for b in bs), 0.0)
        e = min(W.route(g, stops_list[i])[1] for g in W.ok_types(stops_list[i]))
        return (1, 0.0, -e)

    cur = sorted(range(len(stops_list)), key=key)
    p = simulate_order(W, stops_list, cur, delta, fleet, lock)
    if p is None:
        # 起点顺序就排不出零违反的时间表（换容量档时会遇到）。依次退到别的
        # 固定规则找可行的起点；都不可行就交回 schedule()——它有"候选都不满足
        # 时限时退回最早可用"的兜底，永远给得出一份计划。
        for alt in (sorted(range(len(stops_list)),
                           key=lambda i: W.urgency(stops_list[i])),
                    list(range(len(stops_list)))):
            q = simulate_order(W, stops_list, alt, delta, fleet, lock)
            if q is not None:
                cur, p = alt, q
                break
        else:
            return schedule(W, stops_list, delta=delta, fleet=fleet)
    cur_t = max(x["end"] for x in p)
    for _ in range(max_rounds):
        best = None
        n = len(cur)
        for a in range(n):
            for b in range(a + 1, n):
                trial = list(cur)
                trial[a], trial[b] = trial[b], trial[a]
                q = simulate_order(W, stops_list, trial, delta, fleet,
                                   lock)
                if q is None:
                    continue
                t = max(x["end"] for x in q)
                if t < cur_t - 1e-9 and (best is None or t < best[0]):
                    best = (t, trial, q)
        if best is None:
            break
        cur_t, cur, p = best
        if verbose:
            print(f"    交换后 {cur_t:.0f} s")
    return p


def build_groups(W, allow_types=("A", "B"), delta=0.0, optimise=True,
                 fleet=None):
    """本文正式采用的组批 + 合并 + 排程：返回 (stops_list, plan, evaluate 结果)。

    delta 是问题三提出的「中继必须先于第一个缺口就位」所导致的运输整体推迟量。
    它由 d_q3 解出后回代到这里，形成两问之间的不动点迭代（见 d_q3.main）。

    optimise=True 时在**派发顺序**上再做一次局部搜索（见 optimise_order）——
    固定规则的 9153 s 就是这样降到 8188 s 的。
    """
    g, p, e, _ = repair_grouping(W, allow_types=allow_types, delta=delta,
                                 fleet=fleet)
    if optimise:
        p = optimise_order(W, g, delta, fleet=fleet)
        e = evaluate(W, p)
    return g, p, e


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
def schedule(W, stops_list, deadline_aware=True, fleet=None, delta=0.0):
    """按（首批 → 医疗 → 到期时间）排序，逐个派给最早可用的（无人机，电池）。

    deadline_aware：有时限的架次只在「能按时送达」的候选里挑；挑不到才退回
    最早可用的那个。没有这一条时，调度会把 3 个站合并的 C 机长航线堆在
    0 时刻，而全队只有 2 架 C 机——A/B 机闲着，首批却全部超时。
    """
    nf = fleet or {g: sum(1 for d in W.drones if d["type"] == g) for g in TYPES}
    pool = {g: [dict(id=f"{g}{i+1:02d}", free=0.0) for i in range(W.bats[g]["n"])]
            for g in TYPES}
    # 无人机编号必须用附件《逐架无人机清单》的 U01~U08，不能自己编号：
    # verify.py 会拿结果表里的编号去附件里对照，第一版自造的 "A*01" 当场被判失败。
    inv = {g: [d["id"] for d in W.drones if d["type"] == g] for g in TYPES}
    drones = {}
    for g in TYPES:
        for i in range(nf.get(g, 0)):
            did = (inv[g][i] if i < len(inv[g])
                   else f"{g}-EXT{i + 1:02d}")   # 仅机队敏感性实验会走到这里
            drones[did] = dict(id=did, type=g, free=0.0)

    # 派发顺序。0 号键把「首批 / 医疗」提到最前，其中按**期望送达时刻升序**
    # （EDD）；其余软时限架次按**估计能耗降序**派发。
    # 为什么不能直接用 LPT（最长先派）：首批箱多为单站短航线，LPT 把它们一直
    # 往后排，实测 30 个首批箱里 16 个超时——min-makespan 的经典规则在有硬
    # 时限时失效。八种派发规则的实测对比见 d_sched_opt.py 与论文表 8。
    _e = {}
    for st in stops_list:
        _e[id(st)] = min(W.route(g, st)[1] for g in W.ok_types(st))

    def _key(i):
        bs = _boxes_of(stops_list[i])
        if any(b["first"] or b["kind"] == "医疗物资" for b in bs):
            return (0, min(b["t_due"] for b in bs), 0.0)
        return (1, 0.0, -_e[id(stops_list[i])])

    order = sorted(range(len(stops_list)), key=_key)
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
            # 交付不晚于时限：末站时刻 ≈ start + t（保守）。
            # delta 是问题三要求的「中继先到点」导致的整体推迟量——它必须在这里
            # 就扣掉，否则问题二会排出一个"加上推迟就违约"的时间表。
            good = [c for c in cand if c[0] + c[5] + delta <= due + 1e-9]
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
    print("=" * 78)
    print("第一步：组批（先按最少架次装，只拆真正会违约的站）"
          "——容量 %g kg / %g m³" % (W.cap_m, W.cap_v))
    print("=" * 78)
    base = pack_min_sorties(W)
    n_min = len(base)
    split = set()
    g0 = p0 = None
    e0 = None
    for it in range(20):
        g0 = merge_pass(W, base, allow_types=("A",))
        p0 = schedule(W, g0)
        e0 = evaluate(W, p0)
        print(f"  第 {it} 轮：{e0['n_sorties']} 架次 | 完成 {e0['t_done']:.0f}s | "
              f"能耗 {e0['E']:.2f} kWh | 首批违约 {e0['viol_first']} | "
              f"医疗违约 {e0['viol_med']}")
        if e0["viol_first"] == 0 and e0["viol_med"] == 0:
            break
        bad = set()
        for b in W.boxes:
            t = e0["deliv"][b["box"]]
            if b["first"] and t > b["t_first"] + 1e-9:
                bad.add(b["site"])
            elif b["kind"] == "医疗物资" and t > b["t_due"] + 1e-9:
                bad.add(b["site"])
        bad -= split
        if not bad:
            break
        for st in bad:
            base = split_site(W, base, st)
        split |= bad
    print(f"  → 最少架次装箱 {n_min} 架次；只拆了 {len(split)} 个站"
          f"（{sorted(split) if split else '无需拆'}）；合并后 {e0['n_sorties']} 架次")
    g0 = merge_pass(W, base, allow_types=("A",))

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
    print(f"  → 该档下取「{pick}」")
    g1, p1, e1 = variants[pick]

    print()
    print("=" * 78)
    print("第三步之前：派发顺序的局部搜索")
    print("=" * 78)
    print("  固定规则只试得完几种；顺序空间里还有多少余地，只能搜。")
    p1 = optimise_order(W, g1)
    e1 = evaluate(W, p1)
    print(f"  搜索后：{e1['n_sorties']} 架次 / 完成 {e1['t_done']:.0f} s / "
          f"能耗 {e1['E']:.2f} kWh | 首批违约 {e1['viol_first']}、"
          f"医疗违约 {e1['viol_med']}")

    print()
    print("=" * 78)
    print("第四步：排程明细")
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
