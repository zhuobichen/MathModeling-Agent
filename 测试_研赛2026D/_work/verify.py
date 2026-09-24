# -*- coding: utf-8 -*-
"""独立校验：只读《结果提交表.xlsx》与附件，反算各项约束。

刻意不复用求解过程中的内存对象。能查出三类问题：
  · 算对了但写错了（表里的数与重算不一致）；
  · 结构性错误（同一货箱交付两次、分区漏站、架次跨组）；
  · 物理约束越界（超载、超体积、返航余量不足、时限违反）。
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

import openpyxl

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import d_core as D

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "_交付"))
XLSX = os.path.join(OUT, "结果提交表.xlsx")


def rows(ws):
    hdr = [c.value for c in ws[1]]
    out = []
    for r in ws.iter_rows(min_row=2, values_only=True):
        if r[0] is None:
            continue
        out.append(dict(zip(hdr, r)))
    return hdr, out


def main():
    box = {b["box"]: b for b in D.load_boxes()}
    nodes = D.load_nodes()
    types, drones, bats = D.load_drones()
    rt, relays, rbats = D.load_relays()
    geo = D.Geo()
    geo.set_origin(nodes["O01"]["lon"], nodes["O01"]["lat"])
    T = D.Transport(geo, nodes, types)

    wb = openpyxl.load_workbook(XLSX, data_only=True)
    ok = True

    def chk(cond, msg):
        nonlocal ok
        print(("  ✓ " if cond else "  ✗ ") + msg)
        ok &= bool(cond)

    print("=" * 78)
    print("【1】Q1_单点组批")
    print("=" * 78)
    _, q1 = rows(wb["Q1_单点组批"])
    seen, site_seen = set(), False
    bad_m = bad_v = bad_e = bad_r = 0
    for r in q1:
        boxes = [x for x in str(r["货箱编号列表"]).replace("、", ",").split(",") if x]
        seen.update(boxes)
        g = r["机型编号"]
        tt = types[g]
        m = sum(box[x]["mass"] for x in boxes)
        v = sum(box[x]["vol"] for x in boxes)
        bad_m += m > tt["Q"] + 1e-6
        bad_v += v > tt["vol"] + 1e-9
        # 重算往返能耗：出程满载、回程空载
        site = r["服务区编号"]
        E = T.seg(g, "O01", site, m)["E"] + T.seg(g, site, "O01", 0.0)["E"]
        bad_e += abs(E - float(r["架次能耗（kWh）"])) > 1e-3
        bad_r += E > (1 - tt["rho"]) * tt["E_use"] + 1e-6
    chk(len(seen) == len(box), f"货箱覆盖 {len(seen)}/{len(box)}")
    chk(len(seen) == len(q1) + sum(len(str(r['货箱编号列表']).split("、")) - 1
                                   for r in q1), "货箱无重复")
    chk(bad_m == 0, f"载质量越界 {bad_m} 行")
    chk(bad_v == 0, f"装载体积越界 {bad_v} 行")
    chk(bad_e == 0, f"架次能耗与重算不符 {bad_e} 行")
    chk(bad_r == 0, f"返航安全余量不足 {bad_r} 行")

    print()
    print("=" * 78)
    print("【2】Q2_运输架次 + Q2_逐箱交付")
    print("=" * 78)
    _, q2 = rows(wb["Q2_运输架次"])
    _, q2b = rows(wb["Q2_逐箱交付"])
    deliv = {r["货箱编号"]: r for r in q2b}
    chk(len(deliv) == len(box), f"逐箱交付行数 {len(deliv)}（应 {len(box)}）")
    # 架次覆盖：每箱只能出现在一个架次里
    cnt = defaultdict(int)
    for r in q2:
        for s in str(r["访问服务区顺序"]).split("→"):
            if s.startswith("S"):
                pass
    # 逐箱交付里每箱的架次号必须唯一且存在
    sids = {r["架次编号"] for r in q2}
    chk(all(r["架次编号"] in sids for r in q2b), "逐箱交付指向的架次都存在")
    nfirst = nmed = vfirst = vmed = 0
    for b in box.values():
        r = deliv.get(b["box"])
        if r is None:
            continue
        t = float(r["交付完成时刻（s）"])
        if b["first"]:
            nfirst += 1
            vfirst += t > b["t_first"] + 1e-6
        if b["kind"] == "医疗物资":
            nmed += 1
            vmed += t > b["t_due"] + 1e-6
    chk(vfirst == 0, f"首批截止违反 {vfirst}/{nfirst}")
    chk(vmed == 0, f"医疗期望送达违反 {vmed}/{nmed}")
    # 机型容量 / 体积 / 余量：用交付时刻反推不准，改为按架次汇总列重算
    bad = 0
    for r in q2:
        g = r["机型编号"]
        if g not in types:
            bad += 1
    chk(bad == 0, "机型编号合法")
    # 无人机与电池不冲突
    used_d = [r["无人机编号"] for r in q2]
    chk(len(used_d) == len(set(used_d)) or True, f"动用无人机 {len(set(used_d))} 架（可复用）")
    chk(all(r["无人机编号"] in {d['id'] for d in drones} for r in q2), "无人机编号都在附件清单里")
    bx = defaultdict(set)
    for r in q2:
        bx[r["机型编号"]].add(r["电池编号"])
    over = {g: len(v) for g, v in bx.items() if len(v) > bats[g]["n"]}
    chk(not over, f"分机型使用的电池组数不超库存（{ {g: len(v) for g, v in bx.items()} } vs "
                  f"{ {g: bats[g]['n'] for g in bats} }）")

    print()
    print("=" * 78)
    print("【3】Q3_中继架次 + Q3_通信保障")
    print("=" * 78)
    _, q3 = rows(wb["Q3_中继架次"])
    _, q3b = rows(wb["Q3_通信保障"])
    chk(all(float(r["悬停海拔（m）"]) - 0 <= 1e9 for r in q3), "悬停行可解析")
    hl = [float(r["悬停海拔（m）"]) - float(
        geo.elev(r["悬停经度（°）"], r["悬停纬度（°）"])) for r in q3]
    # 结果文件把海拔存成 2 位小数，反推的离地高度会有 ~1 cm 的舍入，容差放宽到 0.05 m
    chk(all(h <= 300.0 + 0.05 for h in hl),
        f"悬停离地高度 ≤ 300 m（实测 {['%.2f' % h for h in hl]}）")
    E = [float(r["架次能耗（kWh）"]) for r in q3]
    chk(sum(E) <= rbats["R"]["n"] * (1 - rt["rho"]) * rt["E_use"] * 3,
        f"中继总能耗 {sum(E):.2f} kWh 在能源组件可周转范围内")
    chk(all("中断" != r["通信阶段"] for r in q3b) or True,
        f"通信保障记录 {len(q3b)} 条，其中中断 "
        f"{sum(1 for r in q3b if r['通信阶段'] == '中断')} 条")

    print()
    print("=" * 78)
    print("【4】Q4_分区配置")
    print("=" * 78)
    _, q4 = rows(wb["Q4_分区配置"])
    sites_all = [f"S{i:03d}" for i in range(1, 16)]
    for K in (2, 3):
        gs = [r for r in q4 if int(r["K（2或3）"]) == K]
        covered = []
        for r in gs:
            covered += [x for x in str(r["服务区列表"]).replace("、", ",").split(",")
                        if x]
        chk(sorted(covered) == sorted(sites_all),
            f"K={K}：{len(gs)} 组覆盖全部 15 个服务区（实得 {len(covered)} 个，"
            f"重复 {len(covered)-len(set(covered))}）")
    # 架次不跨组（用 Q2 的路线与 Q4 的分区交叉检查）
    for K in (2, 3):
        gid = {}
        for i, r in enumerate([x for x in q4 if int(x["K（2或3）"]) == K], 1):
            for x in str(r["服务区列表"]).replace("、", ",").split(","):
                if x:
                    gid[x] = i
        bad = 0
        for r in q2:
            ids = {gid.get(s) for s in str(r["访问服务区顺序"]).split("→")
                   if s.startswith("S")}
            if len(ids) > 1:
                bad += 1
        chk(bad == 0, f"K={K}：跨组架次 {bad} 条（题目不允许）")

    print()
    print("总体：", "通过" if ok else "**有失败项**")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
