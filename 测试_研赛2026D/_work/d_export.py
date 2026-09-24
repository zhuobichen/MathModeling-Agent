# -*- coding: utf-8 -*-
"""按《结果提交模板.xlsx》写出 6 张结果表。

模板 6 个工作表：Q1_单点组批 / Q2_运输架次 / Q2_逐箱交付 / Q3_中继架次 /
Q3_通信保障 / Q4_分区配置。列名以模板为准，不增不改。
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np
import openpyxl

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import d_core as D
import d_relay
from d_q1 import sortie as sortie_of, max_payload
from d_q2 import build_groups, World, TYPES, SITES, initial_groups, merge_pass, schedule, evaluate
import d_q3 as Q3

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "_交付"))
TPL = os.path.abspath(os.path.join(HERE, "..", "附件", "结果提交模板.xlsx"))


def main():
    os.makedirs(OUT, exist_ok=True)
    W = World()
    rt, W.relays, W.rbats = D.load_relays()
    C = D.Comms(W.geo, W.nodes, D.load_comm())
    types = W.types
    W.by_box = {b['box']: b for b in W.boxes}   # 问题一只给了货箱编号

    # ── 问题一 ──
    q1 = json.load(open(os.path.join(HERE, "out_q1.json"), encoding="utf-8"))
    # ── 问题二 ──
    g1, p1, e1 = build_groups(W, allow_types=("A", "B"))
    # ── 问题三：中继 ──
    # 直接读 out_q3.json 的最终方案（3 站）。**不要在这里现算**：d_q3 原来的
    # cluster_stations + station_plan 覆盖判据有高估（d_relay.py 有说明），
    # 结果文件必须与论文引用的是同一套站位。
    with open(os.path.join(HERE, "out_q3.json"), encoding="utf-8") as f:
        q3res = json.load(f)
    stations = q3res["stations"]
    centers = [tuple(x["point"]) for x in stations]
    T_SHIFT = float(q3res.get("t_shift", 0.0))
    # 中继编号：库存只有 2 架，第 3 个站位没有实体机可用（缺口在论文里写明）
    relays_avail = [r["id"] for r in W.relays]
    # ── 问题四 ──
    q4 = json.load(open(os.path.join(HERE, "out_q4.json"), encoding="utf-8"))

    wb = openpyxl.load_workbook(TPL)
    # 清掉模板里的样例行（保留表头）
    for sn in wb.sheetnames:
        ws = wb[sn]
        if ws.max_row > 1:
            ws.delete_rows(2, ws.max_row)

    # ═══ Q1_单点组批 ═══
    ws = wb["Q1_单点组批"]
    r = 2
    sid = 0
    for s in SITES:
        b = q1["best"].get(s)
        if not b:
            continue
        g = b["g"]
        for gp, soc in zip(b["groups"], b["soc"]):
            sid += 1
            m = sum(W.by_box[x]["mass"] for x in gp)
            v = sum(W.by_box[x]["vol"] for x in gp)
            tt = sortie_of(g, s, [W.by_box[x] for x in gp], W.T)
            ws.cell(r, 1, f"P1-{sid:03d}")
            ws.cell(r, 2, s)
            ws.cell(r, 3, g)
            ws.cell(r, 4, "、".join(gp))
            ws.cell(r, 5, round(m, 4))
            ws.cell(r, 6, round(v, 6))
            ws.cell(r, 7, round(tt["t"], 2))
            ws.cell(r, 8, round(tt["E"], 4))
            ws.cell(r, 9, round(soc, 2))
            r += 1
    print(f"Q1_单点组批：{sid} 行")

    # ═══ Q2_运输架次 / Q2_逐箱交付 ═══
    ws2 = wb["Q2_运输架次"]
    ws2b = wb["Q2_逐箱交付"]
    psorted = sorted(p1, key=lambda x: (x["start"], x["drone"]))
    for i, p in enumerate(psorted, 1):
        ws2.cell(i + 1, 1, f"T{i:03d}")
        ws2.cell(i + 1, 2, p["drone"])
        ws2.cell(i + 1, 3, p["g"])
        ws2.cell(i + 1, 4, p["battery"])
        ws2.cell(i + 1, 5, round(p["start"], 2))
        ws2.cell(i + 1, 6, "O01→" + "→".join(s for s, _ in p["stops"]) + "→O01")
        ws2.cell(i + 1, 7, round(p["end"], 2))
        ws2.cell(i + 1, 8, round(p["E"], 4))
    order = sorted(W.boxes, key=lambda b: b["box"])
    for i, b in enumerate(order, 2):
        ws2b.cell(i, 1, b["box"])
        # 找它所在的架次
        sidx = next(k for k, p in enumerate(psorted, 1) if b["box"] in p["boxes"])
        ws2b.cell(i, 2, f"T{sidx:03d}")
        ws2b.cell(i, 3, b["site"])
        ws2b.cell(i, 4, round(e1["deliv"][b["box"]], 2))
    print(f"Q2_运输架次：{len(psorted)} 行；Q2_逐箱交付：{len(order)} 行")

    # ═══ Q3_中继架次 / Q3_通信保障 ═══
    ws3 = wb["Q3_中继架次"]
    for i, x in enumerate(stations, 1):
        ws3.cell(i + 1, 1, f"R{i:03d}")
        ws3.cell(i + 1, 2, relays_avail[i - 1] if i - 1 < len(relays_avail)
                 else f"（缺 {i - len(relays_avail)}）")
        ws3.cell(i + 1, 3, f"RE{i:02d}")
        ws3.cell(i + 1, 4, round(x["t_out"], 2))
        ws3.cell(i + 1, 5, round(x["point"][0], 7))
        ws3.cell(i + 1, 6, round(x["point"][1], 7))
        ws3.cell(i + 1, 7, round(x["point"][2], 2))
        ws3.cell(i + 1, 8, round(x["t_out"] + rt["t_link"], 2))
        ws3.cell(i + 1, 9, round(x["win"][1], 2))
        ws3.cell(i + 1, 10, round(x["t_back"], 2))
        ws3.cell(i + 1, 11, round(x["E"], 4))
    # 通信保障：每条运输架次的每个缺口时段
    ws3b = wb["Q3_通信保障"]
    rr = 2
    for i, p in enumerate(psorted, 1):
        for (t0, t1, smp) in Q3.gaps_of(W, C, p):
            j = None
            for jj, x2 in enumerate(stations):
                if d_relay.gap_covered(W, C, smp, tuple(x2["point"])):
                    j = jj
                    break
            ws3b.cell(rr, 1, f"T{i:03d}")
            if j is None:
                ws3b.cell(rr, 2, "中断")
                ws3b.cell(rr, 3, round(t0 + T_SHIFT, 2))
                ws3b.cell(rr, 4, round(t1 + T_SHIFT, 2))
                ws3b.cell(rr, 5, "无")
            else:
                ws3b.cell(rr, 2, "中继")
                ws3b.cell(rr, 3, round(t0 + T_SHIFT, 2))
                ws3b.cell(rr, 4, round(t1 + T_SHIFT, 2))
                ws3b.cell(rr, 5, "中继无人机")
                ws3b.cell(rr, 6, f"R{j+1:03d}")
            rr += 1
    print(f"Q3_中继架次：{len(stations)} 行；Q3_通信保障：{rr-2} 行")

    # ═══ Q4_分区配置 ═══
    # 各组所需无人机数取**库存约束下的最优分配**（论文 8.3 节）：
    # 把库存按机型分给各组，使硬时限违反数最少。原先按「每组自备峰值机队」
    # 的单一机型估计会得出"缺 3 架 B 型"——那是建模口径造成的假缺口。
    ws4 = wb["Q4_分区配置"]
    rr = 2
    for K in ("2", "3"):
        sch = q4["schemes"][K]
        alloc = sch.get("alloc", {}).get("per_group", [])
        for i, gp in enumerate(sch["groups"], 1):
            ws4.cell(rr, 1, int(K))
            ws4.cell(rr, 2, i)
            ws4.cell(rr, 3, "、".join(gp))
            cp = alloc[i - 1]["cap"] if i - 1 < len(alloc) else {"A": 0, "B": 0, "C": 0}
            ws4.cell(rr, 4, cp.get("A", 0))
            ws4.cell(rr, 5, cp.get("B", 0))
            ws4.cell(rr, 6, cp.get("C", 0))
            # 7~9 列是**电池组数**（不是无人机数）：取该组"零违反且完成时间不超出
            # 满池 5%"所需的最少组数（d_q4.min_batteries）。不能直接写满池——那会
            # 让各组的电池数之和远超库存，不是可行配置。
            bat = alloc[i - 1].get("bat", {}) if i - 1 < len(alloc) else {}
            for ci, g in enumerate(TYPES, 7):
                ws4.cell(rr, ci, bat.get(g, 0))
            rpg = sch.get("relay_per_group", {})
            rmg = sch.get("relay_mod_per_group", {})
            ws4.cell(rr, 10, rpg.get(str(i - 1), rpg.get(i - 1, 0)))
            ws4.cell(rr, 11, rmg.get(str(i - 1), rmg.get(i - 1, 0)))
            rr += 1
    print(f"Q4_分区配置：{rr-2} 行")

    dst = os.path.join(OUT, "结果提交表.xlsx")
    wb.save(dst)
    print("已写出", dst)

    # 供论文核对
    summ = dict(q1=dict(n=sum(b["n"] for b in q1["best"].values()),
                        E=sum(b["E"] for b in q1["best"].values()),
                        t=sum(b["t"] for b in q1["best"].values())),
                q2=dict(n=e1["n_sorties"], t_done=e1["t_done"], E=e1["E"],
                        late=e1["late"], vf=e1["viol_first"],
                        vm=e1["viol_med"], by_type=e1["by_type"]),
                q3=dict(n_station=len(stations),
                        E=sum(x["E"] for x in stations),
                        n_bat=sum(x["n_bat"] for x in stations),
                        n_gap=q3res["n_gaps"],
                        n_relay_min=q3res.get("n_relay_min"),
                        t_shift=q3res.get("t_shift"),
                        t_joint=max(max(p["end"] for p in p1),
                                    max(x["t_back"] for x in stations))),
                q4=q4)
    with open(os.path.join(HERE, "out_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, ensure_ascii=False, indent=1, default=float)
    print("已写出 out_summary.json")
    return summ


if __name__ == "__main__":
    main()
