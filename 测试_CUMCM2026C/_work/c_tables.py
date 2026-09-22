# -*- coding: utf-8 -*-
"""生成论文里按题目表 1 / 表 2 / 表 3 格式给出的结果表。

题面对论文的格式要求是明确的：
  问题 1 —— 以表 1 的格式给出指定时间段的购电量及全天购电量与购电费，
            以表 2 的格式给出指定时间段的充放电量及 0:00、24:00 储电量；
  问题 2 —— 按表 1 和表 2 的格式给出表 3 中 4 个指定日期的结果，
            按表 3 的格式给出紧急购电结果；
  问题 3 —— 同上（另加调整购电量）；
  问题 4 —— 给出相应计算结果。

这里直接产出可直接粘进论文的 markdown 表格文本（appendix_tables.md），
以及一份 out_tables.json 供核对。表格用 HTML <table>——pandoc 只认这个，
markdown 竖线表在 docx 管线里拿不到题注。

注意：管线会把 colspan 展开成重复文字，所以**所有行列数必须一致**，
需要「合并」的位置一律填 — 而不是用 colspan。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import openpyxl

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c_data as D
import mg_core as M

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "_交付"))

# 表 1 指定时间段：10:00-10:10 等 → 步号 = 小时*6
T1_HOURS = [10, 12, 14, 16, 18, 20]
SLOTS = [f"{h}:00-{h}:10" for h in T1_HOURS]
SPEC_DATES = [(3, 20), (6, 21), (9, 23), (12, 21)]
DATE_LBLS = [f"2025-{m:02d}-{d:02d}" for m, d in SPEC_DATES]
MAX_EM_ROWS = 10          # 表 3 格式最多列多少个时段（超出在表下注明）
FIRST_TABLE_NO = 16       # 附录 D 的第一张表接着正文的「表 15」往下编号


# ── 从结果文件读回 ────────────────────────────────────────
def _num(v):
    return float(v) if v not in (None, "") else 0.0


def load_solution_days(fname, start=31):
    """读回某结果文件：逐日 144 步购电量、充放电、SOC、紧急量、计划量。"""
    wb = openpyxl.load_workbook(os.path.join(OUT, fname), data_only=True)
    ws = wb["计划购电量"]
    n = ws.max_row - 1
    P = np.zeros((n, M.T))
    day_kwh = np.zeros(n)
    day_cost = np.zeros(n)
    for i in range(n):
        for j in range(M.T):
            P[i, j] = _num(ws.cell(i + 2, j + 2).value)
        day_kwh[i] = _num(ws.cell(i + 2, M.T + 2).value)
        day_cost[i] = _num(ws.cell(i + 2, M.T + 3).value)

    plan = None
    if "调整购电量" in wb.sheetnames:
        wp = wb["调整购电量"]
        plan = np.zeros((n, M.T))
        for i in range(n):
            for j in range(M.T):
                plan[i, j] = _num(wp.cell(i + 2, j + 2).value)

    w3 = wb["充放电量"]
    ch = np.zeros((n, M.T))
    dis = np.zeros((n, M.T))
    soc0 = np.zeros(n)
    soc1 = np.zeros(n)
    for i in range(n):
        r = 2 + i * 6
        for k, (a, b) in enumerate(D.PERIODS):
            # 表里存的是时段电量(kWh)，这里换回功率(kW) 以便与求解结果同口径
            ch[i, a * 6:b * 6] = _num(w3.cell(r + k, 3).value) / (b - a)
            dis[i, a * 6:b * 6] = _num(w3.cell(r + k, 4).value) / (b - a)
        soc0[i] = _num(w3.cell(r, 6).value)
        soc1[i] = _num(w3.cell(r + 1, 6).value)

    # 日期列是 datetime（与附件 5 模板一致），归一成 'YYYY-MM-DD' 再当键；
    # 同一天的后续时段行日期为空，沿用上一个日期。
    w4 = wb["紧急购电量"]
    em = {}
    cur = None
    for i in range(2, w4.max_row + 1):
        v = w4.cell(i, 1).value
        if v not in (None, ""):
            cur = v.strftime("%Y-%m-%d") if hasattr(v, "strftime") else str(v)
        seg = w4.cell(i, 2).value
        if cur is None or not seg:
            continue
        em.setdefault(cur, []).append((seg, _num(w4.cell(i, 3).value)))
    return dict(P=P, plan=plan, ch=ch, dis=dis, soc0=soc0, soc1=soc1,
                em=em, day_kwh=day_kwh, day_cost=day_cost, start=start)


def _step(hour):
    return hour * 6


def t1_row(buy_kwh_144):
    """表 1 主体：6 个指定时段的购电量。"""
    return {f"{h}:00-{h}:10": float(buy_kwh_144[_step(h)]) for h in T1_HOURS}


def t2_rows(ch_kw, dis_kw):
    """表 2 主体：6 个 4 小时时段的充放电电量(kWh)。"""
    out = []
    for (a, b) in D.PERIODS:
        out.append((f"{a}:00-{b}:00",
                    float(ch_kw[a * 6:b * 6].sum() * M.DT),
                    float(dis_kw[a * 6:b * 6].sum() * M.DT)))
    return out


# ── markdown 片段 ────────────────────────────────────────
def _html(rows):
    out = ["<table>"]
    for i, r in enumerate(rows):
        tag = "th" if i == 0 else "td"
        cells = "".join(f"<{tag}>{c}</{tag}>" for c in r)
        out.append(f"<tr>{cells}</tr>")
    out.append("</table>")
    return "\n".join(out)


def _n4(x):
    return f"{x:.4f}"


def tbl_q1_table1(r1):
    """表 1 格式（问题 1）：三对「时间段|购电量」+ 全天购电量/购电费。"""
    sl = r1["slots"]
    rows = [["时间段", "购电量/kWh", "时间段", "购电量/kWh",
             "时间段", "购电量/kWh"]]
    for a, b in ((0, 1), (2, 3), (4, 5)):
        rows.append([SLOTS[a], _n4(sl[SLOTS[a]]),
                     SLOTS[b], _n4(sl[SLOTS[b]]), "—", "—"])
    rows.append(["全天购电量", _n4(r1["day_kwh"]),
                 "全天购电费", f'{r1["day_cost"]:.4f} 元', "—", "—"])
    return _html(rows)


def tbl_table2(r2):
    """表 2 格式：三对「时间段|充电量|放电量」+ 0:00/24:00 储电量。"""
    rows = [["时间段", "充电量/kWh", "放电量/kWh",
             "时间段", "充电量/kWh", "放电量/kWh"]]
    for a, b in ((0, 3), (1, 4), (2, 5)):
        la, ca, da = r2["periods"][a]
        lb, cb, db = r2["periods"][b]
        rows.append([la, _n4(ca), _n4(da), lb, _n4(cb), _n4(db)])
    rows.append(["0:00 储电量", _n4(r2["soc0"]), "—",
                 "24:00 储电量", _n4(r2["soc1"]), "—"])
    return _html(rows)


def tbl_dates_table1(per_date, key):
    """表 1 格式的 4 日期版：行=指定时段+两项全天值，列=日期。

    key 取 "plan"（0:00 计划）或 "adj"（调整后实际执行），两套数必须分开取，
    否则「全天购电量」会拿成另一套的合计。
    """
    rows = [["时间段"] + DATE_LBLS]
    for s in SLOTS:
        rows.append([s] + [_n4(per_date[d][key][s]) for d in DATE_LBLS])
    rows.append(["全天购电量"] +
                [_n4(per_date[d][key + "_kwh"]) for d in DATE_LBLS])
    rows.append(["全天购电费"] +
                [f'{per_date[d][key + "_cost"]:.4f} 元' for d in DATE_LBLS])
    return _html(rows)


def tbl_dates_table2(per_date, which):
    """表 2 格式的 4 日期版：行=6 个 4 小时时段（+ 储电量），列=日期。

    题面的表 2 是「时间段|充电量|放电量」成对的三栏，四个日期并排就是 9 列，
    在 A4 纵向里挤到表头被截断、读不出来。这里按同样的字段拆成两张
    5 列表（充电量一张、放电量与储电量一张），内容一项不少。
    """
    rows = [["时间段"] + DATE_LBLS]
    for lab, ch, dis in per_date[DATE_LBLS[0]]["periods"]:
        row = [lab]
        for d in DATE_LBLS:
            rec = next(x for x in per_date[d]["periods"] if x[0] == lab)
            row.append(_n4(rec[1] if which == "ch" else rec[2]))
        rows.append(row)
    if which == "dis":
        rows.append(["0:00 储电量"] + [_n4(per_date[d]["soc0"]) for d in DATE_LBLS])
        rows.append(["24:00 储电量"] + [_n4(per_date[d]["soc1"]) for d in DATE_LBLS])
    return _html(rows)


def tbl_dates_table3(per_date):
    """表 3 格式：日期 | 紧急购电时间段 | 紧急购电量（同附录 5 的紧急购电表）。

    题面的表 3 把四个日期并排、每个日期带一对「时间段|购电量」子列，共 8 列，
    同样放不下。此处改用与附件 5「紧急购电量」工作表一致的竖表：日期一列、
    同一天的多个时段逐行往下写——这也是题目表 4 给出的填写示例的样子。
    """
    rows = [["日期", "紧急购电时间段", "紧急购电量/kWh"]]
    for d in DATE_LBLS:
        segs = per_date[d]["emerg"]
        if not segs:
            rows.append([d, "无", "0.0000"])
            continue
        for k, (seg, kwh) in enumerate(segs):
            rows.append([d if k == 0 else "", seg, _n4(kwh)])
    return _html(rows)


def em_note(per_date):
    tot = {d: sum(x[1] for x in per_date[d]["emerg"]) for d in DATE_LBLS}
    n = {d: len(per_date[d]["emerg"]) for d in DATE_LBLS}
    parts = [f"{d} 共 {n[d]} 个时段 {tot[d]:.4f} kWh" for d in DATE_LBLS]
    return "；".join(parts) + "。完整时段见对应的 result*.xlsx。"


# ── 主流程 ───────────────────────────────────────────────
def collect_q1():
    E, P, Q = D.load_att1()
    sol = M.solve_day(E, P, Q, 6000.0, cyclic=True)
    return dict(slots=t1_row(sol["buy"] * M.DT),
                day_kwh=float(sol["buy"].sum() * M.DT),
                day_cost=float(sol["cost"]),
                periods=t2_rows(sol["ch"], sol["dis"]),
                soc0=6000.0, soc1=float(sol["soc"][-1]))


def collect_daywise(fname, price_for_day):
    """按 4 个指定日期挑出表 1/表 2/表 3 所需的数据。

    result*.xlsx 的「计划购电量」表存的是 P0（0:00 按当天预报制定的计划），
    「调整购电量」表（仅问题 3/4-3 有）存的是 Pfin（调整后实际执行的量）。
    """
    sol = load_solution_days(fname)
    per = {}
    for lbl, (mm, dd) in zip(DATE_LBLS, SPEC_DATES):
        i = D.day_index(mm, dd) - sol["start"]
        price = price_for_day(mm, dd)
        P0 = sol["P"][i]                        # kWh/步
        rec = dict(
            plan=t1_row(P0),
            plan_kwh=float(P0.sum()),
            plan_cost=float((price * P0).sum()),
            periods=t2_rows(sol["ch"][i], sol["dis"][i]),
            soc0=float(sol["soc0"][i]), soc1=float(sol["soc1"][i]),
            emerg=sol["em"].get(lbl, []),
        )
        if sol["plan"] is not None:
            Pf = sol["plan"][i]
            rec.update(adj=t1_row(Pf),
                       adj_kwh=float(Pf.sum()),
                       adj_cost=float((price * Pf).sum()))
        per[lbl] = rec
    return per


def main():
    E, P1, Q1 = D.load_att1()
    A4 = D.load_att4()

    def price_fixed(mm, dd):
        return E

    def price_real(mm, dd):
        return A4[D.day_index(mm, dd)]

    blocks = []          # (section 标题, [(表题, 表体)], 表下注)
    res = {}

    # ── 问题 1：正文 5.2 直接用表 1/表 2 格式，不进附录 ──
    q1 = collect_q1()
    res["q1"] = q1
    with open(os.path.join(HERE, "q1_tables.html"), "w", encoding="utf-8",
              newline="\n") as f:
        f.write("<!-- 表 3 的正文：题面要求的表 1 格式 -->\n"
                + tbl_q1_table1(q1)
                + "\n\n<!-- 表 4 的正文：题面要求的表 2 格式 -->\n"
                + tbl_table2(dict(periods=q1["periods"], soc0=q1["soc0"],
                                  soc1=q1["soc1"])) + "\n")

    # ── 问题 2 / 3 / 4-2 / 4-3 ─────────────────────────
    sec_of = {"q2": "D.1 问题 2（固定电价、完美信息）",
              "q3": "D.2 问题 3（固定电价、预报 + 调整）",
              "q42": "D.3 问题 4 对应问题 2（实时电价、完美信息）",
              "q43": "D.4 问题 4 对应问题 3（实时电价、预报 + 调整）"}
    plan = [("q2", "result2.xlsx", price_fixed, "问题 2"),
            ("q3", "result3.xlsx", price_fixed, "问题 3"),
            ("q42", "result4-2.xlsx", price_real, "问题 4-2"),
            ("q43", "result4-3.xlsx", price_real, "问题 4-3")]
    for key, fn, pf, nm in plan:
        per = collect_daywise(fn, pf)
        res[key] = per
        tabs = []
        if key in ("q3", "q43"):
            tabs.append((f"{nm} 指定日期的计划购电量（表 1 格式）"
                         "（2025-03-20、06-21、09-23、12-21）",
                         tbl_dates_table1(per, key="plan")))
            tabs.append((f"{nm} 指定日期的调整购电量（表 1 格式）",
                         tbl_dates_table1(per, key="adj")))
        else:
            tabs.append((f"{nm} 指定日期的购电量（表 1 格式）"
                         "（2025-03-20、06-21、09-23、12-21）",
                         tbl_dates_table1(per, key="plan")))
        tabs.append((f"{nm} 指定日期的充电量（表 2 格式）",
                     tbl_dates_table2(per, "ch")))
        tabs.append((f"{nm} 指定日期的放电量与储电量（表 2 格式）",
                     tbl_dates_table2(per, "dis")))
        tabs.append((f"{nm} 指定日期的紧急购电量（表 3 格式）",
                     tbl_dates_table3(per)))
        blocks.append((f"## {sec_of[key]}", tabs, em_note(per)))

    # ── 打印 ────────────────────────────────────────────
    for sec, tabs, note in blocks:
        print("\n" + sec)
        for cap, body in tabs:
            print(f"\n**{cap}**\n{body}")
        if note:
            print(note)

    # ── 写文件 ──────────────────────────────────────────
    # 表号必须顺着正文的表 1..表 15 往下排，写成「表 D-1」不行：归一化补题注
    # 的正则是 `表\s*(\d+)[：:]`，认不出带字母的编号，会把 14 张附录表当成
    # 没有题注的表，各补一条「表16: …」，与本文写的题目撞成双份。
    lines, n = [], 0
    for sec, tabs, note in blocks:
        lines.append("\n" + sec + "\n")
        for cap, body in tabs:
            n += 1
            lines.append(f"\n**表 {FIRST_TABLE_NO - 1 + n}: {cap}**\n\n{body}\n")
        if note:
            lines.append(f"\n{note}\n")
    with open(os.path.join(HERE, "appendix_tables.md"), "w",
              encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    print(f"\n共 {n} 张附录表")

    with open(os.path.join(HERE, "out_tables.json"), "w",
              encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    print(f"\n已写出 appendix_tables.md（{n} 张表）、q1_tables.html 与 out_tables.json")


if __name__ == "__main__":
    main()
