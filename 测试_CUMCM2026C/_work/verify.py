# -*- coding: utf-8 -*-
"""独立验证：只读「已写出的结果文件」+ 原始附件，反算各项指标并与报告值比对。

刻意不复用求解过程中的内存对象——这样能查出"算出来了但没写对"这类错误
（本次就出过：模板多两列、dict.get 默认值被无条件求值导致 KeyError）。
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np
import openpyxl

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c_data as D
import mg_core as M

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "_交付"))
ATT5 = os.path.abspath(os.path.join(HERE, "..", "附件", "附件5"))


def read_buy_sheet(path, sheet, nrow):
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb[sheet]
    P = np.zeros((nrow, M.T))
    for i in range(nrow):
        for j in range(M.T):
            v = ws.cell(i + 2, j + 2).value
            P[i, j] = float(v) if v is not None else 0.0
    kwh = np.array([float(ws.cell(i + 2, M.T + 2).value or 0) for i in range(nrow)])
    cost = np.array([float(ws.cell(i + 2, M.T + 3).value or 0) for i in range(nrow)])
    return P, kwh, cost


def check_result1():
    print("=" * 76)
    print("【1】问题 1：由 result1.xlsx 反算")
    print("=" * 76)
    E, P, Q = D.load_att1()
    wb = openpyxl.load_workbook(os.path.join(OUT, "result1.xlsx"),
                                data_only=True)
    ws = wb["计划购电量"]
    buy = np.array([float(ws.cell(i + 2, 2).value) for i in range(M.T)])
    # 表2
    ws2 = wb["充放电量"]
    ch = np.zeros(M.T); dis = np.zeros(M.T)
    # 表2 给的是时段内的电量（kWh），换算成功率：kW = kWh / 小时数
    for i, (a, b) in enumerate(D.PERIODS):
        ch[a * 6:b * 6] = float(ws2.cell(i + 2, 2).value) / (b - a)
        dis[a * 6:b * 6] = float(ws2.cell(i + 2, 3).value) / (b - a)
    soc0 = float(ws2.cell(2, 5).value); soc1 = float(ws2.cell(3, 5).value)
    # 反算
    buy_kwh = buy.sum()
    cost = float((E * buy).sum())
    # 能量（净负荷 = 购电 + 放 − 充）
    net = buy + dis * M.DT - ch * M.DT
    print(f"  购电量 {buy_kwh:.4f} kWh    购电费 {cost:.4f} 元")
    print(f"  0:00/24:00 储电量 {soc0:.4f} / {soc1:.4f} kWh  （应相等）")
    print(f"  充放电量：充 {ch.sum()*M.DT:.4f}  放 {dis.sum()*M.DT:.4f} kWh")
    # 从表2 的时段值重算 SOC（只有 0:00 与 24:00 有值，中间不可验；改为看总量）
    # 全局能量守恒：Σ购电 = Σ负荷 − Σ光伏 + (充−放)   （弃光为 0）
    lhs = buy_kwh
    rhs = (P.sum() - Q.sum()) * M.DT + (ch.sum() - dis.sum()) * M.DT
    print(f"  能量守恒：购电 {lhs:.4f}  应等于 (负荷−光伏)+(充−放) {rhs:.4f}"
          f"   差 {abs(lhs-rhs):.2e}")
    print(f"  无储能基线 {(E*np.maximum(P-Q,0)*M.DT).sum():.4f} 元")
    # 容差按**相对**给：结果文件保留 4 位小数，144 步累计舍入可达 ~1e-2 kWh，
    # 用绝对 1e-6 会把自己的舍入当成求解错误。
    rel = abs(lhs - rhs) / max(abs(lhs), 1e-9)
    print(f"  相对差 {rel:.2e}（容差 1e-6）")
    return rel < 1e-6


def check_daywise(fname, prices, start=31, sample=6):
    print()
    print("=" * 76)
    print(f"【】{fname}：逐日抽查")
    print("=" * 76)
    load, pv = D.load_att2()
    P, kwh, cost = read_buy_sheet(os.path.join(OUT, fname), "计划购电量",
                                  D.DAYS - start)
    n = P.shape[0]
    ok = True
    idxs = np.linspace(0, n - 1, sample).astype(int)
    print(f"  {'日序号':>7}{'购电量/kWh':>14}{'表内全天值':>14}{'表内电费':>14}{'重算电费':>14}")
    for i in idxs:
        d = start + i
        pr = prices if np.asarray(prices).ndim == 1 else prices[d]
        rec = float((pr * P[i]).sum())
        print(f"  {d:>7}{P[i].sum():>14.4f}{kwh[i]:>14.4f}{cost[i]:>14.4f}{rec:>14.4f}")
        ok &= abs(cost[i] - rec) < 1e-3
    return ok


def check_totals():
    print()
    print("=" * 76)
    print("【汇总】五个结果文件的整体指标")
    print("=" * 76)
    E, _, _ = D.load_att1()
    _, pv = D.load_att2()
    for f, pr in (("result2.xlsx", E), ("result4-2.xlsx", D.load_att4())):
        P, kwh, cost = read_buy_sheet(os.path.join(OUT, f), "计划购电量",
                                      D.DAYS - 31)
        print(f"  {f:<16} 购电量 {kwh.sum():>14.1f} kWh  购电费 {cost.sum():>14.2f} 元")
    for f, pr in (("result3.xlsx", E), ("result4-3.xlsx", D.load_att4())):
        P, kwh, cost = read_buy_sheet(os.path.join(OUT, f), "计划购电量",
                                      D.DAYS - 31)
        A, akwh, acost = read_buy_sheet(os.path.join(OUT, f), "调整购电量",
                                        D.DAYS - 31)
        print(f"  {f:<16} 计划购电量 {kwh.sum():>14.1f}  调整后 {akwh.sum():>14.1f} kWh"
              f"  结算 {acost.sum():>14.2f} 元")
        # 紧急购电
        wb = openpyxl.load_workbook(os.path.join(OUT, f), data_only=True)
        ws = wb["紧急购电量"]
        tot = sum(float(ws.cell(i, 3).value or 0)
                  for i in range(2, ws.max_row + 1))
        print(f"  {'':<16} 紧急购电量 {tot:>14.1f} kWh")


def check_templates():
    """逐列比对：写出的 5 个结果文件 vs 附件 5 模板。

    论文里「结构与附件 5 模板逐列比对通过」这句话应当有可执行的依据，
    所以这里把模板也读进来比对工作表名、表头、列数与行数，而不是靠人工看。
    比对的是**结构**：模板只填了前几行样例，行数自然对不上，故行数只做
    「不少于模板的样例行数」的检查，不对齐具体值。
    """
    print()
    print("=" * 76)
    print("【结构】与附件 5 模板逐列比对")
    print("=" * 76)
    ok = True
    for fn in ("result1.xlsx", "result2.xlsx", "result3.xlsx",
               "result4-2.xlsx", "result4-3.xlsx"):
        tpl = os.path.join(ATT5, fn)
        got = os.path.join(OUT, fn)
        wt = openpyxl.load_workbook(tpl)
        wg = openpyxl.load_workbook(got, data_only=True)
        msgs = []
        if wg.sheetnames != wt.sheetnames:
            msgs.append(f"工作表名不符：模板 {wt.sheetnames}，实际 {wg.sheetnames}")
        for sn in wt.sheetnames:
            if sn not in wg.sheetnames:
                continue
            st, sg = wt[sn], wg[sn]
            if sg.max_column != st.max_column:
                msgs.append(f"[{sn}] 列数 {sg.max_column} ≠ 模板 {st.max_column}")
            def _norm(v):
                # 模板里有一个孤立的写法错误「7:0-7:10」（分钟没补零），
                # 其余都是「7:00-7:10」。比对前把单个数字的分钟补成两位，
                # 否则会把模板自己的笔误报成我们的差异。
                if isinstance(v, str):
                    return re.sub(r"(?<=:)(\d)(?!\d)", r"0\1", v)
                return v

            ht = [_norm(st.cell(1, c).value) for c in range(1, st.max_column + 1)]
            hg = [_norm(sg.cell(1, c).value) for c in range(1, sg.max_column + 1)]
            # 计划/调整购电量表的时间标签有一处**已知且已在附录 B.1 说明**的
            # 差异：模板把附件 1 的 144 个时刻两两配对当区间，整列右移一格。
            # 这里把「仅这一栏按左移一格逐项相等」认作符合，其余任何表头差异
            # 都算失败——差异必须是可解释的，不能是「反正都不同就算了」。
            if hg != ht:
                # 模板第 i 行 = 本文第 i+1 个区间，故模板的前 143 个标签应
                # 逐个等于本文的标签 2…144；末两列（全天购电量/购电费）须一致。
                shifted = (len(ht) == len(hg) and hg[0] == ht[0]
                           and ht[1:M.T] == hg[2:M.T + 1]
                           and hg[M.T + 1:] == ht[M.T + 1:])
                if shifted:
                    msgs.append(
                        f"[{sn}] 时间标签按已知约定左移一格（附录 B.1，视为符合）")
                else:
                    msgs.append(
                        f"[{sn}] 表头不符：模板 {ht[:3]}…，实际 {hg[:3]}…")
            if sg.max_row < st.max_row:
                msgs.append(f"[{sn}] 行数 {sg.max_row} 少于模板样例 {st.max_row}")
        hard = [m for m in msgs if "视为符合" not in m]
        flag = "OK " if not hard else "!! "
        print(f"  {flag}{fn:<16} {len(wg.sheetnames)} 个工作表")
        for m in msgs:
            print(f"       - {m}")
        ok &= not hard
    return ok


if __name__ == "__main__":
    a = check_result1()
    b = check_daywise("result2.xlsx", D.load_att1()[0])
    c = check_daywise("result4-2.xlsx", D.load_att4())
    check_totals()
    d = check_templates()
    print()
    print("总体:", "通过" if (a and b and c and d) else "**有失败项**")
