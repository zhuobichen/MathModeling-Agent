# -*- coding: utf-8 -*-
"""D 题共用内核：数据装载 + 几何/高程 + 运输与通信物理。

公式全部来自题目附录 2、附录 3。注意题目正文里 `E^hor`、`E^up` 只给了名字与
单位（kWh），没给显式表达式；本文按参数表反推并说明：

  等效航程      L_g(q) = L_g^0 - (L_g^0 - L_g^F)(q/Q_g)^{3/2}     （题目给）
  水平巡航能耗  E^hor = d / L_g(q) * E_g^use                      （按航程分摊电池）
  爬升附加能耗  E^up  = (m_g + q) g h+ / eta_up / 3.6e6           （势能/效率）
  下降附加能耗  = 0                                                （题目明说）

依据：参数表里给了「空载标准航程/满载标准航程」与「电池可用能量」，而航程
唯一的用法就是分摊电量；「爬升能耗效率 0.72、下降能耗效率 0」也只有在势能
口径下才有意义。中继无人机题目明写「水平巡航能耗按巡航功率和巡航时间计算」，
与上式互为印证（中继给的正是巡航功率而非航程）。
"""
from __future__ import annotations

import math
import os

import numpy as np
import openpyxl
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
ATT = os.path.abspath(os.path.join(HERE, "..", "附件"))
BASE = os.path.join(ATT, "无人机应急物资运输基础数据")
GEO = os.path.join(ATT, "镇龙乡地理空间数据", "镇龙乡及周边地理数据")

G0 = 9.81
J2KWH = 3.6e6


# ══════════════════════════════════════════════════════════
# 地理
# ══════════════════════════════════════════════════════════
class Geo:
    """DEM + 节点坐标。本地平面坐标用等距圆柱投影（以 O01 为中心）。"""

    def __init__(self):
        d = sio.loadmat(os.path.join(GEO, "数字高程模型数据（DEM）",
                                     "镇龙乡及周边30米DEM.mat"))
        self.dem = np.asarray(d["dem"], float)
        self.lat = np.asarray(d["latitude"], float).ravel()
        self.lon = np.asarray(d["longitude"], float).ravel()
        self.nodata = float(np.asarray(d["nodata"]).ravel()[0])
        # 栅格中心：lat 递减（北在上），lon 递增
        self.lat0, self.lat1 = self.lat[0], self.lat[-1]
        self.lon0, self.lon1 = self.lon[0], self.lon[-1]
        self.dlat = (self.lat[0] - self.lat[1])
        self.dlon = (self.lon[1] - self.lon[0])
        self.origin = None            # 设为中心点，供 ENU 用

    # ── 高程 ──
    def elev(self, lon, lat):
        """最近邻取高程。返回 (海拔 m)。"""
        r = np.round((self.lat0 - np.asarray(lat, float)) / self.dlat).astype(int)
        c = np.round((np.asarray(lon, float) - self.lon0) / self.dlon).astype(int)
        r = np.clip(r, 0, self.dem.shape[0] - 1)
        c = np.clip(c, 0, self.dem.shape[1] - 1)
        z = self.dem[r, c]
        return np.where(z <= self.nodata + 1, 0.0, z)

    def elev_line(self, lon_a, lat_a, lon_b, lat_b, step_m=30.0):
        """沿水平直线采样，返回沿线地面高程的最大值。"""
        d = self.dist(lon_a, lat_a, lon_b, lat_b)
        n = max(int(d / step_m) + 1, 2)
        t = np.linspace(0.0, 1.0, n)
        lo = lon_a + (lon_b - lon_a) * t
        la = lat_a + (lat_b - lat_a) * t
        return float(np.max(self.elev(lo, la)))

    # ── 平面距离 ──
    def set_origin(self, lon, lat):
        self.origin = (float(lon), float(lat))
        p = math.radians(lat)
        self._mx = (111412.84 * math.cos(p) - 93.5 * math.cos(3 * p)
                    + 0.118 * math.cos(5 * p))       # 每度经度 → m
        self._my = (111132.92 - 559.82 * math.cos(2 * p)
                    + 1.175 * math.cos(4 * p) - 0.0023 * math.cos(6 * p))

    def xy(self, lon, lat):
        """局部平面坐标（m）。"""
        if self.origin is None:
            self.set_origin(lon, lat)
        return ((np.asarray(lon, float) - self.origin[0]) * self._mx,
                (np.asarray(lat, float) - self.origin[1]) * self._my)

    def dist(self, lon_a, lat_a, lon_b, lat_b):
        self.set_origin(self.origin[0], self.origin[1]) if self.origin else \
            self.set_origin(lon_a, lat_a)
        xa, ya = self.xy(lon_a, lat_a)
        xb, yb = self.xy(lon_b, lat_b)
        return float(math.hypot(xb - xa, yb - ya))


# ══════════════════════════════════════════════════════════
# 数据装载
# ══════════════════════════════════════════════════════════
def _sheet(fn, sheet=None):
    wb = openpyxl.load_workbook(os.path.join(BASE, fn), data_only=True)
    return wb[sheet] if sheet else wb[wb.sheetnames[0]]


def _rows(ws, start, ncol):
    out = []
    for r in ws.iter_rows(min_row=start, values_only=True):
        if r[0] is None:
            continue
        out.append(list(r[:ncol]))
    return out


def load_nodes():
    """调度中心 + 15 个服务区。返回 {编号: dict}。"""
    ws = _sheet("调度中心与服务区.xlsx")
    nodes = {}
    for r in _rows(ws, 3, 6):
        if r[0] and str(r[0]).startswith("O"):
            nodes[str(r[0])] = dict(kind="O", name=r[1], lon=float(r[2]),
                                    lat=float(r[3]), alt=float(r[4]), pop=None)
    for r in _rows(ws, 3, 6):
        if r[0] and str(r[0]).startswith("S"):
            nodes[str(r[0])] = dict(kind="S", name=r[1], lon=float(r[2]),
                                    lat=float(r[3]), alt=float(r[4]),
                                    pop=float(r[5]) if r[5] else None)
    return nodes


def load_boxes():
    """逐箱清单。返回 [dict]。"""
    ws = _sheet("物资需求与配送时限.xlsx", "逐箱货箱清单")
    out = []
    for r in _rows(ws, 2, 9):
        out.append(dict(box=str(r[0]), site=str(r[1]), kind=str(r[2]),
                        mass=float(r[3]), vol=float(r[4]),
                        first=(str(r[5]).strip() == "是"),
                        t_first=float(r[6]) if r[6] is not None else None,
                        t_due=float(r[7]), prio=int(r[8])))
    return out


def load_drones():
    """运输机型参数 + 逐架清单 + 共享电池库存。"""
    ws = _sheet("运输无人机数据.xlsx")
    types = {}
    for r in _rows(ws, 3, 18):
        # 电池库存那一段的第一列同样是 'A'/'B'/'C'，别让它进到这里
        if r[0] not in ("A", "B", "C") or len(types) >= 3:
            continue
        types[r[0]] = dict(
            name=r[1], m_empty=float(r[2]), Q=float(r[3]), vol=float(r[4]),
            vc=float(r[5]), L0=float(r[6]), LF=float(r[7]),
            E_use=float(r[8]), rho=float(r[9]) / 100.0,
            t_prep=float(r[10]), t_load=float(r[11]), t_hand0=float(r[12]),
            t_hand1=float(r[13]), v_up=float(r[14]), v_dn=float(r[15]),
            eta_up=float(r[16]), eta_dn=float(r[17]))
    # 后两段按「节标题」切分：三个表的第一列都可能出现 A/B/C，不能只看首列
    drones, bats = [], {}
    sect = None
    for r in ws.iter_rows(min_row=1, values_only=True):
        c0 = str(r[0]).strip() if r[0] is not None else ""
        if c0 in ("逐架无人机清单", "共享电池库存"):
            sect = c0
            continue
        if c0.endswith("编号"):          # 各节内的表头行
            continue
        if sect == "逐架无人机清单" and c0.startswith("U"):
            drones.append(dict(id=c0, type=str(r[1]), loc=str(r[2])))
        elif sect == "共享电池库存" and c0 in ("A", "B", "C"):
            bats[c0] = dict(n=int(r[1]), T_full=float(r[2]))
    return types, drones, bats


def load_relays():
    """中继机型参数 + 逐架清单 + 共享能源组件库存。"""
    ws = _sheet("中继无人机数据.xlsx")
    t = None
    for r in _rows(ws, 3, 19):
        # 能源组件库存那一段的第一列也是 'R'，取到表头行就收手
        if r[0] == "R" and t is None:
            t = dict(name=r[1], m_empty=float(r[2]), m_comm=float(r[3]),
                     m_takeoff=float(r[4]), vc=float(r[5]), P_cruise=float(r[6]),
                     E_use=float(r[7]), rho=float(r[8]) / 100.0,
                     t_prep=float(r[9]), t_link=float(r[10]),
                     t_turn=float(r[11]), v_up=float(r[12]), v_dn=float(r[13]),
                     eta_up=float(r[14]), eta_dn=float(r[15]),
                     P_hover=float(r[16]), P_comm=float(r[17]),
                     h_max=float(r[18]))
    drones, bats = [], {}
    sect = None
    for r in ws.iter_rows(min_row=1, values_only=True):
        c0 = str(r[0]).strip() if r[0] is not None else ""
        if c0 in ("逐架中继无人机清单", "共享能源组件库存"):
            sect = c0
            continue
        if c0.endswith("编号"):
            continue
        if sect == "逐架中继无人机清单" and c0.startswith("R0"):
            drones.append(dict(id=c0, type=str(r[1]), loc=str(r[2])))
        elif sect == "共享能源组件库存" and c0 == "R":
            bats[c0] = dict(n=int(r[1]), T_full=float(r[2]))
    return t, drones, bats


def load_comm():
    """通信参数。"""
    ws = _sheet("通信链路参数.xlsx")
    p = {}
    for r in _rows(ws, 3, 5):
        p[(str(r[0]), str(r[1]))] = float(r[4])
    return p


# ══════════════════════════════════════════════════════════
# 运输物理
# ══════════════════════════════════════════════════════════
class Transport:
    """给定机型参数与地理，计算航段/架次的时间与能耗。"""

    def __init__(self, geo: Geo, nodes: dict, types: dict):
        self.geo, self.nodes, self.types = geo, nodes, types
        self._cache = {}

    def op_alt(self, nid):
        """作业高度：O01 取地面海拔，服务区取地面海拔 + 30 m。"""
        n = self.nodes[nid]
        return n["alt"] + (30.0 if n["kind"] == "S" else 0.0)

    def cruise_alt(self, i, j):
        k = ("alt", i, j)
        if k in self._cache:
            return self._cache[k]
        a, b = self.nodes[i], self.nodes[j]
        z = self.geo.elev_line(a["lon"], a["lat"], b["lon"], b["lat"]) + 50.0
        self._cache[k] = z
        return z

    def seg(self, g, i, j, q):
        """航段 (i→j)，机型 g，载荷 q kg → dict(t, E, E_hor, E_up, ...)。"""
        k = ("seg", g, i, j, round(q, 6))
        if k in self._cache:
            return self._cache[k]
        T = self.types[g]
        a, b = self.nodes[i], self.nodes[j]
        d = self.geo.dist(a["lon"], a["lat"], b["lon"], b["lat"])
        zc = self.cruise_alt(i, j)
        h_up = max(zc - self.op_alt(i), 0.0)
        h_dn = max(zc - self.op_alt(j), 0.0)
        t = h_up / T["v_up"] + d / T["vc"] + h_dn / T["v_dn"]
        Lq = T["L0"] - (T["L0"] - T["LF"]) * (q / T["Q"]) ** 1.5
        E_hor = d / Lq * T["E_use"] if Lq > 0 else float("inf")
        E_up = (T["m_empty"] + q) * G0 * h_up / T["eta_up"] / J2KWH
        r = dict(t=t, E=E_hor + E_up, E_hor=E_hor, E_up=E_up, d=d,
                 h_up=h_up, h_dn=h_dn, z_cruise=zc)
        self._cache[k] = r
        return r

    def sortie_energy(self, g, route):
        """架次能耗 = 各航段能耗之和；route 形如 ['O01','S001','S002','O01']。

        载荷在每投送一个服务区后减少，故各段按分段载荷算。
        """
        T = self.types[g]
        E = 0.0
        q = 0.0
        # 先算该架次的总载荷（route 中间节点）
        return E


def charge_time(soc, T_full):
    """两阶段等效充电模型（题目附录 2）。"""
    if soc >= 1.0:
        return 0.0
    if soc < 0.90:
        return T_full * (0.65 * (0.90 - soc) / 0.90 + 0.35)
    return T_full * 0.35 * (1.0 - soc) / 0.10


# ══════════════════════════════════════════════════════════
# 通信
# ══════════════════════════════════════════════════════════
class Comms:
    """链路预算（题目附录 3）。"""

    def __init__(self, geo: Geo, nodes: dict, p: dict):
        self.geo, self.nodes = geo, nodes
        self.f = p[("传播参数", "载波频率（MHz）")]           # MHz
        self.Lsys = p[("传播参数", "系统损耗（dB）")]
        self.Lobs = p[("传播参数", "地形遮挡附加损耗（dB）")]
        self.Psens = p[("接收参数", "接收灵敏度（dBm）")]
        self.M = p[("接收参数", "衰落裕量（dB）")]
        self.P_th = self.Psens + self.M
        self.ent = {
            "U": dict(Pt=p[("运输无人机", "发射功率（dBm）")],
                      G=p[("运输无人机", "天线增益（dBi）")]),
            "Ra": dict(Pt=p[("中继接入端", "发射功率（dBm）")],
                       G=p[("中继接入端", "天线增益（dBi）")]),
            "Rb": dict(Pt=p[("中继回传端", "发射功率（dBm）")],
                       G=p[("中继回传端", "天线增益（dBi）")]),
            "G01": dict(Pt=p[("固定网关 G01", "发射功率（dBm）")],
                        G=p[("固定网关 G01", "天线增益（dBi）")]),
        }
        self.h_gw = p[("固定网关 G01", "天线离地高度（m）")]
        self._c = {}

    def Lmax(self, a, b):
        """双向门限 min(L_max^{a→b}, L_max^{b→a})。"""
        A, B = self.ent[a], self.ent[b]
        lab = A["Pt"] + A["G"] + B["G"] - self.Lsys - self.P_th
        lba = B["Pt"] + B["G"] + A["G"] - self.Lsys - self.P_th
        return min(lab, lba)

    def blocked(self, pa, pb):
        """地形遮挡：|pa-pb| 连线是否被 DEM 挡住。pa, pb = (lon, lat, 海拔 m)。"""
        k = ("blk", tuple(np.round(pa, 7)), tuple(np.round(pb, 7)))
        if k in self._c:
            return self._c[k]
        n = max(int(self.geo.dist(pa[0], pa[1], pb[0], pb[1]) / 30.0) + 1, 2)
        t = np.linspace(0.0, 1.0, n)
        lo = pa[0] + (pb[0] - pa[0]) * t
        la = pa[1] + (pb[1] - pa[1]) * t
        zz = pa[2] + (pb[2] - pa[2]) * t
        gz = self.geo.elev(lo, la)
        # 端点处不做判断（两端就是天线本身），留一格余量
        m = (gz > zz + 1e-9)
        m[0] = m[-1] = False
        r = bool(np.any(m))
        self._c[k] = r
        return r

    def fspl(self, pa, pb):
        D = self.geo.dist(pa[0], pa[1], pb[0], pb[1]) / 1000.0
        return 32.45 + 20 * math.log10(self.f) + 20 * math.log10(max(D, 1e-6))

    def link_ok(self, ea, pa, eb, pb):
        """端点在 ea/eb，三维位置 pa/pb → 双向链路是否可用。"""
        L = self.fspl(pa, pb) + (self.Lobs if self.blocked(pa, pb) else 0.0)
        return L <= self.Lmax(ea, eb), L


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    geo = Geo()
    nodes = load_nodes()
    types, drones, bats = load_drones()
    geo.set_origin(nodes["O01"]["lon"], nodes["O01"]["lat"])
    T = Transport(geo, nodes, types)
    print("节点", len(nodes), "| 服务区", sum(1 for n in nodes.values()
                                             if n["kind"] == "S"))
    boxes = load_boxes()
    print("货箱", len(boxes), "总质量",
          round(sum(b["mass"] for b in boxes), 1), "kg")
    print("机型", {k: (v["Q"], v["E_use"]) for k, v in types.items()})
    print("无人机", [d["id"] for d in drones], "电池", bats)
    rt, rd, rb = load_relays()
    print("中继", rt["m_takeoff"], rt["E_use"], "架",
          [d["id"] for d in rd], rb)
    c = Comms(geo, nodes, load_comm())
    print("网关门限 U↔G01", round(c.Lmax("U", "G01"), 2), "dB")
    print("O01→S001 地面", nodes["O01"]["alt"], nodes["S001"]["alt"])
    s = T.seg("A", "O01", "S001", 10.0)
    print("A 机 O01→S001 载荷10kg: t=%.1fs E=%.4f kWh (hor %.4f up %.4f) d=%.0fm"
          % (s["t"], s["E"], s["E_hor"], s["E_up"], s["d"]))
