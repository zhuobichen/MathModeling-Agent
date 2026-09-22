# -*- coding: utf-8 -*-
"""对照：只用 0:00 计划、不做 6/12/18 三次调整，值多少钱（正式口径）。"""
from __future__ import annotations
import os, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c_data as D, c_solve as C, mg_core as M
WARM = 31
orig = C.ADJ
C.ADJ = []
tot = em = 0.0
load, pv_act = D.load_att2(); fc = D.load_att3(); E, _, _ = D.load_att1()
soc = 6000.0
for d in range(365):
    r = C.solve_one_day_q3(d, E, load[d], pv_act[d], fc, soc)
    if d >= WARM:
        tot += r["cost"]; em += r["emerg_kwh"]
    soc = r["soc1"]
C.ADJ = orig
print(f"只用 0:00 计划、不调整：{tot:.2f} 元  紧急 {em:.1f} kWh")
print(f"（正式：13802767.33 元  紧急 206928.1 kWh）")
