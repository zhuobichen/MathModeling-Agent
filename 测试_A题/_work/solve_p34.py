"""2022 A题 · 问题3、问题4 求解"""
import json, time
import numpy as np, openpyxl
from scipy.optimize import minimize
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import model_core as M, model3 as M3

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
OUT = {}

# ═══ 问题 3 ═══
print("=" * 62); print("问题 3"); print("=" * 62)
t, Y = M3.solve_p3(3, c=10000.0, c_theta=1000.0)
print(f"采样 {len(t)} 点，t∈[{t[0]:.1f},{t[-1]:.2f}]s，步长 0.2 s")

wb = openpyxl.Workbook(); ws = wb.active; ws.title = "Sheet1"
ws["A1"] = "时间 (s)"; ws["B1"] = "浮子"; ws["F1"] = "振子"
hdr = ["垂荡位移 (m)", "垂荡速度 (m/s)", "纵摇角位移 ", "纵摇角速度 (s-1)"]
for k, hh in enumerate(hdr):
    ws.cell(2, 2 + k, hh); ws.cell(2, 6 + k, hh)
for i in range(len(t)):
    r = i + 3
    ws.cell(r, 1, round(float(t[i]), 4))
    for j, col in enumerate((0, 1, 2, 3)):
        ws.cell(r, 2 + j, round(float(Y[col][i]), 8))     # 浮子 z,vz,θ,ωθ
    for j, col in enumerate((4, 5, 6, 7)):
        ws.cell(r, 6 + j, round(float(Y[col][i]), 8))     # 振子 s,vs,φ,ωφ
ws.column_dimensions["A"].width = 12
for c_ in "BCDEFGHI": ws.column_dimensions[c_].width = 16
wb.save("result3.xlsx")
print("已写出 result3.xlsx")

rows = []
for tm in [10, 20, 40, 60, 100]:
    i = int(round(tm / 0.2))
    rows.append(dict(t=tm, buoy=[float(Y[k, i]) for k in (0, 1, 2, 3)],
                     osc=[float(Y[k, i]) for k in (4, 5, 6, 7)]))
m = t > 120
pl = 10000.0 * Y[5] ** 2; pr = 1000.0 * Y[7] ** 2
P_lin = float(np.trapezoid(pl[m], t[m]) / (t[m][-1] - t[m][0]))
P_rot = float(np.trapezoid(pr[m], t[m]) / (t[m][-1] - t[m][0]))
print(f"稳态平均: 线性阻尼 {P_lin:.4f} W + 旋转阻尼 {P_rot:.6f} W = {P_lin+P_rot:.4f} W")
print(f"{'t(s)':>5} | {'浮子 z':>11} {'浮子 vz':>11} {'浮子 θ':>11} {'浮子 ωθ':>11} | {'振子 s':>11} {'振子 vs':>11} {'中轴 φ':>11} {'中轴 ωφ':>11}")
for r_ in rows:
    print(f"{r_['t']:>5} | " + " ".join(f"{v:>11.6f}" for v in r_["buoy"])
          + " | " + " ".join(f"{v:>11.6f}" for v in r_["osc"]))
OUT["p3_marks"] = rows
OUT["p3_power"] = dict(linear=P_lin, rotary=P_rot, total=P_lin + P_rot)

# ═══ 问题 4 ═══
print(); print("=" * 62); print("问题 4：二维寻优 (c, c_θ)"); print("=" * 62)
t0 = time.time(); cache = {}
def P4(c, cth):
    key = (round(float(c), 2), round(float(cth), 2))
    if key not in cache:
        cache[key] = M3.avg_power3(4, c=key[0], c_theta=key[1])[0]
    return cache[key]

cs = np.linspace(0, 100000, 41); cts = np.linspace(0, 100000, 21)
Z = np.array([[P4(c, ct) for c in cs] for ct in cts])
gi, gj = np.unravel_index(np.argmax(Z), Z.shape)
print(f"网格最优: c={cs[gj]:.0f}, c_θ={cts[gi]:.0f}, P={Z[gi,gj]:.4f} W  ({time.time()-t0:.1f}s)")

res = minimize(lambda x: -P4(np.clip(x[0],0,100000), np.clip(x[1],0,100000)),
               [float(cs[gj]), float(cts[gi])], method="Nelder-Mead",
               options=dict(xatol=1e-2, fatol=1e-8, maxiter=500))
c_opt = float(np.clip(res.x[0],0,100000)); ct_opt = float(np.clip(res.x[1],0,100000))
P_opt = P4(c_opt, ct_opt)
print(f"Nelder-Mead: c*={c_opt:.2f}, c_θ*={ct_opt:.2f}, P*={P_opt:.4f} W")
print(f"\nc_θ 敏感性（固定 c=c*）:")
for ct in [0, 1000, 10000, 100000]:
    print(f"  c_θ={ct:>7}: P = {P4(c_opt, ct):.6f} W")
print(f"c 敏感性（固定 c_θ=0）:")
for c in [10000, 20000, 30000, 40000, 60000]:
    print(f"  c={c:>7}: P = {P4(c, 0):.6f} W")
OUT["p4"] = dict(c_opt=c_opt, c_theta_opt=ct_opt, P_opt=P_opt, evals=len(cache))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
ax[0].plot(t, Y[0], label="浮子垂荡 z"); ax[0].plot(t, Y[4], label="振子滑动 s")
ax[0].set_xlim(0, 60); ax[0].set_xlabel("t (s)"); ax[0].set_ylabel("位移 (m)")
ax[0].set_title("问题3：垂荡与振子滑动"); ax[0].legend(); ax[0].grid(alpha=.3)
im = ax[1].contourf(cs, cts, Z, levels=22, cmap="viridis")
ax[1].plot([c_opt], [ct_opt], "r*", ms=16, label=f"最优 c*={c_opt:.0f}")
ax[1].set_xlabel("直线阻尼系数 c"); ax[1].set_ylabel("旋转阻尼系数 c_θ")
ax[1].set_title("问题4：平均输出功率曲面"); ax[1].legend()
plt.colorbar(im, ax=ax[1], label="P (W)")
plt.tight_layout(); plt.savefig("fig_p34.png", dpi=130)
json.dump(OUT, open("out_p34.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"\n总耗时 {time.time()-t0:.1f}s，周期解评估 {len(cache)} 次")
