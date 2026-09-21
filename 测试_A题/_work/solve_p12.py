"""2022 A题 · 问题1、问题2 求解"""
import json, time
import numpy as np
import openpyxl
from scipy.optimize import minimize_scalar, minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import model_core as M

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
OUT = {}

# ═══════════════ 问题 1 ═══════════════
print("=" * 62); print("问题 1"); print("=" * 62)
t,  Y  = M.solve_heave(1, c=10000.0)
t2, Y2 = M.solve_heave(1, alpha=0.5, c0=10000.0)
print(f"采样 {len(t)} 点，t ∈ [{t[0]:.1f}, {t[-1]:.2f}] s，步长 0.2 s")

def write_result(path, tt, yy):
    wb = openpyxl.Workbook(); ws = wb.active; ws.title = "Sheet1"
    ws["A1"] = "时间 (s)"; ws["B1"] = "浮子"; ws["D1"] = "振子"
    ws["B2"] = "位移 (m)"; ws["C2"] = "速度 (m/s)"; ws["D2"] = "位移 (m)"; ws["E2"] = "速度 (m/s)"
    for i in range(len(tt)):
        r = i + 3
        ws.cell(r, 1, round(float(tt[i]), 4))
        for j in range(4):
            ws.cell(r, 2 + j, round(float(yy[j][i]), 6))
    ws.column_dimensions["A"].width = 12
    for col in "BCDE": ws.column_dimensions[col].width = 14
    wb.save(path)

write_result("result1-1.xlsx", t, Y)
write_result("result1-2.xlsx", t2, Y2)
print("已写出 result1-1.xlsx / result1-2.xlsx")

rows = []
for tm in [10, 20, 40, 60, 100]:
    i = int(round(tm / 0.2))
    d = dict(t=tm)
    for nm, arr in (("a", Y), ("b", Y2)):
        d[nm] = [float(arr[k, i]) for k in range(4)]
    rows.append(d)
OUT["p1_marks"] = rows
print(f"{'t(s)':>5} | {'情形1: 浮子d':>12} {'浮子v':>12} {'振子d':>12} {'振子v':>12} | "
      f"{'情形2: 浮子d':>12} {'浮子v':>12} {'振子d':>12} {'振子v':>12}")
for r in rows:
    print(f"{r['t']:>5} | " + " ".join(f"{v:>12.6f}" for v in r["a"])
          + " | " + " ".join(f"{v:>12.6f}" for v in r["b"]))

# ═══════════════ 问题 2 情形(1) ═══════════════
print(); print("=" * 62); print("问题 2 情形(1)：线性阻尼最优 c"); print("=" * 62)
r1 = minimize_scalar(lambda c: -M.avg_power_linear_exact(2, float(c)),
                     bounds=(1.0, 100000.0), method="bounded", options=dict(xatol=1e-4))
c_opt, P_opt = float(r1.x), -float(r1.fun)
print(f"最优阻尼系数 c* = {c_opt:.4f} N·s/m")
print(f"最大平均输出功率 P̄* = {P_opt:.6f} W")
h = 50.0
print(f"单峰性复核: P(c*-50)={M.avg_power_linear_exact(2,c_opt-h):.4f}  "
      f"P(c*)={M.avg_power_linear_exact(2,c_opt):.4f}  "
      f"P(c*+50)={M.avg_power_linear_exact(2,c_opt+h):.4f}")
OUT["p2_case1"] = dict(c_opt=c_opt, P_opt=P_opt)

# ═══════════════ 问题 2 情形(2) ═══════════════
print(); print("=" * 62); print("问题 2 情形(2)：幂律阻尼最优 (c0, α)"); print("=" * 62)
t0 = time.time()
cache = {}
def P2(c0, al):
    key = (round(c0, 4), round(al, 6))
    if key not in cache:
        cache[key] = M.avg_power_periodic(2, alpha=al, c0=c0)[0]
    return cache[key]

grid_c0 = [10000, 30000, 50000, 80000, 100000]
grid_al = [0.0, 0.25, 0.5, 0.75, 1.0]
best = max(((c0, al, P2(c0, al)) for c0 in grid_c0 for al in grid_al), key=lambda z: z[2])
print(f"粗网格最优: c0={best[0]}, α={best[1]}, P̄={best[2]:.4f} W  ({time.time()-t0:.1f}s, 评估 {len(cache)} 次)")

res = minimize(lambda x: -P2(float(x[0]), float(np.clip(x[1], 0, 1))),
               [best[0], best[1]], method="Nelder-Mead",
               options=dict(xatol=1e-4, fatol=1e-7, maxiter=600))
c0_opt, al_opt = float(res.x[0]), float(np.clip(res.x[1], 0, 1))
P2_opt = P2(c0_opt, al_opt)
print(f"Nelder-Mead 精化: c0*={c0_opt:.4f}, α*={al_opt:.6f}, P̄*={P2_opt:.6f} W  ({time.time()-t0:.1f}s)")
print("边界对照:")
for al in (0.0, 1.0):
    print(f"   α={al}: 最优 P̄ = {max(P2(c, al) for c in np.linspace(5000,100000,40)):.4f} W")
OUT["p2_case2"] = dict(c0_opt=c0_opt, alpha_opt=al_opt, P_opt=P2_opt,
                       evals=len(cache), seconds=time.time() - t0)

# ═══════════════ 图 ═══════════════
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))

cs = np.linspace(1000, 100000, 200)
ax[0].plot(cs, [M.avg_power_linear_exact(2, c) for c in cs], "b-", lw=2)
ax[0].plot([c_opt], [P_opt], "r*", ms=16, label=f"最优 c*={c_opt:.0f}\nP̄*={P_opt:.1f} W")
ax[0].set_xlabel("直线阻尼系数 c (N·s/m)"); ax[0].set_ylabel("稳态平均输出功率 P̄ (W)")
ax[0].set_title("问题2 情形(1)：线性阻尼"); ax[0].legend(); ax[0].grid(alpha=.3)

c0s = np.linspace(8000, 100000, 46); als = np.linspace(0, 1, 26)
Z = np.array([[P2(c0, al) for c0 in c0s] for al in als])
im = ax[1].contourf(c0s, als, Z, levels=24, cmap="viridis")
ax[1].plot([c0_opt], [al_opt], "r*", ms=16, label=f"最优 ({c0_opt:.0f}, {al_opt:.3f})")
ax[1].set_xlabel("比例系数 c0"); ax[1].set_ylabel("幂指数 α")
ax[1].set_title("问题2 情形(2)：幂律阻尼"); ax[1].legend(loc="upper right")
plt.colorbar(im, ax=ax[1], label="P̄ (W)")

ax[2].plot(t, Y[0], label="浮子 情形1"); ax[2].plot(t, Y[2], label="振子 情形1")
ax[2].plot(t, Y2[0], "--", label="浮子 情形2"); ax[2].plot(t, Y2[2], "--", label="振子 情形2")
ax[2].set_xlabel("t (s)"); ax[2].set_ylabel("垂荡位移 (m)"); ax[2].set_xlim(0, 40)
ax[2].set_title("问题1：前 40 s 位移时程"); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
plt.tight_layout(); plt.savefig("fig_p12.png", dpi=130)

json.dump(OUT, open("out_p12.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"\n完成，总耗时 {time.time()-t0:.1f}s（含 2D 曲面 {len(cache)} 次周期解）")
