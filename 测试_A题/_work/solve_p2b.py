"""问题2 情形(2) 修正版：严格在可行域 [0,1e5]×[0,1] 内寻优"""
import json, time
import numpy as np
from scipy.optimize import minimize
import model_core as M

t0 = time.time()
cache = {}
def P2(c0, al):
    c0 = float(np.clip(c0, 0, 100000)); al = float(np.clip(al, 0, 1))
    key = (round(c0, 3), round(al, 6))
    if key not in cache:
        cache[key] = M.avg_power_periodic(2, alpha=al, c0=c0)[0]
    return cache[key]

print("=" * 62); print("问题 2 情形(2)：可行域内最优 (c0, α)"); print("=" * 62)

# ① 细网格全景（确认单峰性与边界行为）
c0s = np.linspace(5000, 100000, 60)
als = np.linspace(0, 1, 41)
Z = np.array([[P2(c, a) for c in c0s] for a in als])
gi, gj = np.unravel_index(np.argmax(Z), Z.shape)
print(f"细网格最优: c0={c0s[gj]:.1f}, α={als[gi]:.4f}, P̄={Z[gi,gj]:.6f} W")
print(f"  (网格 {len(c0s)}×{len(als)}={len(c0s)*len(als)} 点, {time.time()-t0:.1f}s)")

# α=0 应退化为线性(情形1)，作为正确性交叉验证
row0 = max(Z[0, :]); print(f"交叉验证 α=0（应≈情形1的 229.334）: {row0:.6f} W")
print(f"边界 α=1 最优: {Z[-1,:].max():.6f} W")

# ② 带惩罚的 Nelder-Mead（越界重罚，避免跑到域外）
def negP(x):
    pen = 0.0
    if x[0] < 0: pen += 1e3 * (-x[0])
    if x[0] > 100000: pen += 1e3 * (x[0] - 100000)
    if x[1] < 0: pen += 1e6 * (-x[1])
    if x[1] > 1: pen += 1e6 * (x[1] - 1)
    return -P2(x[0], x[1]) + pen

res = minimize(negP, [float(c0s[gj]), float(als[gi])], method="Nelder-Mead",
               options=dict(xatol=1e-3, fatol=1e-8, maxiter=800))
c0_opt = float(np.clip(res.x[0], 0, 100000)); al_opt = float(np.clip(res.x[1], 0, 1))
P_opt = P2(c0_opt, al_opt)
print(f"\nNelder-Mead（带惩罚）: c0*={c0_opt:.4f}, α*={al_opt:.6f}, P̄*={P_opt:.6f} W")
print(f"  越界检查: c0∈[0,1e5]? {0<=c0_opt<=100000}   α∈[0,1]? {0<=al_opt<=1}")

# ③ 局部加密复核
cc = np.linspace(max(0, c0_opt-8000), min(100000, c0_opt+8000), 25)
aa = np.linspace(max(0, al_opt-0.15), min(1, al_opt+0.15), 21)
Z2 = np.array([[P2(c, a) for c in cc] for a in aa])
i2, j2 = np.unravel_index(np.argmax(Z2), Z2.shape)
print(f"局部加密复核: c0={cc[j2]:.1f}, α={aa[i2]:.4f}, P̄={Z2[i2,j2]:.6f} W")

# ④ 结论取三者最优
cands = [(c0_opt, al_opt, P_opt),
         (float(cc[j2]), float(aa[i2]), float(Z2[i2, j2])),
         (float(c0s[gj]), float(als[gi]), float(Z[gi, gj]))]
C0, AL, PP = max(cands, key=lambda z: z[2])
print(f"\n★ 最优解: c0* = {C0:.2f},  α* = {AL:.6f},  最大平均输出功率 = {PP:.4f} W")
print(f"  对照情形(1)线性最优 229.3339 W，增益 {PP-229.3339:+.4f} W（{(PP/229.3339-1)*100:+.3f}%）")

out = json.load(open("out_p12.json", encoding="utf-8"))
out["p2_case2"] = dict(c0_opt=C0, alpha_opt=AL, P_opt=PP,
                       evals=len(cache), seconds=time.time()-t0)
json.dump(out, open("out_p12.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
np.save("p2b_grid.npy", dict(c0=c0s, al=als, Z=Z), allow_pickle=True)
print(f"\n总耗时 {time.time()-t0:.1f}s，评估周期解 {len(cache)} 次")
