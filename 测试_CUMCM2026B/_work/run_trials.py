# -*- coding: utf-8 -*-
"""在本地模拟器上做「演练式」批量测试，统计两项指标。

**重要说明**：官方模拟器需要参赛队号在线登录，其下载链接已失效，且启动测试的
截止时间（北京时间 2026-09-13 17:30）已过，因此**正式测试与其加密日志无法产出**。
本脚本给出的是在自建同协议模拟器上的本地复现结果，不能当作正式测试成绩。

被清除干扰源个数的比例 = 被清除干扰源的个数 / 干扰源总数
平均定位清除时间       = 定位清除总时间 / 被清除干扰源的个数
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import robot
import simulator as S

HERE = os.path.dirname(os.path.abspath(__file__))


def inner_points():
    return [np.asarray(p, float) for p in robot.survey_ring(6, margin=0.70)]


def inner_outer_points(k2=16, r2=1900.0):
    pts = inner_points()
    for i in range(k2):
        th = 2 * math.pi * i / k2 + math.pi / k2
        pts.append(np.array([r2 * math.cos(th), r2 * math.sin(th)]))
    return pts


def batch(tag, pts, dir_frac, seeds, verbose=True):
    rows = []
    for s in seeds:
        srcs = S.sample_case(np.random.default_rng(s), dir_frac=dir_frac)
        sim = S.Simulator(srcs, seed=s)
        rb = robot.Robot(sim)
        t0 = time.time()
        st = rb.run(survey_pts=list(pts), interleave=False)
        wall = time.time() - t0
        st["wall_s"] = wall
        st["seed"] = s
        rows.append(st)
    agg = dict(
        tag=tag, dir_frac=dir_frac, n_trials=len(rows),
        n_points=len(pts),
        ratio_mean=float(np.mean([r["ratio"] for r in rows])),
        ratio_min=float(np.min([r["ratio"] for r in rows])),
        avg_time_mean=float(np.mean([r["avg_time_s"] for r in rows])),
        avg_time_median=float(np.median([r["avg_time_s"] for r in rows])),
        total_time_mean=float(np.mean([r["virtual_time_s"] for r in rows])),
        n_requests_mean=float(np.mean([r["n_requests"] for r in rows])),
        t_move_mean=float(np.mean([r["t_move"] for r in rows])),
        t_measure_mean=float(np.mean([r["t_measure"] for r in rows])),
        t_clear_mean=float(np.mean([r["t_clear"] for r in rows])),
        t_switch_mean=float(np.mean([r["t_switch"] for r in rows])),
        solver_wall_mean=float(np.mean([r["wall_s"] for r in rows])),
        rows=rows,
    )
    if verbose:
        print(f"[{tag}] 定向占比 {dir_frac:.1f}  侦察点 {len(pts)}  "
              f"{agg['n_trials']} 局")
        print(f"   清除率  {agg['ratio_mean']*100:5.1f}%（最低 {agg['ratio_min']*100:.1f}%）")
        print(f"   平均定位清除时间  {agg['avg_time_mean']:7.1f} s"
              f"（中位 {agg['avg_time_median']:.1f}）")
        print(f"   定位清除总时间    {agg['total_time_mean']:7.1f} s")
        print(f"     其中移动 {agg['t_move_mean']:.0f} s  "
              f"检测 {agg['t_measure_mean']:.0f} s  "
              f"清除 {agg['t_clear_mean']:.0f} s  "
              f"换频道 {agg['t_switch_mean']:.0f} s")
        print(f"   指令条数 {agg['n_requests_mean']:.0f}   求解器耗时 {agg['solver_wall_mean']:.2f} s/局")
    return agg


def main():
    seeds = list(range(7000, 7030))
    out = {}
    print("=" * 78)
    print("问题 3（全为全向干扰源）")
    print("=" * 78)
    out["q3"] = batch("q3", inner_points(), 0.0, seeds)

    print()
    print("=" * 78)
    print("问题 4（全向 + 定向混合）")
    print("=" * 78)
    out["q4"] = {}
    for df in (0.3, 0.5, 0.7, 1.0):
        out["q4"][f"df={df}"] = batch(f"q4_df{df}", inner_outer_points(),
                                      df, seeds)
    # 对照：问题 4 若只用内环
    print()
    print("对照组：问题 4 只用内环（说明外环的必要性）")
    out["q4_inner_only"] = {}
    for df in (0.5, 1.0):
        out["q4_inner_only"][f"df={df}"] = batch(f"q4_inner_df{df}",
                                                 inner_points(), df, seeds,
                                                 verbose=True)

    with open(os.path.join(HERE, "out_trials.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print("\n已保存 out_trials.json")


if __name__ == "__main__":
    main()
