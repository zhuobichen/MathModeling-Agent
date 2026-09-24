# -*- coding: utf-8 -*-
"""把 paper.md 经框架的归一化 + md_2_docx 管线编译成 Word。

复用 MathModeling-Agent 里已验证的那条路径：
  归一化（补表格题注、断行、引号） → pandoc + 论文模板 → python-docx 插真表格。
"""
from __future__ import annotations

import os
import shutil
import sys

BACKEND = r"D:\github_project\ZhaoWen_GitHub维护\MathModeling-Agent\backend"
sys.path.insert(0, BACKEND)
sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
DELIV = os.path.abspath(os.path.join(HERE, "..", "_交付"))

TASK_ID = "yansai2026d_paper"


def main():
    os.makedirs(os.path.join(BACKEND, "project", "work_dir", TASK_ID),
                exist_ok=True)
    from app.utils.paper_normalizer import normalize_paper
    from app.utils import common_utils as cu

    src = os.path.join(DELIV, "paper.md")
    with open(src, "r", encoding="utf-8") as f:
        raw = f.read()

    norm = normalize_paper(raw)
    print("归一化：" + "；".join(norm.fixes_applied) if norm.fixes_applied
          else "归一化：无改动")
    print(f"字符数 {len(raw)} → {len(norm.normalized_content)}")

    # get_work_dir 要求目录已存在，且是相对 backend 的相对路径
    os.chdir(BACKEND)
    wd = os.path.abspath(cu.get_work_dir(TASK_ID))
    # 插图随 res.md 一起放进工作目录（pandoc 用 --resource-path 找相对路径）
    fig_dst = os.path.join(wd, "figures")
    if os.path.isdir(fig_dst):
        shutil.rmtree(fig_dst)
    shutil.copytree(os.path.join(DELIV, "figures"), fig_dst)

    with open(os.path.join(wd, "res.md"), "w", encoding="utf-8") as f:
        f.write(norm.normalized_content)

    # 归一化后的 markdown 留一份交付：md_2_docx 之后 res.md 会被写成
    # 带 [TABLE_n] 占位符的中间态，那时再抄出来就是一份读不了的表。
    with open(os.path.join(DELIV, "paper_norm.md"), "w",
              encoding="utf-8") as f:
        f.write(norm.normalized_content)

    cu.md_2_docx(TASK_ID)

    out = os.path.join(wd, "res.docx")
    dst = os.path.join(DELIV, "paper.docx")
    shutil.copy(out, dst)
    print("已生成", dst, f"({os.path.getsize(dst)} 字节)")

    # res.md 会被 md_2_docx 回写成占位符态，这里只报个数好确认表格确实被
    # 抽走插进 docx 了（残留数应等于表格数）。
    with open(os.path.join(wd, "res.md"), "r", encoding="utf-8") as f:
        back = f.read()
    print(f"回写后 res.md {len(back)} 字符，占位符 {back.count('[TABLE_')} 个")


if __name__ == "__main__":
    main()
