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

TASK_ID = "cumcm2026a_paper"


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

    cu.md_2_docx(TASK_ID)

    out = os.path.join(wd, "res.docx")
    dst = os.path.join(DELIV, "paper.docx")
    shutil.copy(out, dst)
    print("已生成", dst, f"({os.path.getsize(dst)} 字节)")

    # res.md 会被 md_2_docx 回写（表格被替换成占位符后又还原），
    # 交付一份归一化后的 markdown 以便核对
    with open(os.path.join(wd, "res.md"), "r", encoding="utf-8") as f:
        back = f.read()
    shutil.copy(os.path.join(wd, "res.md"), os.path.join(DELIV, "paper_norm.md"))
    print(f"回写后 res.md {len(back)} 字符，占位符残留 {back.count('[TABLE_')}")


if __name__ == "__main__":
    main()
