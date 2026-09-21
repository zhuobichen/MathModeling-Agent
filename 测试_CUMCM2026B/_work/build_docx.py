# -*- coding: utf-8 -*-
"""把 paper.md 经框架的归一化 + md_2_docx 管线编译成 Word。"""
from __future__ import annotations

import os
import shutil
import sys

BACKEND = r"D:\github_project\ZhaoWen_GitHub维护\MathModeling-Agent\backend"
sys.path.insert(0, BACKEND)
sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
DELIV = os.path.abspath(os.path.join(HERE, "..", "_交付"))
TASK_ID = "cumcm2026b_paper"


def main():
    os.makedirs(os.path.join(BACKEND, "project", "work_dir", TASK_ID),
                exist_ok=True)
    from app.utils.paper_normalizer import normalize_paper
    from app.utils import common_utils as cu

    raw = open(os.path.join(DELIV, "paper.md"), encoding="utf-8").read()
    norm = normalize_paper(raw)
    print(f"归一化 {len(norm.fixes_applied)} 项；字符 {len(raw)} → "
          f"{len(norm.normalized_content)}")
    for f in norm.fixes_applied:
        if "表" in f or "图" in f:
            print("   ", f)

    os.chdir(BACKEND)
    wd = os.path.abspath(cu.get_work_dir(TASK_ID))
    fig = os.path.join(wd, "figures")
    if os.path.isdir(fig):
        shutil.rmtree(fig)
    shutil.copytree(os.path.join(DELIV, "figures"), fig)
    with open(os.path.join(wd, "res.md"), "w", encoding="utf-8") as f:
        f.write(norm.normalized_content)
    cu.md_2_docx(TASK_ID)
    shutil.copy(os.path.join(wd, "res.docx"), os.path.join(DELIV, "paper.docx"))
    print("已生成", os.path.join(DELIV, "paper.docx"),
          os.path.getsize(os.path.join(DELIV, "paper.docx")), "字节")


if __name__ == "__main__":
    main()
