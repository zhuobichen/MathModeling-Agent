"""MathModel 数学建模数据分析自动化系统 CLI 入口。

Usage:
    python main.py run --question "题目.md" --data "data/"
    python main.py run --question "题目.md" --data "data/" --resume <task_id>
    python main.py resume <task_id>
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.core.workflow import MathModelWorkFlow
from app.utils.log_util import logger


def read_question(question_path: str) -> str:
    """读取题目内容。

    支持 .txt/.md/.pdf 路径，或直接传入题目文本字符串。
    PDF 文件返回原始路径（由 workflow 内部调用 parse_pdf_question 处理），
    文本文件返回读取后的内容。
    """
    if os.path.isfile(question_path):
        if question_path.lower().endswith(".pdf"):
            return question_path  # PDF 保留路径，由 workflow 的 PDF 解析流程处理
        with open(question_path, "r", encoding="utf-8") as f:
            return f.read()
    return question_path


async def run_pipeline(question: str, data: str | None = None, resume: str | None = None,
                       fmt: str = "markdown", no_review: bool = False):
    """运行 5-Agent 流水线，支持从头开始或断点续跑。"""
    from app.schemas.enums import FormatOutPut
    format_output = FormatOutPut.LaTeX if fmt == "latex" else FormatOutPut.Markdown
    workflow = MathModelWorkFlow()

    if resume:
        print("=" * 60)
        print(f"  从断点恢复: {resume}")
        print("=" * 60)
        result = await workflow.resume_execute(resume, data, format_output=format_output)
    else:
        question_text = read_question(question)
        print("=" * 60)
        print("  数学建模数据分析自动化系统")
        print("  5-Agent Pipeline: Parser → Modeler → Coder → Writer → Reviewer")
        print("=" * 60)
        print()
        result = await workflow.execute(question_text, data, format_output=format_output,
                                        review_enabled=not no_review)

    print()
    print("=" * 60)
    print(f"  任务完成! Task ID: {result.get('task_id')}")
    print(f"  工作目录: {result.get('work_dir')}")
    if result.get("review_result"):
        review = result["review_result"]
        print(f"  论文评分: {review.get('overall_score', 'N/A')}/10")
        if review.get("suggestions"):
            print("  改进建议:")
            for s in review["suggestions"]:
                print(f"    - {s}")
    print(f"  论文路径: {result.get('paper_path')}")
    print("=" * 60)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="数学建模数据分析自动化系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py run --question "题目.pdf" --data "data/"
    python main.py run --question "题目.pdf" --data "data/" --resume 20260514-120000-abcdef12
    python main.py resume 20260514-120000-abcdef12
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # run 命令
    run_parser = subparsers.add_parser("run", help="运行完整流水线或断点续跑")
    run_parser.add_argument("--question", "-q", required=True, help="题目文件路径")
    run_parser.add_argument("--data", "-d", default=None, help="数据文件或目录路径")
    run_parser.add_argument("--resume", "-r", default=None, metavar="TASK_ID",
                          help="从指定任务ID的断点续跑（跳过已完成阶段）")
    run_parser.add_argument("--format", "-f", default="markdown", choices=["markdown", "latex"],
                          help="输出格式: markdown→DOCX, latex→PDF (默认: markdown)")
    run_parser.add_argument("--no-review", action="store_true",
                          help="跳过所有人工审核，全自动运行")

    # resume 命令（快捷方式）
    resume_parser = subparsers.add_parser("resume", help="从断点续跑（--resume 的快捷方式）")
    resume_parser.add_argument("task_id", help="要恢复的任务ID")
    resume_parser.add_argument("--data", "-d", default=None, help="补充数据文件路径")
    resume_parser.add_argument("--format", "-f", default="markdown", choices=["markdown", "latex"],
                              help="输出格式: markdown→DOCX, latex→PDF (默认: markdown)")

    args = parser.parse_args()

    if args.command == "run":
        asyncio.run(run_pipeline(args.question, args.data, args.resume, args.format,
                                 no_review=args.no_review))
    elif args.command == "resume":
        asyncio.run(run_pipeline("", args.data, args.task_id, args.format))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
