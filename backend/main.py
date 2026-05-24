"""MathModel 数学建模数据分析自动化系统 CLI 入口。

Usage:
    python main.py run --question "题目.md" --data "data/"
    python main.py run --question "题目内容..." --data "data.csv"
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# 将 backend 目录加入 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

from app.core.workflow import MathModelWorkFlow
from app.utils.log_util import logger


def read_question(question_path: str) -> str:
    """读取题目内容。

    支持 .txt/.md 文件路径，或者直接传入题目文本。
    """
    if os.path.isfile(question_path):
        with open(question_path, "r", encoding="utf-8") as f:
            return f.read()
    return question_path


async def run_pipeline(question: str, data: str | None = None):
    """运行完整的 5-Agent 流水线。

    Args:
        question: 题目文本或文件路径。
        data: 数据文件或目录路径（可选）。
    """
    question_text = read_question(question)

    print("=" * 60)
    print("  数学建模数据分析自动化系统")
    print("  5-Agent Pipeline: Parser → Modeler → Coder → Writer → Reviewer")
    print("=" * 60)
    print()

    workflow = MathModelWorkFlow()
    result = await workflow.execute(question_text, data)

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
Example:
    python main.py run --question "test_data/题目.md" --data "test_data/data.csv"
    python main.py run --question "请分析某城市2010-2025年PM2.5浓度数据，预测未来3年趋势" --data "data/"
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # run 命令
    run_parser = subparsers.add_parser("run", help="运行完整流水线")
    run_parser.add_argument(
        "--question", "-q",
        required=True,
        help="题目文件路径（.md/.txt）或直接题目文本",
    )
    run_parser.add_argument(
        "--data", "-d",
        default=None,
        help="数据文件或目录路径（可选）",
    )

    args = parser.parse_args()

    if args.command == "run":
        asyncio.run(run_pipeline(args.question, args.data))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
