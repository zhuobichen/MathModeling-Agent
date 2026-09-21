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
                       fmt: str = "markdown", no_review: bool = False,
                       driver: str | None = None, pipeline: str | None = None):
    """运行流水线，支持从头开始或断点续跑。

    driver:
      api  —— 调 LLM API 自动跑完整流水线
      file —— 提示词落盘等待外部写入应答（由外部 Agent/人 逐步驱动，可续跑）
      echo —— 占位应答，只验证框架机制，不调用任何模型
    """
    from app.schemas.enums import FormatOutPut
    format_output = FormatOutPut.LaTeX if fmt == "latex" else FormatOutPut.Markdown
    workflow = MathModelWorkFlow()

    if resume:
        print("=" * 60)
        print(f"  从断点恢复: {resume}")
        print("=" * 60)
        result = await workflow.resume_execute(
            resume, data, format_output=format_output,
            driver=driver, pipeline_path=pipeline,
            review_enabled=not no_review,
        )
    else:
        question_text = read_question(question)
        print("=" * 60)
        print("  数学建模数据分析自动化系统")
        print("  Pipeline: parse → model → analyze → write → review → export")
        print("=" * 60)
        print()
        result = await workflow.execute(
            question_text, data, format_output=format_output,
            review_enabled=not no_review,
            driver=driver, pipeline_path=pipeline,
        )

    print()
    print("=" * 60)
    status = result.get("status")
    if status == "awaiting_input":
        print(f"  等待外部输入 · Task ID: {result.get('task_id')}")
        print("  请按下列路径填写应答，然后 resume：")
        for p in result.get("pending") or []:
            print(f"    请求: {p['request']}")
            print(f"    → 写入: {p['response']}")
        print(f"  恢复命令: python main.py resume {result.get('task_id')} "
              + (f"--driver {driver} " if driver else ""))
    else:
        print(f"  任务{('完成' if status == 'completed' else '失败')}! "
              f"Task ID: {result.get('task_id')}")
        print(f"  工作目录: {result.get('work_dir')}")
        if result.get("error"):
            print(f"  错误: {result['error']}")
        if result.get("failures"):
            print(f"  失败阶段: {', '.join(result['failures'])}")
        if result.get("review_result"):
            review = result["review_result"]
            print(f"  论文评分: {review.get('overall_score', 'N/A')}/10")
            for s in (review.get("suggestions") or [])[:5]:
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
    run_parser.add_argument("--driver", default=None, choices=["api", "file", "echo"],
                          help="智能来源: api=调模型自动跑, file=提示词落盘由外部驱动, "
                               "echo=占位(只验证机制)")
    run_parser.add_argument("--pipeline", default=None, metavar="TOML",
                          help="自定义流水线配置文件（默认 pipelines/competition.toml）")

    # resume 命令（快捷方式）
    resume_parser = subparsers.add_parser("resume", help="从断点续跑（--resume 的快捷方式）")
    resume_parser.add_argument("task_id", help="要恢复的任务ID")
    resume_parser.add_argument("--data", "-d", default=None, help="补充数据文件路径")
    resume_parser.add_argument("--format", "-f", default="markdown", choices=["markdown", "latex"],
                              help="输出格式: markdown→DOCX, latex→PDF (默认: markdown)")
    resume_parser.add_argument("--driver", default=None, choices=["api", "file", "echo"],
                              help="智能来源（需与首次运行一致）")
    resume_parser.add_argument("--pipeline", default=None, metavar="TOML",
                              help="自定义流水线配置文件")

    # components 命令：列出已注册组件（便于确认配置里能引用什么）
    subparsers.add_parser("components", help="列出已注册的组件与它们声明的依赖")

    args = parser.parse_args()

    if args.command == "run":
        asyncio.run(run_pipeline(args.question, args.data, args.resume, args.format,
                                 no_review=args.no_review,
                                 driver=args.driver, pipeline=args.pipeline))
    elif args.command == "resume":
        asyncio.run(run_pipeline("", args.data, args.task_id, args.format,
                                 driver=args.driver, pipeline=args.pipeline))
    elif args.command == "components":
        from app.core.components import register_builtins
        from app.core.component import component_registry
        register_builtins()
        print(f"{'组件':<20} {'入':<16} {'出':<16} {'模型角色':<10} 工具")
        print("-" * 90)
        for c in component_registry.describe():
            print(f"{c['name']:<20} {str(c['in']):<16} {c['out']:<16} "
                  f"{str(c['model_role'] or '-'):<10} {','.join(c['tools']) or '-'}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
