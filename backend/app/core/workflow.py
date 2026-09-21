"""流水线入口 —— 薄封装。

本文件此前的 1575 行里，编排逻辑（阶段顺序、质量门、重写路径、断点）已迁移到：

- `app/core/component.py`   组件契约
- `app/core/gate.py`        质量门与统一重写环
- `app/core/pipeline/`      声明式加载与解释执行
- `pipelines/competition.toml`  阶段顺序（改流程改这里，不改 Python）

本文件现在只负责**装配**：准备任务上下文、构造驱动与资源、把外部服务
（Redis / HIL / CLI 审核）以回调形式注入解释器。
"""

import os
import shutil
from pathlib import Path

from app.config.setting import settings
from app.core.components import register_builtins
from app.core.component import component_registry
from app.core.drivers import ApiDriver, EchoDriver, FileDriver
from app.core.llm.llm_factory import LLMFactory
from app.core.pipeline import PipelineInterpreter, RunResult, load_pipeline
from app.core.evaluator import Evaluator
from app.schemas.enums import FormatOutPut
from app.schemas.response import ApprovalMessage, SystemMessage
from app.services.checkpoint_manager import checkpoint_manager
from app.services.redis_manager import redis_manager
from app.tools.interpreter_factory import create_interpreter
from app.tools.notebook_serializer import NotebookSerializer
from app.utils.common_utils import create_task_id
from app.utils.log_util import logger, set_trace_context

DEFAULT_PIPELINE = str(
    Path(__file__).resolve().parent.parent.parent.parent / "pipelines" / "competition.toml"
)


class MathModelWorkFlow:
    """数学建模流水线的对外入口。

    真正的编排在 `PipelineInterpreter`；本类的职责是把外部依赖装配好。
    """

    task_id: str = ""
    work_dir: str = ""

    async def execute(
        self,
        question_text: str,
        data_dir: str | None = None,
        resume: bool = False,
        format_output: FormatOutPut = FormatOutPut.Markdown,
        review_enabled: bool = True,
        driver: str | None = None,
        pipeline_path: str | None = None,
        task_id: str | None = None,
    ) -> dict:
        """执行完整流水线。

        Args:
            question_text: 题目文本或 PDF 路径。
            data_dir: 数据文件目录。
            resume: 为 True 时按 task_id 从断点续跑。
            format_output: Markdown→DOCX 或 LaTeX→PDF。
            review_enabled: 是否启用交互式人工审核。
            driver: 智能来源（api / file / echo），缺省读 pipeline 配置。
            pipeline_path: 自定义流水线文件。
            task_id: 指定任务 ID（续跑或复用工作目录时使用）。
        """
        if resume:
            return await self.resume_execute(
                task_id or self.task_id, data_dir, format_output,
            )

        self._review_enabled = review_enabled
        self._format_output = format_output
        self.task_id = task_id or create_task_id()
        self.work_dir = str(Path("project/work_dir") / self.task_id)
        os.makedirs(self.work_dir, exist_ok=True)
        set_trace_context(self.task_id)
        logger.info(f"任务 {self.task_id} 开始，工作目录: {self.work_dir}")

        if data_dir and os.path.isdir(data_dir):
            self._copy_data(data_dir)

        await self._publish(SystemMessage(
            content=f"任务 {self.task_id} 开始", type="info"))

        pipeline = load_pipeline(pipeline_path or DEFAULT_PIPELINE)
        register_builtins()

        interpreter = await self._build_interpreter(
            pipeline, driver or pipeline.driver, format_output,
        )

        from app.core.artifact import Artifact
        initial = Artifact(kind="raw_question", stage_id="__input__",
                           payload={"question": question_text})
        result = await interpreter.run(initial=initial, resume=False)
        return await self._finalize(result, format_output)

    async def resume_execute(
        self,
        task_id: str,
        data_dir: str | None = None,
        format_output: FormatOutPut = FormatOutPut.Markdown,
        driver: str | None = None,
        pipeline_path: str | None = None,
        review_enabled: bool = False,
    ) -> dict:
        """从断点恢复。与 `execute` 共用同一个解释器，不再是另一份流水线实现。"""
        self.task_id = task_id
        self.work_dir = str(Path("project/work_dir") / task_id)
        self._format_output = format_output
        # 必须在此设置：`_cli_review` 用 getattr(self, "_review_enabled", True) 判定，
        # 漏设会退化为 True，使 --no-review 在续跑路径上失效、每个阶段都打印预览。
        self._review_enabled = review_enabled
        set_trace_context(task_id)

        if not os.path.exists(self.work_dir):
            raise FileNotFoundError(f"任务工作目录不存在: {self.work_dir}")

        if data_dir and os.path.isdir(data_dir):
            self._copy_data(data_dir)

        pipeline = load_pipeline(pipeline_path or DEFAULT_PIPELINE)
        register_builtins()

        interpreter = await self._build_interpreter(
            pipeline, driver or pipeline.driver, format_output,
        )
        await self._publish(SystemMessage(
            content=f"从断点恢复 {task_id}", type="info"))

        result = await interpreter.run(resume=True)
        return await self._finalize(result, format_output)

    # ── 装配 ──

    async def _build_interpreter(self, pipeline, driver_name: str,
                                 format_output: FormatOutPut) -> PipelineInterpreter:
        """构造驱动与资源，注入外部服务回调。"""
        factory = LLMFactory(self.task_id)
        driver = self._make_driver(driver_name, factory)

        resources: dict = {"format_output": str(format_output).lower()}

        # 代码解释器：所有需要执行代码的组件共用同一个实例，
        # 使 kernel 内的变量在子任务之间保持（与原实现一致）。
        notebook = NotebookSerializer(self.work_dir, "notebook")
        resources["interpreter"] = await create_interpreter(
            task_id=self.task_id, work_dir=self.work_dir,
            notebook_serializer=notebook,
            exec_timeout=settings.CODE_EXEC_TIMEOUT,
        )

        # 各角色的 LLM：ApiDriver 与需要 tool-calling 的组件都从这里取，
        # 使"分阶段选模型"的能力（PARSER_MODEL … REVIEWER_MODEL）零成本保留。
        for role in ("parser", "modeler", "coder", "writer", "reviewer"):
            llm = factory.get_llm_by_role(role)
            if llm is not None:
                resources[f"llm:{role}"] = llm
        if not any(k.startswith("llm:") for k in resources):
            logger.warning(
                "未配置任何模型 API Key（backend/.env.private）。"
                "api 模式将无法运行；可用 --driver file 由外部驱动。"
            )

        evaluator_llm = factory.get_evaluator_llm()
        if evaluator_llm is not None:
            resources["evaluator"] = Evaluator(evaluator_llm)

        return PipelineInterpreter(
            pipeline, component_registry, driver,
            task_id=self.task_id, work_dir=self.work_dir,
            settings=settings, resources=resources,
            human_gate_handler=self._human_gate,
            review_handler=self._cli_review,
        )

    def _make_driver(self, name: str, factory: LLMFactory):
        if name == "file":
            logger.info("使用 FileDriver：提示词将落盘，等待外部写入应答")
            return FileDriver(self.work_dir)
        if name == "echo":
            logger.info("使用 EchoDriver：仅验证机制，不调用任何模型")
            return EchoDriver()
        return ApiDriver(factory, fallbacks=factory.get_fallback_llms())

    def _copy_data(self, data_dir: str) -> None:
        for f in os.listdir(data_dir):
            src = os.path.join(data_dir, f)
            dst = os.path.join(self.work_dir, f)
            if os.path.isfile(src):
                shutil.copy2(src, dst)

    # ── 外部服务回调 ──

    async def _human_gate(self, checkpoint_id: str, data: dict) -> dict:
        """HIL 人工确认，Redis 不可用时自动确认继续。"""
        if not settings.HIL_ENABLED:
            return {"action": "confirm"}
        if not settings.HIL_CHECKPOINTS.get(checkpoint_id, True):
            return {"action": "confirm"}
        try:
            await redis_manager.publish_message(
                self.task_id,
                ApprovalMessage(checkpoint_id=checkpoint_id, prompt=data,
                                timeout=settings.HIL_TIMEOUT),
            )
            return await checkpoint_manager.wait_for_decision(
                self.task_id, checkpoint_id, data, settings.HIL_TIMEOUT,
            )
        except Exception as e:
            logger.warning(f"HIL checkpoint '{checkpoint_id}' 失败(自动确认继续): {e}")
            return {"action": "confirm"}

    async def _cli_review(self, stage: str, content_preview: str) -> None:
        """CLI 交互式审核。

        注意：此前的"F=提意见重做"分支**从未生效**（审计发现 4：三处调用都没传
        retry_fn，分支不可达）。这里如实降级为"仅展示、回车继续"，
        不再保留一个做不到的承诺。
        """
        if not getattr(self, "_review_enabled", True):
            return
        print(f"\n{'=' * 60}\n  [审核] {stage}\n{'=' * 60}")
        print(content_preview[:3000])
        print(f"{'=' * 60}")
        try:
            input("[Enter] 继续: ")
        except EOFError:
            logger.debug("非交互环境，跳过 CLI 审核")

    async def _publish(self, msg) -> None:
        try:
            await redis_manager.publish_message(self.task_id, msg)
        except Exception as e:
            logger.debug(f"Redis 不可用，跳过消息推送: {e}")

    # ── 结果 ──

    async def _finalize(self, result: RunResult, format_output: FormatOutPut) -> dict:
        """把 RunResult 转换为既有调用方期望的字典。"""
        paper = result.paper()
        review = result.artifacts.get("review")
        parser = result.artifacts.get("parse")
        modeler = result.artifacts.get("model")

        if result.status == "awaiting_input":
            pend = "\n".join(
                f"  - {p['request']}\n    → 请写入 {p['response']}" for p in result.pending
            )
            logger.warning(
                f"任务 {result.task_id} 等待外部输入（{len(result.pending)} 处）：\n{pend}"
            )
            await self._publish(SystemMessage(
                content=f"等待外部输入（{len(result.pending)} 处）", type="info"))
        elif result.status == "completed":
            score = (review.payload.get("overall_score") if review else None)
            await self._publish(SystemMessage(
                content=f"任务 {result.task_id} 完成! 评分: {score}/10", type="success"))
        else:
            await self._publish(SystemMessage(
                content=f"任务 {result.task_id} 失败: {result.error}", type="error"))

        return {
            "status": result.status,
            "task_id": result.task_id,
            "work_dir": result.work_dir,
            "paper_path": os.path.join(result.work_dir, "res.md"),
            "paper": paper,
            "parser_result": parser.payload if parser else {},
            "modeler_result": modeler.payload if modeler else {},
            "review_result": review.payload if review else {},
            "artifacts": {k: v.model_dump(mode="json") for k, v in result.artifacts.items()},
            "failures": result.failures,
            "pending": result.pending,
            "error": result.error,
        }
