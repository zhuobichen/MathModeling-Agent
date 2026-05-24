"""MathModel 5-Agent 流水线编排模块。"""

import asyncio
import json
import os
from pathlib import Path

from app.config.setting import settings
from app.core.agents.parser_agent import ParserAgent
from app.core.agents.modeler_agent import ModelerAgent
from app.core.agents.coder_agent import CoderAgent
from app.core.agents.writer_agent import WriterAgent
from app.core.agents.reviewer_agent import ReviewerAgent
from app.core.llm.llm_factory import LLMFactory
from app.core.flows import Flows
from app.core.evaluator import Evaluator
from app.schemas.A2A import ParserToModeler, ModelerToCoder, CoderToWriter, WriterResponse, ReviewerResult
from app.schemas.enums import CompTemplate, FormatOutPut
from app.services.redis_manager import redis_manager
from app.services.checkpoint_manager import checkpoint_manager
from app.schemas.response import SystemMessage, ApprovalMessage
from app.tools.interpreter_factory import create_interpreter
from app.tools.notebook_serializer import NotebookSerializer
from app.tools.openalex_scholar import OpenAlexScholar
from app.models.user_output import UserOutput
from app.utils.log_util import logger
from app.utils.common_utils import create_task_id, create_work_dir, md_2_docx
from app.utils.pdf_parser import parse_pdf_question


class MathModelWorkFlow:
    """5-Agent 数学建模流水线: Parser → Modeler → Coder → Writer → Reviewer。"""

    task_id: str
    work_dir: str

    async def execute(self, question_text: str, data_dir: str | None = None) -> dict:
        """执行完整的 5-Agent 流水线。

        Args:
            question_text: 题目文本。
            data_dir: 数据文件目录（可选）。

        Returns:
            包含执行结果和所有中间产物的字典。
        """
        # ---- Phase 0: 初始化 ----
        self.task_id = create_task_id()
        self.work_dir = str(Path("project/work_dir") / self.task_id)
        os.makedirs(self.work_dir, exist_ok=True)
        logger.info(f"任务 {self.task_id} 开始，工作目录: {self.work_dir}")

        # 如果有数据目录，复制数据文件到工作目录
        if data_dir and os.path.isdir(data_dir):
            import shutil
            for f in os.listdir(data_dir):
                src = os.path.join(data_dir, f)
                dst = os.path.join(self.work_dir, f)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)

        await redis_manager.publish_message(
            self.task_id,
            SystemMessage(content=f"任务 {self.task_id} 开始", type="info"),
        )

        # 创建 LLM 实例
        factory = LLMFactory(self.task_id)
        parser_llm, modeler_llm, coder_llm, writer_llm, reviewer_llm = factory.get_all_llms()
        fallbacks = factory.get_fallback_llms()
        evaluator_llm = factory.get_evaluator_llm()

        # 初始化 Evaluator（可选）
        evaluator = Evaluator(evaluator_llm) if evaluator_llm else None

        # ---- Phase 0.5: PDF 解析（提取文本 + 图片识别）----
        # 如果题目文件是 PDF，先提取文本和图片，合并后再传给 ParserAgent
        if question_text.endswith(".pdf") and os.path.exists(question_text):
            logger.info("Phase 0.5: PDF 题目解析（文本提取 + 图片识别）")
            await redis_manager.publish_message(
                self.task_id,
                SystemMessage(content="[0/5] 正在解析PDF题目（提取文本+识别图片）...", type="info"),
            )
            # 复制 PDF 到工作目录
            import shutil
            pdf_dst = os.path.join(self.work_dir, os.path.basename(question_text))
            shutil.copy2(question_text, pdf_dst)
            # 解析 PDF
            parsed_text = await parse_pdf_question(pdf_dst)
            if parsed_text.strip():
                question_text = parsed_text
                logger.info("PDF 解析完成，文字+图片描述已合并")
            else:
                logger.warning("PDF 解析结果为空，使用原始路径")

        # ---- Phase 1: ParserAgent ----
        logger.info("Phase 1: ParserAgent 题目解析")
        await redis_manager.publish_message(
            self.task_id,
            SystemMessage(content="[1/5] ParserAgent 正在解析题目...", type="info"),
        )
        parser = ParserAgent(
            task_id=self.task_id,
            model=parser_llm,
            work_dir=self.work_dir,
        )
        try:
            parser_result = await self._run_with_handoff(
                parser, "run", (question_text,), "parser",
                fallbacks.get("parser"),
                ParserAgent,
                {"task_id": self.task_id, "work_dir": self.work_dir},
            )
        except Exception as e:
            logger.error(f"ParserAgent 失败: {e}")
            return {"status": "error", "stage": "parser", "error": str(e)}

        logger.info(f"题目解析完成: {parser_result.ques_count} 个子问题")

        # ---- Phase 2: ModelerAgent ----
        logger.info("Phase 2: ModelerAgent EDA + 建模方案")
        await redis_manager.publish_message(
            self.task_id,
            SystemMessage(content="[2/5] ModelerAgent 正在进行EDA和建模方案设计...", type="info"),
        )
        modeler = ModelerAgent(task_id=self.task_id, model=modeler_llm)
        try:
            modeler_result = await self._run_with_handoff(
                modeler, "run", (parser_result,), "modeler",
                fallbacks.get("modeler"),
                ModelerAgent,
                {"task_id": self.task_id},
            )
        except Exception as e:
            logger.error(f"ModelerAgent 失败: {e}")
            return {"status": "error", "stage": "modeler", "error": str(e)}

        logger.info("建模方案完成")

        # HIL checkpoint: model_selection
        await self._handle_checkpoint(
            "model_selection",
            {"questions": parser_result.questions, "modeling_plan": modeler_result.questions_solution},
        )

        # ---- Phase 3: CoderAgent ----
        logger.info("Phase 3: CoderAgent 代码生成与执行")
        await redis_manager.publish_message(
            self.task_id,
            SystemMessage(content="[3/5] CoderAgent 正在编写和执行代码...", type="info"),
        )

        # 初始化代码解释器
        notebook = NotebookSerializer(self.work_dir, "notebook")
        interpreter = create_interpreter(
            task_id=self.task_id,
            work_dir=self.work_dir,
            notebook_serializer=notebook,
        )
        await interpreter.initialize()

        coder = CoderAgent(
            task_id=self.task_id,
            model=coder_llm,
            work_dir=self.work_dir,
            code_interpreter=interpreter,
        )

        # 获取子任务列表
        flows = Flows(parser_result.questions)
        solution_flows = flows.get_solution_flows(modeler_result)

        coder_results: dict[str, CoderToWriter] = {}
        all_images: list[str] = []

        for key, flow_info in solution_flows.items():
            logger.info(f"CoderAgent 执行子任务: {key}")
            prompt = flow_info.get("coder_prompt", "")
            try:
                result = await self._run_coder_with_handoff(
                    coder, prompt, key, "coder",
                    fallbacks.get("coder"),
                    self.work_dir,
                    interpreter,
                )
                coder_results[key] = result
                if result.created_images:
                    all_images.extend(result.created_images)
                    logger.info(f"子任务 {key} 生成图片: {result.created_images}")
            except Exception as e:
                logger.error(f"CoderAgent 子任务 {key} 失败: {e}")
                coder_results[key] = CoderToWriter(
                    code_response=f"执行失败: {e}",
                    created_images=[],
                )

        await interpreter.cleanup()
        logger.info("所有代码子任务完成")

        # ---- Phase 4: WriterAgent ----
        logger.info("Phase 4: WriterAgent 论文生成")
        await redis_manager.publish_message(
            self.task_id,
            SystemMessage(content="[4/5] WriterAgent 正在撰写论文...", type="info"),
        )

        writer = WriterAgent(
            task_id=self.task_id,
            model=writer_llm,
            comp_template=CompTemplate.CHINA,
            format_output=FormatOutPut.Markdown,
        )
        user_output = UserOutput(
            work_dir=self.work_dir,
            ques_count=parser_result.ques_count,
        )

        # 先写各子问题的内容
        for key, flow_info in solution_flows.items():
            writer_prompt = flows.get_writer_prompt(
                key,
                coder_results.get(key, CoderToWriter()).code_response or "",
                interpreter,
            )
            result_images = coder_results.get(key, CoderToWriter()).created_images
            try:
                writer_response = await self._run_with_handoff(
                    writer, "run", (writer_prompt, result_images, key), "writer",
                    fallbacks.get("writer"),
                    WriterAgent,
                    {"task_id": self.task_id},
                )
                user_output.set_res(key, writer_response)
                logger.info(f"章节 {key} 写作完成")
            except Exception as e:
                logger.error(f"章节 {key} 写作失败: {e}")
                user_output.set_res(key, WriterResponse(response_content=f"写作失败: {e}"))

        # 写结构性章节
        write_flows = flows.get_write_flows(
            user_output,
            {},  # config_template 可为空
            parser_result.background,
        )
        for key, flow_info in write_flows.items():
            writer_prompt = flow_info.get("writer_prompt", "")
            try:
                writer_response = await self._run_with_handoff(
                    writer, "run", (writer_prompt, None, key), "writer",
                    fallbacks.get("writer"),
                    WriterAgent,
                    {"task_id": self.task_id},
                )
                user_output.set_res(key, writer_response)
                logger.info(f"章节 {key} 写作完成")
            except Exception as e:
                logger.error(f"章节 {key} 写作失败: {e}")
                user_output.set_res(key, WriterResponse(response_content=f"写作失败: {e}"))

        # 保存论文
        paper_md = user_output.save_result()
        logger.info(f"论文已保存到 {self.work_dir}")

        # ---- Phase 5: ReviewerAgent ----
        logger.info("Phase 5: ReviewerAgent 质量评审")
        await redis_manager.publish_message(
            self.task_id,
            SystemMessage(content="[5/5] ReviewerAgent 正在评审论文质量...", type="info"),
        )

        reviewer = ReviewerAgent(task_id=self.task_id, model=reviewer_llm)
        try:
            review_result = await reviewer.run(
                paper_content=paper_md,
                task_summary=json.dumps(parser_result.questions, ensure_ascii=False),
            )
            # 保存评审结果
            review_path = os.path.join(self.work_dir, "review_result.json")
            with open(review_path, "w", encoding="utf-8") as f:
                json.dump(review_result.model_dump(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"ReviewerAgent 评审失败: {e}")
            review_result = ReviewerResult(
                overall_score=0.0,
                passed=False,
                summary=f"评审失败: {e}",
            )

        # 尝试转换为 DOCX
        try:
            md_2_docx(self.task_id)
        except Exception as e:
            logger.warning(f"DOCX 转换失败: {e}")

        # 发布完成消息
        await redis_manager.publish_message(
            self.task_id,
            SystemMessage(
                content=f"任务 {self.task_id} 完成! 综合评分: {review_result.overall_score}/10",
                type="success",
            ),
        )

        return {
            "status": "completed",
            "task_id": self.task_id,
            "work_dir": self.work_dir,
            "parser_result": parser_result.model_dump(),
            "modeler_result": modeler_result.model_dump(),
            "coder_results": {k: v.model_dump() for k, v in coder_results.items()},
            "review_result": review_result.model_dump(),
            "paper_path": os.path.join(self.work_dir, "res.md"),
        }

    # ---- Helper Methods ----

    async def _run_with_handoff(
        self, agent, method_name: str, args: tuple, agent_label: str,
        fallback_llm, agent_cls, agent_init_kwargs: dict,
    ):
        """运行 Agent 方法，失败时使用 fallback LLM 重试。"""
        try:
            method = getattr(agent, method_name)
            return await method(*args)
        except Exception as e:
            if fallback_llm is not None:
                logger.warning(f"{agent_label} 主模型失败，使用 fallback: {e}")
                fallback_agent = agent_cls(model=fallback_llm, **agent_init_kwargs)
                method = getattr(fallback_agent, method_name)
                return await method(*args)
            raise

    async def _run_coder_with_handoff(
        self, coder, prompt: str, subtask_title: str, agent_label: str,
        fallback_llm, work_dir: str, interpreter,
    ) -> CoderToWriter:
        """运行 CoderAgent，失败时使用 fallback 重试。"""
        try:
            return await coder.run(prompt, subtask_title)
        except Exception as e:
            if fallback_llm is not None:
                logger.warning(f"{agent_label} 主模型失败，使用 fallback: {e}")
                from app.core.agents.coder_agent import CoderAgent as CA
                fallback_coder = CA(
                    task_id=self.task_id,
                    model=fallback_llm,
                    work_dir=work_dir,
                    code_interpreter=interpreter,
                )
                return await fallback_coder.run(prompt, subtask_title)
            raise

    async def _handle_checkpoint(self, checkpoint_id: str, data: dict) -> dict:
        """HIL checkpoint：暂停等待用户确认。"""
        if not settings.HIL_ENABLED:
            return {"action": "confirm"}

        checkpoint_enabled = settings.HIL_CHECKPOINTS.get(checkpoint_id, True)
        if not checkpoint_enabled:
            return {"action": "confirm"}

        await redis_manager.publish_message(
            self.task_id,
            ApprovalMessage(
                checkpoint_id=checkpoint_id,
                prompt=data,
                timeout=settings.HIL_TIMEOUT,
            ),
        )
        decision = await checkpoint_manager.wait_for_decision(
            self.task_id,
            checkpoint_id,
            data,
            settings.HIL_TIMEOUT,
        )
        return decision
