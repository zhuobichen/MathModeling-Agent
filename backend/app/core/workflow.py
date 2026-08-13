"""MathModel 5-Agent 流水线编排模块，支持断点续跑。"""

import asyncio
import json
import os
import shutil
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
from app.tools.tool_registry import tool_registry
from app.tools.knowledge_retriever import KnowledgeRetriever
from app.tools.handlers.install_package import _handle_install_package
from app.tools.handlers.read_file import _handle_read_file
from app.tools.handlers.search_knowledge import _handle_search_knowledge
from app.tools.handlers.search_papers import _handle_search_papers
from app.core.functions import search_web_tool, search_knowledge_tool, read_file_tool, install_package_tool, search_papers_tool
from app.core.postprocess import (
    _assign_images_to_sections,
    _extract_figure_metadata,
    _extract_method_contract,
    _fix_figure_captions,
    _remove_duplicate_headings,
    _remove_duplicate_images,
    _renumber_tables_figures,
    _scan_work_dir_images,
)
from app.models.user_output import UserOutput
from app.utils.log_util import logger, set_trace_context
from app.utils.common_utils import create_task_id, create_work_dir, md_2_docx, md_2_pdf
from app.utils.pdf_parser import parse_pdf_question
from app.utils.paper_validator import PaperValidator, validate_paper
from app.utils.paper_normalizer import PaperNormalizer, normalize_paper


CHECKPOINT_FILE = "checkpoint.json"


class MathModelWorkFlow:
    """5-Agent 流水线: Parser → Modeler → Coder → Writer → Reviewer。
    每阶段完成后自动保存 checkpoint，支持 --resume 断点续跑。
    """

    task_id: str
    work_dir: str

    # ---- 公开 API ----

    async def execute(
        self, question_text: str, data_dir: str | None = None,
        resume: bool = False, format_output: FormatOutPut = FormatOutPut.Markdown,
        review_enabled: bool = True,
    ) -> dict:
        """执行完整的 5-Agent 流水线。

        Args:
            question_text: 题目文本或文件路径。
            data_dir: 数据文件目录。
            resume: 是否从断点续跑。
            format_output: 输出格式（Markdown→DOCX 或 LaTeX→PDF）。
            review_enabled: 是否启用人审核。
        """
        # ---- Phase 0: 初始化 ----
        if resume:
            return await self.resume_execute(resume, data_dir, format_output)

        self.task_id = create_task_id()
        self.work_dir = str(Path("project/work_dir") / self.task_id)
        self._format_output = format_output
        self._review_enabled = review_enabled
        os.makedirs(self.work_dir, exist_ok=True)
        # 设置链路追踪上下文，后续所有日志自动携带 trace_id / task_id
        set_trace_context(self.task_id)
        logger.info(f"任务 {self.task_id} 开始，工作目录: {self.work_dir}")

        if data_dir and os.path.isdir(data_dir):
            for f in os.listdir(data_dir):
                src = os.path.join(data_dir, f)
                dst = os.path.join(self.work_dir, f)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)

        await redis_manager.publish_message(
            self.task_id,
            SystemMessage(content=f"任务 {self.task_id} 开始", type="info"),
        )

        factory = LLMFactory(self.task_id)
        parser_llm, modeler_llm, coder_llm, writer_llm, reviewer_llm = factory.get_all_llms()
        fallbacks = factory.get_fallback_llms()
        evaluator_llm = factory.get_evaluator_llm()
        evaluator = Evaluator(evaluator_llm) if evaluator_llm else None

        # ---- RAG 知识库初始化 ----
        knowledge_retriever = None
        if settings.RAG_ENABLED:
            try:
                knowledge_retriever = KnowledgeRetriever()
                logger.info("RAG 知识库已启用")
            except Exception as e:
                logger.warning(f"RAG 知识库初始化失败，将跳过: {e}")

        # ---- 注册 Tool Registry 处理器 ----
        if knowledge_retriever:
            tool_registry.register(
                "search_knowledge",
                lambda args, tid: _handle_search_knowledge(args, knowledge_retriever),
                search_knowledge_tool,
            )
        # search_web 仅在有 Tavily API key 时注册
        if settings.TAVILY_API_KEY:
            from app.tools.web_searcher import WebSearcher
            web_searcher = WebSearcher(api_key=settings.TAVILY_API_KEY)
            tool_registry.register(
                "search_web",
                lambda args, tid: web_searcher.search(**args),
                search_web_tool,
            )
            logger.info("Web 搜索工具已注册")

        # 注册 read_file 工具
        tool_registry.register(
            "read_file",
            lambda args, tid: _handle_read_file(args, self.work_dir),
            read_file_tool,
        )
        # 注册 install_package 工具
        tool_registry.register(
            "install_package",
            lambda args, tid: _handle_install_package(args),
            install_package_tool,
        )

        # ---- Phase 0.5: PDF 解析 ----
        try:
            question_text = await self._run_pdf_parse(question_text)
        except Exception as e:
            logger.exception(f"Phase 0.5 PDF 解析失败: {e}")
            self._save_error_checkpoint("pdf_parse", e)

        # ---- Phase 1: ParserAgent ----
        try:
            parser_result = await self._run_parser(parser_llm, fallbacks, question_text)
        except Exception as e:
            logger.exception(f"Phase 1 ParserAgent 失败: {e}")
            self._save_error_checkpoint("parser", e)
            raise

        # Inject RAG knowledge into modeler prompt
        rag_context = ""
        if knowledge_retriever:
            try:
                rag_query = parser_result.background + " " + str(parser_result.questions)
                if settings.RAG_EXPANSION_ENABLED and modeler_llm:
                    # 使用查询扩展增强检索召回率
                    rag_results = await knowledge_retriever.retrieve_with_expansion(
                        query=rag_query,
                        top_k=settings.RAG_TOP_K,
                        context=[parser_result.background],
                        llm=modeler_llm,
                    )
                else:
                    rag_results = await knowledge_retriever.retrieve(
                        query=rag_query,
                        top_k=settings.RAG_TOP_K,
                    )
                if rag_results:
                    rag_context = "\n\n## [知识库参考]\n" + "\n".join(
                        f"### {r.method_name}\n{r.content[:500]}" for r in rag_results
                    )
                    logger.info(f"RAG 检索到 {len(rag_results)} 条相关算法知识")
            except Exception as e:
                logger.warning(f"RAG 检索失败: {e}")
        # ParserAgent 自反思：检查子问题是否完整、数据文件是否遗漏
        parser_review_prompt = """请审查以上JSON输出是否存在以下问题：
1. 子问题数量是否正确（对照题目）
2. 每个子问题的类型判断是否合理
3. 数据文件是否全部识别（特别是Excel多sheet）
4. JSON格式是否合法（引号、逗号、括号匹配）
如有问题请直接输出修正后的完整JSON，无问题请原样输出。"""
        parser_output_str = json.dumps(parser_result.model_dump(mode="json"), ensure_ascii=False)[:4000]
        reviewed_parser = await self._self_review(
            parser_output_str, parser_review_prompt, parser_llm, "ParserAgent",
        )
        if reviewed_parser != parser_output_str:
            try:
                new_result = ParserToModeler(**json.loads(reviewed_parser))
                if new_result.ques_count > 0:
                    parser_result = new_result
                    logger.info("ParserAgent 自反思修正已应用")
            except Exception:
                pass

        self._save_checkpoint("parser", parser_result=parser_result.model_dump(mode="json"))
        await self._cli_review("ParserAgent",
            json.dumps(parser_result.questions, ensure_ascii=False, indent=2)[:3000])

        # ---- Phase 2: ModelerAgent（含 RAG 注入 + Evaluator 反馈）----
        try:
            modeler_result = await self._run_modeler_with_feedback(
                modeler_llm, fallbacks, parser_result, evaluator, rag_context
            )
        except Exception as e:
            logger.exception(f"Phase 2 ModelerAgent 失败: {e}")
            self._save_error_checkpoint("modeler", e)
            raise
        self._save_checkpoint("modeler",
            parser_result=parser_result.model_dump(mode="json"),
            modeler_result=modeler_result.model_dump(mode="json"),
        )

        # HIL checkpoint
        await self._handle_checkpoint(
            "model_selection",
            {"questions": parser_result.questions, "modeling_plan": modeler_result.questions_solution},
        )
        await self._cli_review("ModelerAgent",
            json.dumps(modeler_result.questions_solution, ensure_ascii=False, indent=2)[:3000])

        # ---- Phase 3: CoderAgent ----
        try:
            coder_results, all_images, interpreter = await self._run_coder(
                coder_llm, fallbacks, parser_result, modeler_result
            )
        except Exception as e:
            logger.exception(f"Phase 3 CoderAgent 失败: {e}")
            self._save_error_checkpoint("coder", e)
            raise
        self._save_checkpoint("coder",
            parser_result=parser_result.model_dump(mode="json"),
            modeler_result=modeler_result.model_dump(mode="json"),
            coder_results={k: v.model_dump(mode="json") for k, v in coder_results.items()},
        )

        # ---- Phase 4: WriterAgent ----
        try:
            paper_md, user_output = await self._run_writer(
                writer_llm, fallbacks, parser_result, modeler_result, coder_results, interpreter
            )
        except Exception as e:
            logger.exception(f"Phase 4 WriterAgent 失败: {e}")
            self._save_error_checkpoint("writer", e)
            raise
        self._save_checkpoint("writer",
            parser_result=parser_result.model_dump(mode="json"),
            modeler_result=modeler_result.model_dump(mode="json"),
            coder_results={k: v.model_dump(mode="json") for k, v in coder_results.items()},
        )
        await self._cli_review("WriterAgent", paper_md[:3000])

        # ---- Phase 5: ReviewerAgent ----
        try:
            review_result = await self._run_reviewer(reviewer_llm, paper_md, parser_result)
        except Exception as e:
            logger.exception(f"Phase 5 ReviewerAgent 失败: {e}")
            self._save_error_checkpoint("reviewer", e)
            raise
        self._save_checkpoint("reviewer",
            parser_result=parser_result.model_dump(mode="json"),
            review_result=review_result.model_dump(mode="json"),
        )

        # ---- Phase 5.5: Reviewer 反馈闭环 ----
        # 当评审分数低于阈值时，将建议反馈给 WriterAgent 重写低分章节
        REVIEW_REWRITE_THRESHOLD = 6.0
        if review_result.overall_score < REVIEW_REWRITE_THRESHOLD and review_result.suggestions:
            logger.info(
                f"论文评分 {review_result.overall_score} < {REVIEW_REWRITE_THRESHOLD}，"
                f"启动 Reviewer 反馈重写"
            )
            # 找出低分维度对应的章节
            weak_dimensions = [
                dim_name for dim_name, dim_data in review_result.dimensions.items()
                if dim_data.get("score", 10) < 6.0
            ]
            # 维度→章节映射
            dim_section_map = {
                "数据支撑": ["eda", "ques1", "ques2", "ques3", "ques4"],
                "建模逻辑": ["ques1", "ques2", "ques3", "ques4"],
                "可视化": ["eda", "sensitivity_analysis"],
                "去AI味": ["firstPage", "RepeatQues", "analysisQues", "judge"],
                "完整性": ["firstPage", "RepeatQues", "analysisQues"],
            }
            sections_to_rewrite = set()
            for dim in weak_dimensions:
                if dim in dim_section_map:
                    sections_to_rewrite.update(dim_section_map[dim])

            if sections_to_rewrite and hasattr(self, '_writer'):
                suggestions_text = "\n".join(
                    f"- {s}" for s in review_result.suggestions
                )
                rewrite_context = (
                    f"[审稿意见] 论文总分 {review_result.overall_score}/10，以下维度需改进：\n"
                    + "\n".join(
                        f"- {d}: {review_result.dimensions.get(d, {}).get('score', 'N/A')}/10"
                        for d in weak_dimensions
                    )
                    + f"\n\n改进建议：\n{suggestions_text}\n\n"
                    + "请根据上述意见重写对应章节，确保：\n"
                    + "1. 每个结论有具体数据支撑\n"
                    + "2. 使用'本文'而非'我们'\n"
                    + "3. 表格使用 HTML <table> 标签\n"
                    + "4. 禁止使用空泛词汇（此外、显著的、具有重要意义）\n"
                )
                for section_key in sections_to_rewrite:
                    if section_key in user_output.res:
                        logger.info(f"重写章节: {section_key}（Reviewer 反馈）")
                        original = user_output.res[section_key].get("response_content", "")
                        rewrite_prompt = (
                            f"{getattr(self, '_context_anchor', '')}"
                            f"以下是需要重写的章节 ({section_key})：\n\n"
                            f"{rewrite_context}\n\n"
                            f"原始内容供参考：\n{original[:1000]}\n"
                        )
                        try:
                            wr = await self._run_with_handoff(
                                self._writer, "run", (rewrite_prompt, None, f"{section_key}_rewrite"),
                                "writer", fallbacks.get("writer"),
                                WriterAgent, {"task_id": self.task_id},
                            )
                            user_output.res[section_key]["response_content"] = wr.response_content
                            logger.info(f"章节 {section_key} 重写完成（Reviewer 反馈）")
                        except Exception as e:
                            logger.warning(f"章节 {section_key} 重写失败: {e}")

                paper_md = user_output.save_result()
                paper_md = _remove_duplicate_headings(paper_md)
                paper_md = _renumber_tables_figures(paper_md)
                _write_paper_clean(paper_md, self.work_dir)
                # 重写后重新评审
                try:
                    review_result = await self._run_reviewer(reviewer_llm, paper_md, parser_result)
                    logger.info(f"重写后论文评分: {review_result.overall_score}/10")
                except Exception as e:
                    logger.warning(f"重写后评审失败: {e}")

        # 输出：Markdown→DOCX 或 LaTeX→PDF
        if format_output == FormatOutPut.LaTeX:
            pdf_path = md_2_pdf(self.task_id)
            if pdf_path:
                logger.info(f"PDF 输出: {pdf_path}")
        else:
            try:
                md_2_docx(self.task_id)
            except Exception as e:
                logger.warning(f"DOCX 转换失败: {e}")

        await redis_manager.publish_message(
            self.task_id,
            SystemMessage(
                content=f"任务 {self.task_id} 完成! 评分: {review_result.overall_score}/10",
                type="success",
            ),
        )

        return {
            "status": "completed",
            "task_id": self.task_id,
            "work_dir": self.work_dir,
            "parser_result": parser_result.model_dump(mode="json"),
            "modeler_result": modeler_result.model_dump(mode="json"),
            "coder_results": {k: v.model_dump(mode="json") for k, v in coder_results.items()},
            "review_result": review_result.model_dump(mode="json"),
            "paper_path": os.path.join(self.work_dir, "res.md"),
        }

    async def resume_execute(self, task_id: str, data_dir: str | None = None,
                              format_output: FormatOutPut = FormatOutPut.Markdown) -> dict:
        """从指定任务的工作目录恢复执行。

        Args:
            task_id: 之前任务的 ID。
            data_dir: 数据目录（可选，用于补充数据文件）。

        Returns:
            包含执行结果的字典。
        """
        self.task_id = task_id
        self.work_dir = str(Path("project/work_dir") / task_id)
        self._format_output = format_output
        # 恢复时也设置链路追踪上下文
        set_trace_context(task_id)

        if not os.path.exists(self.work_dir):
            raise FileNotFoundError(f"任务工作目录不存在: {self.work_dir}")

        ckpt_path = os.path.join(self.work_dir, CHECKPOINT_FILE)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"断点文件不存在: {ckpt_path}。该任务可能已完成或从未启动。")

        with open(ckpt_path, "r", encoding="utf-8") as f:
            ckpt = json.load(f)

        completed = ckpt.get("completed_stages", [])
        logger.info(f"从断点恢复 {task_id}，已完成: {completed}")

        factory = LLMFactory(self.task_id)
        parser_llm, modeler_llm, coder_llm, writer_llm, reviewer_llm = factory.get_all_llms()
        fallbacks = factory.get_fallback_llms()

        # 恢复已有结果
        parser_result = ParserToModeler(**ckpt["parser_result"]) if "parser_result" in ckpt else None
        modeler_result = ModelerToCoder(**ckpt["modeler_result"]) if "modeler_result" in ckpt else None
        coder_results: dict[str, CoderToWriter] = {}
        if "coder_results" in ckpt:
            coder_results = {k: CoderToWriter(**v) for k, v in ckpt["coder_results"].items()}
        review_result = ReviewerResult(**ckpt["review_result"]) if "review_result" in ckpt else None

        # 判断需要从哪个阶段开始
        need_coder = "coder" not in completed
        need_writer = "writer" not in completed
        need_reviewer = "reviewer" not in completed

        await redis_manager.publish_message(
            self.task_id,
            SystemMessage(content=f"从断点恢复 {task_id}，已完成: {', '.join(completed)}", type="info"),
        )

        # Coder phase
        if need_coder and parser_result and modeler_result:
            logger.info("恢复: 从 CoderAgent 开始")
            new_coder_results, all_images, interpreter = await self._run_coder(
                coder_llm, fallbacks, parser_result, modeler_result,
                skip_completed=list(coder_results.keys()),
            )
            coder_results.update(new_coder_results)
            self._save_checkpoint("coder",
                parser_result=parser_result.model_dump(mode="json"),
                modeler_result=modeler_result.model_dump(mode="json"),
                coder_results={k: v.model_dump(mode="json") for k, v in coder_results.items()},
            )
        else:
            interpreter = None

        # Writer phase
        if need_writer and parser_result and modeler_result:
            logger.info("恢复: 从 WriterAgent 开始")
            paper_md, user_output = await self._run_writer(
                writer_llm, fallbacks, parser_result, modeler_result, coder_results, interpreter
            )
            self._save_checkpoint("writer",
                parser_result=parser_result.model_dump(mode="json"),
                modeler_result=modeler_result.model_dump(mode="json"),
                coder_results={k: v.model_dump(mode="json") for k, v in coder_results.items()},
            )
        else:
            paper_md = os.path.join(self.work_dir, "res.md")
            if os.path.exists(paper_md):
                with open(paper_md, "r", encoding="utf-8") as f:
                    paper_md = f.read()
            else:
                paper_md = ""

        # Reviewer phase
        if need_reviewer and paper_md and parser_result:
            logger.info("恢复: 从 ReviewerAgent 开始")
            review_result = await self._run_reviewer(reviewer_llm, paper_md, parser_result)
            self._save_checkpoint("reviewer",
                parser_result=parser_result.model_dump(mode="json") if parser_result else {},
                review_result=review_result.model_dump(mode="json"),
            )

        # 全部完成
        if not need_coder and not need_writer and not need_reviewer:
            logger.info("该任务所有阶段已完成，无需恢复")
            if review_result is None:
                review_result = ReviewerResult(overall_score=0, passed=False, summary="评审未执行")

        # 输出路线
        if format_output == FormatOutPut.LaTeX:
            pdf_path = md_2_pdf(self.task_id)
            if pdf_path:
                logger.info(f"PDF 输出: {pdf_path}")
        else:
            try:
                md_2_docx(self.task_id)
            except Exception as e:
                logger.warning(f"DOCX 转换失败: {e}")

        await redis_manager.publish_message(
            self.task_id,
            SystemMessage(
                content=f"任务 {self.task_id} 恢复完成! 评分: {review_result.overall_score if review_result else '?'}/10",
                type="success",
            ),
        )

        return {
            "status": "completed",
            "task_id": self.task_id,
            "work_dir": self.work_dir,
            "parser_result": parser_result.model_dump(mode="json") if parser_result else {},
            "modeler_result": modeler_result.model_dump(mode="json") if modeler_result else {},
            "coder_results": {k: v.model_dump(mode="json") for k, v in coder_results.items()},
            "review_result": review_result.model_dump(mode="json") if review_result else {},
            "paper_path": os.path.join(self.work_dir, "res.md"),
        }

    # ---- 各阶段实现 ----

    async def _run_pdf_parse(self, question_text: str) -> str:
        """Phase 0.5: PDF 文本提取 + 图片识别。"""
        if not (question_text.endswith(".pdf") and os.path.exists(question_text)):
            return question_text

        logger.info("Phase 0.5: PDF 题目解析")
        await redis_manager.publish_message(
            self.task_id,
            SystemMessage(content="[0/5] 正在解析PDF题目...", type="info"),
        )
        pdf_dst = os.path.join(self.work_dir, os.path.basename(question_text))
        shutil.copy2(question_text, pdf_dst)
        parsed = await parse_pdf_question(pdf_dst)
        if parsed.strip():
            logger.info("PDF 解析完成")
            return parsed
        logger.warning("PDF 解析结果为空")
        return question_text

    async def _run_parser(
        self, parser_llm, fallbacks, question_text: str
    ) -> ParserToModeler:
        """Phase 1: 题目解析。"""
        logger.info("Phase 1: ParserAgent")
        await redis_manager.publish_message(
            self.task_id,
            SystemMessage(content="[1/5] ParserAgent 正在解析题目...", type="info"),
        )
        parser = ParserAgent(task_id=self.task_id, model=parser_llm, work_dir=self.work_dir)
        result = await self._run_with_handoff(
            parser, "run", (question_text,), "parser",
            fallbacks.get("parser"),
            ParserAgent, {"task_id": self.task_id, "work_dir": self.work_dir},
        )
        logger.info(f"题目解析完成: {result.ques_count} 个子问题")
        return result

    async def _run_modeler(
        self, modeler_llm, fallbacks, parser_result: ParserToModeler
    ) -> ModelerToCoder:
        """Phase 2: EDA + 建模方案设计。"""
        logger.info("Phase 2: ModelerAgent")
        await redis_manager.publish_message(
            self.task_id,
            SystemMessage(content="[2/5] ModelerAgent 正在进行EDA和建模方案设计...", type="info"),
        )
        modeler = ModelerAgent(task_id=self.task_id, model=modeler_llm)
        result = await self._run_with_handoff(
            modeler, "run", (parser_result,), "modeler",
            fallbacks.get("modeler"),
            ModelerAgent, {"task_id": self.task_id},
        )
        logger.info("建模方案完成")
        return result

    async def _run_coder(
        self, coder_llm, fallbacks, parser_result: ParserToModeler,
        modeler_result: ModelerToCoder, skip_completed: list[str] | None = None,
    ) -> tuple[dict[str, CoderToWriter], list[str], object]:
        """Phase 3: 代码生成与执行。

        Args:
            skip_completed: 已完成且无需重跑的 subtask key 列表。

        Returns:
            (coder_results, all_images, interpreter)
        """
        logger.info("Phase 3: CoderAgent")
        await redis_manager.publish_message(
            self.task_id,
            SystemMessage(content="[3/5] CoderAgent 正在编写和执行代码...", type="info"),
        )

        notebook = NotebookSerializer(self.work_dir, "notebook")
        interpreter = await create_interpreter(
            task_id=self.task_id, work_dir=self.work_dir, notebook_serializer=notebook,
        )

        coder = CoderAgent(
            task_id=self.task_id, model=coder_llm,
            work_dir=self.work_dir, code_interpreter=interpreter,
        )

        flows = Flows(parser_result.questions)
        solution_flows = flows.get_solution_flows(modeler_result)

        coder_results: dict[str, CoderToWriter] = {}
        all_images: list[str] = []
        skip = set(skip_completed or [])

        # 从 Modeler 方案提取方法契约，注入每个 Coder 子任务 prompt
        method_contract = _extract_method_contract(modeler_result)

        for key, flow_info in solution_flows.items():
            if key in skip:
                logger.info(f"跳过已完成的子任务: {key}")
                continue

            logger.info(f"CoderAgent 子任务: {key}")
            prompt = flow_info.get("coder_prompt", "")
            if method_contract:
                prompt = f"[⚠️ 建模方案指定的实现方法，必须使用]\n{method_contract}\n\n{prompt}"
            try:
                result = await self._run_coder_with_handoff(
                    coder, prompt, key, "coder",
                    fallbacks.get("coder"), self.work_dir, interpreter,
                )
                coder_results[key] = result
                if result.created_images:
                    all_images.extend(result.created_images)
            except Exception as e:
                logger.error(f"CoderAgent 子任务 {key} 失败: {e}")
                coder_results[key] = CoderToWriter(code_response=f"失败: {e}", created_images=[])
            finally:
                # 无论成功失败都存 checkpoint，确保断点续跑不会丢失已完成的进度
                self._save_checkpoint(f"coder.{key}",
                    parser_result=parser_result.model_dump(mode="json"),
                    modeler_result=modeler_result.model_dump(mode="json"),
                    coder_results={k: v.model_dump(mode="json") for k, v in coder_results.items()},
                )

        # CoderAgent 自反思：检查代码输出是否完整、结果是否有矛盾
        if coder_results and coder_llm:
            coder_review_prompt = """请审查以上所有子任务的代码执行结果，检查：
1. 每个子问题是否都有代码输出（不是"超过最大重试次数"或"失败"）
2. 不同子问题的数值结果是否矛盾（如qua1用的样本量和qua2不一致）
3. 模型评估指标是否齐全（R²/RMSE/准确率/AUC等）
4. 图表是否按规范保存（figures/目录、fig.suptitle中文标题）
如有严重问题请说明，无问题回复"PASS"。"""
            coder_output_str = json.dumps(
                {k: v.code_response[:500] for k, v in coder_results.items()},
                ensure_ascii=False,
            )[:3000]
            reviewed = await self._self_review(
                coder_output_str, coder_review_prompt, coder_llm, "CoderAgent",
            )
            if reviewed != coder_output_str and "PASS" not in reviewed:
                logger.warning(f"CoderAgent 自反思发现问题: {reviewed[:200]}")

        # 注意: 不在这里 cleanup interpreter，WriterAgent 还需要通过
        # get_code_output() 读取缓存的代码输出。interpreter 在流程结束后由 Python GC 回收。
        return coder_results, all_images, interpreter

    async def _run_writer(
        self, writer_llm, fallbacks, parser_result: ParserToModeler,
        modeler_result: ModelerToCoder, coder_results: dict[str, CoderToWriter],
        interpreter,
    ) -> tuple[str, UserOutput]:
        """Phase 4: 论文生成。"""
        logger.info("Phase 4: WriterAgent")
        await redis_manager.publish_message(
            self.task_id,
            SystemMessage(content="[4/5] WriterAgent 正在撰写论文...", type="info"),
        )

        # 加载论文模板
        import tomllib
        template_path = Path(__file__).parent.parent / "config" / "md_template.toml"
        config_template = {}
        if template_path.exists():
            with open(template_path, "rb") as f:
                config_template = tomllib.load(f)
            logger.info(f"论文模板已加载: {len(config_template)} 个章节")

        # 注册 search_papers 工具
        scholar = OpenAlexScholar(task_id=self.task_id)
        tool_registry.register(
            "search_papers",
            lambda args, tid: _handle_search_papers(args, scholar),
            search_papers_tool,
        )

        # 预加载图片元数据（必须在 WriterAgent 之前）
        all_images = _scan_work_dir_images(self.work_dir)
        figure_metadata = _extract_figure_metadata(self.work_dir)
        figure_descriptions: dict[str, str] = {}

        self._writer = WriterAgent(
            task_id=self.task_id, model=writer_llm,
            comp_template=CompTemplate.CHINA,
            format_output=getattr(self, '_format_output', FormatOutPut.Markdown),
            figure_descriptions=figure_descriptions,
            figure_metadata=figure_metadata,
            work_dir=self.work_dir,
        )
        writer = self._writer
        user_output = UserOutput(work_dir=self.work_dir, ques_count=parser_result.ques_count)

        flows = Flows(parser_result.questions)
        solution_flows = flows.get_solution_flows(modeler_result)
        # 构建模型的建立与求解摘要（供模板占位符使用）
        model_build_solve = json.dumps(
            {k: (v.code_response or "")[:1500] for k, v in coder_results.items()},
            ensure_ascii=False,
        )

        # 构建题目上下文锚点 + 方法契约——保证全文方法名一致
        method_contract = _extract_method_contract(modeler_result)
        self._context_anchor = (
            f"[题目背景] {parser_result.background}\n"
            f"[子问题列表] {json.dumps(parser_result.questions, ensure_ascii=False)[:2000]}\n\n"
            f"[⚠️ 方法契约] 以下方法名全文统一使用，不可替换:\n{method_contract}\n\n"
        )

        # 互斥分配：每张图只归属一个章节
        section_order = list(solution_flows.keys())
        image_assignments = _assign_images_to_sections(section_order, all_images)

        for key, flow_info in solution_flows.items():
            coder_writer = coder_results.get(key, CoderToWriter())
            modeler_solution = modeler_result.questions_solution.get(key, "")
            writer_prompt = flows.get_writer_prompt(
                key, coder_writer.code_response or "", interpreter,
                config_template.get(key, ""),
                model_build_solve,
                modeler_solution,
                format_output=str(getattr(self, '_format_output', 'markdown')),
            )
            writer_prompt = self._context_anchor + writer_prompt
            result_images = image_assignments.get(key, []) or all_images[:2]
            try:
                wr = await self._run_with_handoff(
                    writer, "run", (writer_prompt, result_images, key), "writer",
                    fallbacks.get("writer"),
                    WriterAgent, {"task_id": self.task_id},
                )
                user_output.set_res(key, wr)
                logger.info(f"章节 {key} 写作完成")
            except Exception as e:
                logger.exception(f"章节 {key} 写作失败: {e}")
                user_output.set_res(key, WriterResponse(response_content=f"写作失败: {e}"))

        # 结构性章节（传入模板，确保占位符被替换）
        write_flows = flows.get_write_flows(
            user_output, config_template, parser_result.background
        )
        for key, flow_info in write_flows.items():
            writer_prompt = flow_info.get("writer_prompt", "")
            try:
                wr = await self._run_with_handoff(
                    writer, "run", (writer_prompt, None, key), "writer",
                    fallbacks.get("writer"),
                    WriterAgent, {"task_id": self.task_id},
                )
                user_output.set_res(key, wr)
            except Exception as e:
                logger.exception(f"章节 {key} 写作失败: {e}")
                user_output.set_res(key, WriterResponse(response_content=f"写作失败: {e}"))

        paper_md = user_output.save_result()
        # 后处理：移除连续重复的同级标题
        paper_md = _remove_duplicate_headings(paper_md)
        # 后处理：移除 LLM 跨章节重复插入的图片
        paper_md = _remove_duplicate_images(paper_md)
        # 后处理：用Coder代码标题强制修正论文图注
        paper_md = _fix_figure_captions(paper_md, self.work_dir)
        # 后处理：全文统一重编号表和图
        paper_md = _renumber_tables_figures(paper_md)
        # 写回磁盘（save_result 写入的是原始版本，需覆盖）
        _write_paper_clean(paper_md, self.work_dir)
        logger.info(f"论文已保存到 {self.work_dir}")
        
        # ---- Phase 4.5: 论文格式验证和标准化 ----
        paper_md = await self._validate_and_normalize_paper(
            paper_md, user_output, writer, fallbacks
        )
        
        return paper_md, user_output

    async def _run_reviewer(
        self, reviewer_llm, paper_md: str, parser_result: ParserToModeler | None
    ) -> ReviewerResult:
        """Phase 5: 质量评审。"""
        logger.info("Phase 5: ReviewerAgent")
        await redis_manager.publish_message(
            self.task_id,
            SystemMessage(content="[5/5] ReviewerAgent 正在评审论文...", type="info"),
        )
        reviewer = ReviewerAgent(task_id=self.task_id, model=reviewer_llm)
        try:
            summary = json.dumps(parser_result.questions, ensure_ascii=False) if parser_result else ""
            result = await reviewer.run(paper_content=paper_md, task_summary=summary)

            # VL 图验证：千问看图对比论文描述
            vl_results = await _verify_figures_with_vl(paper_md, self.work_dir)
            if vl_results:
                # 计算匹配度
                import re
                match_count = 0
                vl_feedback: list[str] = []
                for fname, vl_desc in vl_results.items():
                    # 找论文中该图的描述
                    paper_caption = ""
                    for m in re.finditer(
                        r'!\[[^\]]*\]\(' + re.escape("figures/" + fname) + r'\)',
                        paper_md,
                    ):
                        after = paper_md[m.end():m.end()+200]
                        cm = re.search(r'\*\*图\d+[：:]\s*(.+?)\*\*', after)
                        if cm:
                            paper_caption = cm.group(1)
                        break
                    # n-gram匹配：提取2-3字中文词组对比
                    if paper_caption and vl_desc:
                        vl_words = set(re.findall(r'[\u4e00-\u9fff]{2,3}', vl_desc))
                        paper_words = set(re.findall(r'[\u4e00-\u9fff]{2,3}', paper_caption))
                        overlap = vl_words & paper_words
                        if len(overlap) >= 2:
                            match_count += 1
                        else:
                            vl_feedback.append(
                                f"图{fname}: 论文写'{paper_caption[:40]}'，"
                                f"但图中实际内容为'{vl_desc[:40]}'"
                            )

                total = len(vl_results)
                accuracy = match_count / total if total > 0 else 1.0
                logger.info(f"VL图验证: {match_count}/{total} 匹配 ({accuracy:.0%})")

                # 调整 visualization 和 data_support 评分
                if accuracy < 0.5:
                    dims = result.dimensions
                    for key in ["visualization", "data_support"]:
                        if key in dims:
                            old = dims[key].get("score", 6)
                            new = max(1, old - 3)
                            dims[key]["score"] = new
                            dims[key]["issues"].append(
                                f"VL验图: {match_count}/{total}张图描述与实际内容不符"
                            )
                    result.overall_score = sum(d.get("score", 0) for d in dims.values()) / len(dims)
                    result.suggestions.append(
                        f"【图内容不匹配】{match_count}/{total}张图的描述与实际图片内容不符。"
                        + "; ".join(vl_feedback[:3])
                    )

            review_path = os.path.join(self.work_dir, "review_result.json")
            with open(review_path, "w", encoding="utf-8") as f:
                json.dump(result.model_dump(mode="json"), f, ensure_ascii=False, indent=2)
            return result
        except Exception as e:
            logger.exception(f"ReviewerAgent 评审失败: {e}")
            return ReviewerResult(overall_score=0.0, passed=False, summary=f"评审失败: {e}")

    async def _validate_and_normalize_paper(
        self, paper_md: str, user_output: UserOutput,
        writer=None, fallbacks: dict | None = None,
    ) -> str:
        """Phase 4.5: 验证和标准化论文格式。

        1. 验证论文格式
        2. 自动标准化修复
        3. 如果验证失败，尝试单章节回退重写（调用 LLM）

        Args:
            paper_md: Markdown 论文内容。
            user_output: 用户输出对象（包含各章节结果）。
            writer: WriterAgent 实例（用于回退重写）。
            fallbacks: 备用 LLM 配置。

        Returns:
            验证和标准化后的论文内容。
        """
        logger.info("Phase 4.5: 论文格式验证")
        
        # 验证论文格式
        validator = PaperValidator(self.work_dir)
        validation_result = validator.validate(paper_md)
        
        if validation_result.passed:
            logger.info(f"论文格式验证通过，评分: {validation_result.score:.1f}/100")
        else:
            logger.warning(
                f"论文格式验证失败，发现 {validation_result.error_count} 个错误，"
                f"评分: {validation_result.score:.1f}/100"
            )
            for error in validation_result.errors:
                logger.warning(
                    f"  - [{error.section}] {error.error_type}: {error.message}"
                )
        
        # 标准化修复
        normalizer = PaperNormalizer()
        norm_result = normalizer.normalize(paper_md)
        
        if norm_result.has_changes:
            logger.info(f"论文标准化完成，应用了 {len(norm_result.fixes_applied)} 项修复")
            paper_md = norm_result.normalized_content
            
            # 保存标准化后的论文
            paper_path = os.path.join(self.work_dir, "res.md")
            with open(paper_path, "w", encoding="utf-8") as f:
                f.write(paper_md)
            logger.info("标准化后的论文已保存")
        
        # 如果验证失败，尝试单章节回退重写
        if not validation_result.passed and validation_result.score < 80:
            logger.info("开始单章节回退重写...")
            paper_md = await self._retry_failed_sections(
                validation_result, user_output, paper_md, writer, fallbacks
            )
        
        return paper_md

    async def _retry_failed_sections(
        self, validation_result, user_output: UserOutput, paper_md: str,
        writer=None, fallbacks: dict | None = None,
    ) -> str:
        """单章节回退重写机制。

        根据验证错误定位到问题章节，通过 WriterAgent 重写失败的部分。

        Args:
            validation_result: 验证结果。
            user_output: 用户输出对象。
            paper_md: 当前论文内容。
            writer: WriterAgent 实例（用于重写）。
            fallbacks: 备用 LLM 配置。

        Returns:
            重写后的论文内容。
        """
        sections_to_retry = self._determine_failed_sections(validation_result)

        if not sections_to_retry:
            logger.info("无法确定问题章节，跳过回退重写")
            return paper_md

        logger.info(f"需要重写的章节: {sections_to_retry}")

        for section_key in sections_to_retry:
            if section_key not in user_output.res:
                continue

            logger.info(f"重写章节: {section_key}（LLM 重写）")

            section_errors = [
                f"[{e.error_type}] {e.message}"
                for e in validation_result.errors
                if e.section == section_key or section_key in e.message
            ]
            fix_suggestions = [
                e.fix_suggestion
                for e in validation_result.errors
                if e.section == section_key or section_key in e.message
                if e.fix_suggestion
            ]

            original_content = user_output.res[section_key].get("response_content", "")
            rewrite_prompt = (
                "你的论文格式需要修正，请根据以下反馈重写该章节：\n\n"
                + "\n".join(f"- {err}" for err in section_errors) + "\n\n"
                + "修正建议：\n"
                + "\n".join(f"- {fix}" for fix in fix_suggestions) + "\n\n"
                + "原始内容供参考：\n"
                + original_content[:2000] + "\n\n"
                + "请输出修正后的完整章节内容，保持与其他章节一致的风格。"
            )

            if writer is not None:
                try:
                    wr = await self._run_with_handoff(
                        writer, "run", (rewrite_prompt, None, f"{section_key}_fix"),
                        "writer", (fallbacks or {}).get("writer"),
                        WriterAgent, {"task_id": self.task_id},
                    )
                    user_output.res[section_key]["response_content"] = wr.response_content
                    logger.info(f"章节 {section_key} LLM 重写完成")
                except Exception as e:
                    logger.warning(f"章节 {section_key} LLM 重写失败，使用原始内容: {e}")
            else:
                # 无 writer 可用时，追加格式提示到原文
                feedback = (
                    f"该章节格式存在问题，需要修正：\n"
                    + "\n".join(f"- {err}" for err in section_errors) + "\n\n"
                )
                user_output.res[section_key]["response_content"] = (
                    original_content + f"\n\n[格式修正提示] {feedback}"
                )
                logger.warning(f"章节 {section_key} 无可用的 WriterAgent，仅追加格式提示")

        paper_md = user_output.save_result()
        paper_md = _remove_duplicate_headings(paper_md)
        paper_md = _remove_duplicate_images(paper_md)
        paper_md = _renumber_tables_figures(paper_md)
        _write_paper_clean(paper_md, self.work_dir)

        validator = PaperValidator(self.work_dir)
        revalidation_result = validator.validate(paper_md)

        if revalidation_result.passed:
            logger.info("单章节回退重写后验证通过")
        else:
            logger.warning(
                f"单章节回退重写后仍有问题，评分: {revalidation_result.score:.1f}/100"
            )

        return paper_md

    def _determine_failed_sections(self, validation_result) -> list[str]:
        """根据验证错误确定需要重写的章节。

        根据错误类型和位置推断受影响的章节。
        
        Args:
            validation_result: 验证结果。
            
        Returns:
            需要重写的章节列表。
        """
        failed_sections = set()
        
        # 错误类型到章节的映射
        error_section_mapping = {
            "标题层级": ["RepeatQues", "analysisQues", "modelAssumption", "symbol"],
            "表格格式": ["symbol", "sensitivity_analysis"],
            "图片引用": ["sensitivity_analysis"],
            "公式格式": ["ques1", "ques2", "ques3", "ques4", "ques5", "ques6"],
            "章节完整性": ["firstPage", "RepeatQues"],
            "章节编号": ["RepeatQues", "analysisQues", "judge"],
        }
        
        for error in validation_result.errors:
            for error_type, sections in error_section_mapping.items():
                if error_type in error.error_type or error_type in error.section:
                    failed_sections.update(sections)
        
        # 如果无法确定具体章节，默认重写结构性问题章节
        if not failed_sections and validation_result.error_count > 5:
            failed_sections = {"RepeatQues", "analysisQues"}
        
        return list(failed_sections)

    async def _write_sections_parallel(
        self, writer_llm, fallbacks, sections: dict,
        coder_results: dict, modeler_result: ModelerToCoder,
        interpreter, config_template: dict,
        model_build_solve: str, flows: Flows,
    ) -> list[tuple[str, WriterResponse]]:
        """并行执行独立章节写作。

        为每个章节创建独立的 WriterAgent 实例，通过 asyncio.gather 并行执行。
        各章节间无数据依赖，并行写作可将该阶段耗时降低 50-70%。

        Returns:
            [(key, WriterResponse), ...] 保持输入顺序的结果列表。
        """
        # 扫描工作目录所有图片，弥补 CoderAgent 图片追踪缺失的问题
        all_images = _scan_work_dir_images(self.work_dir)

        async def _write_one(key: str, flow_info: dict) -> tuple[str, WriterResponse]:
            coder_writer = coder_results.get(key, CoderToWriter())
            # 按文件名前缀匹配，每个章节只拿自己相关的图
            section_images = _assign_images_to_section(
                key, all_images, coder_writer.created_images,
            )
            modeler_solution = modeler_result.questions_solution.get(key, "")
            writer_prompt = flows.get_writer_prompt(
                key, coder_writer.code_response or "", interpreter,
                config_template.get(key, ""),
                model_build_solve,
                modeler_solution,
                format_output=str(getattr(self, '_format_output', 'markdown')),
            )
            writer_prompt = self._context_anchor + writer_prompt
            # 每个并行任务创建独立的 WriterAgent，避免 chat_history 共享
            section_writer = WriterAgent(
                task_id=self.task_id, model=writer_llm,
                comp_template=CompTemplate.CHINA,
                format_output=getattr(self, '_format_output', FormatOutPut.Markdown),
            )
            try:
                wr = await self._run_with_handoff(
                    section_writer, "run",
                    (writer_prompt, section_images, key),
                    "writer", fallbacks.get("writer"),
                    WriterAgent, {"task_id": self.task_id},
                )
                return key, wr
            except Exception as e:
                logger.exception(f"章节 {key} 并行写作失败: {e}")
                return key, WriterResponse(response_content=f"写作失败: {e}")

        # asyncio.gather 按顺序返回（即使内部并发执行）
        return list(await asyncio.gather(*[
            _write_one(key, flow_info) for key, flow_info in sections.items()
        ]))

    # ---- 人工审核机制 ----

    async def _cli_review(self, stage: str, content_preview: str,
                          retry_fn=None, retry_args: tuple = ()) -> None:
        """CLI 交互式审核。A/Enter=通过, F=提意见重做。

        当用户提意见时，重跑该阶段的 Agent，将反馈注入 prompt 重新生成。
        """
        if not getattr(self, '_review_enabled', True):
            return

        print(f"\n{'='*60}")
        print(f"  [审核] {stage}")
        print(f"{'='*60}")
        print(content_preview[:3000] if len(content_preview) > 3000 else content_preview)
        print(f"{'='*60}")

        while True:
            choice = input("\n[Enter]通过  [F]提意见重做: ").strip().lower()
            if choice == '' or choice == 'a':
                return
            elif choice == 'f' and retry_fn:
                feedback = input("修改意见: ").strip()
                if not feedback:
                    continue
                print(f">>> 根据反馈重新生成...")
                try:
                    new_args = list(retry_args)
                    if new_args:
                        new_args[-1] = str(new_args[-1]) + f"\n\n[修改要求] {feedback}"
                    new_result = await retry_fn(*new_args)
                    preview = str(new_result)[:2500]
                    print(f"\n--- 新结果 ---")
                    print(preview)
                    print(f"---")
                except Exception as e:
                    print(f"重试失败: {e}")
            else:
                print("回车=通过, F=提意见")

    # ---- Checkpoint 机制 ----

    def _save_checkpoint(self, stage: str, **data) -> None:
        """保存断点：将已完成阶段和数据序列化到 checkpoint.json。

        每步完成后自动调用，确保失败时能从此处恢复。

        Args:
            stage: 当前阶段标识（如 "parser", "modeler", "coder.eda"）。
            **data: 要持久化的阶段结果数据。
        """
        ckpt_path = os.path.join(self.work_dir, CHECKPOINT_FILE)

        existing: dict = {}
        if os.path.exists(ckpt_path):
            with open(ckpt_path, "r", encoding="utf-8") as f:
                existing = json.load(f)

        # 累积已完成阶段列表
        completed: list[str] = existing.get("completed_stages", [])
        if stage not in completed:
            completed.append(stage)

        checkpoint = {
            **existing,
            "task_id": self.task_id,
            "current_stage": stage,
            "completed_stages": completed,
        }
        checkpoint.update(data)

        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
        logger.debug(f"Checkpoint saved: {stage}")

    def _save_error_checkpoint(self, stage: str, error: Exception) -> None:
        """保存失败信息到 checkpoint，用于后续排查。

        Args:
            stage: 失败阶段标识。
            error: 异常对象。
        """
        ckpt_path = os.path.join(self.work_dir, CHECKPOINT_FILE)
        existing: dict = {}
        if os.path.exists(ckpt_path):
            with open(ckpt_path, "r", encoding="utf-8") as f:
                existing = json.load(f)

        checkpoint = {
            **existing,
            "task_id": self.task_id,
            "failed_stage": stage,
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
        logger.info(f"错误信息已保存到 checkpoint: [{stage}] {type(error).__name__}: {error}")

    # ---- 内部辅助 ----

    async def _run_with_handoff(
        self, agent, method_name: str, args: tuple, agent_label: str,
        fallback_llm, agent_cls, agent_init_kwargs: dict,
    ):
        """运行 Agent 方法，失败时使用 fallback LLM。"""
        try:
            method = getattr(agent, method_name)
            return await method(*args)
        except Exception as e:
            if fallback_llm is not None:
                logger.warning(f"{agent_label} 主模型失败，使用 fallback: {e}", exc_info=True)
                fallback_agent = agent_cls(model=fallback_llm, **agent_init_kwargs)
                method = getattr(fallback_agent, method_name)
                return await method(*args)
            raise

    async def _run_coder_with_handoff(
        self, coder, prompt: str, subtask_title: str, agent_label: str,
        fallback_llm, work_dir: str, interpreter,
    ) -> CoderToWriter:
        """运行 CoderAgent，失败时使用 fallback。"""
        try:
            return await coder.run(prompt, subtask_title)
        except Exception as e:
            if fallback_llm is not None:
                logger.warning(f"{agent_label} 主模型失败，使用 fallback: {e}", exc_info=True)
                from app.core.agents.coder_agent import CoderAgent as CA
                fallback_coder = CA(
                    task_id=self.task_id, model=fallback_llm,
                    work_dir=work_dir, code_interpreter=interpreter,
                )
                return await fallback_coder.run(prompt, subtask_title)
            raise

    async def _handle_checkpoint(self, checkpoint_id: str, data: dict) -> dict:
        """HIL checkpoint，Redis 不可用时自动确认继续。"""
        if not settings.HIL_ENABLED:
            return {"action": "confirm"}
        if not settings.HIL_CHECKPOINTS.get(checkpoint_id, True):
            return {"action": "confirm"}

        try:
            await redis_manager.publish_message(
                self.task_id,
                ApprovalMessage(
                    checkpoint_id=checkpoint_id, prompt=data,
                    timeout=settings.HIL_TIMEOUT,
                ),
            )
            return await checkpoint_manager.wait_for_decision(
                self.task_id, checkpoint_id, data, settings.HIL_TIMEOUT,
            )
        except Exception as e:
            logger.warning(f"HIL checkpoint '{checkpoint_id}' 失败(自动确认继续): {e}")
            return {"action": "confirm"}

    # ---- Evaluator 反馈循环 ----

    async def _run_modeler_with_feedback(
        self, modeler_llm, fallbacks, parser_result, evaluator, rag_context: str = ""
    ) -> ModelerToCoder:
        """运行 ModelerAgent，Agent 自主通过 search_knowledge 工具检索知识。"""
        modeler = ModelerAgent(task_id=self.task_id, model=modeler_llm)

        result = await self._run_with_handoff(
            modeler, "run_with_tools", (parser_result, None), "modeler",
            fallbacks.get("modeler"),
            ModelerAgent, {"task_id": self.task_id},
        )

        # ModelerAgent 自反思：检查方法选择是否一致、是否覆盖所有问题
        modeler_review_prompt = """请审查以上建模方案是否存在以下问题：
1. 四个子问题是否全部覆盖
2. 每个问题的方法选择是否有充分理由
3. 全文方法名是否统一（如聚类只用GMM或只用层次聚类，不要混用）
4. 敏感性分析方案是否包含关键参数
如有问题请直接输出修正后的完整JSON，无问题请原样输出。"""
        modeler_output_str = json.dumps(result.model_dump(mode="json"), ensure_ascii=False)[:4000]
        reviewed_modeler = await self._self_review(
            modeler_output_str, modeler_review_prompt, modeler_llm, "ModelerAgent",
        )
        if reviewed_modeler != modeler_output_str:
            try:
                new_json = json.loads(reviewed_modeler)
                eda_plan = new_json.pop("eda", "")
                new_result = ModelerToCoder(
                    questions_solution=new_json,
                    eda_plan=json.dumps(eda_plan, ensure_ascii=False) if isinstance(eda_plan, dict) else str(eda_plan),
                )
                if new_result.questions_solution:
                    result = new_result
                    logger.info("ModelerAgent 自反思修正已应用")
            except Exception:
                pass

        # Evaluator 评估 + 反馈重跑
        if evaluator and settings.MAX_FEEDBACK_ROUNDS > 0:
            result = await self._evaluate_and_rerun(
                evaluator, result, "modeler", modeler,
                fallbacks.get("modeler"), ModelerAgent, {"task_id": self.task_id},
                agent_input=parser_result,
            )

        return result

    async def _self_review(
        self, content: str, review_prompt: str, model,
        agent_name: str, max_rounds: int = 2,
    ) -> str:
        """让 LLM 自反思自己的输出，发现问题就修正。

        比 Evaluator 轻量：不需要单独的评估模型，只用同一 LLM 多调一次。
        适用于 Parser/Modeler/Coder 等非结构化输出的质量把关。

        Args:
            content: Agent 的初始输出。
            review_prompt: 告诉 LLM 从什么角度审查自己。
            model: LLM 实例。
            agent_name: Agent 名称（日志用）。
            max_rounds: 最多反思轮数。

        Returns:
            改进后的输出（如果 LLM 认为没问题则返回原文）。
        """
        if not model:
            return content

        for r in range(max_rounds):
            history = [
                {"role": "user", "content": content},
                {"role": "user", "content": review_prompt},
            ]
            try:
                resp = await model.chat(
                    history=history,
                    tools=None, tool_choice=None,
                    agent_name=f"{agent_name}-Review",
                )
                improved = resp.choices[0].message.content
                if not improved or len(improved) < 10:
                    return content
                # 如果改进后的内容比原文长很多（>2x），说明 LLM 在啰嗦，保留原文
                if len(improved) > len(content) * 2.5:
                    logger.info(f"{agent_name} 自反思: 输出膨胀 ({len(content)}→{len(improved)})，保留原文")
                    return content
                logger.info(f"{agent_name} 自反思: 第{r+1}轮修正 ({len(content)}→{len(improved)} 字)")
                content = improved
            except Exception as e:
                logger.warning(f"{agent_name} 自反思失败: {e}")
                return content
        return content

    async def _evaluate_and_rerun(
        self, evaluator, result, agent_name: str,
        agent, fallback_llm, agent_cls, agent_init_kwargs,
        agent_input=None,
    ):
        """用 Evaluator 评估 Agent 输出，不达标则反馈重跑。

        注意：当前仅对 ModelerToCoder 类型的结果做评估。
        评估不达标时，用反馈内容作为额外 prompt 重新请求 LLM。
        agent_input: 传给 agent.run() 的原始输入（如 ParserToModeler 对象）
        """
        for round_num in range(settings.MAX_FEEDBACK_ROUNDS):
            output_str = json.dumps(
                result.model_dump(mode="json") if hasattr(result, 'model_dump') else str(result),
                ensure_ascii=False,
            )[:4000]
            eval_result = await evaluator.evaluate(f"{agent_name} Agent 的输出", output_str)

            if eval_result.passed or eval_result.score >= settings.EVALUATION_THRESHOLD:
                logger.info(f"{agent_name} 评估通过: {eval_result.score:.2f} (round {round_num+1})")
                return result

            logger.warning(
                f"{agent_name} 评估不达标 ({eval_result.score:.2f} < {settings.EVALUATION_THRESHOLD})，"
                f"反馈: {eval_result.feedback[:100]}"
            )
            # 反馈注入：将 evaluator 的建议作为修正指令传给 agent
            feedback_prompt = f"你的上一次输出质量不够好，审稿人给出以下反馈：\n{eval_result.feedback}\n\n请根据反馈修正你的输出，保持JSON格式不变。"
            try:
                # 重新创建 agent 实例以清空 chat_history
                new_agent = agent_cls(model=agent.model, **agent_init_kwargs)
                if hasattr(new_agent, 'run') and agent_input is not None:
                    # 将 feedback 注入 agent_input.questions（会被 agent 传给 LLM）
                    if hasattr(agent_input, 'questions') and isinstance(agent_input.questions, dict):
                        agent_input.questions = {
                            "_feedback": eval_result.feedback,
                            **agent_input.questions,
                        }
                    result = await new_agent.run(agent_input)
                elif hasattr(new_agent, 'run'):
                    result = await new_agent.run(
                        "你的输出需要改进，请重新生成: " + feedback_prompt[:500]
                    )
                else:
                    break
            except Exception as e:
                logger.warning(f"{agent_name} 反馈重跑失败: {e}")
                break

        return result


# ---- 模块级辅助函数 ----


async def _describe_figures(
    work_dir: str, context: str = ""
) -> dict[str, str]:
    """用千问 VL 识图模型为工作目录下每张图片生成内容描述。

    DeepSeekV4Pro 无法"看到"图片，通过此函数将图表转为文字描述，
    注入 WriterAgent 提示词中，使 LLM 能根据实际图表内容撰写分析。

    Args:
        work_dir: 工作目录路径。
        context: 题目背景文本，帮助模型理解图表上下文。

    Returns:
        {filename: 描述文字} 字典。
    """
    import base64
    from app.config.setting import settings

    api_key = settings.VISION_API_KEY
    model = settings.VISION_MODEL
    base_url = settings.VISION_BASE_URL

    if not api_key or not model:
        logger.warning("识图模型未配置，跳过图片识别")
        return {}

    image_files = _scan_work_dir_images(work_dir)
    if not image_files:
        return {}

    # 检查缓存
    cache_path = os.path.join(work_dir, "figure_descriptions.json")
    cached: dict[str, str] = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            # 只复用缓存中仍然存在的图片
            cached = {k: v for k, v in cached.items() if k in image_files}
        except Exception:
            cached = {}

    # 过滤已缓存的
    need_describe = [img for img in image_files if img not in cached]
    if not need_describe:
        logger.info(f"图片描述: {len(cached)} 张全部命中缓存")
        return cached

    logger.info(f"图片描述: {len(need_describe)} 张待识别（共 {len(image_files)} 张）")

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)
    except ImportError:
        logger.warning("openai 库未安装，跳过图片识别")
        return cached

    import asyncio

    # 每批最多 5 张图
    batch_size = 5
    for i in range(0, len(need_describe), batch_size):
        batch = need_describe[i:i + batch_size]
        content_parts = [
            {
                "type": "text",
                "text": (
                    "请按顺序描述以下每张图片的内容。用'---'分隔每张图的描述。\n"
                    "这些图片来自数学建模论文的数据分析，包括数据分布图、统计检验结果图、"
                    "聚类可视化图、分类评估图、相关性分析图等。\n"
                    "对每张图请描述：图表类型、坐标轴含义和单位、主要数据趋势和关键数值、"
                    "图中明显的分组/异常点/聚类结构。\n"
                    f"题目背景: {context[:800] if context else '无'}\n"
                    "每张图描述控制在150字以内。"
                ),
            }
        ]

        for img in batch:
            abs_path = os.path.join(work_dir, img)
            ext = img.rsplit(".", 1)[-1].lower()
            mime_map = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "gif": "image/gif", "svg": "image/svg+xml"}
            mime = mime_map.get(ext, "image/png")
            try:
                with open(abs_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                })
            except Exception as e:
                logger.warning(f"图片读取失败 {img}: {e}")

        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": content_parts}],
                    max_tokens=4096,
                ),
            )
            text = response.choices[0].message.content or ""

            # 按 "---" 分隔，按顺序匹配批次中的图片
            parts = [p.strip() for p in text.split("---") if p.strip()]
            for j, desc in enumerate(parts):
                if j < len(batch):
                    cached[batch[j]] = desc
                    logger.debug(f"图片描述 [{batch[j]}]: {desc[:80]}...")

        except Exception as e:
            logger.warning(f"图片识别调用失败 (批次 {i//batch_size + 1}): {e}")

    # 写入缓存
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cached, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    logger.info(f"图片描述完成: {len(cached)} 张")
    return cached


def _write_paper_clean(content: str, work_dir: str) -> None:
    """将后处理完成的论文写回 res.md。"""
    import os
    path = os.path.join(work_dir, "res.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)








async def _verify_figures_with_vl(
    paper_content: str, work_dir: str,
) -> dict[str, str]:
    """用千问 VL 验证论文中每张图的描述是否与图片实际内容一致。

    返回 {filename: vl_description}，供 ReviewerAgent 对照评分。
    """
    import base64, re
    from app.config.setting import settings

    api_key = settings.VISION_API_KEY
    model = settings.VISION_MODEL
    base_url = settings.VISION_BASE_URL
    if not api_key or not model:
        return {}

    # 提取论文中的所有图片引用
    fig_refs: list[tuple[str, str, str]] = []  # (filename, paper_desc, line_context)
    for m in re.finditer(r'!\[(.*?)\]\(([^)]+)\)', paper_content):
        paper_desc = m.group(1)
        filepath = m.group(2)
        # 找图注（紧跟的 **图N: xxx**）
        after = paper_content[m.end():m.end()+200]
        cm = re.search(r'\*\*图\d+[：:]\s*(.+?)\*\*', after)
        caption = cm.group(1) if cm else paper_desc
        fig_refs.append((os.path.basename(filepath), caption, filepath))

    if not fig_refs:
        return {}

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)
    except ImportError:
        return {}

    results: dict[str, str] = {}
    import asyncio

    for fname, paper_caption, filepath in fig_refs:
        abs_path = os.path.join(work_dir, filepath)
        if not os.path.exists(abs_path):
            results[fname] = "[图片文件不存在]"
            continue

        try:
            with open(abs_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": "用一句话描述这张图的内容(图表类型+核心数据)，不超过40字。不要评价，只描述。"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ]}],
                    max_tokens=80,
                ),
            )
            vl_desc = response.choices[0].message.content.strip()
            results[fname] = vl_desc
            logger.debug(f"VL验证 [{fname}]: {vl_desc[:80]}")
        except Exception as e:
            logger.warning(f"VL验证失败 {fname}: {e}")
            results[fname] = f"[VL调用失败: {e}]"

    # 写入缓存
    cache_path = os.path.join(work_dir, "vl_verification.json")
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    logger.info(f"VL图验证完成: {len(results)} 张")
    return results


