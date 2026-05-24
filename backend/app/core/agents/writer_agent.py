"""WriterAgent —— Stage 4: 论文生成 + 去AI味处理。"""

from app.core.agents.agent import Agent
from app.core.llm.llm import LLM
from app.core.prompts.writer import get_writer_prompt
from app.schemas.enums import CompTemplate, FormatOutPut
from app.tools.openalex_scholar import OpenAlexScholar
from app.utils.log_util import logger
from app.services.redis_manager import redis_manager
from app.schemas.response import SystemMessage, WriterMessage
import json
from app.core.functions import writer_tools
from app.schemas.A2A import WriterResponse

DEFAULT_WRITER_MAX_RETRIES = 3


class WriterAgent(Agent):
    """WriterAgent 负责撰写完整论文，包含去AI味后处理。"""

    def __init__(
        self,
        task_id: str,
        model: LLM,
        max_chat_turns: int | None = None,
        comp_template: CompTemplate = CompTemplate.CHINA,
        format_output: FormatOutPut = FormatOutPut.Markdown,
        scholar: OpenAlexScholar | None = None,
        max_memory: int = 25,
        max_retries: int = DEFAULT_WRITER_MAX_RETRIES,
    ) -> None:
        super().__init__(task_id, model, max_chat_turns, max_memory)
        self.format_out_put = format_output
        self.comp_template = comp_template
        self.scholar = scholar
        self.is_first_run = True
        self.system_prompt = get_writer_prompt(format_output)
        self.available_images: list[str] = []
        self.max_retries = max_retries

    async def run(  # type: ignore[reportIncompatibleMethodOverride]
        self,
        prompt: str,
        available_images: list[str] | None = None,
        sub_title: str | None = None,
    ) -> WriterResponse:
        """执行写作任务。

        Args:
            prompt: 写作提示（包含编码结果和建模方案）。
            available_images: 可用图片路径列表。
            sub_title: 子任务标题。

        Returns:
            WriterResponse 对象。
        """
        logger.info(f"WriterAgent 开始: {sub_title}")

        if self.is_first_run:
            self.is_first_run = False
            await self.append_chat_history(
                {"role": "system", "content": self.system_prompt}
            )

        if available_images:
            self.available_images = available_images
            image_lines = "\n".join(
                [f"- ![{img}]({img})" for img in available_images]
            )
            image_prompt = (
                f"\n\n【必须插入的图片列表】\n"
                f"{image_lines}\n"
                f"每张图片后需配至少100字的分析解读。插入格式为 ![描述](文件名)。\n"
            )
            prompt = prompt + image_prompt

        self.current_chat_turns += 1
        await self.append_chat_history({"role": "user", "content": prompt})

        last_error: str = ""
        for attempt in range(self.max_retries + 1):
            try:
                response = await self.model.chat(
                    history=self.chat_history,
                    tools=writer_tools,
                    tool_choice="auto",
                    agent_name=self.__class__.__name__,
                    sub_title=sub_title,
                )

                footnotes = []
                response_content: str = ""

                if (
                    hasattr(response.choices[0].message, "tool_calls")
                    and response.choices[0].message.tool_calls
                ):
                    tool_call = response.choices[0].message.tool_calls[0]
                    tool_id = tool_call.id
                    if tool_call.function.name == "search_papers":
                        query = json.loads(tool_call.function.arguments)["query"]
                        await self.append_chat_history(
                            response.choices[0].message.model_dump()
                        )
                        assert self.scholar is not None, "scholar 未初始化"
                        papers = await self.scholar.search_papers(query)
                        papers_str = self.scholar.papers_to_str(papers)
                        await self.append_chat_history(
                            {
                                "role": "tool",
                                "content": papers_str,
                                "tool_call_id": tool_id,
                                "name": "search_papers",
                            }
                        )
                        next_response = await self.model.chat(
                            history=self.chat_history,
                            tools=writer_tools,
                            tool_choice="auto",
                            agent_name=self.__class__.__name__,
                            sub_title=sub_title,
                        )
                        response_content = next_response.choices[0].message.content
                else:
                    response_content = response.choices[0].message.content

                self.chat_history.append(
                    {"role": "assistant", "content": response_content}
                )
                logger.info(f"WriterAgent 完成: {sub_title}")
                return WriterResponse(
                    response_content=response_content, footnotes=footnotes
                )

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"WriterAgent 重试 {attempt+1}/{self.max_retries+1}: {last_error}"
                )
                if attempt < self.max_retries:
                    await redis_manager.publish_message(
                        self.task_id,
                        SystemMessage(
                            content=f"写作手重试中({attempt+1}/{self.max_retries})...",
                            type="warning",
                        ),
                    )

        error_msg = f"WriterAgent 超过最大重试次数({self.max_retries}), 最后错误: {last_error}"
        logger.error(error_msg)
        return WriterResponse(response_content=error_msg, footnotes=[])

    async def summarize(self) -> str:
        """总结对话内容。"""
        try:
            await self.append_chat_history(
                {"role": "user", "content": "请简单总结以上完成什么任务取得什么结果:"}
            )
            response = await self.model.chat(
                history=self.chat_history, agent_name=self.__class__.__name__
            )
            await self.append_chat_history(
                {"role": "assistant", "content": response.choices[0].message.content}
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"WriterAgent 总结失败: {e}")
            return "已完成主要任务处理。"
