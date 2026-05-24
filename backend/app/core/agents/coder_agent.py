"""CoderAgent —— Stage 3: 代码生成+执行+自动纠错+结果验证。"""

from app.core.agents.agent import Agent
from app.config.setting import settings
from app.utils.log_util import logger
from app.services.redis_manager import redis_manager
from app.schemas.response import SystemMessage, InterpreterMessage
from app.tools.base_interpreter import BaseCodeInterpreter
from app.core.llm.llm import LLM
from app.schemas.A2A import CoderToWriter
from app.core.skill_loader import skill_loader
from app.utils.common_utils import get_current_files
import json
from app.core.prompts.shared import ErrorClassifier, get_reflection_prompt
from app.core.functions import coder_tools
from app.tools.tool_registry import tool_registry


class CoderAgent(Agent):
    """CoderAgent 负责生成/执行 Python 代码，自动纠错，并验证结果合理性。"""

    def __init__(
        self,
        task_id: str,
        model: LLM,
        work_dir: str,
        max_chat_turns: int | None = settings.MAX_CHAT_TURNS,
        max_retries: int | None = settings.MAX_RETRIES,
        code_interpreter: BaseCodeInterpreter | None = None,
    ) -> None:
        super().__init__(task_id, model, max_chat_turns)
        self.work_dir = work_dir
        self.max_retries = max_retries
        self.is_first_run = True
        import platform
        self.system_prompt = skill_loader.load("coder", include_references=False, PLATFORM=platform.system())
        self.code_interpreter = code_interpreter
        # 核心工具 + 按需工具
        self.available_tools = list(coder_tools)
        from app.core.functions import AGENT_TOOL_CONFIG
        config = AGENT_TOOL_CONFIG.get("CoderAgent", {"always": [], "optional": []})
        self.available_tools.extend(tool_registry.get_schemas(config["optional"]))

    async def run(self, prompt: str, subtask_title: str) -> CoderToWriter:  # type: ignore[reportIncompatibleMethodOverride]
        """执行代码子任务，包含错误自纠和结果验证。

        Args:
            prompt: 子任务描述（来自 ModelerAgent）。
            subtask_title: 子任务标题（如 eda, ques1 等）。

        Returns:
            CoderToWriter 对象，包含代码执行结果和生成的图片。
        """
        logger.info(f"CoderAgent 开始子任务: {subtask_title}")
        assert self.code_interpreter is not None, "code_interpreter 未初始化"
        self.code_interpreter.add_section(subtask_title)

        if self.is_first_run:
            self.is_first_run = False
            await self.append_chat_history(
                {"role": "system", "content": self.system_prompt}
            )
            await self.append_chat_history(
                {
                    "role": "user",
                    "content": f"当前工作目录: {self.work_dir}\n数据文件: {get_current_files(self.work_dir)}",
                }
            )

        await self.append_chat_history({"role": "user", "content": prompt})

        retry_count = 0
        last_error_message = ""

        while True:
            if self.max_retries is not None and retry_count >= self.max_retries:
                logger.error(f"超过最大重试次数: {self.max_retries}")
                return CoderToWriter(
                    code_response=f"任务失败，超过最大重试次数{self.max_retries}",
                    created_images=[],
                )

            if self.max_chat_turns is not None and self.current_chat_turns >= self.max_chat_turns:
                logger.error(f"超过最大对话轮次({self.max_chat_turns})")
                return CoderToWriter(
                    code_response=f"任务失败，超过最大对话轮次{self.max_chat_turns}",
                    created_images=[],
                )

            self.current_chat_turns += 1

            try:
                response = await self.model.chat(
                    history=self.chat_history,
                    tools=self.available_tools,
                    tool_choice="auto",
                    agent_name=self.__class__.__name__,
                )

                if (
                    hasattr(response.choices[0].message, "tool_calls")
                    and response.choices[0].message.tool_calls
                ):
                    await self.append_chat_history(
                        response.choices[0].message.model_dump()
                    )

                    for tool_call in response.choices[0].message.tool_calls:
                        tool_id = tool_call.id
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)

                        if tool_name == "execute_code":
                            code = tool_args["code"]
                            await redis_manager.publish_message(
                                self.task_id,
                                InterpreterMessage(input={"code": code}),
                            )

                            text_to_gpt, error_occurred, error_message = (
                                await self.code_interpreter.execute_code(code)
                            )

                            if error_occurred:
                                await self.append_chat_history({
                                    "role": "tool",
                                    "tool_call_id": tool_id,
                                    "name": "execute_code",
                                    "content": error_message,
                                })
                                retry_count += 1
                                last_error_message = error_message
                                error_type, suggestion = ErrorClassifier.classify(error_message)
                                logger.warning(
                                    f"代码执行错误 [{error_type}] (重试 {retry_count}/{self.max_retries}): "
                                    f"{error_message[:200]}"
                                )
                                reflection_prompt = get_reflection_prompt(
                                    error_message, code, error_type, suggestion,
                                )
                                await redis_manager.publish_message(
                                    self.task_id,
                                    SystemMessage(content="CoderAgent 反思纠错", type="error"),
                                )
                                await self.append_chat_history(
                                    {"role": "user", "content": reflection_prompt}
                                )
                            else:
                                await self.append_chat_history({
                                    "role": "tool",
                                    "tool_call_id": tool_id,
                                    "name": "execute_code",
                                    "content": text_to_gpt,
                                })
                        else:
                            try:
                                result = await tool_registry.dispatch(
                                    tool_name, tool_args, self.task_id
                                )
                            except ValueError as e:
                                result = f"工具调用失败: {e}"
                            await self.append_chat_history({
                                "role": "tool",
                                "tool_call_id": tool_id,
                                "name": tool_name,
                                "content": result,
                            })
                    # 所有 tool_calls 已处理，继续下一轮 LLM 对话
                    continue
                else:
                    logger.info(f"CoderAgent 子任务完成: {subtask_title}")
                    return CoderToWriter(
                        code_response=response.choices[0].message.content,
                        created_images=await self.code_interpreter.get_created_images(subtask_title),
                    )

            except Exception as e:
                logger.error(f"CoderAgent 执行异常: {e}")
                retry_count += 1
                last_error_message = str(e)
                continue
