"""ModelerAgent —— Stage 2: 数据探索(EDA) + 建模方案设计。"""

from app.core.agents.agent import Agent
from app.core.llm.llm import LLM
from app.core.skill_loader import skill_loader
from app.config.setting import settings
from app.schemas.A2A import ParserToModeler, ModelerToCoder
from app.services.redis_manager import redis_manager
from app.schemas.response import SystemMessage
from app.utils.log_util import logger
from app.tools.tool_registry import tool_registry
import json

from app.utils.json_repair import repair_json


def _get_tools_for(agent_name: str) -> list[dict]:
    """从 AGENT_TOOL_CONFIG 获取 Agent 的工具 schema 列表。"""
    from app.core.functions import AGENT_TOOL_CONFIG
    config = AGENT_TOOL_CONFIG.get(agent_name, {"always": [], "optional": []})
    schemas = tool_registry.get_schemas(config["always"])
    schemas.extend(tool_registry.get_schemas(config["optional"]))
    return schemas


class ModelerAgent(Agent):
    """ModelerAgent 负责 EDA 规划和建模方案设计。"""

    def __init__(
        self,
        task_id: str,
        model: LLM,
        max_chat_turns: int = 30,
        max_retries: int = settings.MAX_MODELER_RETRIES,
    ) -> None:
        super().__init__(task_id, model, max_chat_turns)
        self.system_prompt = skill_loader.get_system_prompt("modeler", include_references=False)
        self.max_retries = max_retries
        self.available_tools = _get_tools_for("ModelerAgent")

    async def run(self, parser_result: ParserToModeler) -> ModelerToCoder:  # type: ignore[reportIncompatibleMethodOverride]
        """基于题目解析结果设计 EDA 和建模方案。

        Args:
            parser_result: ParserAgent 传递的结构化问题信息。

        Returns:
            ModelerToCoder 对象，包含各问题的 EDA 方案和建模方案。
        """
        await self.append_chat_history(
            {"role": "system", "content": self.system_prompt}
        )
        await self.append_chat_history(
            {
                "role": "user",
                "content": json.dumps(parser_result.questions, ensure_ascii=False),
            }
        )

        for attempt in range(self.max_retries + 1):
            response = await self.model.chat(
                history=self.chat_history,
                agent_name=self.__class__.__name__,
            )

            json_str = response.choices[0].message.content
            if not json_str:
                raise ValueError("ModelerAgent 返回空响应")

            questions_solution = repair_json(json_str)
            if questions_solution:
                logger.info("ModelerAgent 成功生成建模方案")
                eda_plan = questions_solution.pop("eda", "")
                return ModelerToCoder(
                    questions_solution=questions_solution,
                    eda_plan=json.dumps(eda_plan, ensure_ascii=False) if isinstance(eda_plan, dict) else str(eda_plan),
                )

            logger.warning(f"ModelerAgent JSON解析失败 (尝试 {attempt+1}/{self.max_retries+1})")
            await self.append_chat_history({"role": "assistant", "content": json_str})
            await self.append_chat_history(
                {
                    "role": "user",
                    "content": '你返回的JSON格式有误，请严格按照JSON格式重新输出。注意字符串值内的双引号必须转义为\\"。',
                }
            )

        raise ValueError(f"ModelerAgent 超过最大重试次数({self.max_retries})")

    async def run_with_tools(
        self,
        parser_result: ParserToModeler,
        tools: list[dict] | None = None,
    ) -> ModelerToCoder:
        """支持工具调用（知识库搜索）的建模方案生成。"""
        from app.tools.tool_registry import tool_registry

        await self.append_chat_history(
            {"role": "system", "content": self.system_prompt}
        )
        await self.append_chat_history(
            {
                "role": "user",
                "content": json.dumps(parser_result.questions, ensure_ascii=False),
            }
        )

        for attempt in range(self.max_retries + 1):
            response = await self.model.chat(
                history=self.chat_history,
                tools=tools,
                tool_choice="auto" if tools else None,
                agent_name=self.__class__.__name__,
            )

            msg = response.choices[0].message

            if hasattr(msg, "tool_calls") and msg.tool_calls:
                await self.append_chat_history(msg.model_dump())
                for tool_call in msg.tool_calls:
                    tool_id = tool_call.id
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)

                    logger.info(f"ModelerAgent 调用工具: {tool_name}")
                    await redis_manager.publish_message(
                        self.task_id,
                        SystemMessage(content=f"建模手调用{tool_name}工具"),
                    )

                    try:
                        result = await tool_registry.dispatch(tool_name, tool_args, self.task_id)
                    except ValueError as e:
                        result = f"工具调用失败: {e}"

                    await self.append_chat_history(
                        {"role": "tool", "tool_call_id": tool_id, "name": tool_name, "content": result}
                    )

                next_response = await self.model.chat(
                    history=self.chat_history,
                    agent_name=self.__class__.__name__,
                )
                json_str = next_response.choices[0].message.content
            else:
                json_str = msg.content

            if not json_str:
                raise ValueError("ModelerAgent 返回空响应")

            questions_solution = repair_json(json_str)
            if questions_solution:
                logger.info("ModelerAgent 成功生成建模方案（含工具调用）")
                eda_plan = questions_solution.pop("eda", "")
                return ModelerToCoder(
                    questions_solution=questions_solution,
                    eda_plan=str(eda_plan),
                )

            logger.warning(f"ModelerAgent JSON解析失败 (尝试 {attempt+1}/{self.max_retries+1})")
            await self.append_chat_history({"role": "assistant", "content": json_str})
            await self.append_chat_history(
                {
                    "role": "user",
                    "content": '你返回的JSON格式有误，请严格按照JSON格式重新输出。',
                }
            )

        raise ValueError(f"ModelerAgent 超过最大重试次数({self.max_retries})")
