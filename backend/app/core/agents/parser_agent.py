"""ParserAgent —— Stage 1: 题目解析与任务分解。"""

from app.core.agents.agent import Agent
from app.core.llm.llm import LLM
from app.core.skill_loader import skill_loader
from app.schemas.A2A import ParserToModeler
from app.utils.log_util import logger
from app.utils.common_utils import get_current_files
from app.utils.json_repair import repair_json
from app.tools.tool_registry import tool_registry, _get_tools_for
import json
import re


class ParserAgent(Agent):
    """ParserAgent 负责解析题目文本和附件数据，分解为结构化子任务。"""

    def __init__(
        self,
        task_id: str,
        model: LLM,
        work_dir: str,
        max_chat_turns: int = 3,
        max_retries: int = 5,
    ) -> None:
        super().__init__(task_id, model, max_chat_turns)
        self.work_dir = work_dir
        self.system_prompt = skill_loader.load("parser", include_references=False)
        self.max_retries = max_retries
        self.available_tools = _get_tools_for("ParserAgent")

    async def run(self, question_text: str) -> ParserToModeler:  # type: ignore[reportIncompatibleMethodOverride]
        """解析题目文本并返回结构化任务描述。

        Args:
            question_text: 原始题目文本。

        Returns:
            ParserToModeler 对象，包含解析后的结构化问题信息。
        """
        await self.append_chat_history(
            {"role": "system", "content": self.system_prompt}
        )

        # 列出工作目录中的所有数据文件
        data_files_info = get_current_files(self.work_dir)
        prompt = f"## 题目内容\n{question_text}\n\n## 工作目录中的文件\n{data_files_info}\n\n请解析题目并返回JSON格式结果。"

        await self.append_chat_history({"role": "user", "content": prompt})

        for attempt in range(self.max_retries + 1):
            response = await self.model.chat(
                history=self.chat_history,
                tools=self.available_tools if self.available_tools else None,
                tool_choice="auto" if self.available_tools else None,
                agent_name=self.__class__.__name__,
            )

            msg = response.choices[0].message

            # 处理工具调用（如 read_file）
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                await self.append_chat_history(msg.model_dump())
                for tc in msg.tool_calls:
                    t_name = tc.function.name
                    t_args = json.loads(tc.function.arguments)
                    try:
                        result = await tool_registry.dispatch(t_name, t_args, self.task_id)
                    except ValueError as e:
                        result = f"工具调用失败: {e}"
                    await self.append_chat_history({
                        "role": "tool", "tool_call_id": tc.id,
                        "name": t_name, "content": result,
                    })
                continue

            json_str = msg.content
            if not json_str:
                raise ValueError("ParserAgent 返回空响应")

            result = repair_json(json_str)
            if result:
                logger.info(f"ParserAgent 成功解析题目，识别 {result.get('ques_count', '?')} 个子问题")
                return ParserToModeler(
                    title=result.get("title", ""),
                    background=result.get("background", ""),
                    questions=result,
                    ques_count=result.get("ques_count", 0),
                    data_files=result.get("data_files", []),
                    question_types={
                        f"ques{p['id']}": p.get("type", "other")
                        for p in result.get("sub_problems", [])
                    },
                )

            logger.warning(f"ParserAgent JSON解析失败 (尝试 {attempt+1}/{self.max_retries+1})")
            await self.append_chat_history({"role": "assistant", "content": json_str})
            await self.append_chat_history(
                {
                    "role": "user",
                    "content": "你返回的JSON格式有误，请严格按照JSON格式重新输出，不要包含```json代码块标记。",
                }
            )

        raise ValueError(f"ParserAgent 超过最大重试次数({self.max_retries})，无法解析JSON")
