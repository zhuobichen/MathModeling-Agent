"""ReviewerAgent —— Stage 5: 论文质量评审。"""

from app.core.agents.agent import Agent
from app.core.llm.llm import LLM
from app.core.skill_loader import skill_loader
from app.schemas.A2A import ReviewerResult
from app.utils.log_util import logger
from app.utils.json_repair import repair_json
from app.tools.tool_registry import tool_registry
import json
import re


class ReviewerAgent(Agent):
    """ReviewerAgent 负责从 5 个维度评审生成的论文质量。"""

    def __init__(
        self,
        task_id: str,
        model: LLM,
        max_chat_turns: int | None = None,
        max_retries: int = 3,
    ) -> None:
        super().__init__(task_id, model, max_chat_turns)
        self.system_prompt = skill_loader.get_system_prompt("reviewer")
        self.max_retries = max_retries
        self.available_tools = tool_registry.get_schemas(["search_papers"])

    async def run(self, paper_content: str, task_summary: str = "") -> ReviewerResult:  # type: ignore[reportIncompatibleMethodOverride]
        """评审论文质量。

        Args:
            paper_content: 完整的论文 Markdown 内容。
            task_summary: 任务摘要（题目背景、建模方案等）。

        Returns:
            ReviewerResult 对象，包含各维度评分和改进建议。
        """
        await self.append_chat_history(
            {"role": "system", "content": self.system_prompt}
        )

        # 截断过长的论文内容（保留前 35000 字和后 5000 字）
        if len(paper_content) > 40000:
            truncated = paper_content[:35000] + "\n...(中间内容省略)...\n" + paper_content[-5000:]
        else:
            truncated = paper_content

        review_prompt = f"## 任务背景\n{task_summary[:500]}\n\n## 论文内容（全文）\n{truncated}\n\n请对以上论文进行5维度评审，返回JSON格式。"
        await self.append_chat_history({"role": "user", "content": review_prompt})

        for attempt in range(self.max_retries + 1):
            response = await self.model.chat(
                history=self.chat_history,
                tools=self.available_tools if self.available_tools else None,
                tool_choice="auto" if self.available_tools else None,
                agent_name=self.__class__.__name__,
            )

            json_str = response.choices[0].message.content
            if not json_str:
                raise ValueError("ReviewerAgent 返回空响应")

            result = repair_json(json_str)
            if result:
                logger.info(f"ReviewerAgent 评审完成，综合评分: {result.get('overall_score', '?')}")
                return ReviewerResult(
                    overall_score=result.get("overall_score", 0.0),
                    passed=result.get("passed", True),
                    dimensions=result.get("dimensions", {}),
                    suggestions=result.get("suggestions", []),
                    summary=result.get("summary", ""),
                )

            logger.warning(f"ReviewerAgent JSON解析失败 (尝试 {attempt+1}/{self.max_retries+1})")
            await self.append_chat_history({"role": "assistant", "content": json_str})
            await self.append_chat_history(
                {"role": "user", "content": "你返回的JSON格式有误，请严格按照JSON格式重新输出。"}
            )

        raise ValueError(f"ReviewerAgent 超过最大重试次数({self.max_retries})")
