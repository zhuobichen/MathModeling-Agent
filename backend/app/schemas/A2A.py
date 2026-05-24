"""Agent 间通信数据模型定义。

所有 A2A 消息均继承 AgentMessage 基类，提供：
- schema_version: 协议版本号，支持向后兼容演进
- timestamp: 消息时间戳，用于调试和链路追踪
- metadata: 扩展元数据，可携带 trace_id / agent 标识等
"""

from datetime import datetime, timezone
from pydantic import BaseModel, Field
from typing import Any


class AgentMessage(BaseModel):
    """统一 Agent 间通信基类，所有 A2A 消息继承此类。

    schema_version 允许协议逐步演进，新版可新增字段而不破坏旧版解析。
    timestamp 自动创建，便于追踪消息生命周期。
    metadata 字典可携带 trace_id、来源agent 等扩展信息。
    """

    schema_version: str = Field(default="1.0")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


class ParserToModeler(AgentMessage):
    """ParserAgent 传递给 ModelerAgent 的数据结构。"""

    title: str = ""
    background: str = ""
    questions: dict
    ques_count: int
    data_files: list[dict] = []
    question_types: dict[str, str] = {}


class ModelerToCoder(AgentMessage):
    """ModelerAgent 传递给 CoderAgent 的数据结构。"""

    questions_solution: dict[str, str]
    eda_plan: str = ""


class CoderToWriter(AgentMessage):
    """CoderAgent 传递给 WriterAgent 的数据结构。"""

    code_response: str | None = None
    code_output: str | None = None
    created_images: list[str] | None = None


class WriterResponse(AgentMessage):
    """WriterAgent 的响应数据结构。"""

    response_content: Any
    footnotes: list[tuple[str, str]] | None = None


class ReviewerResult(AgentMessage):
    """ReviewerAgent 的评审结果。"""

    overall_score: float = 0.0
    passed: bool = True
    dimensions: dict[str, dict] = {}
    suggestions: list[str] = []
    summary: str = ""


class EvaluationResult(AgentMessage):
    """评估器的评估结果。"""

    passed: bool = True
    score: float = 1.0
    feedback: str = ""
    should_handoff: bool = False
    reason: str = ""
