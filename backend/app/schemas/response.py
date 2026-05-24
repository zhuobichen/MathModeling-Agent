"""响应数据模型定义，包括消息类型和代码执行结果。"""

from typing import Literal, Union
from app.schemas.enums import AgentType
from pydantic import BaseModel, Field
from uuid import uuid4


class Message(BaseModel):
    """消息基类。"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    msg_type: Literal["system", "agent", "user", "tool"]
    content: str | None = None


class ToolMessage(Message):
    msg_type: Literal["system", "agent", "user", "tool"] = "tool"
    tool_name: Literal["execute_code", "search_scholar"]
    input: dict | None = None
    output: list | None = None


class SystemMessage(Message):
    msg_type: Literal["system", "agent", "user", "tool"] = "system"
    type: Literal["info", "warning", "success", "error"] = "info"


class UserMessage(Message):
    msg_type: Literal["system", "agent", "user", "tool"] = "user"


class AgentMessage(Message):
    msg_type: Literal["system", "agent", "user", "tool"] = "agent"
    agent_type: AgentType


class ParserMessage(AgentMessage):
    agent_type: AgentType = AgentType.PARSER


class ModelerMessage(AgentMessage):
    agent_type: AgentType = AgentType.MODELER


class CodeExecution(BaseModel):
    """代码执行结果基类。"""
    res_type: Literal["stdout", "stderr", "result", "error"]
    msg: str | None = None


class StdOutModel(CodeExecution):
    res_type: Literal["stdout", "stderr", "result", "error"] = "stdout"


class StdErrModel(CodeExecution):
    res_type: Literal["stdout", "stderr", "result", "error"] = "stderr"


class ResultModel(CodeExecution):
    res_type: Literal["stdout", "stderr", "result", "error"] = "result"
    format: Literal[
        "text", "html", "markdown", "png", "jpeg",
        "svg", "pdf", "latex", "json", "javascript",
    ]


class ErrorModel(CodeExecution):
    res_type: Literal["stdout", "stderr", "result", "error"] = "error"
    name: str
    value: str
    traceback: str


OutputItem = Union[StdOutModel, StdErrModel, ResultModel, ErrorModel]


class ScholarMessage(ToolMessage):
    tool_name: Literal["execute_code", "search_scholar"] = "search_scholar"
    input: dict | None = None
    output: list[str] | None = None


class InterpreterMessage(ToolMessage):
    tool_name: Literal["execute_code", "search_scholar"] = "execute_code"
    input: dict | None = None
    output: list[OutputItem] | None = None


class CoderMessage(AgentMessage):
    agent_type: AgentType = AgentType.CODER


class WriterMessage(AgentMessage):
    agent_type: AgentType = AgentType.WRITER
    sub_title: str | None = None


class ReviewerMessage(AgentMessage):
    agent_type: AgentType = AgentType.REVIEWER


class ApprovalMessage(Message):
    """HIL 审批消息。"""
    msg_type: Literal["system", "agent", "user", "tool", "approval"] = "approval"
    checkpoint_id: str = ""
    prompt: dict = Field(default_factory=dict)
    options: list[str] = Field(
        default_factory=lambda: ["confirm", "edit", "regenerate", "ask", "skip", "abort"]
    )
    timeout: int = 300


MessageType = Union[
    SystemMessage,
    UserMessage,
    ParserMessage,
    ModelerMessage,
    CoderMessage,
    WriterMessage,
    ReviewerMessage,
    ApprovalMessage,
]
