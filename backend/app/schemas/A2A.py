"""Agent 间通信数据模型定义。"""

from pydantic import BaseModel
from typing import Any


class ParserToModeler(BaseModel):
    """ParserAgent 传递给 ModelerAgent 的数据结构。"""
    title: str = ""
    background: str = ""
    questions: dict
    ques_count: int
    data_files: list[dict] = []
    question_types: dict[str, str] = {}


class ModelerToCoder(BaseModel):
    """ModelerAgent 传递给 CoderAgent 的数据结构。"""
    questions_solution: dict[str, str]
    eda_plan: str = ""


class CoderToWriter(BaseModel):
    """CoderAgent 传递给 WriterAgent 的数据结构。"""
    code_response: str | None = None
    code_output: str | None = None
    created_images: list[str] | None = None


class WriterResponse(BaseModel):
    """WriterAgent 的响应数据结构。"""
    response_content: Any
    footnotes: list[tuple[str, str]] | None = None


class ReviewerResult(BaseModel):
    """ReviewerAgent 的评审结果。"""
    overall_score: float = 0.0
    passed: bool = True
    dimensions: dict[str, dict] = {}
    suggestions: list[str] = []
    summary: str = ""


class EvaluationResult(BaseModel):
    """评估器的评估结果。"""
    passed: bool = True
    score: float = 1.0
    feedback: str = ""
    should_handoff: bool = False
    reason: str = ""
