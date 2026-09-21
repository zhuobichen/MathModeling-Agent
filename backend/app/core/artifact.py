"""中间产物与问题条目定义。

设计要点（见《架构范式转换设计文档》3.1 / 3.2）：

- `Artifact` 是组件间流转的**唯一**载体，取代此前"阶段间靠 Pydantic 类型互相引用"
  的耦合方式。它继承 `AgentMessage`，保留 schema_version / timestamp / metadata，
  使现有链路追踪能力不丢失。
- `Issue` 携带**章节归属**（`section`）与机器可判的类别（`code`）。
  这是消除 `dim_section_map` / `error_section_mapping` 两张硬编码映射表的地基：
  有了 section，重写目标由 Issue 自己携带，下游不需要查表猜。
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.A2A import AgentMessage


class Issue(BaseModel):
    """一条带归属的问题条目。

    检查器（Gate / 校验器 / 评审）产生 Issue，`repair` 按 `section` 分组决定重写谁。
    """

    section: str | None = None
    """章节归属。None 表示全局问题（如"数据文件未识别"）。"""

    severity: Literal["error", "warning", "info"] = "warning"

    code: str = ""
    """机器可判的类别，如 "figure_missing" / "data_unsupported" / "heading_level"。
    用于统计与规则匹配，避免对自然语言做字符串包含判断。"""

    message: str = ""
    fix_hint: str = ""
    source: str = ""
    """产出该 Issue 的 Gate 名，便于定位是哪道门报的。"""

    def __str__(self) -> str:
        loc = f"[{self.section}] " if self.section else ""
        return f"{loc}{self.message}"


class Artifact(AgentMessage):
    """组件间流转的统一中间产物。

    `payload` 的形状由 `kind` 约定；迁移期直接复用现有 A2A 类型
    （ParserToModeler / ModelerToCoder / ...）作为 payload 的 schema，
    因此现有 Agent 的输入输出签名几乎不需要改动。
    """

    kind: str
    """如 "question_set" / "model_plan" / "analysis_result" / "section_draft" / "paper"。"""

    stage_id: str = ""
    """产出该产物的 stage id（fan_out 时形如 "analyze.ques1"）。"""

    payload: dict[str, Any] = Field(default_factory=dict)

    refs: list[str] = Field(default_factory=list)
    """工作目录中的产物路径（figures/*.png、notebook、数据文件等）。"""

    metrics: dict[str, float] = Field(default_factory=dict)
    """供 Gate 使用的量化指标（如 ques_count、figure_count）。"""

    issues: list[Issue] = Field(default_factory=list)
    """累积的问题。Gate 产生的 Issue 会合并到这里。"""

    extras: dict[str, Any] = Field(default_factory=dict)
    """不适合放进 payload 的附带数据，例如物化的 code_output
    （见设计文档 3.6 方案 B：消除 interpreter 跨阶段隐式通道）。"""

    skipped: bool = False

    # ── 便捷构造 ──

    @classmethod
    def of(cls, kind: str, stage_id: str = "", **payload: Any) -> "Artifact":
        """从零散字段快速构造。"""
        return cls(kind=kind, stage_id=stage_id, payload=payload)

    def merge_issues(self, issues: list[Issue]) -> "Artifact":
        """并入一批 Issue，返回自身（便于链式调用）。"""
        self.issues.extend(issues)
        return self

    def section_issues(self) -> dict[str, list[Issue]]:
        """按章节归属分组，供 RepairLoop 决定重写范围。

        无归属（section=None）的 Issue 归入 "" 键——重写器可据此判断
        "这是全局问题，不该由章节重写解决"。
        """
        grouped: dict[str, list[Issue]] = {}
        for iss in self.issues:
            grouped.setdefault(iss.section or "", []).append(iss)
        return grouped

    def blocking(self) -> list[Issue]:
        """仅返回 error 级 Issue。"""
        return [i for i in self.issues if i.severity == "error"]

    def ok(self) -> bool:
        """无 error 级 Issue 即为通过。"""
        return not self.blocking()


def issue_from_validation(validation_result: Any, default_section: str | None = None) -> list[Issue]:
    """把 PaperValidator 的校验结果适配成 Issue[]。

    设计文档 3.2 提到的"唯一需要触碰检查器解析逻辑的地方"——
    把只返回自然语言/错误类型的校验结果转换为带章节归属的 Issue。

    Args:
        validation_result: `PaperValidator.validate()` 的返回值。
        default_section: 校验结果未给出章节时的兜底归属。

    Returns:
        Issue 列表。
    """
    issues: list[Issue] = []
    for err in getattr(validation_result, "errors", []) or []:
        section = getattr(err, "section", None) or default_section
        issues.append(
            Issue(
                section=section,
                severity="error",
                code=getattr(err, "error_type", "") or "validation",
                message=getattr(err, "message", str(err)),
                fix_hint=getattr(err, "fix_suggestion", "") or "",
                source="PaperValidator",
            )
        )
    return issues
