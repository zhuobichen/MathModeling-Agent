"""质量门 —— 把散落在编排器里的多套质量机制收成可插拔的门。

设计要点（见《架构范式转换设计文档》3.4）：

- **Gate 只产出 Issue，不决定重写什么。** 谁来重写由 Pipeline 声明
  （`loop = { gate = ..., repair_with = ... }`）。
- 因为 `Issue` 携带 `section`，`SectionRepairGate` / `RepairLoop` 不需要任何
  硬编码的"维度→章节"或"错误类型→章节"映射表。

对应关系：

| 本模块的 Gate        | 取代的既有机制                                   |
|----------------------|--------------------------------------------------|
| `SelfReviewGate`     | `workflow._self_review`（3 处内联 prompt）        |
| `EvaluatorGate`      | `workflow._evaluate_and_rerun`                   |
| `ScoreGate`          | Reviewer 反馈重写（阈值 6.0 + dim_section_map）   |
| `FormatGate`         | Phase 4.5（PaperValidator + PaperNormalizer）     |
| `VLFigureGate`       | `workflow._verify_figures_with_vl` + 匹配度打分   |
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel, Field

from app.core.artifact import Artifact, Issue
from app.core.drivers import CompletionRequest
from app.utils.log_util import logger


class GateResult(BaseModel):
    """一道门的判定结果。"""

    passed: bool = True
    score: float | None = None
    issues: list[Issue] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)

    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    def summary(self) -> str:
        if self.passed:
            return f"通过{f'（{self.score:.1f}）' if self.score is not None else ''}"
        return f"未通过：{len(self.issues)} 个问题" + (
            f"，{len(self.errors())} 个 error" if self.errors() else ""
        )


class Gate(Protocol):
    """质量门协议。"""

    name: str

    async def check(self, art: Artifact, ctx) -> GateResult:
        ...


# ─────────────────────────────────────────────────────────────
# 内置门
# ─────────────────────────────────────────────────────────────


@dataclass
class SelfReviewGate:
    """让同一个智能来源审查自己的输出，发现问题就以修正后的内容替换。

    取代 `_self_review`：此前审查 prompt 以内联字符串散落在 3 个调用点，
    现在收敛为一个配置项。
    """

    prompt: str
    name: str = "self_review"
    max_rounds: int = 2
    subject: str = "payload"

    async def check(self, art: Artifact, ctx) -> GateResult:
        content = self._subject(art)
        if not content:
            return GateResult(passed=True, raw={"skipped": "空内容"})

        history = [
            {"role": "user", "content": content},
            {"role": "user", "content": self.prompt},
        ]
        try:
            improved = await ctx.driver.complete(
                CompletionRequest(
                    prompt=self.prompt,
                    agent_name=self.name,
                    model_role=None,
                    stage_id=ctx.stage_id,
                    round=0,
                    context={"subject_excerpt": content[:2000]},
                    expect="json" if self._looks_like_json(content) else "text",
                )
            )
        except Exception as e:
            logger.warning(f"{self.name}: 自审失败（跳过，不阻塞）: {e}")
            return GateResult(passed=True, raw={"error": str(e)})

        raw = {"original_chars": len(content), "reviewed_chars": len(improved or "")}
        if not improved or len(improved) < 10:
            return GateResult(passed=True, raw=raw)

        # 膨胀保护：改进后远超原文时视为啰嗦，保留原文
        # （这是从既有实现继承的护栏，见设计文档 6.3 的删除候选项）
        if len(improved) > len(content) * 2.5:
            logger.info(f"{self.name}: 输出膨胀 ({len(content)}→{len(improved)})，保留原文")
            return GateResult(passed=True, raw={**raw, "reverted": True})

        changed = improved.strip() != content.strip()
        if changed:
            self._apply(art, improved)
        return GateResult(
            passed=True,
            raw={**raw, "changed": changed},
            issues=[] if not changed else [
                Issue(severity="info", code="self_review_applied",
                      message="自审后内容被替换", source=self.name)
            ],
        )

    def _subject(self, art: Artifact) -> str:
        if self.subject == "payload":
            return json.dumps(art.payload, ensure_ascii=False)
        if self.subject == "extras":
            return str(art.extras.get("code_output", ""))
        return str(art.payload.get(self.subject, ""))

    def _apply(self, art: Artifact, improved: str) -> None:
        if self.subject == "payload":
            try:
                art.payload = json.loads(improved)
            except Exception:
                art.payload = {"raw": improved}
        elif self.subject == "extras":
            art.extras["code_output"] = improved
        else:
            art.payload[self.subject] = improved

    @staticmethod
    def _looks_like_json(s: str) -> bool:
        t = s.lstrip()
        return t.startswith("{") or t.startswith("[")


@dataclass
class EvaluatorGate:
    """用评估模型给输出打分，不达标则报 error 级 Issue。

    取代 `_evaluate_and_rerun`。此前它只支持 `ModelerToCoder` 一种类型，
    且靠修改 agent 入参注入反馈；现在对任意 Artifact 生效，反馈走 Issue。
    """

    threshold: float = 0.6
    name: str = "evaluator"

    async def check(self, art: Artifact, ctx) -> GateResult:
        evaluator = ctx.resources.get("evaluator")
        if evaluator is None:
            return GateResult(passed=True, raw={"skipped": "未配置 evaluator"})

        body = json.dumps(art.payload, ensure_ascii=False)[:4000]
        try:
            res = await evaluator.evaluate(f"{ctx.stage_id} 的输出", body)
        except Exception as e:
            logger.warning(f"{self.name}: 评估失败（跳过）: {e}")
            return GateResult(passed=True, raw={"error": str(e)})

        score = float(getattr(res, "score", 1.0))
        passed = bool(getattr(res, "passed", True)) or score >= self.threshold
        issues = []
        if not passed:
            issues.append(
                Issue(
                    severity="error",
                    code="evaluation_below_threshold",
                    message=f"评估 {score:.2f} < {self.threshold}："
                            f"{getattr(res, 'feedback', '')[:500]}",
                    fix_hint=getattr(res, "feedback", ""),
                    source=self.name,
                )
            )
        return GateResult(passed=passed, score=score, issues=issues,
                          raw={"feedback": getattr(res, "feedback", "")})


@dataclass
class ScoreGate:
    """按评分阈值判定。

    取代 Reviewer 反馈重写里的 `overall_score < 6.0` 判断，
    以及配套的 `dim_section_map` —— 现在章节归属由 Issue 自己携带。
    """

    threshold: float = 6.0
    name: str = "score_gate"
    score_key: str = "overall_score"

    async def check(self, art: Artifact, ctx) -> GateResult:
        score = art.payload.get(self.score_key)
        if score is None:
            score = art.metrics.get(self.score_key)
        if score is None:
            return GateResult(passed=True, raw={"skipped": "无评分"})

        score = float(score)
        passed = score >= self.threshold
        issues: list[Issue] = []

        if not passed:
            # 弱维度 → Issue，章节归属来自 Reviewer 自己给出的维度 key
            dims = art.payload.get("dimensions") or {}
            for dim_name, dim_data in dims.items():
                if not isinstance(dim_data, dict):
                    continue
                dscore = dim_data.get("score", 10)
                if isinstance(dscore, (int, float)) and dscore < self.threshold:
                    for msg in (dim_data.get("issues") or []) or [""]:
                        issues.append(
                            Issue(
                                section=self._section_of(dim_name, dim_data),
                                severity="error",
                                code=f"weak_dimension:{dim_name}",
                                message=f"维度 {dim_name} 得分 {dscore}: {msg}",
                                source=self.name,
                            )
                        )
            for s in art.payload.get("suggestions") or []:
                issues.append(
                    Issue(severity="error", code="review_suggestion",
                          message=str(s), source=self.name)
                )
        return GateResult(passed=passed, score=score, issues=issues,
                          raw={"threshold": self.threshold})

    @staticmethod
    def _section_of(dim_name: str, dim_data: dict) -> str | None:
        """取章节归属。

        优先用 Reviewer 显式给的 `section`/`sections` 字段；没有则返回 None
        （表示全局问题）。**这里刻意不做 dim→section 的硬编码映射**——
        要修的正确做法是让 Reviewer 输出章节归属，而不是在代码里再建一张表。
        """
        sec = dim_data.get("section") or dim_data.get("sections")
        if isinstance(sec, str):
            return sec
        if isinstance(sec, list) and sec:
            return str(sec[0])
        return None


@dataclass
class FormatGate:
    """论文格式校验 + 标准化。取代 Phase 4.5。

    标准化是**修复**（写回内容），校验产生 Issue。
    """

    name: str = "format_gate"
    normalize: bool = True

    async def check(self, art: Artifact, ctx) -> GateResult:
        from app.utils.paper_validator import PaperValidator
        from app.utils.paper_normalizer import PaperNormalizer

        content = art.payload.get("content") or art.payload.get("paper_md") or ""
        if not content:
            return GateResult(passed=True, raw={"skipped": "无正文"})

        validator = PaperValidator(ctx.work_dir)
        vr = validator.validate(content)
        issues = _issues_from_validator(vr)

        raw = {"score": getattr(vr, "score", None),
               "errors": getattr(vr, "error_count", len(issues))}

        if self.normalize:
            try:
                nr = PaperNormalizer().normalize(content)
                if getattr(nr, "has_changes", False):
                    art.payload["content"] = nr.normalized_content
                    raw["normalized"] = list(getattr(nr, "fixes_applied", []))
            except Exception as e:
                logger.warning(f"{self.name}: 标准化失败: {e}")
                raw["normalize_error"] = str(e)

        return GateResult(
            passed=bool(getattr(vr, "passed", not issues)),
            score=getattr(vr, "score", None),
            issues=issues,
            raw=raw,
        )


@dataclass
class VLFigureGate:
    """用识图模型验证论文中每张图的描述是否与图片实际内容一致。

    取代 `_verify_figures_with_vl` 及其内联的 n-gram 匹配打分逻辑。
    """

    tolerance: float = 0.5
    name: str = "vl_figures"
    partial_score_penalty: int = 3

    async def check(self, art: Artifact, ctx) -> GateResult:
        content = art.payload.get("content") or art.payload.get("paper_md") or ""
        if not content:
            return GateResult(passed=True, raw={"skipped": "无正文"})

        try:
            from app.core.vl import verify_figures_with_vl
            vl = await verify_figures_with_vl(content, ctx.work_dir)
        except Exception as e:
            logger.warning(f"{self.name}: VL 验证不可用（跳过）: {e}")
            return GateResult(passed=True, raw={"skipped": str(e)})

        if not vl:
            return GateResult(passed=True, raw={"skipped": "无图"})

        matched, feedback = _match_figure_captions(content, vl)
        total = len(vl)
        accuracy = matched / total if total else 1.0
        passed = accuracy >= self.tolerance

        issues: list[Issue] = []
        if not passed:
            for fname, desc in _mismatched(content, vl):
                issues.append(
                    Issue(
                        section=_section_of_figure(content, fname),
                        severity="error",
                        code="figure_mismatch",
                        message=f"图 {fname} 描述与图片实际内容不符：{desc[:60]}",
                        source=self.name,
                    )
                )
            issues.append(
                Issue(severity="error", code="figure_mismatch_ratio",
                      message=f"{total - matched}/{total} 张图的描述与实际内容不符，"
                              f"匹配率 {accuracy:.0%} < {self.tolerance:.0%}",
                      source=self.name)
            )

        return GateResult(passed=passed, score=accuracy, issues=issues,
                          raw={"matched": matched, "total": total, "feedback": feedback})


# ─────────────────────────────────────────────────────────────
# 统一的重写环
# ─────────────────────────────────────────────────────────────


@dataclass
class RepairLoop:
    """声明式重试环：gate 不过 → repair → 重新 gate，最多 max_rounds 轮。

    取代三套各自实现的重写路径（Reviewer 反馈重写、格式回退重写、
    Evaluator 重跑），它们此前各有一份"找出坏章节"的硬编码映射表。
    """

    gate: Gate
    repair_component: str | None = None
    max_rounds: int = 1
    name: str = ""

    async def run(self, art: Artifact, ctx) -> tuple[Artifact, GateResult, int]:
        result = await self.gate.check(art, ctx)
        art.merge_issues(result.issues)

        if result.passed:
            return art, result, 0

        if self.repair_component is None:
            logger.info(f"{self.gate.name}: 未声明 repair，仅报告问题")
            return art, result, 0

        from app.core.component import component_registry

        try:
            repair_comp = component_registry.get(self.repair_component)
        except KeyError as e:
            logger.warning(f"RepairLoop: {e}；跳过修复")
            return art, result, 0

        for round_no in range(1, self.max_rounds + 1):
            fixable = [i for i in result.issues if i.severity == "error"]
            if not fixable:
                break
            logger.info(
                f"{self.gate.name}: 第 {round_no}/{self.max_rounds} 轮修复，"
                f"{len(fixable)} 个问题，涉及章节 "
                f"{sorted({i.section or '(全局)' for i in fixable})}"
            )
            ctx.feedback = fixable
            try:
                art = await repair_comp(art, ctx)
            except Exception as e:
                logger.warning(f"RepairLoop: 修复失败: {e}")
                break
            finally:
                ctx.feedback = []

            result = await self.gate.check(art, ctx)
            art.merge_issues(result.issues)
            if result.passed:
                logger.info(f"{self.gate.name}: 第 {round_no} 轮修复后通过")
                return art, result, round_no

        logger.info(f"{self.gate.name}: 修复 {self.max_rounds} 轮后仍未通过")
        return art, result, self.max_rounds


# ─────────────────────────────────────────────────────────────
# 辅助
# ─────────────────────────────────────────────────────────────


_ERROR_TYPE_TO_SECTION = {
    # 仅在检查器无法给出章节归属时的最后兜底，**不是**主要映射手段。
    "章节完整性": "firstPage",
    "章节编号": "RepeatQues",
}


def _issues_from_validator(vr: Any) -> list[Issue]:
    issues: list[Issue] = []
    for err in getattr(vr, "errors", []) or []:
        etype = getattr(err, "error_type", "") or "validation"
        section = getattr(err, "section", None)
        if not section:
            for key, sec in _ERROR_TYPE_TO_SECTION.items():
                if key in etype:
                    section = sec
                    break
        issues.append(
            Issue(
                section=section,
                severity="error",
                code=etype,
                message=getattr(err, "message", str(err)),
                fix_hint=getattr(err, "fix_suggestion", "") or "",
                source="PaperValidator",
            )
        )
    return issues


def _figure_refs(content: str) -> list[tuple[str, str]]:
    """提取 (文件名, 图注) 列表。"""
    import os
    import re

    refs: list[tuple[str, str]] = []
    for m in re.finditer(r"!\[(.*?)\]\(([^)]+)\)", content):
        desc = m.group(1)
        fname = os.path.basename(m.group(2))
        after = content[m.end():m.end() + 200]
        cm = re.search(r"\*\*图\d+[：:]\s*(.+?)\*\*", after)
        refs.append((fname, cm.group(1) if cm else desc))
    return refs


def _match_figure_captions(content: str, vl: dict[str, str]) -> tuple[int, list[str]]:
    """n-gram 匹配图注与识图描述，返回 (匹配数, 不符说明)。"""
    import re

    matched = 0
    feedback: list[str] = []
    captions = dict(_figure_refs(content))
    for fname, vl_desc in vl.items():
        caption = captions.get(fname, "")
        if caption and vl_desc:
            vl_words = set(re.findall(r"[一-鿿]{2,3}", vl_desc))
            cap_words = set(re.findall(r"[一-鿿]{2,3}", caption))
            if len(vl_words & cap_words) >= 2:
                matched += 1
            else:
                feedback.append(f"图{fname}: 论文写「{caption[:40]}」，实际「{vl_desc[:40]}」")
    return matched, feedback


def _mismatched(content: str, vl: dict[str, str]) -> list[tuple[str, str]]:
    captions = dict(_figure_refs(content))
    out = []
    for fname, vl_desc in vl.items():
        caption = captions.get(fname)
        if caption is None:
            continue
        if not caption or not vl_desc:
            continue
        import re
        if len(set(re.findall(r"[一-鿿]{2,3}", vl_desc))
               & set(re.findall(r"[一-鿿]{2,3}", caption))) < 2:
            out.append((fname, vl_desc))
    return out


def _section_of_figure(content: str, fname: str) -> str | None:
    """找出某张图所在的章节（就近向上找最近的标题）。"""
    import re

    idx = content.find(fname)
    if idx < 0:
        return None
    heads = list(re.finditer(r"^#{1,3}\s+(.+)$", content[:idx], re.MULTILINE))
    return heads[-1].group(1).strip() if heads else None


def build_gate(spec: str, **params) -> Gate:
    """按名字构造 Gate。TOML 中写成 `"score_gate(threshold=6.0)"` 会被解析到这里。"""
    table: dict[str, type] = {
        "self_review": SelfReviewGate,
        "evaluator": EvaluatorGate,
        "score_gate": ScoreGate,
        "format_gate": FormatGate,
        "vl_figures": VLFigureGate,
    }
    if spec not in table:
        raise ValueError(f"未知的 gate: {spec!r}（可选 {sorted(table)}）")
    return table[spec](**params)
