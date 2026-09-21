"""Pipeline 声明式定义 —— TOML 加载与校验。

设计要点（见《架构范式转换设计文档》3.5）：

- **代码只提供组件，不提供顺序。** 增删改阶段是改 TOML，不是改 Python 控制流。
- 语义规则（务必与文档一致）：
    `needs`      顺序约束（拓扑排序用），不是数据依赖；数据一律走 ctx.get(stage_id)
    `optional`   失败/跳过时记入 ctx.failures 并继续，不阻塞下游
    `fan_out`    按指定集合展开为多个实例；checkpoint 粒度跟随展开
    `loop`       声明式重试环：gate 不过 → repair_with → 重新 gate
    `checkpoint` 是否在该阶段完成后落盘
"""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field
from typing import Any

from app.core.gate import RepairLoop, build_gate
from app.utils.log_util import logger


class PipelineError(Exception):
    """Pipeline 定义有误（拓扑成环、引用不存在的组件等）。"""


@dataclass
class FanOut:
    """按集合展开阶段实例。"""

    over: str
    """源，形如 "parse.questions"（stage_id.payload路径）。"""

    key: str = "fan_out_key"

    def source_stage(self) -> str:
        return self.over.split(".", 1)[0]

    def payload_path(self) -> str:
        parts = self.over.split(".", 1)
        return parts[1] if len(parts) > 1 else ""


@dataclass
class LoopSpec:
    """声明式重试环。"""

    gate: str
    repair_with: str | None = None
    max_rounds: int = 1
    gate_params: dict[str, Any] = field(default_factory=dict)

    def build(self, gate_prompt_resolver=None) -> RepairLoop:
        params = dict(self.gate_params)
        if gate_prompt_resolver is not None:
            params = gate_prompt_resolver(params)
        return RepairLoop(
            gate=build_gate(self.gate, **params),
            repair_component=self.repair_with,
            max_rounds=self.max_rounds,
            name=self.gate,
        )


@dataclass
class StageSpec:
    """一个阶段的声明。"""

    id: str
    component: str
    needs: list[str] = field(default_factory=list)
    after: list[str] = field(default_factory=list)
    """挂在该阶段之后的门，形如 "self_review(prompt_key='x')"。"""

    before: list[str] = field(default_factory=list)
    input_from: str | None = None
    """显式指定输入产物来自哪个 stage；缺省取第一个 need。"""

    fan_out: FanOut | None = None
    loop: LoopSpec | None = None
    inject: list[str] = field(default_factory=list)
    """从上游产物抽取字段注入 ctx.resources["injected"]，形如 "model.method_contract"。"""

    optional: bool = False
    checkpoint: bool = True
    human_gate: str | None = None
    mode: str | None = None
    """阶段级驱动模式覆盖：deterministic | autonomous。"""

    retries: int = 0
    resources: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def resolve_input(self) -> str | None:
        if self.input_from:
            return self.input_from
        if self.needs:
            return self.needs[0]
        return None


@dataclass
class Pipeline:
    """一条完整的流水线。"""

    name: str
    stages: list[StageSpec]
    driver: str = "api"
    schema: str = "1.0"
    sections: str | None = None
    defaults: dict[str, Any] = field(default_factory=dict)
    source_path: str = ""

    def stage(self, stage_id: str) -> StageSpec:
        for s in self.stages:
            if s.id == stage_id:
                return s
        raise PipelineError(f"未定义的 stage: {stage_id!r}")

    def ids(self) -> list[str]:
        return [s.id for s in self.stages]

    def topo_sort(self) -> list[StageSpec]:
        """按 needs 拓扑排序。成环或引用不存在的 stage 时报错。

        排序稳定：同层按声明顺序，保证 checkpoint 的可复现性。
        """
        by_id = {s.id: s for s in self.stages}
        for s in self.stages:
            for dep in s.needs:
                if dep not in by_id:
                    raise PipelineError(
                        f"stage {s.id!r} 的 needs 引用了不存在的 stage {dep!r}"
                    )
            if s.fan_out and s.fan_out.source_stage() not in by_id:
                raise PipelineError(
                    f"stage {s.id!r} 的 fan_out 源 {s.fan_out.source_stage()!r} 不存在"
                )

        # Kahn，同层按声明顺序
        order_index = {s.id: i for i, s in enumerate(self.stages)}
        indeg = {s.id: len(set(s.needs)) for s in self.stages}
        ready = sorted([s.id for s in self.stages if indeg[s.id] == 0],
                       key=lambda i: order_index[i])
        out: list[StageSpec] = []
        while ready:
            sid = ready.pop(0)
            out.append(by_id[sid])
            for s in self.stages:
                if sid in s.needs:
                    indeg[s.id] -= 1
                    if indeg[s.id] == 0:
                        ready.append(s.id)
            ready.sort(key=lambda i: order_index[i])

        if len(out) != len(self.stages):
            stuck = [s.id for s in self.stages if s not in out]
            raise PipelineError(f"needs 存在环，无法排序。涉及: {stuck}")
        return out


# ─────────────────────────────────────────────────────────────
# 解析
# ─────────────────────────────────────────────────────────────


_GATE_RE = re.compile(r"^\s*([a-zA-Z_][\w]*)\s*(?:\((.*)\))?\s*$", re.DOTALL)


def parse_gate_spec(spec: str) -> tuple[str, dict[str, Any]]:
    """解析 `"self_review(prompt_key='x', max_rounds=2)"` → (name, params)。"""
    m = _GATE_RE.match(spec)
    if not m:
        raise PipelineError(f"无法解析的 gate 声明: {spec!r}")
    name, argstr = m.group(1), m.group(2)
    params: dict[str, Any] = {}
    if argstr:
        for part in _split_args(argstr):
            if "=" not in part:
                raise PipelineError(f"gate 参数必须是 key=value 形式: {part!r}（在 {spec!r} 中）")
            k, _, v = part.partition("=")
            params[k.strip()] = _coerce(v.strip())
    return name, params


def _split_args(s: str) -> list[str]:
    """按逗号切分参数，忽略引号内的逗号。"""
    out, buf, quote = [], [], None
    for ch in s:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
            buf.append(ch)
        elif ch == ",":
            out.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        out.append("".join(buf).strip())
    return [p for p in out if p]


def _coerce(v: str) -> Any:
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
        return v[1:-1]
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def load_pipeline(path: str) -> Pipeline:
    """从 TOML 文件加载 Pipeline。"""
    with open(path, "rb") as f:
        data = tomllib.load(f)

    name = data.get("name") or os.path.splitext(os.path.basename(path))[0]
    stages_raw = data.get("stage") or []
    if not stages_raw:
        raise PipelineError(f"{path} 中没有任何 [[stage]] 定义")

    stages: list[StageSpec] = []
    seen: set[str] = set()
    for raw in stages_raw:
        sid = raw.get("id")
        comp = raw.get("component")
        if not sid:
            raise PipelineError(f"{path} 中存在缺少 id 的 stage")
        if not comp:
            raise PipelineError(f"stage {sid!r} 缺少 component")
        if sid in seen:
            raise PipelineError(f"stage id 重复: {sid!r}")
        seen.add(sid)

        fo = raw.get("fan_out")
        fan_out = None
        if fo:
            if "over" not in fo:
                raise PipelineError(f"stage {sid!r} 的 fan_out 缺少 over")
            fan_out = FanOut(over=fo["over"], key=fo.get("key", "fan_out_key"))

        lp = raw.get("loop")
        loop = None
        if lp:
            gspec = lp.get("gate")
            if not gspec:
                raise PipelineError(f"stage {sid!r} 的 loop 缺少 gate")
            gname, gparams = parse_gate_spec(gspec)
            loop = LoopSpec(
                gate=gname,
                repair_with=lp.get("repair_with"),
                max_rounds=int(lp.get("max_rounds", 1)),
                gate_params=gparams,
            )

        stages.append(
            StageSpec(
                id=sid,
                component=comp,
                needs=list(raw.get("needs") or []),
                after=list(raw.get("after") or []),
                before=list(raw.get("before") or []),
                input_from=raw.get("input_from"),
                fan_out=fan_out,
                loop=loop,
                inject=list(raw.get("inject") or []),
                optional=bool(raw.get("optional", False)),
                checkpoint=bool(raw.get("checkpoint", True)),
                human_gate=raw.get("human_gate"),
                mode=raw.get("mode"),
                retries=int(raw.get("retries", 0)),
                resources=dict(raw.get("resources") or {}),
                raw=raw,
            )
        )

    p = Pipeline(
        name=name,
        stages=stages,
        driver=data.get("driver", "api"),
        schema=str(data.get("schema", "1.0")),
        sections=data.get("sections"),
        defaults=dict(data.get("defaults") or {}),
        source_path=path,
    )
    # 立即做一次拓扑校验，让配置错误在启动时就暴露
    p.topo_sort()
    logger.info(f"Pipeline 已加载: {p.name}（{len(p.stages)} 个阶段，driver={p.driver}）")
    return p


def load_gate_prompts(pipeline_path: str, key: str) -> str:
    """加载 gate 的提示词。

    约定：`pipelines/gates/<key>.md`。相对 pipeline 文件所在目录解析。
    """
    base = os.path.dirname(os.path.abspath(pipeline_path))
    for cand in (
        os.path.join(base, "gates", f"{key}.md"),
        os.path.join(base, f"{key}.md"),
    ):
        if os.path.exists(cand):
            with open(cand, "r", encoding="utf-8") as f:
                return f.read()
    raise PipelineError(
        f"找不到 gate 提示词 {key!r}（查找过 {base}/gates/{key}.md）"
    )
