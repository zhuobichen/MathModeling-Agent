"""组件契约 —— 可拆解、可组装的最小环节。

设计要点（见《架构范式转换设计文档》3.3）：

- **LLM 组件与纯函数组件共用同一契约。** 此前 postprocess 的纯函数是游离的模块级
  函数、Agent 是另一套形状；粒度不统一就无法统一组装。
- **组件自声明依赖**（tools / skill / model_role），取代 `AGENT_TOOL_CONFIG`
  那种"把工具绑在 Agent 名字上"的全局注册表。
- **`model_role` 与现有配置一一映射**（parser/modeler/coder/writer/reviewer），
  因此分阶段选模型的能力零成本保留。
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol

from app.core.artifact import Artifact, Issue
from app.core.drivers import CompletionRequest, Driver
from app.utils.log_util import logger


# ─────────────────────────────────────────────────────────────
# 运行上下文
# ─────────────────────────────────────────────────────────────


@dataclass
class StageContext:
    """一次流水线运行中，组件可见的全部上下文。

    组件**只应**通过本对象获取外部依赖，不得直接读全局单例——
    这是让组件可独立测试、可被自主编排复用的前提。
    """

    task_id: str
    work_dir: str
    driver: Driver
    artifacts: dict[str, Artifact] = field(default_factory=dict)
    """stage_id → Artifact。fan_out 的实例以 "stage.item" 为键。"""

    failures: dict[str, str] = field(default_factory=dict)
    """stage_id → 错误信息。optional 阶段失败时记录于此，不中断整体。"""

    feedback: list[Issue] = field(default_factory=list)
    """RepairLoop 注入的待修问题。取代此前往 agent_input.questions 里塞
    `_feedback` 键的数据污染式做法。"""

    resources: dict[str, Any] = field(default_factory=dict)
    """受管的有状态资源（如 code_interpreter）。见设计文档 3.6。"""

    settings: Any = None
    run_id: str = ""
    stage_id: str = ""
    """当前正在执行的 stage id，供组件构造产物时回填。"""

    def get(self, stage_id: str, default: Artifact | None = None) -> Artifact | None:
        return self.artifacts.get(stage_id, default)

    def require(self, stage_id: str) -> Artifact:
        art = self.artifacts.get(stage_id)
        if art is None:
            raise KeyError(
                f"stage {stage_id!r} 的产物不存在（已完成的 stage: "
                f"{sorted(self.artifacts)}）"
            )
        return art

    def payload(self, stage_id: str, default: Any = None) -> Any:
        art = self.artifacts.get(stage_id)
        return art.payload if art is not None else default

    def fan_out_keys(self, stage_id: str) -> list[str]:
        """取某个 fan_out 阶段展开出的全部实例键（形如 "analyze.ques1"）。"""
        prefix = stage_id + "."
        return sorted(k for k in self.artifacts if k.startswith(prefix))

    def fan_out_items(self, stage_id: str) -> list[Artifact]:
        return [self.artifacts[k] for k in self.fan_out_keys(stage_id)]

    def issues_from(self, *stage_ids: str) -> list[Issue]:
        """汇总若干阶段的 Issue。"""
        out: list[Issue] = []
        for sid in stage_ids:
            for art in ([self.artifacts[sid]] if sid in self.artifacts
                        else self.fan_out_items(sid)):
                out.extend(art.issues)
        return out


# ─────────────────────────────────────────────────────────────
# 组件契约
# ─────────────────────────────────────────────────────────────


class ComponentRunner(Protocol):
    async def __call__(self, art: Artifact, ctx: StageContext) -> Artifact: ...


@dataclass(frozen=True)
class Component:
    """一个可拆解、可组装的环节。"""

    name: str
    """全局唯一名，TOML 中以 `component = "<name>"` 引用。"""

    run: ComponentRunner

    in_kind: str | tuple[str, ...] = "*"
    """接受的 `Artifact.kind`。`"*"` 表示不校验。"""

    out_kind: str = "artifact"
    description: str = ""

    tools: tuple[str, ...] = ()
    """自声明所需工具名。取代 AGENT_TOOL_CONFIG。"""

    skill: str | None = None
    """提示词文件名（不含 .md），由 skill_loader 加载。纯函数组件为 None。"""

    model_role: str | None = None
    """模型角色 → settings.{ROLE}_MODEL。纯函数组件为 None。"""

    fallback: bool = True

    produces: tuple[str, ...] = ()
    """语义化的产物声明，供编排层做兼容性校验与文档生成。"""

    def accepts(self, art: Artifact) -> bool:
        if self.in_kind == "*":
            return True
        kinds = self.in_kind if isinstance(self.in_kind, tuple) else (self.in_kind,)
        return art.kind in kinds

    async def __call__(self, art: Artifact, ctx: StageContext) -> Artifact:
        if not self.accepts(art):
            raise TypeError(
                f"组件 {self.name!r} 不接受 kind={art.kind!r}（期望 {self.in_kind}）"
            )
        out = await self.run(art, ctx)
        if out.stage_id == "":
            out.stage_id = ctx.stage_id
        return out


# ─────────────────────────────────────────────────────────────
# 注册表
# ─────────────────────────────────────────────────────────────


class ComponentRegistry:
    """组件注册表。TOML 只引用名字，实现由代码提供。"""

    def __init__(self) -> None:
        self._components: dict[str, Component] = {}

    def register(self, comp: Component) -> Component:
        if comp.name in self._components:
            logger.warning(f"ComponentRegistry: 覆盖已注册组件 {comp.name}")
        self._components[comp.name] = comp
        return comp

    def get(self, name: str) -> Component:
        if name not in self._components:
            raise KeyError(
                f"未注册的组件 {name!r}。已注册: {sorted(self._components)}"
            )
        return self._components[name]

    def all(self) -> list[Component]:
        return list(self._components.values())

    def names(self) -> list[str]:
        return sorted(self._components)

    def describe(self) -> list[dict[str, Any]]:
        """给运维/文档用的组件清单。"""
        return [
            {
                "name": c.name,
                "in": c.in_kind,
                "out": c.out_kind,
                "model_role": c.model_role,
                "skill": c.skill,
                "tools": list(c.tools),
                "desc": c.description,
            }
            for c in self.all()
        ]


# 全局注册表（具体组件在 app/core/components/ 中注册）
component_registry = ComponentRegistry()


# ─────────────────────────────────────────────────────────────
# 三类通用实现
# ─────────────────────────────────────────────────────────────


class ScriptExecutionError(RuntimeError):
    """外部提供的脚本在解释器中执行失败。"""


def _call_maybe_async(fn: Callable, *args, **kwargs) -> Any:
    """纯函数组件可能写成同步或异步，统一处理。"""
    result = fn(*args, **kwargs)
    if inspect.isawaitable(result):
        return result
    async def _wrap():
        return result
    return _wrap()


def make_fn_component(
    name: str,
    fn: Callable[[Artifact, StageContext], Artifact | Any],
    *,
    in_kind: str | tuple[str, ...] = "*",
    out_kind: str = "artifact",
    description: str = "",
) -> Component:
    """纯函数组件：无 LLM、无工具。用于后处理、格式转换、校验等确定性环节。"""

    async def _run(art: Artifact, ctx: StageContext) -> Artifact:
        result = await _call_maybe_async(fn, art, ctx)
        if isinstance(result, Artifact):
            return result
        return Artifact(kind=out_kind, stage_id=ctx.stage_id, payload={"value": result})

    return Component(
        name=name,
        run=_run,
        in_kind=in_kind,
        out_kind=out_kind,
        description=description or getattr(fn, "__doc__", "") or "",
    )


@dataclass
class AgentComponent:
    """LLM 组件：由 (skill, prompt 构造, 结果解析) 三元组描述。

    这是对"5 个 Agent 类各自调 LLM"的统一。提示词与解析逻辑来自既有实现，
    但不再需要为每个 Agent 维护一个类。
    """

    name: str
    skill: str
    prompt_fn: Callable[[Artifact, StageContext], str]
    parse_fn: Callable[[str, Artifact, StageContext], dict[str, Any]]
    out_kind: str
    in_kind: str | tuple[str, ...] = "*"
    model_role: str | None = None
    tools: tuple[str, ...] = ()
    expect: str = "text"
    description: str = ""
    system_override: str | None = None

    async def __call__(self, art: Artifact, ctx: StageContext) -> Artifact:
        system = await self._system(ctx)
        prompt = self.prompt_fn(art, ctx)
        if ctx.feedback:
            prompt = self._inject_feedback(prompt, ctx.feedback)

        raw = await ctx.driver.complete(
            CompletionRequest(
                system=system,
                prompt=prompt,
                agent_name=self.name,
                model_role=self.model_role,
                stage_id=ctx.stage_id,
                sub_title=ctx.stage_id,
                round=len(ctx.feedback),
                tools=None,
                expect=self.expect,
                context={"work_dir": ctx.work_dir, "in_kind": art.kind},
            )
        )

        payload = self.parse_fn(raw, art, ctx)
        return Artifact(
            kind=self.out_kind,
            stage_id=ctx.stage_id,
            payload=payload,
            metrics={k: float(v) for k, v in payload.items()
                     if isinstance(v, (int, float)) and not isinstance(v, bool)},
        )

    async def _system(self, ctx: StageContext) -> str:
        if self.system_override is not None:
            return self.system_override
        try:
            from app.core.skill_loader import skill_loader
            return skill_loader.load(self.skill, include_references=False)
        except Exception as e:
            logger.warning(f"{self.name}: 加载 skill {self.skill!r} 失败: {e}")
            return ""

    @staticmethod
    def _inject_feedback(prompt: str, feedback: list[Issue]) -> str:
        """把待修问题显式注入提示词。

        取代此前 `agent_input.questions["_feedback"] = ...` 的数据污染式注入。
        """
        lines = "\n".join(f"- {i}" for i in feedback)
        return f"{prompt}\n\n[待修正问题]\n{lines}\n"


def make_agent_component(**kwargs: Any) -> Component:
    """把 AgentComponent 适配成统一的 Component。"""
    ac = AgentComponent(**kwargs)
    return Component(
        name=ac.name,
        run=ac,
        in_kind=ac.in_kind,
        out_kind=ac.out_kind,
        tools=ac.tools,
        skill=ac.skill,
        model_role=ac.model_role,
        description=ac.description or f"LLM 组件 ({ac.skill})",
    )


# ─────────────────────────────────────────────────────────────
# 工具循环组件（需要实际执行代码的环节）
# ─────────────────────────────────────────────────────────────


class ToolLoopComponent:
    """带工具调用循环的组件。

    两种驱动下的行为：

    - `api`  驱动：走标准 tool-calling 循环（与原 CoderAgent 等价）。
    - `file` 驱动：期望外部**提供可执行脚本**，由本组件在解释器中执行并收集产物。
      这把"写代码"这件事交给外部 Agent，而框架只负责执行与产物采集——
      与设计文档 G5"同一组件支持两种驱动"一致。
    """

    def __init__(
        self,
        name: str,
        skill: str,
        prompt_fn: Callable[[Artifact, StageContext], str],
        *,
        in_kind: str | tuple[str, ...] = "*",
        out_kind: str = "analysis_result",
        model_role: str | None = "coder",
        tools: tuple[str, ...] = ("execute_code",),
        max_rounds: int = 12,
        description: str = "",
    ):
        self.name = name
        self.skill = skill
        self.prompt_fn = prompt_fn
        self.in_kind = in_kind
        self.out_kind = out_kind
        self.model_role = model_role
        self.tools = tools
        self.max_rounds = max_rounds
        self.description = description

    async def __call__(self, art: Artifact, ctx: StageContext) -> Artifact:
        if ctx.driver.name == "file":
            return await self._run_script_mode(art, ctx)
        return await self._run_api_mode(art, ctx)

    # ── api 模式：复用既有 CoderAgent 的 tool-calling 循环 ──

    async def _run_api_mode(self, art: Artifact, ctx: StageContext) -> Artifact:
        """委托给 `CoderAgent` 执行。

        刻意**不重新实现**工具循环：`CoderAgent` 里的错误自纠、重试计数、
        记忆压缩都经过实践检验，重写一遍只会引入行为差异。
        组件化的目标是"谁来调、按什么顺序调"，不是重写环节内部。
        """
        from app.core.agents.coder_agent import CoderAgent

        interpreter = ctx.resources.get("interpreter")
        if interpreter is None:
            raise RuntimeError(
                f"{self.name}: api 模式需要 ctx.resources['interpreter']"
            )
        llm = ctx.resources.get(f"llm:{self.model_role}") or ctx.resources.get("llm")
        if llm is None:
            raise RuntimeError(
                f"{self.name}: 缺少 model_role={self.model_role!r} 的 LLM 资源"
            )

        coder = CoderAgent(
            task_id=ctx.task_id, model=llm,
            work_dir=ctx.work_dir, code_interpreter=interpreter,
        )
        prompt = self.prompt_fn(art, ctx)
        result = await coder.run(prompt, ctx.stage_id)

        return Artifact(
            kind=self.out_kind,
            stage_id=ctx.stage_id,
            payload={"code_response": result.code_response or ""},
            refs=list(result.created_images or []),
            extras={"code_output": interpreter.get_code_output(ctx.stage_id)},
        )

    # ── file 模式：外部提供脚本，框架执行 ──

    async def _run_script_mode(self, art: Artifact, ctx: StageContext) -> Artifact:
        interpreter = ctx.resources.get("interpreter")
        if interpreter is None:
            raise RuntimeError(f"{self.name}: file 模式同样需要 interpreter 资源")

        prompt = self.prompt_fn(art, ctx)
        script = await ctx.driver.complete(
            CompletionRequest(
                system="",
                prompt=prompt,
                agent_name=self.name,
                model_role=self.model_role,
                stage_id=ctx.stage_id,
                expect="code",
                context={
                    "work_dir": ctx.work_dir,
                    "说明": "请返回可直接执行的完整 Python 脚本（不要 Markdown 代码围栏）",
                },
            )
        )
        code = _strip_code_fence(script)
        interpreter.add_section(ctx.stage_id)
        text, error, err_msg = await interpreter.execute_code(code)

        # 执行失败必须抛出，不能"正常返回一个装错误的产物"。
        # 后者会让解释器把失败实例计入 completed_stages，续跑时被跳过，
        # 表现为"跑完了但结果缺失"——这类静默失败极难排查。
        if error:
            raise ScriptExecutionError(
                f"{self.name}[{ctx.stage_id}] 脚本执行失败: {err_msg[:500]}"
            )

        return Artifact(
            kind=self.out_kind,
            stage_id=ctx.stage_id,
            payload={"code_response": text, "code": code},
            refs=await interpreter.get_created_images(ctx.stage_id),
            extras={"code_output": interpreter.get_code_output(ctx.stage_id)},
        )


def _strip_code_fence(text: str) -> str:
    """去掉外部回答里可能带的 Markdown 代码围栏。"""
    t = text.strip()
    if t.startswith("```"):
        lines = t.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines)
    return t


def make_tool_loop_component(**kwargs: Any) -> Component:
    tlc = ToolLoopComponent(**kwargs)
    return Component(
        name=tlc.name,
        run=tlc,
        in_kind=tlc.in_kind,
        out_kind=tlc.out_kind,
        tools=tlc.tools,
        skill=tlc.skill,
        model_role=tlc.model_role,
        description=f"工具循环组件 ({tlc.skill})",
    )
