"""Pipeline 解释器 —— 把声明式配置变成实际执行。

取代 `MathModelWorkFlow.execute()` 的 300 行单函数与 `resume_execute()` 的
130 行近似重复：两者塌缩为"解释器 + 一份断点"。

关键行为：

- 拓扑执行、`fan_out` 展开、门与重试环、`optional` 不阻塞、checkpoint 自动落盘
- `AwaitingInput` → 存盘并以 `awaiting_input` 状态退出；外部补上应答后 resume 继续
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from app.core.artifact import Artifact
from app.core.component import ComponentRegistry, StageContext
from app.core.drivers import AwaitingInput, Driver
from app.core.pipeline.loader import (
    Pipeline,
    PipelineError,
    StageSpec,
    load_gate_prompts,
)
from app.utils.log_util import logger


CHECKPOINT_FILE = "checkpoint.json"
CHECKPOINT_SCHEMA = "2.0"


@dataclass
class RunResult:
    """一次运行的结局。"""

    status: str
    """completed | awaiting_input | failed"""

    task_id: str
    work_dir: str
    artifacts: dict[str, Artifact] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)
    pending: list[dict[str, str]] = field(default_factory=list)
    error: str | None = None

    def payload(self, stage_id: str, default: Any = None) -> Any:
        art = self.artifacts.get(stage_id)
        return art.payload if art is not None else default

    def paper(self) -> str:
        for sid in ("paper_post", "write", "export"):
            art = self.artifacts.get(sid)
            if art and art.payload.get("content"):
                return art.payload["content"]
        return ""


class PipelineInterpreter:
    """执行一条 Pipeline。"""

    def __init__(
        self,
        pipeline: Pipeline,
        registry: ComponentRegistry,
        driver: Driver,
        *,
        task_id: str,
        work_dir: str,
        settings: Any = None,
        resources: dict[str, Any] | None = None,
        human_gate_handler: Callable[[str, dict], Any] | None = None,
        review_handler: Callable[[str, str], Any] | None = None,
    ):
        self.pipeline = pipeline
        self.registry = registry
        self.driver = driver
        self.task_id = task_id
        self.work_dir = work_dir
        self.settings = settings
        self.resources = dict(resources or {})
        self.human_gate_handler = human_gate_handler
        self.review_handler = review_handler

        self.ctx = StageContext(
            task_id=task_id,
            work_dir=work_dir,
            driver=driver,
            resources=self.resources,
            settings=settings,
            run_id=task_id,
        )
        self._completed: list[str] = []
        self._pending: list[dict[str, str]] = []

    # ── 公开 API ──

    async def run(self, initial: Artifact | None = None, resume: bool = False) -> RunResult:
        """执行整条流水线。"""
        os.makedirs(self.work_dir, exist_ok=True)

        if resume:
            self._load_checkpoint()
            logger.info(
                f"从断点恢复 {self.task_id}，已完成 {len(self._completed)} 个阶段: "
                f"{self._completed}"
            )
        else:
            self._completed = []
            self.ctx.artifacts = {}

        if initial is not None:
            self.ctx.artifacts.setdefault("__input__", initial)

        try:
            ordered = self.pipeline.topo_sort()
        except PipelineError as e:
            return RunResult("failed", self.task_id, self.work_dir, error=str(e))

        for spec in ordered:
            if self._stage_done(spec):
                logger.info(f"跳过已完成阶段: {spec.id}")
                continue
            try:
                await self._run_stage(spec)
            except AwaitingInput as ai:
                self._pending = self.driver.pending()
                self._save_checkpoint(status="awaiting_input")
                logger.warning(
                    f"阶段 {spec.id} 等待外部输入，已存盘。"
                    f"请填写 {ai.response_path} 后 resume。"
                )
                return RunResult(
                    "awaiting_input", self.task_id, self.work_dir,
                    artifacts=self.ctx.artifacts,
                    failures=self.ctx.failures,
                    pending=self._pending,
                )
            except Exception as e:
                if spec.optional:
                    self.ctx.failures[spec.id] = f"{type(e).__name__}: {e}"
                    logger.warning(f"可选阶段 {spec.id} 失败（继续）: {e}")
                    self.ctx.artifacts[spec.id] = Artifact(
                        kind="skipped", stage_id=spec.id, skipped=True
                    )
                    continue
                logger.exception(f"阶段 {spec.id} 失败: {e}")
                self._save_checkpoint(status="failed")
                return RunResult(
                    "failed", self.task_id, self.work_dir,
                    artifacts=self.ctx.artifacts,
                    failures=self.ctx.failures,
                    error=f"[{spec.id}] {type(e).__name__}: {e}",
                )

        self._save_checkpoint(status="completed")
        return RunResult(
            "completed", self.task_id, self.work_dir,
            artifacts=self.ctx.artifacts,
            failures=self.ctx.failures,
        )

    # ── 阶段执行 ──

    def _stage_done(self, spec: StageSpec) -> bool:
        if spec.id in self._completed:
            return True
        if spec.fan_out:
            # fan_out 阶段：全部实例都完成才算完成
            keys = self._fan_out_keys(spec)
            if keys and all(f"{spec.id}.{k}" in self._completed for k in keys):
                return True
        return False

    async def _run_stage(self, spec: StageSpec) -> None:
        component = self.registry.get(spec.component)

        if spec.fan_out:
            await self._run_fan_out(spec, component)
            return

        self.ctx.stage_id = spec.id
        art_in = self._input_artifact(spec)

        if not component.accepts(art_in):
            raise TypeError(
                f"stage {spec.id!r}: 组件 {component.name!r} 不接受 kind={art_in.kind!r}"
            )

        self._apply_injections(spec)
        log_stage(spec.id, component.name, f"开始（in={art_in.kind}）")

        art_out = await component(art_in, self.ctx)
        art_out = await self._apply_gates(spec, art_out)

        self.ctx.artifacts[spec.id] = art_out
        self._completed.append(spec.id)
        log_stage(spec.id, component.name, f"完成（out={art_out.kind}）")

        await self._after_stage(spec, art_out)
        if spec.checkpoint:
            self._save_checkpoint()

    async def _run_fan_out(self, spec: StageSpec, component) -> None:
        """按集合展开为多个实例。

        注意"展开源"与"输入产物"是两回事：`run_analysis` 按 `parse.questions`
        展开，但它的输入是 `model` 的建模方案。因此这里分开解析：
        实例列表来自 fan_out 源，传给组件的输入走 `_input_artifact`。
        """
        assert spec.fan_out is not None
        src_id = spec.fan_out.source_stage()
        src = self.ctx.artifacts.get(src_id)
        if src is None:
            raise PipelineError(
                f"stage {spec.id!r} 的 fan_out 源 {src_id!r} 尚无产物"
            )

        items = resolve_path(src.payload, spec.fan_out.payload_path())
        if items is None:
            logger.warning(
                f"stage {spec.id!r}: fan_out 源路径 "
                f"{spec.fan_out.over!r} 解析为空，跳过"
            )
            return
        if isinstance(items, dict):
            keys = list(items.keys())
        elif isinstance(items, list):
            keys = [str(i) for i in range(len(items))]
        else:
            raise PipelineError(
                f"stage {spec.id!r}: fan_out 源必须是 dict 或 list，"
                f"实际为 {type(items).__name__}"
            )

        art_in = self._input_artifact(spec)
        failed: list[str] = []

        for key in keys:
            instance_id = f"{spec.id}.{key}"
            if instance_id in self._completed:
                logger.info(f"跳过已完成实例: {instance_id}")
                continue

            self.ctx.stage_id = instance_id
            # 组件统一从 "fan_out_key" 读取当前项；按 TOML 命名的别名一并提供，
            # 使配置可以自定义键名，而组件不必知道配置里叫什么。
            self.ctx.resources["fan_out_key"] = key
            self.ctx.resources[spec.fan_out.key] = key
            self.ctx.resources["fan_out_item"] = (
                items[key] if isinstance(items, dict) else items[int(key)]
            )
            self._apply_injections(spec)

            try:
                art_out = await component(art_in, self.ctx)
                art_out.stage_id = instance_id
                art_out = await self._apply_gates(spec, art_out)
                self.ctx.artifacts[instance_id] = art_out
            except AwaitingInput:
                raise
            except Exception as e:
                # 单个实例失败不中断整体——等价于原 _run_coder 的 per-subtask try/except
                logger.error(f"实例 {instance_id} 失败（继续）: {e}")
                self.ctx.failures[instance_id] = f"{type(e).__name__}: {e}"
                failed.append(instance_id)
                # 失败实例**不**计入 completed_stages，否则续跑会跳过它们
                art_out = Artifact(
                    kind="failed", stage_id=instance_id,
                    payload={"error": str(e)},
                )
                self.ctx.artifacts[instance_id] = art_out
                if spec.checkpoint:
                    self._save_checkpoint()
                continue

            self._completed.append(instance_id)
            if spec.checkpoint:
                self._save_checkpoint()

        log_stage(spec.id, component.name, f"完成 {len(keys) - len(failed)}/{len(keys)} 个实例")
        # 全部实例都失败不是"个别子任务出错"，而是系统性故障（配置错、依赖缺失等）。
        # 此时必须让 stage 失败，否则流水线会带着一堆失败产物一路"成功"下去。
        if keys and len(failed) == len(keys):
            raise PipelineError(
                f"stage {spec.id!r} 的 {len(keys)} 个实例全部失败，"
                f"首个错误: {self.ctx.failures.get(failed[0], '?')}"
            )

    async def _apply_gates(self, spec: StageSpec, art: Artifact) -> Artifact:
        """执行 after 门与 loop 重试环。"""
        for gspec in spec.after:
            from app.core.pipeline.loader import parse_gate_spec
            gname, gparams = parse_gate_spec(gspec)
            gparams = self._resolve_gate_prompt(gparams)
            from app.core.gate import build_gate
            gate = build_gate(gname, **gparams)
            res = await gate.check(art, self.ctx)
            art.merge_issues(res.issues)
            logger.info(f"门 {gname} @ {spec.id}: {res.summary()}")

        # RepairLoop 需要 gate 对象，且 gate 是复用的——重建一份以保证无状态残留
        loop_spec = self._loop_for(spec)
        if loop_spec is not None:
            art, res, rounds = await loop_spec.run(art, self.ctx)
            logger.info(
                f"重试环 {loop_spec.gate.name} @ {spec.id}: "
                f"{res.summary()}（{rounds} 轮修复）"
            )
        return art

    def _loop_for(self, spec: StageSpec) -> Any:
        """取该阶段的 loop（必要时从 loop_from 关联阶段继承）。"""
        if spec.loop is None:
            return None
        # 每次重建，避免 prompt 参数在多次运行间串味
        params = self._resolve_gate_prompt(dict(spec.loop.gate_params))
        from app.core.gate import RepairLoop, build_gate
        return RepairLoop(
            gate=build_gate(spec.loop.gate, **params),
            repair_component=spec.loop.repair_with,
            max_rounds=spec.loop.max_rounds,
            name=spec.loop.gate,
        )

    def _resolve_gate_prompt(self, params: dict[str, Any]) -> dict[str, Any]:
        """把 `prompt_key` 解析成实际提示词文本。"""
        key = params.pop("prompt_key", None)
        if key:
            params["prompt"] = load_gate_prompts(self.pipeline.source_path, key)
        return params

    def _input_artifact(self, spec: StageSpec) -> Artifact:
        src = spec.resolve_input()
        if src is None:
            return self.ctx.artifacts.get(
                "__input__", Artifact(kind="empty", stage_id="__input__")
            )
        art = self.ctx.artifacts.get(src)
        if art is None:
            # optional 上游被跳过时容忍，否则报错
            raise PipelineError(
                f"stage {spec.id!r} 的输入 stage {src!r} 无产物"
                f"（已产出: {sorted(self.ctx.artifacts)}）"
            )
        return art

    def _apply_injections(self, spec: StageSpec) -> None:
        """把上游产物的字段注入 ctx.resources["injected"]。"""
        if not spec.inject:
            return
        injected: dict[str, Any] = self.ctx.resources.setdefault("injected", {})
        for path in spec.inject:
            stage_id, _, field = path.partition(".")
            src = self.ctx.artifacts.get(stage_id)
            if src is None:
                logger.debug(f"inject: {stage_id!r} 无产物，跳过 {path!r}")
                continue
            val = resolve_path(src.payload, field) if field else src.payload
            if val is not None:
                injected[path] = val

    def _fan_out_keys(self, spec: StageSpec) -> list[str]:
        assert spec.fan_out is not None
        src = self.ctx.artifacts.get(spec.fan_out.source_stage())
        if src is None:
            return []
        items = resolve_path(src.payload, spec.fan_out.payload_path())
        if isinstance(items, dict):
            return [str(k) for k in items]
        if isinstance(items, list):
            return [str(i) for i in range(len(items))]
        return []

    async def _after_stage(self, spec: StageSpec, art: Artifact) -> None:
        """人工审核节点。"""
        if spec.human_gate and self.human_gate_handler is not None:
            try:
                await self.human_gate_handler(spec.human_gate, art.payload)
            except Exception as e:
                logger.warning(f"HIL {spec.human_gate} 失败（继续）: {e}")
        if self.review_handler is not None:
            try:
                preview = json.dumps(art.payload, ensure_ascii=False)[:3000]
                await self.review_handler(spec.id, preview)
            except Exception as e:
                logger.debug(f"CLI 审核跳过: {e}")

    # ── Checkpoint ──

    def _ckpt_path(self) -> str:
        return os.path.join(self.work_dir, CHECKPOINT_FILE)

    def _save_checkpoint(self, status: str = "running") -> None:
        path = self._ckpt_path()
        existing: dict[str, Any] = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception:
                existing = {}

        payload = {
            "schema": CHECKPOINT_SCHEMA,
            "task_id": self.task_id,
            "pipeline": self.pipeline.name,
            "status": status,
            "current_stage": self.ctx.stage_id,
            "completed_stages": self._completed,
            "failures": self.ctx.failures,
            "artifacts": {
                k: v.model_dump(mode="json")
                for k, v in self.ctx.artifacts.items()
                if not k.startswith("__")
            },
            "pending": getattr(self.driver, "_pending", []) if status == "awaiting_input" else [],
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        # 保留旧断点里的自定义键，避免覆盖掉外部写入的信息
        for k, v in existing.items():
            if k not in payload and not k.startswith(("parser_result", "modeler_result",
                                                      "coder_results", "review_result")):
                payload[k] = v

        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)  # 原子替换，避免半写坏断点

    def _load_checkpoint(self) -> None:
        path = self._ckpt_path()
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"断点文件不存在: {path}。该任务可能从未启动。"
            )
        with open(path, "r", encoding="utf-8") as f:
            ckpt = json.load(f)

        schema = ckpt.get("schema")
        if schema != CHECKPOINT_SCHEMA:
            migrated = self._migrate_checkpoint(ckpt)
            if migrated is None:
                raise PipelineError(
                    f"断点 schema={schema!r} 无法迁移到 {CHECKPOINT_SCHEMA!r}。"
                    "请从头运行该任务。"
                )
            ckpt = migrated
            logger.info("旧版断点已迁移")

        self._completed = list(ckpt.get("completed_stages") or [])
        self.ctx.failures = dict(ckpt.get("failures") or {})
        self.ctx.artifacts = {
            k: Artifact(**v) for k, v in (ckpt.get("artifacts") or {}).items()
        }

    def _migrate_checkpoint(self, ckpt: dict) -> dict | None:
        """把旧版（v1，顶层平铺 parser_result/modeler_result/...）迁移到 v2。

        旧版没有 Artifact 包装，这里按已知的键名映射到对应 stage 的产物，
        使旧任务仍可续跑（设计文档附录 C 的不可回归项）。
        """
        old_keys = {"parser_result", "modeler_result", "coder_results", "review_result"}
        if not old_keys & set(ckpt):
            return None

        artifacts: dict[str, Any] = {}
        mapping = {
            "parser_result": ("parse", "question_set"),
            "modeler_result": ("model", "model_plan"),
            "review_result": ("review", "review_result"),
        }
        for old_key, (stage_id, kind) in mapping.items():
            if old_key in ckpt:
                artifacts[stage_id] = Artifact(
                    kind=kind, stage_id=stage_id, payload=ckpt[old_key]
                ).model_dump(mode="json")

        if "coder_results" in ckpt:
            for key, val in (ckpt["coder_results"] or {}).items():
                sid = f"analyze.{key}"
                artifacts[sid] = Artifact(
                    kind="analysis_result", stage_id=sid, payload=val
                ).model_dump(mode="json")

        return {
            "schema": CHECKPOINT_SCHEMA,
            "task_id": ckpt.get("task_id"),
            "completed_stages": list(ckpt.get("completed_stages") or []),
            "artifacts": artifacts,
            "failures": {},
            "migrated_from": ckpt.get("schema") or "1.0",
        }


def resolve_path(obj: Any, path: str) -> Any:
    """按点号路径取值，如 resolve_path(payload, "questions")。

    支持形如 "items.0.name"；任一层缺失返回 None。
    """
    if not path:
        return obj
    cur = obj
    for part in path.split("."):
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, (list, tuple)):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            cur = getattr(cur, part, None)
    return cur


def log_stage(stage_id: str, component: str, msg: str) -> None:
    logger.info(f"[{stage_id}] {component}: {msg}")
