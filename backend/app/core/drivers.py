"""智能来源驱动层 —— 把"谁来产生这一步的输出"变成可插拔的。

背景（见《架构范式转换设计文档》4.x）：组件契约把"环节"统一了，但组件内部
仍然需要一个智能来源。此前它被写死为"调用 LLM API"。本模块把它抽象成 `Driver`：

- `ApiDriver`   —— 调 LLM API（原有行为，用于自动化流水线）
- `FileDriver`  —— 把提示词落成文件并等待外部写入答案（用于由外部 Agent/人 逐步驱动）
- `EchoDriver`  —— 返回占位内容（用于验证框架机制，不消耗任何模型）

`FileDriver` 与现有断点续跑机制天然契合：
    请求无应答 → 抛 `AwaitingInput` → 解释器存 checkpoint 并退出
    → 外部写入 answer 文件 → `--resume` 重跑到该步 → 命中应答 → 继续

因此"由外部 Agent 驱动"不是降级模式，而是**同样可续跑**的一种驱动方式。
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.utils.log_util import logger


class AwaitingInput(Exception):
    """外部驱动下，某一步的应答尚未提供。

    解释器捕获此异常后应保存 checkpoint 并以 "awaiting_input" 状态退出，
    待外部写入应答文件后 resume 重跑。
    """

    def __init__(self, request_path: str, response_path: str, stage_id: str):
        self.request_path = request_path
        self.response_path = response_path
        self.stage_id = stage_id
        super().__init__(
            f"等待外部输入: 请填写 {response_path} 后 resume（请求见 {request_path}）"
        )


@dataclass
class CompletionRequest:
    """一次智能调用的请求描述。"""

    system: str = ""
    prompt: str = ""
    agent_name: str = "agent"
    model_role: str | None = None
    stage_id: str = ""
    sub_title: str = ""
    round: int = 0
    """同一 (stage, agent) 下的第几次尝试。loop/repair 会递增，保证文件名稳定可复现。"""
    tools: list[dict] | None = None
    expect: str = "text"
    """期望的产物形态，写入请求文件供外部参考：text / json / code / latex。"""
    context: dict[str, Any] = field(default_factory=dict)
    """附加上下文，会写进请求文件（如可用工具、工作目录文件清单）。"""


class Driver(Protocol):
    """智能来源协议。"""

    name: str

    async def complete(self, req: CompletionRequest) -> str:
        """产生一次输出。无法产生时抛 `AwaitingInput`。"""
        ...

    def pending(self) -> list[dict[str, str]]:
        """列出尚未应答的请求（仅 FileDriver 有意义）。"""
        ...


# ─────────────────────────────────────────────────────────────
# ApiDriver —— 原有行为，调 LLM
# ─────────────────────────────────────────────────────────────


class ApiDriver:
    """通过 LLM API 产生输出。行为等价于此前的直接调用。"""

    name = "api"

    def __init__(self, llm_factory=None, fallbacks: dict | None = None):
        self.llm_factory = llm_factory
        self.fallbacks = fallbacks or {}

    def _llm_for(self, role: str | None):
        if self.llm_factory is None:
            return None
        return self.llm_factory.get_llm_by_role(role) if role else None

    async def complete(self, req: CompletionRequest) -> str:
        llm = self._llm_for(req.model_role)
        if llm is None:
            raise RuntimeError(
                f"ApiDriver: 无法为 model_role={req.model_role!r} 取得 LLM，"
                "请检查 .env.private 中的模型配置"
            )

        history = []
        if req.system:
            history.append({"role": "system", "content": req.system})
        history.append({"role": "user", "content": req.prompt})

        try:
            resp = await llm.chat(
                history=history,
                tools=req.tools,
                tool_choice="auto" if req.tools else None,
                agent_name=req.agent_name,
                sub_title=req.sub_title,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            fb = self.fallbacks.get(req.model_role or "")
            if fb is None:
                raise
            logger.warning(f"{req.agent_name} 主模型失败，使用 fallback: {e}")
            resp = await fb.chat(
                history=history,
                tools=req.tools,
                tool_choice="auto" if req.tools else None,
                agent_name=req.agent_name,
                sub_title=req.sub_title,
            )
            return resp.choices[0].message.content or ""

    def pending(self) -> list[dict[str, str]]:
        return []


# ─────────────────────────────────────────────────────────────
# FileDriver —— 由外部 Agent / 人 提供输出
# ─────────────────────────────────────────────────────────────


class FileDriver:
    """把请求落成文件，等待外部写入应答。

    目录布局（位于 work_dir/_driver/）::

        <seq>__<stage>__<agent>__r<round>__<hash8>.request.json   # 结构化请求
        <seq>__<stage>__<agent>__r<round>__<hash8>.request.md     # 人类可读的提示词
        <seq>__<stage>__<agent>__r<round>__<hash8>.response.md    # ← 外部写入这里
        <seq>__<stage>__<agent>__r<round>__<hash8>.meta.json      # 应答元信息（可选）

    文件名中的 hash8 是提示词内容的短哈希：
      - 提示词相同 → 同名 → resume 时能命中已有应答（**续跑成立**）
      - 提示词不同 → 换名 → 旧应答不会被误用（**正确性成立**）
    """

    name = "file"

    def __init__(self, work_dir: str, subdir: str = "_driver"):
        self.work_dir = work_dir
        self.dir = os.path.join(work_dir, subdir)
        os.makedirs(self.dir, exist_ok=True)
        self._seq = 0
        self._pending: list[dict[str, str]] = []

    def _key(self, req: CompletionRequest) -> str:
        """稳定的请求键：`<stage>__<agent>__r<round>`。

        **刻意不用提示词内容做哈希。** 提示词里含工作目录文件清单等随运行变化的
        内容——若参与命名，第一次运行产生的 `_driver/`、`checkpoint.json`
        会改变第二次运行的清单，进而改变文件名，导致续跑时找不到已写好的应答，
        陷入"每次续跑都要重新作答"的死循环。

        提示词是否变化改由 `prompt_hash` 记录并告警，不影响命名稳定性。
        """
        stage = (req.stage_id or "stage").replace("/", "_").replace("\\", "_")
        agent = req.agent_name.replace("/", "_").replace("\\", "_")
        return f"{stage}__{agent}__r{req.round}"

    @staticmethod
    def _prompt_hash(req: CompletionRequest) -> str:
        return hashlib.sha1(
            (req.system + "\x00" + req.prompt).encode("utf-8")
        ).hexdigest()[:12]

    def _paths(self, req: CompletionRequest) -> tuple[str, str, str]:
        base = os.path.join(self.dir, self._key(req))
        return base + ".request.md", base + ".request.json", base + ".response.md"

    async def complete(self, req: CompletionRequest) -> str:
        req_md, req_json, resp_md = self._paths(req)

        # 1. 写/更新请求
        self._write_request(req, req_md, req_json)

        # 2. 找应答
        if os.path.exists(resp_md):
            with open(resp_md, "r", encoding="utf-8") as f:
                content = f.read()
            if content.strip():
                stale = self._check_stale(req, req_json)
                logger.info(
                    f"FileDriver: 命中外部应答 {os.path.basename(resp_md)}"
                    + ("（提示词已变化，仍沿用该应答）" if stale else "")
                )
                return content
            logger.warning(f"FileDriver: 应答文件为空，视为未提供: {resp_md}")

        # 3. 无应答 → 挂起
        self._pending.append({
            "stage_id": req.stage_id,
            "agent": req.agent_name,
            "request": req_md,
            "response": resp_md,
            "expect": req.expect,
        })
        raise AwaitingInput(req_md, resp_md, req.stage_id)

    def _check_stale(self, req: CompletionRequest, req_json: str) -> bool:
        """提示词相对上次请求是否已变化（仅告警，不影响是否复用应答）。"""
        try:
            with open(req_json, "r", encoding="utf-8") as f:
                old = json.load(f)
        except Exception:
            return False
        cur = self._prompt_hash(req)
        if old.get("prompt_hash") and old["prompt_hash"] != cur:
            logger.warning(
                f"FileDriver: {req.stage_id} 的提示词已变化"
                f"（{old['prompt_hash']} → {cur}），但已有应答，沿用旧应答。"
                "若应答明显不适用，删除对应 .response.md 后重跑。"
            )
            return True
        return False

    def _write_request(self, req: CompletionRequest, req_md: str, req_json: str) -> None:
        """写请求文件。请求已存在时更新内容（提示词可能随上游产物变化）。"""
        self._seq += 1
        with open(req_md, "w", encoding="utf-8") as f:
            f.write(self._render_request(req))
        with open(req_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "stage_id": req.stage_id,
                    "agent_name": req.agent_name,
                    "model_role": req.model_role,
                    "round": req.round,
                    "expect": req.expect,
                    "tools": [t.get("function", {}).get("name") for t in (req.tools or [])],
                    "context": req.context,
                    "prompt_chars": len(req.prompt),
                    "prompt_hash": self._prompt_hash(req),
                    "response_file": os.path.basename(req_json).replace(
                        ".request.json", ".response.md"
                    ),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def _render_request(self, req: CompletionRequest) -> str:
        parts = [
            f"# 待作答请求 · {req.stage_id or '-'} · {req.agent_name}",
            "",
            f"- **stage**: `{req.stage_id}`",
            f"- **agent**: `{req.agent_name}`",
            f"- **model_role**: `{req.model_role}`",
            f"- **round**: {req.round}",
            f"- **期望产物**: {req.expect}",
            f"- **写入位置**: 同目录下的同名 `.response.md`",
            "",
        ]
        if req.context:
            parts += ["## 上下文", "", "```json",
                      json.dumps(req.context, ensure_ascii=False, indent=2)[:3000], "```", ""]
        if req.tools:
            names = [t.get("function", {}).get("name") for t in req.tools]
            parts += [f"## 可用工具（本步期望由你实际执行，而非仅描述）", "",
                      ", ".join(f"`{n}`" for n in names), ""]
        if req.system:
            parts += ["## System Prompt", "", "```", req.system, "```", ""]
        parts += ["## User Prompt", "", "```", req.prompt, "```", ""]
        return "\n".join(parts)

    def pending(self) -> list[dict[str, str]]:
        return list(self._pending)


# ─────────────────────────────────────────────────────────────
# EchoDriver —— 机制验证用，不消耗模型
# ─────────────────────────────────────────────────────────────


class EchoDriver:
    """回显占位内容。用于在不接任何模型的情况下验证框架机制（拓扑/续跑/门）。"""

    name = "echo"

    def __init__(self, canned: dict[str, str] | None = None):
        self.canned = canned or {}
        self.calls: list[str] = []

    async def complete(self, req: CompletionRequest) -> str:
        key = f"{req.stage_id}__{req.agent_name}"
        self.calls.append(key)
        if key in self.canned:
            return self.canned[key]
        if req.stage_id in self.canned:
            return self.canned[req.stage_id]
        logger.info(f"EchoDriver: 占位应答 {key}")
        if req.expect == "json":
            return "{}"
        return f"[ECHO placeholder for {key}]"

    def pending(self) -> list[dict[str, str]]:
        return []


def make_driver(kind: str, **kwargs) -> Driver:
    """按名称构造 driver。"""
    if kind == "api":
        return ApiDriver(**kwargs)
    if kind == "file":
        return FileDriver(**kwargs)
    if kind == "echo":
        return EchoDriver(**kwargs)
    raise ValueError(f"未知 driver: {kind!r}（可选 api / file / echo）")
