"""框架自检 —— 不接任何 LLM，验证组件化流水线的机制是否正确。

覆盖设计文档 P1 的验收项：

  1. TOML 加载与拓扑排序（含成环检测）
  2. 组件解析（配置里引用的组件必须已注册）
  3. fan_out 展开与 checkpoint 粒度
  4. 门（Gate）产出带章节归属的 Issue
  5. RepairLoop 按 Issue.section 触发修复
  6. 断点写入 + resume 跳过已完成阶段
  7. echo 驱动下完整跑通一条真实流水线

运行：
    ./.venv/Scripts/python.exe scripts/selftest_framework.py
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.artifact import Artifact, Issue
from app.core.component import (
    Component,
    StageContext,
    component_registry,
    make_fn_component,
)
from app.core.drivers import CompletionRequest, EchoDriver
from app.core.gate import GateResult, RepairLoop, SelfReviewGate, build_gate
from app.core.pipeline import PipelineInterpreter, load_pipeline
from app.core.pipeline.loader import PipelineError, parse_gate_spec

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"  — {detail}" if detail else ""))


# ─────────────────────────────────────────────────────────────
# 测试用组件
# ─────────────────────────────────────────────────────────────

CALLS: list[str] = []


async def _src(art: Artifact, ctx: StageContext) -> Artifact:
    CALLS.append("src")
    # 产出 3 个项目，供下游 fan_out 展开
    return Artifact(
        kind="items", stage_id=ctx.stage_id,
        payload={"items": {"a": "A 项", "b": "B 项", "c": "C 项"}},
    )


async def _per_item(art: Artifact, ctx: StageContext) -> Artifact:
    key = ctx.resources.get("item_key")
    CALLS.append(f"per_item:{key}")
    return Artifact(kind="done", stage_id=ctx.stage_id,
                    payload={"item": key, "text": f"处理 {key}"})


async def _paper(art: Artifact, ctx: StageContext) -> Artifact:
    CALLS.append("paper")
    n = len(ctx.fan_out_items("expand"))
    return Artifact(kind="paper", stage_id=ctx.stage_id,
                    payload={"content": f"# 论文\n\n共 {n} 项\n\n## ques1\n内容一\n"},
                    metrics={"n": float(n)})


REPAIRS: list[str] = []


async def _repair(art: Artifact, ctx: StageContext) -> Artifact:
    """按 Issue.section 修复——验证不带任何硬编码映射表。"""
    secs = sorted({i.section for i in ctx.feedback if i.section})
    REPAIRS.append(",".join(secs))
    art.payload["content"] = "已修复：" + art.payload.get("content", "")
    return art


class _FlakyGate:
    """第一次检查必失败，修复后通过——用于验证 RepairLoop 真的重试了。"""

    name = "flaky_gate"

    def __init__(self):
        self.calls = 0

    async def check(self, art: Artifact, ctx) -> GateResult:
        self.calls += 1
        if self.calls == 1:
            return GateResult(
                passed=False, score=3.0,
                issues=[Issue(section="ques1", severity="error",
                              code="test_issue", message="第一章不合格", source="flaky_gate")],
            )
        return GateResult(passed=True, score=9.0)


# ─────────────────────────────────────────────────────────────
# 测试用流水线
# ─────────────────────────────────────────────────────────────

TEST_PIPELINE = """
schema = "1.0"
name = "selftest"
driver = "echo"

[[stage]]
id = "src"
component = "t_src"

[[stage]]
id = "expand"
component = "t_per_item"
needs = ["src"]
input_from = "src"
fan_out = { over = "src.items", key = "item_key" }

[[stage]]
id = "paper"
component = "t_paper"
needs = ["expand"]
input_from = "src"
loop = { max_rounds = 2, gate = "flaky_gate", repair_with = "t_repair" }
"""


def register_test_components():
    for name, fn, ik, ok in [
        ("t_src", _src, "*", "items"),
        ("t_per_item", _per_item, "items", "done"),
        ("t_paper", _paper, "*", "paper"),
        ("t_repair", _repair, "paper", "paper"),
    ]:
        if name not in component_registry.names():
            component_registry.register(
                make_fn_component(name, fn, in_kind=ik, out_kind=ok))


# ─────────────────────────────────────────────────────────────
# 各项检查
# ─────────────────────────────────────────────────────────────


def test_gate_spec_parsing():
    print("\n[1] Gate 声明解析")
    n, p = parse_gate_spec("self_review(prompt_key='parser_review', max_rounds=2)")
    check("解析 gate 名", n == "self_review", n)
    check("解析字符串参数", p.get("prompt_key") == "parser_review", str(p))
    check("解析整数参数", p.get("max_rounds") == 2, str(p))

    n2, p2 = parse_gate_spec("score_gate(threshold=6.0)")
    check("解析浮点参数", p2.get("threshold") == 6.0, str(p2))

    n3, p3 = parse_gate_spec("vl_figures")
    check("无参数 gate", n3 == "vl_figures" and p3 == {}, f"{n3} {p3}")

    try:
        build_gate("self_review", prompt="x")
        check("构造 self_review", True)
    except Exception as e:
        check("构造 self_review", False, str(e))

    try:
        build_gate("nonexistent_gate")
        check("未知 gate 报错", False, "应当抛异常")
    except ValueError:
        check("未知 gate 报错", True)


def test_topo_sort(tmpdir: str):
    print("\n[2] 拓扑排序与成环检测")
    p = os.path.join(tmpdir, "t.toml")
    with open(p, "w", encoding="utf-8") as f:
        f.write(TEST_PIPELINE)
    pipe = load_pipeline(p)
    order = [s.id for s in pipe.topo_sort()]
    check("拓扑顺序正确", order == ["src", "expand", "paper"], str(order))

    # 成环
    bad = TEST_PIPELINE.replace('needs = ["src"]', 'needs = ["paper"]')
    bp = os.path.join(tmpdir, "bad.toml")
    with open(bp, "w", encoding="utf-8") as f:
        f.write(bad)
    try:
        load_pipeline(bp)
        check("成环被检出", False, "应当报错")
    except PipelineError as e:
        check("成环被检出", "环" in str(e), str(e)[:60])

    # 引用不存在的 stage
    bad2 = TEST_PIPELINE.replace('needs = ["expand"]', 'needs = ["nope"]')
    bp2 = os.path.join(tmpdir, "bad2.toml")
    with open(bp2, "w", encoding="utf-8") as f:
        f.write(bad2)
    try:
        load_pipeline(bp2)
        check("悬空 needs 被检出", False, "应当报错")
    except PipelineError:
        check("悬空 needs 被检出", True)


def test_missing_component(tmpdir: str):
    print("\n[3] 未注册组件被检出")
    p = os.path.join(tmpdir, "t.toml")
    with open(p, "w", encoding="utf-8") as f:
        f.write(TEST_PIPELINE)
    pipe = load_pipeline(p)
    try:
        component_registry.get("definitely_not_registered")
        check("未注册组件报错", False, "应当抛 KeyError")
    except KeyError:
        check("未注册组件报错", True)
    _ = pipe


async def test_full_run(tmpdir: str):
    print("\n[4] 完整运行（echo 驱动，无 LLM）")
    p = os.path.join(tmpdir, "t.toml")
    with open(p, "w", encoding="utf-8") as f:
        f.write(TEST_PIPELINE)
    pipe = load_pipeline(p)

    work = os.path.join(tmpdir, "work")
    os.makedirs(work, exist_ok=True)

    CALLS.clear()
    REPAIRS.clear()
    flaky = _FlakyGate()
    gate_table["flaky_gate"] = flaky
    # 让 loader.build_gate 能构造测试用门
    import app.core.gate as gate_mod
    orig = gate_mod.build_gate
    gate_mod.build_gate = lambda spec, **kw: (
        gate_table[spec] if spec in gate_table else orig(spec, **kw))

    interp = PipelineInterpreter(
        pipe, component_registry, EchoDriver(),
        task_id="selftest", work_dir=work,
    )
    res = await interp.run(initial=Artifact(kind="raw", payload={}))

    check("运行状态 completed", res.status == "completed", res.status)
    check("src 阶段执行", "src" in res.artifacts)
    check("fan_out 展开 3 个实例",
          len([k for k in res.artifacts if k.startswith("expand.")]) == 3,
          str(sorted(k for k in res.artifacts if k.startswith("expand."))))
    check("每个实例拿到自己的 key",
          all(res.artifacts[f"expand.{k}"].payload.get("item") == k
              for k in ("a", "b", "c")))
    check("下游能汇总 fan_out 实例",
          res.artifacts["paper"].payload["content"].find("共 3 项") >= 0)
    check("RepairLoop 触发了修复", flaky.calls == 2, f"gate 调用 {flaky.calls} 次")
    check("修复按 Issue.section 定位", REPAIRS == ["ques1"], str(REPAIRS))
    check("修复后内容已变更",
          res.artifacts["paper"].payload["content"].startswith("已修复"))

    ck = os.path.join(work, "checkpoint.json")
    check("checkpoint 已写入", os.path.exists(ck))
    with open(ck, encoding="utf-8") as f:
        data = json.load(f)
    check("checkpoint schema 版本", data.get("schema") == "2.0", str(data.get("schema")))
    check("completed_stages 含 fan_out 子项",
          "expand.a" in data.get("completed_stages", []),
          str(data.get("completed_stages")))
    check("checkpoint 保存了产物",
          "paper" in data.get("artifacts", {}),
          str(sorted(data.get("artifacts", {}).keys())))

    # ── resume：应当跳过全部已完成阶段 ──
    print("\n[5] 断点续跑")
    CALLS.clear()
    interp2 = PipelineInterpreter(
        pipe, component_registry, EchoDriver(),
        task_id="selftest", work_dir=work,
    )
    res2 = await interp2.run(resume=True)
    check("resume 后状态 completed", res2.status == "completed", res2.status)
    check("resume 未重复执行任何阶段",
          CALLS == [], f"重新执行了 {CALLS}")
    check("resume 恢复了产物", "paper" in res2.artifacts)
    check("resume 恢复内容一致",
          res2.artifacts["paper"].payload.get("content")
          == res.artifacts["paper"].payload.get("content"))

    gate_mod.build_gate = orig


async def test_partial_resume(tmpdir: str):
    """删掉一个 fan_out 子项的完成记录，验证只重跑该子项。"""
    print("\n[6] 部分续跑（只补缺失的实例）")
    p = os.path.join(tmpdir, "t.toml")
    pipe = load_pipeline(p)
    work = os.path.join(tmpdir, "work")

    ck = os.path.join(work, "checkpoint.json")
    with open(ck, encoding="utf-8") as f:
        data = json.load(f)
    data["completed_stages"] = [s for s in data["completed_stages"]
                                if s not in ("expand.b", "paper")]
    data["artifacts"].pop("paper", None)
    with open(ck, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    import app.core.gate as gate_mod
    orig = gate_mod.build_gate
    gate_mod.build_gate = lambda spec, **kw: (
        gate_table[spec] if spec in gate_table else orig(spec, **kw))

    CALLS.clear()
    interp = PipelineInterpreter(
        pipe, component_registry, EchoDriver(),
        task_id="selftest", work_dir=work,
    )
    res = await interp.run(resume=True)
    gate_mod.build_gate = orig

    check("只重跑缺失实例", CALLS == ["per_item:b", "paper"], str(CALLS))
    check("补跑后状态 completed", res.status == "completed", res.status)


def test_real_pipeline():
    print("\n[7] 真实流水线配置（competition.toml）")
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(root, "pipelines", "competition.toml")
    check("competition.toml 存在", os.path.exists(path), path)
    if not os.path.exists(path):
        return

    try:
        pipe = load_pipeline(path)
    except Exception as e:
        check("competition.toml 可解析", False,
              f"{type(e).__name__}: {str(e)[:200]}")
        return
    check("competition.toml 可解析", True)

    order = [s.id for s in pipe.topo_sort()]
    check("拓扑顺序", order == ["pdf", "parse", "model", "plan_sections", "analyze",
                               "write", "assemble", "paper_post", "review",
                               "polish", "export"], str(order))

    from app.core.components.builtin import register_builtins
    register_builtins()
    missing = [s.component for s in pipe.stages
               if s.component not in component_registry.names()]
    check("所有 stage 引用的组件都已注册", not missing, f"缺失: {missing}")

    check("fan_out 阶段有 input_from",
          all(s.input_from for s in pipe.stages if s.fan_out),
          str([s.id for s in pipe.stages if s.fan_out and not s.input_from]))

    # gate 提示词可加载
    from app.core.pipeline.loader import load_gate_prompts
    for key in ("parser_review", "modeler_review", "coder_review"):
        try:
            txt = load_gate_prompts(path, key)
            check(f"gate 提示词 {key}", len(txt) > 10, f"{len(txt)} 字符")
        except Exception as e:
            check(f"gate 提示词 {key}", False, str(e))

    print("\n     已注册组件:")
    for c in component_registry.describe():
        print(f"       {c['name']:<20} {str(c['in']):<16} → {c['out']:<16} "
              f"model_role={c['model_role'] or '-'}")


gate_table: dict[str, object] = {}


async def main():
    print("=" * 70)
    print("  组件化流水线框架 · 自检")
    print("=" * 70)

    tmpdir = tempfile.mkdtemp(prefix="selftest_")
    try:
        register_test_components()
        test_gate_spec_parsing()
        test_topo_sort(tmpdir)
        test_missing_component(tmpdir)
        await test_full_run(tmpdir)
        await test_partial_resume(tmpdir)
        test_real_pipeline()
    except Exception:
        print("\n自检过程抛出异常:")
        traceback.print_exc()
        FAIL.append("unhandled_exception")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print("\n" + "=" * 70)
    print(f"  结果: {len(PASS)} 通过, {len(FAIL)} 失败")
    if FAIL:
        for f in FAIL:
            print(f"    失败: {f}")
    print("=" * 70)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
