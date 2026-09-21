"""内置组件 —— 把原本散在 workflow.py / flows.py 里的环节收敛为可组装组件。

原则：**提示词构造与解析复用既有实现**（`Flows`、`prompts/`），
本次改造只改"谁来调、按什么顺序调"，不改"调什么"——保证等价性可验证。

组件清单见 `register_builtins()`。
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from app.core.artifact import Artifact, Issue
from app.core.component import (
    Component,
    StageContext,
    component_registry,
    make_agent_component,
    make_fn_component,
    make_tool_loop_component,
)
from app.utils.log_util import logger


# ─────────────────────────────────────────────────────────────
# 辅助：从各种形态的文本中抽出 JSON
# ─────────────────────────────────────────────────────────────


def parse_json_loose(text: str) -> dict[str, Any]:
    """尽力从模型输出里解析 JSON（容忍 ```json 围栏与前后噪声）。"""
    if not text:
        return {}
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.DOTALL)
    if m:
        t = m.group(1).strip()
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else {"value": obj}
    except Exception:
        pass
    start, end = t.find("{"), t.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(t[start:end + 1])
            return obj if isinstance(obj, dict) else {"value": obj}
        except Exception:
            pass
    logger.warning(f"parse_json_loose: 无法解析为 JSON，长度 {len(text)}")
    return {"raw": text}


def normalize_question_set(obj: dict[str, Any]) -> dict[str, Any]:
    """统一 parser 的两种输出形态。

    `skills/parser.md` 要求输出 `sub_problems`（列表），而下游组件按
    `{ques1: 描述}` 取用（`Flows` 两者都支持，但新组件需要统一形态）。
    这里在入口处归一化一次，避免每个下游各写一套兼容分支。
    """
    if not isinstance(obj, dict):
        return {}
    if obj.get("questions"):
        return obj
    subs = obj.get("sub_problems")
    if isinstance(subs, list):
        q: dict[str, str] = {}
        for i, sp in enumerate(subs, start=1):
            if isinstance(sp, dict):
                qid = sp.get("id") or i
                q[f"ques{qid}"] = sp.get("question") or sp.get("desc") or ""
            else:
                q[f"ques{i}"] = str(sp)
        obj["questions"] = q
        obj.setdefault("ques_count", len(q))
    return obj


def sections_of(ctx: StageContext) -> dict[str, Any]:
    """取章节定义（plan_sections 的产物）。"""
    art = ctx.artifacts.get("plan_sections")
    if art is None:
        raise RuntimeError("plan_sections 尚未产出，无法解析章节列表")
    return art.payload.get("sections") or {}


def _fan_key(ctx: StageContext, name: str = "fan_out_key") -> str:
    key = ctx.resources.get(name)
    if key is None:
        raise RuntimeError(f"fan_out key {name!r} 未注入（stage_id={ctx.stage_id}）")
    return str(key)


# ─────────────────────────────────────────────────────────────
# 1. PDF 解析（纯函数）
# ─────────────────────────────────────────────────────────────


async def _parse_pdf(art: Artifact, ctx: StageContext) -> Artifact:
    """Phase 0.5：PDF 文本提取 + 图片识别。非 PDF 输入原样透传。"""
    question = art.payload.get("question", "")
    if not (isinstance(question, str) and question.endswith(".pdf") and os.path.exists(question)):
        return Artifact(kind="raw_question", stage_id=ctx.stage_id,
                        payload={"question": question})

    from app.utils.pdf_parser import parse_pdf_question
    import shutil

    dst = os.path.join(ctx.work_dir, os.path.basename(question))
    shutil.copy2(question, dst)
    parsed = await parse_pdf_question(dst)
    if parsed.strip():
        logger.info("PDF 解析完成")
        return Artifact(kind="raw_question", stage_id=ctx.stage_id,
                        payload={"question": parsed}, refs=[os.path.basename(dst)])
    logger.warning("PDF 解析结果为空，回退为原始路径")
    return Artifact(kind="raw_question", stage_id=ctx.stage_id,
                    payload={"question": question})


# ─────────────────────────────────────────────────────────────
# 2. 题目解析
# ─────────────────────────────────────────────────────────────


def _parser_prompt(art: Artifact, ctx: StageContext) -> str:
    question = art.payload.get("question", "")
    files = _list_work_files(ctx.work_dir)
    return (
        f"{question}\n\n"
        f"## 工作目录中的文件\n{files}\n\n"
        "请输出 JSON（字段：title, background, questions, ques_count, "
        "data_files, question_types）。questions 为 {ques1: 描述, ...}。"
    )


def _parse_question(art: Artifact, ctx: StageContext) -> dict[str, Any]:
    data = parse_json_loose(text=art.payload.get("__raw__", ""))
    return data


# 框架自身产生的文件，不应出现在给模型的"数据文件"清单里。
# 否则它们会污染提示词（且随运行变化，影响可复现性）。
_FRAMEWORK_ARTIFACTS = {
    "checkpoint.json", "res.md", "review_result.json", "figure_descriptions.json",
    "vl_verification.json", "paper.docx", "paper.pdf",
}
_FRAMEWORK_DIRS = {"_driver", "figures", "notebook", "__pycache__", "project", "logs"}


def _list_work_files(work_dir: str) -> str:
    """列出工作目录中的数据文件（排除框架产物）。"""
    try:
        out = []
        for n in sorted(os.listdir(work_dir)):
            if n.startswith(("_", ".")) or n in _FRAMEWORK_ARTIFACTS:
                continue
            if n in _FRAMEWORK_DIRS or n.endswith((".ipynb", ".tmp")):
                continue
            if os.path.isdir(os.path.join(work_dir, n)):
                continue
            out.append(f"- {n}")
        return "\n".join(out) or "(空)"
    except Exception:
        return "(无法读取)"


def _modeler_prompt(art: Artifact, ctx: StageContext) -> str:
    q = art.payload
    questions = q.get("questions") or {}
    qlist = "\n".join(f"- {k}: {v}" for k, v in questions.items())
    return (
        f"## 题目背景\n{q.get('background', '')}\n\n"
        f"## 子问题\n{qlist}\n\n"
        f"## 工作目录文件\n{_list_work_files(ctx.work_dir)}\n\n"
        "请为每个子问题给出建模方案，并给出 EDA 方案与敏感性分析方案。\n"
        "输出 JSON：{questions_solution: {ques1: 方案, ...}, eda_plan: \"...\"}\n"
        "注意：全文方法名必须统一，不要同一问题混用多种同类方法。"
    )


def _design_model(art: Artifact, ctx: StageContext) -> dict[str, Any]:
    obj = parse_json_loose(art.payload.get("__raw__", ""))
    sol = obj.get("questions_solution") or {}
    eda = obj.get("eda_plan") or obj.get("eda") or ""
    if isinstance(eda, dict):
        eda = json.dumps(eda, ensure_ascii=False)
    return {"questions_solution": sol, "eda_plan": str(eda)}


# ─────────────────────────────────────────────────────────────
# 3. 章节规划（纯函数）—— 取代 flows.get_seq() 与各处硬编码
# ─────────────────────────────────────────────────────────────

# 结构性章节。新增章节在此加一段即可（见设计文档 5.3）。
STRUCTURAL_SECTIONS: list[dict[str, Any]] = [
    {"key": "firstPage", "order": 1, "title": "首页（标题/摘要/关键词）",
     "kind": "structural", "heading_level": "h1", "heading_scope": "self"},
    {"key": "RepeatQues", "order": 2, "title": "问题重述",
     "kind": "structural", "heading_level": "h1", "heading_scope": "self"},
    {"key": "analysisQues", "order": 3, "title": "问题分析",
     "kind": "structural", "heading_level": "h1", "heading_scope": "self"},
    {"key": "modelAssumption", "order": 4, "title": "模型假设",
     "kind": "structural", "heading_level": "h1", "heading_scope": "self"},
    {"key": "symbol", "order": 5, "title": "符号说明与数据预处理",
     "kind": "structural", "heading_level": "h1", "heading_scope": "self"},
]


async def _plan_sections(art: Artifact, ctx: StageContext) -> Artifact:
    """根据子问题数量生成完整章节表。取代 flows.get_seq() 的硬编码顺序。"""
    parse = ctx.require("parse")
    questions = parse.payload.get("questions") or {}
    qkeys = sorted(
        [k for k in questions if k.startswith("ques") and k[4:].isdigit()],
        key=lambda k: int(k[4:]),
    )

    sections: dict[str, Any] = {}
    for s in STRUCTURAL_SECTIONS:
        sections[s["key"]] = dict(s)

    sections["eda"] = {
        "key": "eda", "order": 6, "title": "探索性数据分析",
        "kind": "analysis", "heading_level": "h2", "heading_scope": "child",
        "depends_on": "eda",
    }
    order = 7
    for k in qkeys:
        sections[k] = {
            "key": k, "order": order, "title": f"{k} 建模与求解",
            "kind": "analysis", "heading_level": "h2", "heading_scope": "child",
            "depends_on": k,
        }
        order += 1
    sections["sensitivity_analysis"] = {
        "key": "sensitivity_analysis", "order": order, "title": "敏感性分析",
        "kind": "analysis", "heading_level": "h1", "heading_scope": "self",
        "depends_on": "sensitivity_analysis",
    }
    sections["judge"] = {
        "key": "judge", "order": order + 1, "title": "模型评价与改进",
        "kind": "structural", "heading_level": "h1", "heading_scope": "self",
    }
    # 参考文献：论文格式规范要求的必需章节。
    # 新增一个章节只需在此处加一段——这正是把章节定义收进注册表的目的（G3）。
    sections["references"] = {
        "key": "references", "order": order + 2, "title": "参考文献",
        "kind": "structural", "heading_level": "h1", "heading_scope": "self",
    }

    # 分析子任务清单：eda + 各子问题 + 敏感性分析
    # 等价于原 Flows.get_solution_flows() 产出的键集合，供 analyze 阶段 fan_out 展开。
    analysis_tasks: dict[str, str] = {"eda": "探索性数据分析"}
    for k in qkeys:
        analysis_tasks[k] = str(questions.get(k, ""))
    analysis_tasks["sensitivity_analysis"] = "关键参数的敏感性分析"

    return Artifact(kind="section_plan", stage_id=ctx.stage_id,
                    payload={"sections": sections, "analysis_tasks": analysis_tasks},
                    metrics={"section_count": float(len(sections)),
                             "task_count": float(len(analysis_tasks))})


# ─────────────────────────────────────────────────────────────
# 4. 分析执行（fan_out）
# ─────────────────────────────────────────────────────────────


def _analysis_prompt(art: Artifact, ctx: StageContext) -> str:
    """构造一个子任务的执行提示。

    复用 `Flows.get_solution_flows()` 的既有措辞，保证与改造前等价。
    """
    from app.core.flows import Flows

    key = _fan_key(ctx)
    model = ctx.require("model")
    parse = ctx.require("parse")

    flows = Flows(parse.payload.get("questions") or {})
    model_art = Artifact(kind="model_plan", stage_id="model", payload=model.payload)
    try:
        from app.schemas.A2A import ModelerToCoder
        typed = ModelerToCoder(**model.payload)
        all_flows = flows.get_solution_flows(typed)
    except Exception as e:
        logger.warning(f"无法用 Flows 构造提示（{e}），回退为通用提示")
        all_flows = {}

    base = (all_flows.get(key) or {}).get("coder_prompt", "")
    if not base:
        sol = (model.payload.get("questions_solution") or {}).get(key, "")
        base = f"## 任务: {key}\n\n请根据以下方案编写代码并求解：\n\n{sol}"

    contract = (ctx.resources.get("injected") or {}).get("model.method_contract", "")
    prefix = f"[⚠️ 建模方案指定的实现方法，必须使用]\n{contract}\n\n" if contract else ""
    return (
        f"{prefix}{base}\n\n"
        f"## 工作目录\n{ctx.work_dir}\n"
        f"## 数据文件\n{_list_work_files(ctx.work_dir)}\n\n"
        "要求：图表保存到 figures/ 目录；用 print() 输出关键指标；"
        "完成后打印「" + key + "完成」。"
    )


# ─────────────────────────────────────────────────────────────
# 5. 论文写作（fan_out）
# ─────────────────────────────────────────────────────────────


def _write_prompt(art: Artifact, ctx: StageContext) -> str:
    from app.core.flows import Flows

    key = _fan_key(ctx)
    parse = ctx.require("parse")
    model = ctx.require("model")
    sections = sections_of(ctx)
    spec = sections.get(key) or {}

    flows = Flows(parse.payload.get("questions") or {})
    modeler_sol = (model.payload.get("questions_solution") or {}).get(key, "")

    # 分析结果：章节 key 与 analyze 实例键同名
    analysis = ctx.artifacts.get(f"analyze.{key}")
    code_response = ""
    code_output = ""
    if analysis is not None:
        code_response = analysis.payload.get("code_response") or ""
        code_output = analysis.extras.get("code_output") or ""

    template = _section_template(key)
    try:
        from app.schemas.A2A import ModelerToCoder
        typed = ModelerToCoder(**model.payload)
        prompt = flows.get_writer_prompt(
            key, code_response, _CodeOutputShim(code_output), template,
            _model_build_solve(ctx), modeler_sol,
        )
    except Exception as e:
        logger.warning(f"Flows.get_writer_prompt 失败（{e}），使用通用提示")
        prompt = (
            f"## 写作任务: 撰写 [{key}]（{spec.get('title', '')}）\n\n"
            f"### 建模方案\n{modeler_sol[:2000]}\n\n"
            f"### 代码输出\n{code_output[:2000]}\n\n"
            "请撰写该章节，结论必须有数据支撑，使用「本文」而非「我们」。"
        )

    anchor = _context_anchor(ctx)
    return f"{anchor}{prompt}"


class _CodeOutputShim:
    """适配 `Flows.get_writer_prompt` 对 code_interpreter 的依赖。

    改造前 Writer 通过 interpreter 取代码输出（跨阶段隐式通道，审计发现 6）。
    现在代码输出已物化进 Artifact.extras，用这个薄适配层满足既有签名，
    从而**不改 flows.py 的提示词逻辑**。
    """

    def __init__(self, output: str):
        self._output = output

    def get_code_output(self, key: str) -> str:
        return self._output or "(无可用的代码输出缓存)"


def _context_anchor(ctx: StageContext) -> str:
    """全文一致的上下文锚点：题目背景 + 子问题 + 方法契约。"""
    parse = ctx.require("parse")
    contract = (ctx.resources.get("injected") or {}).get("model.method_contract", "")
    return (
        f"[题目背景] {parse.payload.get('background', '')}\n"
        f"[子问题列表] "
        f"{json.dumps(parse.payload.get('questions') or {}, ensure_ascii=False)[:2000]}\n\n"
        + (f"[⚠️ 方法契约] 以下方法名全文统一使用，不可替换:\n{contract}\n\n" if contract else "")
    )


def _section_template(key: str) -> str:
    """读取章节模板（config/md_template.toml）。"""
    import tomllib
    from pathlib import Path

    path = Path(__file__).parent.parent.parent / "config" / "md_template.toml"
    if not path.exists():
        return ""
    try:
        with open(path, "rb") as f:
            return (tomllib.load(f) or {}).get(key, "")
    except Exception:
        return ""


def _model_build_solve(ctx: StageContext) -> str:
    out = {}
    for item in ctx.fan_out_items("analyze"):
        key = item.stage_id.split(".", 1)[-1]
        out[key] = (item.payload.get("code_response") or "")[:1500]
    return json.dumps(out, ensure_ascii=False)


def _write_section(art: Artifact, ctx: StageContext) -> dict[str, Any]:
    return {"content": art.payload.get("__raw__", "")}


# ─────────────────────────────────────────────────────────────
# 6. 评审
# ─────────────────────────────────────────────────────────────


async def _review_paper(art: Artifact, ctx: StageContext) -> Artifact:
    """评审论文。评审模型直接给出带章节归属的 dimensions。

    设计文档 3.2 的要求：让检查器**直接产出章节归属**，
    而不是由下游用硬编码映射表去猜（原 dim_section_map / error_section_mapping）。
    """
    from app.core.drivers import CompletionRequest

    content = art.payload.get("content") or ""
    parse = ctx.artifacts.get("parse")
    summary = json.dumps(parse.payload.get("questions") or {}, ensure_ascii=False) if parse else ""

    prompt = (
        "请评审以下数学建模论文，输出 JSON：\n"
        "{\n"
        '  "overall_score": 0-10,\n'
        '  "passed": true/false,\n'
        '  "summary": "总评",\n'
        '  "dimensions": {\n'
        '     "<章节key>": {"score": 0-10, "section": "<同key>", "issues": ["问题描述"]}\n'
        "  },\n"
        '  "suggestions": ["改进建议"]\n'
        "}\n\n"
        "**关键**：dimensions 的键必须是论文中实际存在的章节标识（如 ques1、eda、"
        "sensitivity_analysis、symbol），并在 section 字段回填同一标识——"
        "下游据此定位需要重写的章节。若某问题不属于任何章节，section 填 null。\n\n"
        f"## 题目子问题\n{summary[:1500]}\n\n"
        f"## 论文全文\n{content[:24000]}"
    )

    raw = await ctx.driver.complete(
        CompletionRequest(
            prompt=prompt, agent_name="reviewer", model_role="reviewer",
            stage_id=ctx.stage_id, expect="json",
            context={"char_count": len(content)},
        )
    )
    data = parse_json_loose(raw)
    if "overall_score" not in data:
        data["overall_score"] = 0.0
        data["summary"] = data.get("raw", "")[:500]
    return Artifact(
        kind="review_result", stage_id=ctx.stage_id, payload=data,
        metrics={"overall_score": float(data.get("overall_score") or 0.0)},
    )


# ─────────────────────────────────────────────────────────────
# 7. 章节修复（供 RepairLoop 调用）
# ─────────────────────────────────────────────────────────────


async def _repair_sections_async(art: Artifact, ctx: StageContext) -> dict[str, Any]:
    """`_repair_sections` 的异步版本（组件契约支持 async）。"""
    content = art.payload.get("content") or ""
    sections = sections_of(ctx)
    grouped = art.section_issues()

    targets = [s for s in grouped if s and s in sections]
    unresolved = [s for s in grouped if s and s not in sections]
    if not targets:
        logger.info(
            f"section_repair: 无带章节归属的问题（全局 {len(grouped.get('', []))} 条，"
            f"未知章节 {unresolved}），不重写"
        )
        return {"content": content, "repaired": []}

    from app.core.drivers import CompletionRequest

    repaired: list[str] = []
    for key in targets:
        issues = grouped[key]
        block = "\n".join(f"- {i.message}" for i in issues)
        hint = "\n".join(f"- {i.fix_hint}" for i in issues if i.fix_hint)
        old = _extract_section(content, key, sections[key].get("title", ""))
        prompt = (
            f"以下章节（{key}）存在质量问题，请重写：\n\n"
            f"## 问题\n{block}\n"
            + (f"\n## 修正建议\n{hint}\n" if hint else "")
            + f"\n## 当前内容（供参考）\n{old[:3000]}\n\n"
            "要求：每个结论有数据支撑；使用「本文」而非「我们」；"
            "表格用 HTML <table>；禁止空泛词汇。只输出重写后的章节内容。"
        )
        try:
            new = await ctx.driver.complete(
                CompletionRequest(
                    prompt=prompt, agent_name="section_repair", model_role="writer",
                    stage_id=f"{ctx.stage_id}.{key}", expect="text",
                    context={"section": key, "issues": [i.message for i in issues]},
                )
            )
            if new and new.strip():
                content = _replace_section(content, key, new.strip(), sections[key])
                repaired.append(key)
                logger.info(f"章节 {key} 已重写")
        except Exception as e:
            logger.warning(f"章节 {key} 重写失败: {e}")
    return {"content": content, "repaired": repaired}


def _extract_section(content: str, key: str, title: str = "") -> str:
    """按章节 key/标题抽取章节正文。"""
    if not content:
        return ""
    pats = [rf"^#+\s*[^\n]*{re.escape(key)}[^\n]*$"]
    if title:
        pats.append(rf"^#+\s*[^\n]*{re.escape(title)}[^\n]*$")
    for pat in pats:
        m = re.search(pat, content, re.MULTILINE)
        if not m:
            continue
        rest = content[m.end():]
        nxt = re.search(r"^#{1,3}\s+", rest, re.MULTILINE)
        return rest[:nxt.start()] if nxt else rest
    return ""


def _replace_section(content: str, key: str, new_body: str, spec: dict) -> str:
    """替换章节正文，保留原标题行。"""
    title = spec.get("title", "")
    for pat in (rf"^#+\s*[^\n]*{re.escape(key)}[^\n]*$",
                rf"^#+\s*[^\n]*{re.escape(title)}[^\n]*$") if title else (
                rf"^#+\s*[^\n]*{re.escape(key)}[^\n]*$",):
        m = re.search(pat, content, re.MULTILINE)
        if not m:
            continue
        rest = content[m.end():]
        nxt = re.search(r"^#{1,3}\s+", rest, re.MULTILINE)
        end = m.end() + (nxt.start() if nxt else len(rest))
        return content[:m.end()] + "\n\n" + new_body + "\n\n" + content[end:]
    # 找不到就追加
    return content + f"\n\n## {title or key}\n\n{new_body}\n"


# ─────────────────────────────────────────────────────────────
# 8. 后处理与输出（纯函数）
# ─────────────────────────────────────────────────────────────


async def _paper_postprocess(art: Artifact, ctx: StageContext) -> Artifact:
    """论文后处理链。等价于原 `_run_writer` 尾部的 4 步后处理。"""
    from app.core.postprocess import (
        _fix_figure_captions,
        _remove_duplicate_headings,
        _remove_duplicate_images,
        _renumber_tables_figures,
    )

    content = art.payload.get("content") or ""
    before = len(content)
    content = _remove_duplicate_headings(content)
    content = _remove_duplicate_images(content)
    content = _fix_figure_captions(content, ctx.work_dir)
    content = _renumber_tables_figures(content)
    logger.info(f"后处理: {before} → {len(content)} 字符")

    _write_res_md(content, ctx.work_dir)
    return Artifact(kind="paper", stage_id=ctx.stage_id, payload={"content": content})


async def _assemble_paper(art: Artifact, ctx: StageContext) -> Artifact:
    """把各章节按 order 拼成完整论文。"""
    sections = sections_of(ctx)
    ordered = sorted(sections.values(), key=lambda s: s.get("order", 999))
    parts: list[str] = []
    for spec in ordered:
        key = spec["key"]
        sart = ctx.artifacts.get(f"write.{key}")
        if sart is None:
            logger.warning(f"章节 {key} 无产物，跳过")
            continue
        body = (sart.payload.get("content") or "").strip()
        if not body:
            continue
        parts.append(body)
    content = "\n\n".join(parts)
    _write_res_md(content, ctx.work_dir)
    return Artifact(kind="paper", stage_id=ctx.stage_id,
                    payload={"content": content},
                    metrics={"paper_chars": float(len(content))})


def _write_res_md(content: str, work_dir: str) -> None:
    with open(os.path.join(work_dir, "res.md"), "w", encoding="utf-8") as f:
        f.write(content)


async def _export(art: Artifact, ctx: StageContext) -> Artifact:
    """导出 DOCX / PDF。

    **注意**：`md_2_docx` 内部的 `_extract_html_tables` 会把 res.md 中的 HTML 表格
    替换成 `[TABLE_n]` 占位符并**写回原文件**（供 pandoc 转换用，表格数据另行恢复）。
    这对 DOCX 是正确的，但会**永久破坏 res.md 这个交付物**——转换完成后表格全没了。

    因此在转换之后必须用 artifact 里的完整内容重写一次 res.md。
    """
    from app.utils.common_utils import md_2_docx, md_2_pdf

    fmt = (ctx.resources.get("format_output") or "markdown")
    content = art.payload.get("content") or ""
    _write_res_md(content, ctx.work_dir)

    out = None
    try:
        if fmt == "latex":
            out = md_2_pdf(ctx.task_id)
        else:
            out = md_2_docx(ctx.task_id)
    except Exception as e:
        logger.warning(f"导出失败: {e}")

    # 还原 res.md（见上）。md_2_docx / md_2_pdf 都可能清洗过它。
    _write_res_md(content, ctx.work_dir)
    logger.info(f"导出完成: {out or '（函数无返回值，产物见工作目录）'}")
    return Artifact(kind="exported", stage_id=ctx.stage_id,
                    payload={"content": content, "output": str(out) if out else ""})


# ─────────────────────────────────────────────────────────────
# 注册
# ─────────────────────────────────────────────────────────────


def register_builtins() -> None:
    """注册全部内置组件。幂等。"""
    if "parse_question" in component_registry.names():
        return

    # 纯函数型
    component_registry.register(make_fn_component(
        "parse_pdf", _parse_pdf, in_kind="raw_question", out_kind="raw_question",
        description="PDF 题目解析（非 PDF 透传）"))
    component_registry.register(make_fn_component(
        "plan_sections", _plan_sections, in_kind="*", out_kind="section_plan",
        description="按子问题生成完整章节表"))
    component_registry.register(make_fn_component(
        "assemble_paper", _assemble_paper, in_kind="*", out_kind="paper",
        description="按 order 拼接各章节为完整论文"))
    component_registry.register(make_fn_component(
        "paper_postprocess", _paper_postprocess, in_kind="paper", out_kind="paper",
        description="后处理链：去重标题/图、修图注、统一重编号"))
    component_registry.register(make_fn_component(
        "section_repair", _repair_sections_async, in_kind="paper", out_kind="paper",
        description="按 Issue.section 重写受影响章节"))
    component_registry.register(make_fn_component(
        "export", _export, in_kind="paper", out_kind="exported",
        description="导出 DOCX / PDF"))

    # LLM 组件
    component_registry.register(make_agent_component(
        name="parse_question", skill="parser",
        prompt_fn=lambda art, ctx: _parser_prompt(art, ctx),
        parse_fn=lambda raw, art, ctx: normalize_question_set(parse_json_loose(raw)),
        out_kind="question_set", in_kind="raw_question",
        model_role="parser", expect="json",
        description="题目解析与子问题分解"))

    component_registry.register(make_agent_component(
        name="design_model", skill="modeler",
        prompt_fn=_modeler_prompt,
        parse_fn=lambda raw, art, ctx: _design_model(
            Artifact(kind="x", payload={"__raw__": raw}), ctx),
        out_kind="model_plan", in_kind="question_set",
        model_role="modeler", expect="json",
        description="EDA 方案与建模方案设计"))

    component_registry.register(make_agent_component(
        name="write_section", skill="writer",
        prompt_fn=_write_prompt,
        parse_fn=lambda raw, art, ctx: {"content": raw},
        out_kind="section_draft", in_kind="*",
        model_role="writer", expect="text",
        description="论文章节写作"))

    component_registry.register(Component(
        name="review_paper", run=_review_paper,
        in_kind="paper", out_kind="review_result",
        model_role="reviewer", description="论文质量评审"))

    # 工具循环型（api 模式跑 tool loop；file 模式执行外部提供的脚本）
    component_registry.register(make_tool_loop_component(
        name="run_analysis", skill="coder",
        prompt_fn=_analysis_prompt,
        in_kind="model_plan", out_kind="analysis_result",
        model_role="coder", tools=("execute_code",),
        description="子问题代码实现与执行"))

    logger.info(f"已注册内置组件: {component_registry.names()}")
