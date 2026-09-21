"""识图（Vision-Language）辅助 —— 让无视觉能力的写作环节"看见"图表。

从 `workflow.py` 抽出，原因见《架构范式转换设计文档》：

- `_verify_figures_with_vl` 原本是模块级私有函数，被编排器调用；
  抽为独立模块后，`VLFigureGate` 可复用，且可单独测试。
- `_describe_figures` 在原实现中**从未被调用**（审计发现 3），
  但其需求是真实的（Writer 需要知道图里画了什么）。此处保留实现并注明，
  避免在清理死代码时把需求一并删掉（见设计文档 6.4）。
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re

from app.config.setting import settings
from app.utils.log_util import logger

_MIME = {
    "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
    "gif": "image/gif", "svg": "image/svg+xml", "webp": "image/webp",
}


def _client():
    """构造识图客户端。未配置时返回 None。"""
    api_key = settings.VISION_API_KEY
    model = settings.VISION_MODEL
    base_url = settings.VISION_BASE_URL
    if not api_key or not model:
        logger.debug("识图模型未配置，跳过")
        return None, None
    try:
        from openai import OpenAI
        return OpenAI(api_key=api_key, base_url=base_url), model
    except ImportError:
        logger.warning("openai 库未安装，跳过识图")
        return None, None


def _b64(path: str) -> str | None:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logger.warning(f"图片读取失败 {path}: {e}")
        return None


def extract_figure_refs(paper_content: str) -> list[tuple[str, str, str]]:
    """提取论文中的 (文件名, 图注, 引用路径)。"""
    refs: list[tuple[str, str, str]] = []
    for m in re.finditer(r"!\[(.*?)\]\(([^)]+)\)", paper_content):
        desc, filepath = m.group(1), m.group(2)
        after = paper_content[m.end():m.end() + 200]
        cm = re.search(r"\*\*图\d+[：:]\s*(.+?)\*\*", after)
        refs.append((os.path.basename(filepath), cm.group(1) if cm else desc, filepath))
    return refs


async def verify_figures_with_vl(paper_content: str, work_dir: str) -> dict[str, str]:
    """用识图模型描述论文中每张图的实际内容，供对照评分。

    只做"图里实际是什么"的客观描述，**不做匹配判断**——匹配与打分由
    `VLFigureGate` 负责，避免把判定逻辑埋在 IO 函数里。

    Returns:
        {文件名: 一句话描述}；未配置识图模型时返回空字典。
    """
    client, model = _client()
    if client is None:
        return {}

    refs = extract_figure_refs(paper_content)
    if not refs:
        return {}

    results: dict[str, str] = {}
    loop = asyncio.get_running_loop()

    for fname, _caption, filepath in refs:
        abs_path = os.path.join(work_dir, filepath)
        if not os.path.exists(abs_path):
            results[fname] = "[图片文件不存在]"
            continue
        b64 = _b64(abs_path)
        if b64 is None:
            continue
        mime = _MIME.get(fname.rsplit(".", 1)[-1].lower(), "image/png")
        try:
            resp = await loop.run_in_executor(
                None,
                lambda b=b64, m=mime: client.chat.completions.create(
                    model=model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text",
                             "text": "用一句话描述这张图的内容(图表类型+核心数据)，"
                                     "不超过40字。不要评价，只描述。"},
                            {"type": "image_url",
                             "image_url": {"url": f"data:{m};base64,{b}"}},
                        ],
                    }],
                    max_tokens=80,
                ),
            )
            results[fname] = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            logger.warning(f"VL 验证失败 {fname}: {e}")
            results[fname] = f"[VL调用失败: {e}]"

    _cache(work_dir, "vl_verification.json", results)
    logger.info(f"VL 图验证完成: {len(results)} 张")
    return results


async def describe_figures(
    work_dir: str, context: str = "", images: list[str] | None = None,
) -> dict[str, str]:
    """为每张图生成内容描述，注入写作提示词。

    这是原 `workflow._describe_figures` 的实现，此前从未被调用（审计发现 3）。
    保留是为了不丢失需求：**让写作环节知道图里画了什么**，
    而不是事后用 VL 检查去抓错（见设计文档 6.4）。

    Args:
        work_dir: 工作目录。
        context: 题目背景，帮助模型理解图表语境。
        images: 待描述的图片相对路径；None 表示扫描 work_dir/images 相关目录。

    Returns:
        {相对路径: 描述}；命中 `figure_descriptions.json` 缓存的直接复用。
    """
    from app.core.postprocess import _scan_work_dir_images

    client, model = _client()
    if client is None:
        return {}

    image_files = images if images is not None else _scan_work_dir_images(work_dir)
    if not image_files:
        return {}

    cache_path = os.path.join(work_dir, "figure_descriptions.json")
    cached: dict[str, str] = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            cached = {k: v for k, v in cached.items() if k in image_files}
        except Exception:
            cached = {}

    need = [i for i in image_files if i not in cached]
    if not need:
        logger.info(f"图描述: {len(cached)} 张全部命中缓存")
        return cached

    loop = asyncio.get_running_loop()
    batch_size = 5
    for i in range(0, len(need), batch_size):
        batch = need[i:i + batch_size]
        parts: list[dict] = [{
            "type": "text",
            "text": (
                "请按顺序描述以下每张图片的内容。用'---'分隔每张图的描述。\n"
                "这些图片来自数学建模论文的数据分析，包括数据分布图、统计检验结果图、"
                "聚类可视化图、分类评估图、相关性分析图等。\n"
                "对每张图请描述：图表类型、坐标轴含义和单位、主要数据趋势和关键数值、"
                "图中明显的分组/异常点/聚类结构。\n"
                f"题目背景: {context[:800] if context else '无'}\n"
                "每张图描述控制在150字以内。"
            ),
        }]
        for img in batch:
            b64 = _b64(os.path.join(work_dir, img))
            if b64:
                mime = _MIME.get(img.rsplit(".", 1)[-1].lower(), "image/png")
                parts.append({"type": "image_url",
                              "image_url": {"url": f"data:{mime};base64,{b64}"}})
        try:
            resp = await loop.run_in_executor(
                None,
                lambda p=parts: client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": p}],
                    max_tokens=4096,
                ),
            )
            text = resp.choices[0].message.content or ""
            for j, desc in enumerate(p.strip() for p in text.split("---") if p.strip()):
                if j < len(batch):
                    cached[batch[j]] = desc
        except Exception as e:
            logger.warning(f"图描述调用失败（批次 {i // batch_size + 1}）: {e}")

    _cache(work_dir, "figure_descriptions.json", cached)
    logger.info(f"图描述完成: {len(cached)} 张")
    return cached


def _cache(work_dir: str, filename: str, data: dict) -> None:
    try:
        with open(os.path.join(work_dir, filename), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
