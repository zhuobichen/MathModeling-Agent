"""论文后处理纯函数集合。

从 workflow.py 抽离出的无状态后处理函数，负责：
- 图片去重 / 图注修正 / 表图重编号
- 重复标题移除
- 工作目录图片扫描与章节互斥分配
- 方法契约提取 / 图片元数据提取
"""

import os

from app.utils.log_util import logger
def _extract_method_contract(modeler_result) -> str:
    """从 ModelerAgent 方案中提取方法契约——每个问题的主方法名。

    扫描建模方案中所有出现的已知方法名，构建全文统一的术语表。
    WriterAgent 必须使用这些名字，即使代码输出提到其他方法也要统一改写。
    """
    import re
    solutions = getattr(modeler_result, "questions_solution", {})
    all_text = " ".join(str(v) for v in solutions.values())

    # 已知方法关键字 → 标准化名称
    METHOD_NAMES = {
        "GPR": "高斯过程回归(GPR)",
        "高斯过程回归": "高斯过程回归(GPR)",
        "决策树": "决策树(CART)",
        "CART": "决策树(CART)",
        "随机森林": "随机森林",
        "XGBoost": "XGBoost",
        "GMM": "高斯混合模型(GMM)",
        "高斯混合": "高斯混合模型(GMM)",
        "层次聚类": "层次聚类",
        "K-means": "K-means",
        "KMeans": "K-means",
        "Spearman": "Spearman秩相关",
        "Mann-Whitney": "Mann-Whitney U检验",
        "卡方": "卡方检验",
        "Fisher": "Fisher精确检验",
        "对应分析": "对应分析(MCA)",
        "PCA": "主成分分析(PCA)",
        "PCA/K-means": "PCA+K-means",
        "TOPSIS": "TOPSIS",
        "Apriori": "Apriori关联规则",
    }

    found: dict[str, str] = {}
    for keyword, std_name in METHOD_NAMES.items():
        if re.search(keyword, all_text, re.IGNORECASE):
            found[std_name] = std_name

    if found:
        lines = [f"- {name}" for name in found]
        return "全文统一方法名:\n" + "\n".join(lines)
    return "(未从建模方案中提取到方法契约)"


def _extract_figure_metadata(work_dir: str) -> dict[str, str]:
    """从 Jupyter notebook 中提取每张图的生成意图。

    CoderAgent 生成代码时会用 print() 描述图表、用 fig.suptitle() 写标题。
    这些信息比视觉模型识别更准确——代码知道自己在画什么。
    """
    import re, json

    nb_path = os.path.join(work_dir, "notebook.ipynb")
    if not os.path.exists(nb_path):
        return {}

    try:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
    except Exception:
        return {}

    metadata: dict[str, str] = {}
    for cell in nb.get("cells", []):
        source = "".join(cell.get("source", []))
        outputs = cell.get("outputs", [])

        # 1) 从 fig.suptitle / plt.title 提取图表中文标题
        for m in re.finditer(
            r'(?:fig\.suptitle|plt\.title|ax\.set_title)\s*\([\"\'](.+?)[\"\']',
            source,
        ):
            title = m.group(1).strip()
            # 找最近的 savefig
            save_m = re.search(
                r'(?:plt|fig)\.savefig\([\"\'](.+?)[\"\']',
                source[m.start():],
            )
            if save_m:
                fname = save_m.group(1).split("/")[-1]
                if fname not in metadata or len(title) > len(metadata.get(fname, "")):
                    metadata[fname] = title

        # 2) 从 print 输出提取图表关键数值描述
        for out in outputs:
            text = out.get("text", "")
            if isinstance(text, list):
                text = "".join(text)
            # 找 print 输出的关键指标（R², p值, 准确率等）
            metrics = re.findall(
                r"([\u4e00-\u9fff\w]+[：:]\s*[\d.]+%?)", text
            )
            if metrics:
                # 找最近的 savefig
                for m in re.finditer(
                    r'(?:plt|fig)\.savefig\([\"\'](.+?)[\"\']', source
                ):
                    fname = m.group(1).split("/")[-1]
                    if fname not in metadata:
                        metadata[fname] = "; ".join(metrics[:5])

    if metadata:
        logger.info(f"图片元数据: 从 notebook 提取 {len(metadata)} 条")
    return metadata


def _remove_duplicate_images(content: str) -> str:
    """移除论文中重复出现的图片标记，每张图片只保留第一次出现。

    LLM 可能从共享 chat_history 中"记住"前面章节的图并在后续章节重复插入。
    此函数对合并后的全文做最终去重：同一文件名的 ![]() 标记只保留第一次。
    """
    import re

    seen: set[str] = set()
    removed = 0

    def _replace(m: re.Match) -> str:
        nonlocal removed
        filepath = m.group(2)
        basename = filepath.split("/")[-1]
        if basename not in seen:
            seen.add(basename)
            return m.group(0)
        removed += 1
        return ""  # 移除整个 ![]() 标记

    content = re.sub(r'!\[([^\]]*)\]\(([^\)]+)\)', _replace, content)
    if removed:
        logger.info(f"图片去重: 移除 {removed} 处重复图片标记")
    return content


def _fix_figure_captions(content: str, work_dir: str) -> str:
    """用 VL 识图结果或代码标题强制替换论文中编造的图注。

    优先使用 VL 识图结果（千问实际看到的图内容），
    其次使用代码标题（fig.suptitle）。
    按文件名匹配替换论文中的 **图N: 编造描述** 为真实描述。
    """
    import re, json

    # 代码标题（fig.suptitle）是标准答案，优先使用
    coder_data = _extract_figure_metadata(work_dir)

    # 代码标题缺失时，用 VL 验证缓存兜底
    vl_cache_path = os.path.join(work_dir, "vl_verification.json")
    vl_data: dict[str, str] = {}
    if os.path.exists(vl_cache_path):
        try:
            with open(vl_cache_path, "r", encoding="utf-8") as f:
                vl_data = json.load(f)
        except Exception:
            pass

    all_meta: dict[str, str] = {}
    for fname in set(list(coder_data.keys()) + list(vl_data.keys())):
        coder_desc = coder_data.get(fname, "")
        if coder_desc:
            all_meta[fname] = re.sub(r'^图\d+[：:]\s*', '', coder_desc)[:60]
        else:
            vl_desc = vl_data.get(fname, "")
            if vl_desc and len(vl_desc) > 5:
                all_meta[fname] = vl_desc[:40]

    if not all_meta:
        return content

    fixed = 0
    for fname, real_desc in all_meta.items():
        # 去掉 "图N: " 或 "图N：" 前缀，提取后面的真实描述
        short_title = real_desc
        if not short_title:
            continue
        basename = fname.split("/")[-1]

        # 在论文中找该文件的引用: ![xxx](xxx/fname)
        img_pat = re.compile(
            r'(!\[[^\]]*\]\(' + re.escape(fname) + r'\))'
        )
        # 找紧跟其后的 **图N: xxx** 行
        caption_pat = re.compile(
            r'(\*\*图\d+[：:])\s*.+?(\*\*)',
            re.MULTILINE,
        )

        # 找到图片引用位置，然后找最近的图注（从后往前替换避免偏移）
        replacements: list[tuple[int, int, str]] = []  # (start, end, new_text)
        for m in img_pat.finditer(content):
            after = content[m.end():m.end()+200]
            cm = caption_pat.search(after)
            if cm and short_title not in cm.group(0):
                new_caption = f"{cm.group(1)} {short_title}{cm.group(2)}"
                start = m.end() + cm.start()
                end = m.end() + cm.end()
                replacements.append((start, end, new_caption))

        # 从后往前替换
        for start, end, new_text in reversed(replacements):
            content = content[:start] + new_text + content[end:]
            fixed += 1
            logger.info(f"图注修正: {fname} → '{short_title[:60]}'")

    if fixed:
        logger.info(f"图注修正完成: {fixed} 处")
    return content


def _renumber_tables_figures(content: str) -> str:
    """全文统一重编号表和图的题注及正文引用。

    每个章节独立写作时各自从"表1""图1"开始编号。
    此函数按出现顺序重新分配全局编号，确保全文无重复。
    """
    import re

    # 匹配题注: **表1: xxx** 或 **图1: xxx**
    caption_pat = re.compile(r'\*\*(表|图)(\d+)(:\s*.+?\*\*)')

    # 收集所有题注位置，按出现顺序分配全局编号
    table_idx = 1
    fig_idx = 1
    replacements: list[tuple[int, int, str, str, str]] = []  # (start, end, typ, old_num, new_str)

    for m in caption_pat.finditer(content):
        typ = m.group(1)
        old_num = m.group(2)
        new_num = table_idx if typ == "表" else fig_idx
        if int(old_num) == new_num:
            # 编号已经正确，不需要改，但仍需消耗计数器
            if typ == "表":
                table_idx += 1
            else:
                fig_idx += 1
            continue
        new_str = f"**{typ}{new_num}{m.group(3)}"
        replacements.append((m.start(), m.end(), typ, old_num, new_str))
        if typ == "表":
            table_idx += 1
        else:
            fig_idx += 1

    if not replacements:
        return content

    # 从后往前替换，避免偏移
    for start, end, typ, old_num, new_str in reversed(replacements):
        content = content[:start] + new_str + content[end:]

    logger.info(f"表图重编号: {len(replacements)} 处修正 (表: {table_idx-1}, 图: {fig_idx-1})")
    return content


def _remove_duplicate_headings(content: str) -> str:
    """移除论文中重复的同编号标题。

    检测规则：提取标题的数字编号前缀（如 # 四、→4, ## 5.2→5.2），
    同一编号只保留第一次出现的标题，后续出现的视为并行写作导致的重复，
    整行移除（不含后续内容）。
    """
    import re

    # 中文数字 → 阿拉伯数字
    CN_NUM = "一二三四五六七八九十"
    def _extract_num(text: str) -> str:
        # 去除标题前的标点（如 四、→ 四）
        first_char = re.sub(r"[、，。；：\s]", "", text[0]) if text else ""
        for i, cn in enumerate(CN_NUM):
            if text.startswith(cn) or first_char == cn:
                return str(i + 1)
        m = re.match(r"(\d[\d\.]*)", text)
        return m.group(1) if m else text[:10]

    lines = content.split("\n")
    seen_nums: dict[str, int] = {}  # 编号 → 行号
    duplicates: set[int] = set()

    for i, line in enumerate(lines):
        m = re.match(r"^(#{1,3})\s+(.+)$", line.strip())
        if not m:
            continue
        level = m.group(1)
        text = m.group(2).strip()
        num = _extract_num(text)
        key = f"{level}:{num}"

        if key in seen_nums:
            duplicates.add(i)
        else:
            seen_nums[key] = i

    if not duplicates:
        return content

    result = [l for i, l in enumerate(lines) if i not in duplicates]
    logger.info(f"标题去重: 移除 {len(duplicates)} 处重复标题")
    return "\n".join(result)


def _scan_work_dir_images(work_dir: str) -> list[str]:
    """扫描工作目录下所有图片文件，返回相对路径列表。"""
    images: list[str] = []
    if not os.path.isdir(work_dir):
        return images
    for root, _dirs, files in os.walk(work_dir):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".svg")):
                rel_path = os.path.relpath(os.path.join(root, f), work_dir)
                images.append(rel_path.replace("\\", "/"))
    images.sort()
    logger.info(f"工作目录图片扫描: {len(images)} 张 ({work_dir})")
    return images


def _assign_images_to_sections(
    section_order: list[str], all_images: list[str],
) -> dict[str, list[str]]:
    """将工作目录图片按文件名前缀互斥分配给各章节。

    按 section_order 顺序处理，每张图只分配给第一个匹配的章节，
    后续章节不再获取该图，确保全文无跨章重复。
    """
    unassigned = list(all_images)
    result: dict[str, list[str]] = {k: [] for k in section_order}

    for key in section_order:
        remaining: list[str] = []
        for img in unassigned:
            basename = img.split("/")[-1].lower()
            matched = False

            if key == "eda":
                if any(kw in basename for kw in ("eda", "weathering", "missing", "overview")):
                    matched = True
            elif key.startswith("ques"):
                n = key[4:]
                if f"figure{n}_" in basename or f"figure{n}." in basename:
                    matched = True
            elif key == "sensitivity_analysis":
                # 不主动认领，余量最终统一分配
                pass

            if matched:
                result[key].append(img)
            else:
                remaining.append(img)
        unassigned = remaining

    # 余量全部给 sensitivity_analysis
    if "sensitivity_analysis" in result:
        result["sensitivity_analysis"] = unassigned

    # 兜底：空白章节给少量图
    for key in section_order:
        if not result[key]:
            result[key] = all_images[:2]

    total = sum(len(v) for v in result.values())
    logger.debug(f"图片互斥分配: {total} 张 → { {k: len(v) for k, v in result.items()} }")
    return result
