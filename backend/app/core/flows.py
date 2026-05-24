"""任务流编排模块，负责为每个阶段生成 Prompt。"""

from app.schemas.A2A import ModelerToCoder
from app.tools.base_interpreter import BaseCodeInterpreter


class Flows:
    """管理 5-Agent 流水线中每个阶段的 Prompt 生成。"""

    def __init__(self, questions: dict[str, str | int]) -> None:
        self.questions = questions

    def get_questions_quesx(self) -> dict[str, str]:
        """过滤出子问题的描述（ques1, ques2, ...）。

        优先从顶层 `ques1`/`ques2` key 读取，
        兼容从 `sub_problems` 列表中提取。

        Returns:
            以 quesX 为键、问题描述为值的字典。
        """
        # 方案A：顶层有 ques1, ques2, ... key（MathModelAgent 格式）
        result = {
            k: str(v)
            for k, v in self.questions.items()
            if k.startswith("ques") and k[4:].isdigit()
        }
        if result:
            return result

        # 方案B：从 sub_problems 列表中提取（我们的 Parser 格式）
        sub_problems = self.questions.get("sub_problems", [])
        if isinstance(sub_problems, list):
            for sp in sub_problems:
                qid = sp.get("id", 0)
                question = sp.get("question", "")
                result[f"ques{qid}"] = str(question)
        return result

    def get_solution_flows(
        self, modeler_response: ModelerToCoder
    ) -> dict[str, dict]:
        """生成代码实现阶段的子任务列表。

        Args:
            modeler_response: ModelerAgent 的输出，含 EDA 方案和各子问题建模方案。

        Returns:
            以子任务名称为键（eda, ques1, ...）、包含 coder_prompt 的字典。
        """
        solutions = modeler_response.questions_solution
        flows: dict[str, dict] = {}

        # EDA 阶段
        if modeler_response.eda_plan:
            flows["eda"] = {
                "coder_prompt": f"""## 任务: 探索性数据分析 (EDA)

请根据以下 EDA 方案执行完整的数据探索分析：

{modeler_response.eda_plan}

要求:
1. 读取所有数据文件，输出数据概览（形状、类型、缺失值、统计量）
2. 生成数据分布图（直方图、箱线图）和相关性热力图
3. 检测异常值并给出处理建议
4. 用 print() 输出所有关键统计发现
5. 所有图表保存到 figures/ 目录
6. 完成后打印"EDA完成" """
            }

        # 各子问题
        quesx = self.get_questions_quesx()
        for key, question in quesx.items():
            solution = solutions.get(key, "")
            if solution:
                flows[key] = {
                    "coder_prompt": f"""## 任务: {question}

请根据以下建模方案编写代码并求解：

{solution}

要求:
1. 严格按照建模方案的步骤实现
2. 输出模型评估指标（R², RMSE, MAE 等）
3. 生成可视化图表并保存到 figures/
4. 进行结果自验证（数值范围、指标阈值）
5. 完成后打印"{key}完成" """
                }

        # 敏感性分析
        sensitivity = solutions.get("sensitivity_analysis", "")
        if sensitivity:
            flows["sensitivity_analysis"] = {
                "coder_prompt": f"""## 任务: 敏感性分析

请根据以下方案进行敏感性分析：

{sensitivity}

要求:
1. 对关键参数进行 ±20% 扰动测试
2. 生成敏感性曲线或龙卷风图
3. 用 print() 输出敏感性分析结论
4. 图表保存到 figures/ """
            }

        return flows

    def get_writer_prompt(
        self,
        key: str,
        coder_response: str,
        code_interpreter: BaseCodeInterpreter,
        config_template: str = "",
        model_build_solve: str = "",
        modeler_solution: str = "",
        format_output: str = "markdown",
    ) -> str:
        """为 WriterAgent 生成论文章节写作 Prompt。

        Args:
            key: 章节标识（eda, ques1, ...）。
            coder_response: CoderAgent 的响应文本。
            code_interpreter: 代码解释器，用于获取 section 输出。
            config_template: 该章节的模板。
            model_build_solve: 所有子问题的建模求解摘要。
            modeler_solution: 建模手对该子问题的建模方案（含方法选择理由）。
            format_output: 输出格式（"markdown" 或 "latex"）。
        """
        code_output = (
            code_interpreter.get_code_output(key)
            if code_interpreter is not None
            else "(断点恢复，无可用的代码输出缓存)"
        )

        # 替换模板中的占位符
        section_template = (config_template or "")
        if "{模型的建立与求解}" in section_template:
            section_template = section_template.replace(
                "{模型的建立与求解}",
                f"{modeler_solution[:1500]}\n\n{model_build_solve[:2000]}",
            )
        if "{题目}" in section_template:
            section_template = section_template.replace("{题目}", str(self.questions)[:2000])
        if "{问题}" in section_template:
            section_template = section_template.replace("{问题}", str(self.questions)[:2000])

        # LaTeX 模式：将模板中的 Markdown 标题转为 LaTeX 命令
        if format_output == "latex":
            section_template = section_template.replace("\n## ", "\n\\subsection{").replace("\n# ", "\n\\section{")
            # 闭合未闭合的 LaTeX 标题
            for line in section_template.split("\n"):
                if line.startswith("\\subsection{") and not line.endswith("}"):
                    section_template = section_template.replace(line, line + "}")
                if line.startswith("\\section{") and not line.endswith("}"):
                    section_template = section_template.replace(line, line + "}")

        # 从代码结果+输出中提取关键数值（双重来源确保不遗漏）
        key_metrics = self._extract_key_metrics(coder_response, code_output)
        model_description = self._extract_model_description(coder_response)

        # 章节标题规则
        chapter_rule = ""
        if key == "eda":
            chapter_rule = "\n**注意：你只写 ## 4.2 描述性统计。章标题 # 四、 由 symbol 节负责，你不要重复。**"
        elif key == "symbol":
            chapter_rule = "\n**你写 # 四、符号说明与数据预处理 章标题 + ## 4.1 符号说明 小节。**"
        elif key.startswith("ques"):
            chapter_rule = (
                f"\n**注意：你只写 ## 5.{key[4:]} 子节，章标题 # 五、 由系统自动添加。**"
            )
        elif key == "sensitivity_analysis":
            chapter_rule = "\n**你写 # 六、敏感性分析 章标题 + 所有小节。**"

        # LaTeX 模式：用户消息不使用任何 Markdown 格式
        if format_output == "latex":
            return f"""WRITING TASK: Write the [{key}] section in LaTeX.

MODELING PLAN (why this method was chosen):
{modeler_solution[:2000]}

TEMPLATE AND STRUCTURE:
{section_template[:1500]}

MODELS AND METHODS:
{model_description or "(from code execution results)"}

KEY METRICS AND DATA:
{key_metrics or "(from code output)"}

CODE OUTPUT:
{code_output[:1500]}

FIGURES: Use \\includegraphics for each figure. Caption with \\caption.

CRITICAL: Output ONLY LaTeX. Use \\section{{}}, \\subsection{{}}, \\begin{{tabular}}+\\toprule, \\begin{{equation}}, \\textbf{{}}. ABSOLUTELY NO Markdown (#, ##, **, etc.)"""
        else:
            return f"""## 写作任务: 撰写论文的 [{key}] 部分
{chapter_rule}

### 建模方案（建模手的分析——告诉你"为什么选这个方法"）
{modeler_solution[:2000]}

### 该章节模板与结构要求
{section_template[:1500]}

### 模型与方法（从代码中提取）
{model_description or "(从代码执行结果中获取)"}

### ⚠️ 关键指标与数据（必须逐条写入论文，禁止编造或改写数值）⚠️
以下是代码实际输出的确切数值，你必须原样引用到正文中。
每条指标后括号内注明"（见代码输出）"，确保与图表数据一致。
{key_metrics or "(代码输出中无关键指标) "}

### 代码输出详情（补充参考）
{code_output[:1500]}

### 图注格式要求（强制）
每张图片插入后，图名必须单独一行，格式：\n**图X: 中文描述**\n图名不加粗、不加特殊格式，紧跟在图片标记下方。

### 输出格式
使用 Markdown 格式，表格用 HTML <table> 标签。"""

    @staticmethod
    def _extract_key_metrics(coder_response: str, code_output: str = "") -> str:
        """从 Coder 响应 + 代码输出中提取关键指标行。

        强制提取所有带数字的结论语句，确保 Writer 拿到的是确切数值。
        """
        lines = (coder_response or "").split("\n") + (code_output or "").split("\n")
        metrics = []
        for line in lines:
            lower = line.lower()
            if any(kw in lower for kw in [
                "r²", "rmse", "mae", "mse", "accuracy", "准确",
                "precision", "recall", "f1", "p值", "p-value", "p <", "p=",
                "系数", "coefficient", "r2", "得分", "score", "auc", "roc",
                "轮廓", "silhouette", "ari", "nmi", "χ²", "卡方",
                "= 0.", "=0.", "±", "保留率", "流失率", "缺失率",
            ]):
                stripped = line.strip()
                if stripped and len(stripped) < 200:
                    metrics.append(stripped)
            if len(metrics) >= 30:
                break
        return "\n".join(metrics[:30]) if metrics else ""

    @staticmethod
    def _extract_model_description(coder_response: str) -> str:
        """从 Coder 响应中提取模型描述行。"""
        if not coder_response:
            return ""
        lines = coder_response.split("\n")
        desc = []
        for line in lines:
            lower = line.lower()
            if any(kw in lower for kw in ["=== ", "模型", "model", "方法", "method",
                "算法", "algorithm", "评估", "evaluation", "验证", "validation"]):
                desc.append(line.strip())
            if len(desc) >= 15:
                break
        return "\n".join(desc[:15]) if desc else ""

    def get_write_flows(
        self,
        user_output,
        config_template: dict[str, str],
        background: str,
    ) -> dict[str, dict]:
        """生成论文结构性章节的写作任务。

        Args:
            user_output: UserOutput 实例。
            config_template: 论文模板字典，key 为章节名，value 为模板内容。
            background: 题目背景。

        Returns:
            以章节名称为键的写作任务字典。
        """
        model_build_solve = user_output.get_model_build_solve()

        def _fill_template(key: str) -> str:
            """读取模板并替换占位符。"""
            tpl = config_template.get(key, "")
            tpl = tpl.replace("{模型的建立与求解}", model_build_solve[:3000])
            tpl = tpl.replace("{题目}", background[:2000])
            tpl = tpl.replace("{问题}", background[:2000])
            return tpl

        flows = {
            "firstPage": {
                "writer_prompt": f"""## 写作任务: 撰写论文首页（标题+摘要+关键词）

{_fill_template('firstPage')}

请撰写论文首页，包含标题、摘要（300-500字，结构: 问题→方法→结果→结论）和3-5个关键词。"""
            },
            "RepeatQues": {
                "writer_prompt": f"""## 写作任务: 问题重述

{_fill_template('RepeatQues')}

请用自己的话重述题目，不要直接复制原文，200-300字。"""
            },
            "analysisQues": {
                "writer_prompt": f"""## 写作任务: 问题分析

{_fill_template('analysisQues')}

请分析每个子问题的类型、难点和整体求解思路。"""
            },
            "modelAssumption": {
                "writer_prompt": f"""## 写作任务: 模型假设

{_fill_template('modelAssumption')}

请列出3-5条合理的模型假设，每条假设说明理由。使用有序列表格式。"""
            },
            "symbol": {
                "writer_prompt": f"""## 写作任务: 符号说明

{_fill_template('symbol')}

**你只写 ## 4.1 符号说明 小节，不要写 ## 4.2 描述性统计。**
用 HTML <table> 语法制作符号表，列：符号 | 含义 | 单位，符号用 $...$ 包裹。"""
            },
            "judge": {
                "writer_prompt": f"""## 写作任务: 模型评价与改进

{_fill_template('judge')}

**你写 # 七、模型评价与改进 章标题 + 所有小节。**
请撰写模型评价章节，包括：
1. 模型优点（3-4条，每条结合具体数据）
2. 模型缺点（2-3条，诚实评估）
3. 改进方向"""
            },
        }
        return flows

    def get_seq(self) -> list[str]:
        """返回论文章节顺序。"""
        return [
            "firstPage",
            "RepeatQues",
            "analysisQues",
            "modelAssumption",
            "symbol",
            "eda",
            *sorted(
                [k for k in self.questions if k.startswith("ques") and k[4:].isdigit()],
                key=lambda x: int(x[4:]),
            ),
            "sensitivity_analysis",
            "judge",
        ]
