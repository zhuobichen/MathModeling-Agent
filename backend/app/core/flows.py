"""任务流编排模块，负责为每个阶段生成 Prompt。"""

from app.schemas.A2A import ModelerToCoder
from app.tools.base_interpreter import BaseCodeInterpreter


class Flows:
    """管理 5-Agent 流水线中每个阶段的 Prompt 生成。"""

    def __init__(self, questions: dict[str, str | int]) -> None:
        self.questions = questions

    def get_questions_quesx(self) -> dict[str, str]:
        """过滤出子问题的描述（ques1, ques2, ...）。

        Returns:
            以 quesX 为键、问题描述为值的字典。
        """
        return {
            k: str(v)
            for k, v in self.questions.items()
            if k.startswith("ques") and k[4:].isdigit()
        }

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
    ) -> str:
        """为 WriterAgent 生成论文章节写作 Prompt。

        Args:
            key: 章节标识（eda, ques1, ...）。
            coder_response: CoderAgent 的响应文本。
            code_interpreter: 代码解释器，用于获取 section 输出。

        Returns:
            写作 Prompt。
        """
        code_output = code_interpreter.get_code_output(key)

        return f"""## 写作任务: 撰写论文的 [{key}] 部分

### 代码执行结果
代码响应: {coder_response[:2000]}

代码输出: {code_output[:2000]}

### 论文模板参考
{config_template[:1000]}

请根据以上代码执行结果，撰写论文对应章节。注意：
1. 使用段落式写作，避免无序列表
2. 每张图需要至少100字分析
3. 遵循去AI味写作规则（不要使用禁用词汇和句式）
4. 数据要具体，不能笼统描述"""

    def get_write_flows(
        self,
        user_output,
        config_template: str,
        background: str,
    ) -> dict[str, dict]:
        """生成论文结构性章节的写作任务。

        Args:
            user_output: UserOutput 实例。
            config_template: 论文模板。
            background: 题目背景。

        Returns:
            以章节名称为键的写作任务字典。
        """
        model_build_solve = user_output.get_model_build_solve()

        flows = {
            "firstPage": {
                "writer_prompt": f"""## 写作任务: 撰写论文首页（标题+摘要+关键词）

### 题目背景
{background[:500]}

### 建模与求解摘要
{model_build_solve[:1000]}

### 模板
{config_template.get('firstPage', '')[:1000]}

请撰写论文首页，包含标题、摘要（300-500字，结构: 问题→方法→结果→结论）和3-5个关键词。"""
            },
            "RepeatQues": {
                "writer_prompt": f"""## 写作任务: 问题重述

### 题目背景
{background[:500]}

请用自己的话重述题目，不要直接复制原文，200-300字。"""
            },
            "analysisQues": {
                "writer_prompt": f"""## 写作任务: 问题分析

### 题目背景
{background[:500]}

请分析每个子问题的类型、难点和整体求解思路。"""
            },
            "modelAssumption": {
                "writer_prompt": """## 写作任务: 模型假设

请列出3-5条合理的模型假设，每条假设说明理由。使用有序列表格式。"""
            },
            "symbol": {
                "writer_prompt": """## 写作任务: 符号说明

请制作符号表（三线表格式），包含符号、含义、单位。"""
            },
            "judge": {
                "writer_prompt": f"""## 写作任务: 模型评价与改进

### 建模方案
{model_build_solve[:1000]}

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
