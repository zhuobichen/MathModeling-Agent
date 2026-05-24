"""共享的 Prompt 工具函数。"""

import re


class ErrorClassifier:
    """代码错误分类器，识别Python错误类型并推荐针对性修复策略。

    用于 CoderAgent 错误自愈：先分类错误，再生成针对性修复提示，
    而非将原始 error traceback 笼统地丢给 LLM。
    """

    ERROR_PATTERNS: list[tuple[str, str, str]] = [
        (
            "import_error",
            r"ModuleNotFoundError|ImportError",
            "检查依赖名称拼写，使用 pip install 安装缺失包，或换用标准库替代方案",
        ),
        (
            "type_error",
            r"TypeError|ValueError|AttributeError",
            "检查变量类型、函数参数格式、None值处理；确认 DataFrame 列数据类型",
        ),
        (
            "index_error",
            r"IndexError|KeyError",
            "检查DataFrame列名（注意大小写、前后空格）；检查列表/数组索引范围",
        ),
        (
            "syntax_error",
            r"SyntaxError|IndentationError",
            "检查语法表达式和缩进，注意括号匹配和字符串引号",
        ),
        (
            "file_error",
            r"FileNotFoundError|FileExistsError|OSError|PermissionError",
            "确认文件路径正确、文件存在于工作目录；检查文件权限",
        ),
        (
            "name_error",
            r"NameError|UnboundLocalError",
            "检查变量/函数是否在使用前定义，注意变量作用域",
        ),
        (
            "memory_error",
            r"MemoryError",
            "数据量过大，使用分块处理(chunksize)、采样或稀疏矩阵",
        ),
        (
            "timeout_error",
            r"TimeoutError|timeout|timed out|Kernel.*died",
            "代码耗时过长或内核崩溃，优化算法复杂度，减少数据规模，增加超时时间",
        ),
    ]

    @classmethod
    def classify(cls, error_message: str) -> tuple[str, str]:
        """根据错误信息匹配已知错误模式。

        Args:
            error_message: Python traceback 或错误描述文本。

        Returns:
            (error_type, suggestion): 错误类型标识和中文修复建议。
            匹配失败时返回 ("unknown", 通用建议)。
        """
        for error_type, pattern, suggestion in cls.ERROR_PATTERNS:
            if re.search(pattern, error_message):
                return error_type, suggestion
        return "unknown", "请仔细阅读错误信息最后一行，逐行排查代码逻辑"


def get_reflection_prompt(
    error_message: str,
    code: str,
    error_type: str = "",
    suggestion: str = "",
) -> str:
    """生成代码错误反思修复提示词，支持错误分类后的针对性引导。

    Args:
        error_message: 错误信息（traceback）。
        code: 导致错误的代码。
        error_type: 错误分类标识（由 ErrorClassifier.classify 返回）。
        suggestion: 针对该错误类型的修复建议。

    Returns:
        包含错误分类和修复指南的反思提示词。
    """
    # 针对性引导，根据错误类型给出不同指令
    type_specific_guide = ""
    if error_type == "import_error":
        type_specific_guide = (
            "\n**处理方式**: 优先检查拼写 → 尝试标准库替代 → 最后才尝试 pip install"
        )
    elif error_type == "type_error":
        type_specific_guide = (
            "\n**处理方式**: 在关键操作前添加类型检查和转换（如 pd.to_numeric, astype）"
        )
    elif error_type == "index_error":
        type_specific_guide = (
            "\n**处理方式**: 先 print(df.columns.tolist()) 确认列名 → 去除前后空格 → 使用 df.columns.str.strip()"
        )
    elif error_type == "memory_error":
        type_specific_guide = (
            "\n**处理方式**: 使用 chunksize 分块读取 → 采样 → 删除中间变量释放内存"
        )
    elif error_type == "timeout_error":
        type_specific_guide = (
            "\n**处理方式**: 先在小数据上测试 → 减少循环嵌套 → 使用向量化操作"
        )

    classification_block = ""
    if error_type and error_type != "unknown":
        classification_block = f"""
## 错误分类: {error_type}
**修复建议**: {suggestion}{type_specific_guide}
"""

    return f"""你的代码执行时发生了错误。请仔细分析错误原因并直接给出修复后的完整代码。

{classification_block}
## 错误信息
```
{error_message[:2000]}
```

## 出错的代码
```python
{code[:2000]}
```

## 修复规则
1. 根据上述错误分类和建议进行针对性修复
2. 直接输出完整的修复后代码，不要省略任何部分
3. 如果同一错误连续出现 2 次，请换用不同的实现方式
4. 不要请求用户帮助——必须独立完成修复"""


def get_completion_check_prompt(prompt: str, text: str) -> str:
    """生成任务完成检查提示词。

    Args:
        prompt: 原始任务提示。
        text: 已生成的文本输出。

    Returns:
        完成检查提示词。
    """
    return f"""请检查以下任务是否已完全完成：

原始任务:
{prompt[:1000]}

当前输出:
{text[:1000]}

请判断：
1. 所有要求的数据处理是否完成
2. 所有图表是否生成并保存
3. 所有指标是否计算并输出
4. 是否有遗漏的步骤

如果还有遗漏，请继续完成。如果全部完成，回复\u201c任务完成\u201d\u3002"""
