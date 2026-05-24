"""共享的 Prompt 工具函数。"""


def get_reflection_prompt(error_message: str, code: str) -> str:
    """生成代码错误反思修复提示词。

    Args:
        error_message: 错误信息（traceback）。
        code: 导致错误的代码。

    Returns:
        反思提示词。
    """
    return f"""你的代码执行时发生了错误。请仔细分析错误原因并修复代码。

## 错误信息
```
{error_message[:2000]}
```

## 出错的代码
```python
{code[:2000]}
```

## 修复指南
1. 仔细阅读错误信息的最后一行，确定错误类型
2. 如果是 ImportError: 检查库名是否正确，或使用替代库
3. 如果是 NameError: 检查变量/函数是否在使用前定义
4. 如果是 ValueError/TypeError: 检查数据类型和参数格式
5. 如果是 KeyError: 检查列名是否正确（注意大小写、空格）
6. 如果是 FileNotFoundError: 确认文件在当前工作目录中

请直接给出修复后的完整代码，不要请求用户帮助。如果多次修复失败，请简化方案或使用替代方法。"""


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

如果还有遗漏，请继续完成。如果全部完成，回复"任务完成"。"""
