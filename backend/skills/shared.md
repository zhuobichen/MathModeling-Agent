---
name: shared
description: 共享工具提示词，包含代码错误反思修复和任务完成检查
agent: Shared
version: "1.0"
---

# 共享提示词

## 一、代码错误反思修复 (reflection_prompt)

当代码执行出错时使用。错误信息、错误分类和修复建议由 ErrorClassifier 动态注入。

格式：
```
你的代码执行时发生了错误。请仔细分析错误原因并直接给出修复后的完整代码。

[错误分类信息]

## 错误信息
{error_message}

## 出错的代码
{code}

## 修复规则
1. 根据上述错误分类和建议进行针对性修复
2. 直接输出完整的修复后代码，不要省略任何部分
3. 如果同一错误连续出现 2 次，请换用不同的实现方式
4. 不要请求用户帮助——必须独立完成修复
```

## 二、任务完成检查 (completion_check_prompt)

当需要检查任务是否完全完成时使用。

格式：
```
请检查以下任务是否已完全完成：

原始任务:
{prompt}

当前输出:
{text}

请判断：
1. 所有要求的数据处理是否完成
2. 所有图表是否生成并保存
3. 所有指标是否计算并输出
4. 是否有遗漏的步骤

如果还有遗漏，请继续完成。如果全部完成，回复"任务完成"。
```

## 三、错误分类模式

ErrorClassifier 支持的 8 种 Python 错误模式：

| 错误类型 | 匹配模式 | 修复建议 |
|---------|---------|---------|
| import_error | ModuleNotFoundError, ImportError | 检查依赖安装 |
| type_error | TypeError, ValueError, AttributeError | 检查变量类型 |
| index_error | IndexError, KeyError | 检查索引/列名 |
| syntax_error | SyntaxError, IndentationError | 检查语法缩进 |
| file_error | FileNotFoundError, OSError | 检查文件路径 |
| name_error | NameError, UnboundLocalError | 检查变量定义 |
| memory_error | MemoryError | 分块处理/采样 |
| timeout_error | TimeoutError, timed out | 优化算法/减少数据 |
