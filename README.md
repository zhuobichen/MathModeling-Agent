# 数学建模竞赛 · 数据分析题全流程自动化系统

> 从输入题目到输出完整论文的多 Agent 自动求解 Pipeline。

## 定位

面向数学建模竞赛数据分析类题目，覆盖「题目解析 → 数据理解 → 建模方案 → 代码实现 → 结果验证 → 论文生成」全流程，由多个 LLM Agent 协作完成。

## 架构

| 模块 | 职责 |
|------|------|
| `backend/app/core/agents/` | 多角色 Agent（建模 / 编码 / 解析 / 评审 / 写作） |
| `backend/app/core/workflow.py` | 流水线编排 |
| `backend/app/tools/` | 本地解释器、文献检索、网页搜索等工具 |
| `backend/skills/` | 各阶段提示词与参考规范 |
| `backend/main.py` | 服务入口 |

## 技术栈

Python + FastAPI + Redis + LLM（多模型），支持本地代码执行与 notebook 序列化。

## 文档

- `数学建模数据分析自动化系统设计文档.md` — 完整设计文档
