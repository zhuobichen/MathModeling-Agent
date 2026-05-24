"""Skill 加载器 —— 从 Markdown 文件加载 Agent 系统提示词。

三层渐进式加载模型:
  1. 索引 (index): 启动时加载所有 skill 的 name + description (~100 token/skill)
  2. 加载 (load): Agent 初始化时按需加载完整 SKILL.md
  3. 读取 (read): 仅在需要时读取 references/ 中的重型文档

Skill 文件格式:
  ---
  name: coder
  description: 代码生成与执行
  agent: CoderAgent
  version: "1.0"
  context:
    KEY: "{KEY}"  # 占位符，加载时替换
  ---
  # System Prompt 正文...
"""

import os
import re
from pathlib import Path
from typing import Any

from app.utils.log_util import logger


class SkillLoader:
    """从 skills/ 目录加载和管理 Agent 提示词。"""

    def __init__(self, skills_dir: str | None = None):
        if skills_dir is None:
            # 默认: backend/skills/
            skills_dir = str(
                Path(__file__).parent.parent.parent / "skills"
            )
        self.skills_dir = skills_dir
        self._cache: dict[str, str] = {}  # name → raw content
        self._index: list[dict[str, str]] = []  # [{name, description, agent}]

    # ── 公开 API ──

    def build_index(self) -> list[dict[str, str]]:
        """扫描 skills/ 目录，建立 skill 索引。

        所有 skill 的 name + description 都会被加载（约 100 token/skill），
        用于意图匹配。即使未使用也要付出此成本。

        Returns:
            [{name, description, agent}, ...]
        """
        if not os.path.isdir(self.skills_dir):
            logger.warning(f"Skills 目录不存在: {self.skills_dir}")
            return []

        self._index = []
        for fname in sorted(os.listdir(self.skills_dir)):
            if not fname.endswith(".md"):
                continue
            path = os.path.join(self.skills_dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw = f.read()
                frontmatter = self._parse_frontmatter(raw)
                name = frontmatter.get("name", fname[:-3])
                self._index.append({
                    "name": name,
                    "description": frontmatter.get("description", ""),
                    "agent": frontmatter.get("agent", ""),
                    "version": frontmatter.get("version", "1.0"),
                })
                logger.debug(f"Skill 已索引: {name}")
            except Exception as e:
                logger.warning(f"Skill 索引失败 {fname}: {e}")

        return self._index

    def load(self, name: str, context: dict[str, str] | None = None,
             include_references: bool = True, **kwargs: str) -> str:
        """按需加载指定 skill 的完整内容。

        Args:
            name: skill 名称（如 "coder", "writer"）。
            context: 占位符替换字典（如 {"PLATFORM": "Windows"}）。

        Returns:
            skill 正文（不包含 frontmatter）。
        """
        context = {**(context or {}), **kwargs}

        # 检查缓存
        cache_key = f"{name}:{hash(frozenset(context.items()))}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # 查找文件
        fname = f"{name}.md"
        path = os.path.join(self.skills_dir, fname)
        if not os.path.exists(path):
            # 尝试模糊匹配 (agent name → skill name)
            for f in os.listdir(self.skills_dir):
                if f.endswith(".md"):
                    try:
                        with open(os.path.join(self.skills_dir, f), "r", encoding="utf-8") as fh:
                            raw = fh.read()
                        fm = self._parse_frontmatter(raw)
                        if fm.get("agent", "").lower() == name.lower() or fm.get("name") == name:
                            path = os.path.join(self.skills_dir, f)
                            break
                    except Exception:
                        continue

        if not os.path.exists(path):
            logger.warning(f"Skill 文件不存在: {name}.md")
            return ""

        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()

            content = self._strip_frontmatter(raw)

            # 替换 frontmatter 中的 context 占位符
            fm = self._parse_frontmatter(raw)
            fm_context = fm.get("context", {})
            if isinstance(fm_context, dict):
                for key, tmpl in fm_context.items():
                    if isinstance(tmpl, str) and "{" in tmpl:
                        # 仅当 context 提供值时替换
                        if key in context:
                            content = content.replace(tmpl.format(**{key: context[key]}), context[key])
                        else:
                            # 否则移除模板语法
                            content = content.replace("{" + key + "}", "")

            # 也支持正文中的 {KEY} 直接替换
            for key, value in context.items():
                content = content.replace("{" + key + "}", value)

            # 自动加载 references/{name}/ 目录下的所有参考文件
            ref_dir = os.path.join(self.skills_dir, "references", name)
            if include_references and os.path.isdir(ref_dir):
                ref_parts: list[str] = []
                for ref_file in sorted(os.listdir(ref_dir)):
                    if ref_file.endswith(".md"):
                        ref_path = os.path.join(ref_dir, ref_file)
                        try:
                            with open(ref_path, "r", encoding="utf-8") as rf:
                                ref_raw = rf.read()
                            ref_content = self._strip_frontmatter(ref_raw)
                            ref_parts.append(ref_content)
                            logger.debug(f"Skill 参考已加载: {name}/{ref_file}")
                        except Exception as e:
                            logger.warning(f"Skill 参考加载失败 {ref_file}: {e}")
                if ref_parts:
                    content += "\n\n---\n\n" + "\n\n".join(ref_parts)

            self._cache[cache_key] = content
            logger.info(f"Skill 已加载: {name} ({len(content)} 字符)")
            return content

        except Exception as e:
            logger.error(f"Skill 加载失败 {name}: {e}")
            return ""

    def get_system_prompt(self, name: str, include_references: bool = True, **context: str) -> str:
        """获取 Agent 系统提示词（快捷方法）。"""
        return self.load(name, context, include_references=include_references)

    def reload(self, name: str) -> None:
        """清除指定 skill 的缓存，下次 load 时重新读取。"""
        keys = [k for k in self._cache if k.startswith(f"{name}:")]
        for k in keys:
            del self._cache[k]
        logger.info(f"Skill 缓存已清除: {name}")

    # ── 内部 ──

    @staticmethod
    def _parse_frontmatter(raw: str) -> dict[str, Any]:
        """解析 YAML 风格的 frontmatter。

        frontmatter 被 `---` 包裹，位于文件开头。
        """
        m = re.match(r"^---\s*\n(.*?)\n---\s*\n", raw, re.DOTALL)
        if not m:
            return {}

        result: dict[str, Any] = {}
        yaml_text = m.group(1)

        current_key: str | None = None
        current_indent: int = 0
        nested: dict[str, Any] = {}

        for line in yaml_text.split("\n"):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            indent = len(line) - len(line.lstrip())

            # 顶级 key: value
            if ":" in stripped and not stripped.startswith("  "):
                if current_key:
                    result[current_key] = nested if nested else ""
                key, _, value = stripped.partition(":")
                current_key = key.strip()
                value = value.strip().strip('"')
                if value:
                    result[current_key] = value
                    nested = {}
                    current_key = None
                else:
                    nested = {}
                    current_indent = indent
            # 嵌套 key: value
            elif current_key and ":" in stripped:
                key, _, value = stripped.partition(":")
                nested[key.strip()] = value.strip().strip('"')

        if current_key:
            result[current_key] = nested if nested else ""

        return result

    @staticmethod
    def _strip_frontmatter(raw: str) -> str:
        """去除 frontmatter，返回正文。"""
        return re.sub(r"^---\s*\n.*?\n---\s*\n", "", raw, count=1, flags=re.DOTALL).strip()


# 全局单例
skill_loader = SkillLoader()
