"""WriterAgent —— Stage 4: 论文生成 + 去AI味处理。"""

from app.core.agents.agent import Agent
from app.core.llm.llm import LLM
from app.core.skill_loader import skill_loader
from app.schemas.enums import CompTemplate, FormatOutPut
from app.tools.openalex_scholar import OpenAlexScholar
from app.utils.log_util import logger
from app.services.redis_manager import redis_manager
from app.schemas.response import SystemMessage, WriterMessage
import json
from app.tools.tool_registry import tool_registry
from app.schemas.A2A import WriterResponse

DEFAULT_WRITER_MAX_RETRIES = 3


class WriterAgent(Agent):
    """WriterAgent 负责撰写完整论文，包含去AI味后处理。"""

    def __init__(
        self,
        task_id: str,
        model: LLM,
        max_chat_turns: int | None = None,
        comp_template: CompTemplate = CompTemplate.CHINA,
        format_output: FormatOutPut = FormatOutPut.Markdown,
        scholar: OpenAlexScholar | None = None,
        max_memory: int = 25,
        max_retries: int = DEFAULT_WRITER_MAX_RETRIES,
        figure_descriptions: dict[str, str] | None = None,
        figure_metadata: dict[str, str] | None = None,
        work_dir: str = "",
    ) -> None:
        super().__init__(task_id, model, max_chat_turns, max_memory)
        self.format_out_put = format_output
        self.comp_template = comp_template
        self.work_dir = work_dir
        self.scholar = scholar
        self.is_first_run = True
        self.system_prompt = skill_loader.get_system_prompt(
            "writer", FORMAT="markdown" if format_output == FormatOutPut.Markdown else "latex"
        )
        self.available_images: list[str] = []
        self.max_retries = max_retries
        self.figure_descriptions: dict[str, str] = figure_descriptions or {}
        self.figure_metadata: dict[str, str] = figure_metadata or {}

    async def run(  # type: ignore[reportIncompatibleMethodOverride]
        self,
        prompt: str,
        available_images: list[str] | None = None,
        sub_title: str | None = None,
    ) -> WriterResponse:
        """执行写作任务。

        Args:
            prompt: 写作提示（包含编码结果和建模方案）。
            available_images: 可用图片路径列表。
            sub_title: 子任务标题。

        Returns:
            WriterResponse 对象。
        """
        logger.info(f"WriterAgent 开始: {sub_title}")

        if self.is_first_run:
            self.is_first_run = False
            await self.append_chat_history(
                {"role": "system", "content": self.system_prompt}
            )

        if available_images:
            self.available_images = available_images
            # 用表格 + 代码块格式展示，避免 LLM 把 ![]() 当成"格式示例"而非"确切文件名"
            image_rows = "\n".join(
                f"| {i+1} | `{img}` |"
                for i, img in enumerate(available_images)
            )
            # 图标题：来自 CoderAgent 代码中的 suptitle/print 输出
            desc_section = ""
            desc_lines = []
            for img in available_images:
                coder_desc = self.figure_metadata.get(img, "")
                if coder_desc:
                    desc_lines.append(f"- **{img}**: {coder_desc[:200]}")
            if desc_lines:
                desc_section = (
                    "\n\n### ⚠️ 每张图的标题必须使用以下代码输出的确切描述（禁止编造）⚠️\n"
                    + "\n".join(desc_lines)
                    + "\n\n**图注要求**：`**图X: 描述**` 中的描述必须与上表中对应文件名的描述完全一致，"
                    + "不得自己编写新的描述。代码输出是什么就写什么。"
                )
            image_prompt = (
                f"\n\n## 必须插入的图片（使用确切文件名，禁止编造）\n\n"
                f"以下是本小节**唯一可用**的图片文件，你必须按顺序逐一插入。\n"
                f"插入语法：`![描述文字](文件路径)`。**文件路径必须与下表完全一致**：\n\n"
                f"| 序号 | 文件路径（必须原样使用）|\n"
                f"|------|---------------------------|\n"
                f"{image_rows}\n"
                f"{desc_section}\n\n"
                f"要求：\n"
                f"1. 每张图插入后紧跟一行 `**图X: 描述**`（X 为图片序号）\n"
                f"2. **图注描述必须与上表中该文件的描述完全一致，禁止自己编造新描述**\n"
                f"3. 每张图后配至少 100 字的分析解读\n"
                f"4. **禁止**使用任何未在上表中列出的文件名\n"
                f"5. **禁止**修改文件名（包括大小写、后缀、路径前缀）\n"
            )
            prompt = prompt + image_prompt

        self.current_chat_turns += 1
        await self.append_chat_history({"role": "user", "content": prompt})

        last_error: str = ""
        for attempt in range(self.max_retries + 1):
            try:
                response = await self.model.chat(
                    history=self.chat_history,
                    tools=tool_registry.get_schemas(["search_papers", "search_knowledge"]),
                    tool_choice="auto",
                    agent_name=self.__class__.__name__,
                    sub_title=sub_title,
                )

                footnotes: list = []
                response_content: str = ""

                if (
                    hasattr(response.choices[0].message, "tool_calls")
                    and response.choices[0].message.tool_calls
                ):
                    await self.append_chat_history(
                        response.choices[0].message.model_dump()
                    )
                    for tool_call in response.choices[0].message.tool_calls:
                        tool_id = tool_call.id
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)
                        try:
                            result = await tool_registry.dispatch(tool_name, tool_args, self.task_id)
                        except ValueError as e:
                            result = f"工具调用失败: {e}"
                        await self.append_chat_history(
                            {
                                "role": "tool",
                                "content": result,
                                "tool_call_id": tool_id,
                                "name": tool_name,
                            }
                        )
                    next_response = await self.model.chat(
                        history=self.chat_history,
                        tools=tool_registry.get_schemas(["search_papers", "search_knowledge"]),
                        tool_choice="auto",
                        agent_name=self.__class__.__name__,
                        sub_title=sub_title,
                    )
                    response_content = next_response.choices[0].message.content
                else:
                    response_content = response.choices[0].message.content

                # 提取脚注（[^N]: content 格式）
                if response_content:
                    from app.utils.common_utils import split_footnotes
                    response_content, footnotes = split_footnotes(response_content)
                    # 图片文件名自动修正：将 LLM 编造的文件名替换为真实文件名
                    response_content = self._fix_image_filenames(response_content)

                self.chat_history.append(
                    {"role": "assistant", "content": response_content}
                )
                logger.info(f"WriterAgent 完成: {sub_title}")
                return WriterResponse(
                    response_content=response_content, footnotes=footnotes
                )

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"WriterAgent 重试 {attempt+1}/{self.max_retries+1}: {last_error}"
                )
                if attempt < self.max_retries:
                    await redis_manager.publish_message(
                        self.task_id,
                        SystemMessage(
                            content=f"写作手重试中({attempt+1}/{self.max_retries})...",
                            type="warning",
                        ),
                    )

        error_msg = f"WriterAgent 超过最大重试次数({self.max_retries}), 最后错误: {last_error}"
        logger.error(error_msg)
        return WriterResponse(response_content=error_msg, footnotes=[])

    async def _append_multimodal_message(
        self, text: str, images: list[str],
    ) -> None:
        """将文本+图片编码为多模态消息并追加到 chat_history。

        千问 Qwen3.6 Plus 等视觉模型可直接理解图片内容，
        无需额外识图步骤。每张图以 base64 编码内嵌。
        """
        import base64, os

        content_parts: list[dict] = [{"type": "text", "text": text}]
        img_count = 0
        for img_path in images:
            try:
                ext = img_path.rsplit(".", 1)[-1].lower()
                mime_map = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "gif": "image/gif"}
                mime = mime_map.get(ext, "image/png")
                abs_path = os.path.join(self.work_dir, img_path) if self.work_dir else img_path
                if not os.path.exists(abs_path):
                    continue
                with open(abs_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                })
                img_count += 1
            except Exception as e:
                logger.warning(f"图片编码失败 {img_path}: {e}")

        await self.append_chat_history({"role": "user", "content": content_parts})
        logger.info(f"WriterAgent 多模态消息: {len(text)} 字 + {img_count} 张图")

    def _fix_image_filenames(self, content: str) -> str:
        """将 LLM 编造的图片文件名自动修正为 CoderAgent 生成的真实文件名。

        LLM 在写作时可能忽略 prompt 中给出的确切文件名而自行编造，
        此方法提取 content 中的所有 ![xxx](path) 引用，
        按出现顺序逐一替换为 self.available_images 中的真实文件路径。

        Args:
            content: 生成的文本内容。

        Returns:
            修正后的文本内容。
        """
        import re

        if not self.available_images:
            return content

        # 提取所有图片引用: ![desc](filepath)
        image_refs = re.findall(r'!\[([^\]]*)\]\(([^\)]+)\)', content)
        if not image_refs:
            return content

        # 区分真实文件名和编造文件名
        available_basenames = {
            img.split("/")[-1] for img in self.available_images
        }

        fake_count = 0
        for desc, filepath in image_refs:
            basename = filepath.split("/")[-1]
            if basename not in available_basenames:
                fake_count += 1

        if fake_count == 0:
            return content  # 全部正确，无需修正

        # 按出现顺序逐一替换
        real_idx = 0
        fixed = content

        def _replace_one(match: re.Match) -> str:
            nonlocal real_idx
            desc = match.group(1)
            filepath = match.group(2)
            basename = filepath.split("/")[-1]

            if basename in available_basenames:
                # 文件名正确，不替换但推进索引
                if real_idx < len(self.available_images):
                    real_idx += 1
                return match.group(0)

            # 文件名错误：用下一个真实文件替换
            if real_idx < len(self.available_images):
                real_file = self.available_images[real_idx]
                real_idx += 1
                logger.warning(
                    f"图片文件名自动修正: '{filepath}' → '{real_file}'"
                )
                return f"![{desc}]({real_file})"

            return match.group(0)

        fixed = re.sub(r'!\[([^\]]*)\]\(([^\)]+)\)', _replace_one, fixed)

        if fake_count > 0:
            logger.warning(
                f"图片文件名修正完成: {fake_count} 个编造文件名已替换为真实文件"
            )

        return fixed

