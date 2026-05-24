"""PDF 图片提取与多模态识别工具。

将 PDF 中的嵌入图片提取出来，调用多模态 LLM 识别为文字描述，
防止题目中的图表/公式截图信息丢失。
"""

import base64
import io
import os
from pathlib import Path
from typing import Optional

from app.config.setting import settings
from app.utils.log_util import logger


def extract_images_from_pdf(pdf_path: str) -> list[dict]:
    """从 PDF 中提取所有嵌入图片。

    使用 PyMuPDF (fitz) 提取 PDF 中嵌入的图片，
    返回图片的 base64 编码和位置信息。

    Args:
        pdf_path: PDF 文件路径。

    Returns:
        图片信息列表，每项包含 {page, b64, ext}。

    Raises:
        ImportError: PyMuPDF 未安装时。
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.warning("PyMuPDF 未安装，跳过 PDF 图片提取")
        return []

    images = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                ext = base_image["ext"]
                b64 = base64.b64encode(image_bytes).decode("utf-8")
                images.append({
                    "page": page_num + 1,
                    "index": img_index,
                    "ext": ext,
                    "b64": b64,
                })
        doc.close()
        logger.info(f"从 PDF 中提取了 {len(images)} 张图片")
    except Exception as e:
        logger.warning(f"PDF 图片提取失败: {e}")

    return images


async def recognize_images(images: list[dict], context: str = "") -> str:
    """调用多模态 LLM 识别图片内容为文字描述。

    使用 OpenAI 兼容接口调用千问 DashScope 进行识图。

    Args:
        images: extract_images_from_pdf 返回的图片列表。
        context: 附件文本上下文（如 PDF 中已提取的文字），帮助模型理解图片。

    Returns:
        所有图片识别结果的汇总文本。
    """
    if not images:
        return ""

    api_key = settings.VISION_API_KEY
    model = settings.VISION_MODEL
    base_url = settings.VISION_BASE_URL

    if not api_key or not model:
        logger.warning("识图模型未配置（VISION_API_KEY/VISION_MODEL），跳过图片识别")
        return ""

    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai 库未安装，跳过图片识别")
        return ""

    # 构建多模态消息内容
    content_parts = [
        {
            "type": "text",
            "text": f"""请识别以下从PDF中提取的图片内容。这些图片来自一份数学建模竞赛题目。

上下文信息:
{context[:2000] if context else "无"}

请对每张图片的内容进行详细描述，包括：
1. 如果是表格/数据：描述表格结构、列名、关键数值
2. 如果是流程图：描述流程的各个环节和关系
3. 如果是公式：用 LaTeX 格式写出公式
4. 如果是图表：描述图表类型、轴标签、数据趋势
5. 如果有文字标注：完整转述所有文字内容

请用中文回答。""",
        }
    ]

    # 添加图片（最多 10 张，避免 token 过大）
    for img in images[:10]:
        ext = img.get("ext", "png")
        content_parts.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{ext};base64,{img['b64']}",
            },
        })

    messages = [{"role": "user", "content": content_parts}]

    try:
        logger.info(f"正在识别 {min(len(images), 10)} 张 PDF 图片...")
        client = OpenAI(api_key=api_key, base_url=base_url)
        # 在线程池中运行同步 OpenAI 调用，避免阻塞事件循环
        import asyncio
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=4096,
            ),
        )
        text = response.choices[0].message.content
        logger.info(f"图片识别完成，获取 {len(text) if text else 0} 字符")
        return text or ""
    except Exception as e:
        logger.warning(f"图片识别调用失败: {e}")
        return ""


def extract_text_from_pdf(pdf_path: str) -> str:
    """从 PDF 中提取纯文本内容。

    Args:
        pdf_path: PDF 文件路径。

    Returns:
        提取的文本内容。
    """
    text = ""
    # 方案1: PyMuPDF
    try:
        import fitz
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        doc.close()
        if text.strip():
            return text
    except ImportError:
        pass

    # 方案2: pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        if text.strip():
            return text
    except ImportError:
        pass

    return text


async def parse_pdf_question(pdf_path: str) -> str:
    """解析 PDF 题目文件，提取文本 + 识别图片，合并为完整题目文本。

    这是 PDF 题目的统一入口：
    1. 提取 PDF 中的文本
    2. 提取 PDF 中嵌入的图片
    3. 调用多模态模型识别图片为文字
    4. 合并文本和图片描述

    Args:
        pdf_path: PDF 文件路径。

    Returns:
        包含文本和图片描述的完整题目内容。
    """
    if not os.path.exists(pdf_path):
        logger.warning(f"PDF 文件不存在: {pdf_path}")
        return ""

    logger.info(f"解析 PDF 题目: {pdf_path}")

    # Step 1: 提取文本
    text = extract_text_from_pdf(pdf_path)
    logger.info(f"PDF 文本提取: {len(text)} 字符")

    # Step 2: 提取图片
    images = extract_images_from_pdf(pdf_path)

    # Step 3: 识别图片
    image_text = ""
    if images:
        image_text = await recognize_images(images, context=text)

    # Step 4: 合并
    if image_text:
        full_text = f"{text}\n\n## PDF中图片的描述（由AI识别）\n{image_text}"
        logger.info(f"PDF 完整解析: 文本 {len(text)} 字符 + 图片描述 {len(image_text)} 字符")
        return full_text

    logger.info(f"PDF 解析完成（无图片或无需识别）: {len(text)} 字符")
    return text
