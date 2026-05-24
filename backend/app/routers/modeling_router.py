"""FastAPI 路由 —— 提交建模任务。"""

import asyncio
import os
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from app.core.workflow import MathModelWorkFlow
from app.task_store import task_store
from app.utils.log_util import logger

router = APIRouter(tags=["modeling"])

WORK_DIR_ROOT = Path("project/work_dir")


@router.post("/modeling")
async def submit_modeling(
    background_tasks: BackgroundTasks,
    question: str = Form(default="", description="题目文本（直接粘贴）"),
    question_file: UploadFile | None = File(default=None, description="题目文件（PDF/TXT/MD）"),
    data_files: list[UploadFile] = File(default=[], description="数据附件"),
):
    """提交数学建模任务。

    支持两种提交题目方式：
    - question: 直接粘贴题目文本
    - question_file: 上传题目文件（PDF/TXT/MD）

    上传的文件会被复制到任务工作目录供 Agent 使用。
    """
    # 验证输入
    if not question and not question_file:
        raise HTTPException(status_code=400, detail="请提供题目文本或上传题目文件")

    # 创建任务
    from app.utils.common_utils import create_task_id
    task_id = create_task_id()
    work_dir = WORK_DIR_ROOT / task_id
    work_dir.mkdir(parents=True, exist_ok=True)
    task_store.create(task_id)

    # 保存题目
    if question_file:
        question_path = work_dir / question_file.filename
        content = await question_file.read()
        with open(question_path, "wb") as f:
            f.write(content)
        if question_file.filename.lower().endswith(".pdf"):
            question_text = str(question_path)  # PDF 走特殊解析
        else:
            question_text = content.decode("utf-8")
    else:
        question_text = question

    # 保存数据文件
    data_dir = None
    if data_files:
        data_dir = work_dir / "data"
        data_dir.mkdir(exist_ok=True)
        for f in data_files:
            content = await f.read()
            filepath = data_dir / f.filename
            with open(filepath, "wb") as fh:
                fh.write(content)
        data_dir = str(data_dir)

    # 后台运行流水线
    task_store.update(task_id, status="running", stage="初始化", progress="0/5")
    background_tasks.add_task(_run_pipeline, task_id, question_text, data_dir)

    return {
        "task_id": task_id,
        "status": "running",
        "message": "任务已提交，正在后台执行",
        "check_url": f"/task/{task_id}",
    }


async def _run_pipeline(task_id: str, question_text: str, data_dir: str | None) -> None:
    """后台执行流水线，更新任务状态到 task_store。"""
    try:
        workflow = MathModelWorkFlow()
        workflow.task_id = task_id
        workflow.work_dir = str(WORK_DIR_ROOT / task_id)

        task_store.update(task_id, status="running", stage="PDF解析", progress="0/5")
        result = await workflow.execute(question_text, data_dir)

        task_store.update(
            task_id,
            status="completed",
            stage="完成",
            progress="5/5",
            result=result,
        )
        logger.info(f"任务 {task_id} 完成")
    except Exception as e:
        logger.error(f"任务 {task_id} 失败: {e}")
        task_store.update(
            task_id,
            status="error",
            stage="失败",
            error=str(e),
        )
