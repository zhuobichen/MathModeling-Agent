"""FastAPI 路由 —— 任务状态查询、文件下载、健康检查。"""

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from app.task_store import task_store

router = APIRouter(tags=["task"])

WORK_DIR_ROOT = Path("project/work_dir")


@router.get("/health")
async def health():
    """健康检查。"""
    return {"status": "ok", "service": "MathModel 数学建模自动化系统"}


@router.get("/task/{task_id}")
async def get_task(task_id: str):
    """查询任务状态和结果。

    Returns:
        任务状态 JSON，包含 status/stage/progress/result 等字段。
    """
    task = task_store.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    response = {
        "task_id": task["task_id"],
        "status": task["status"],
        "stage": task.get("stage", ""),
        "progress": task.get("progress", "0/5"),
        "error": task.get("error"),
    }

    # 如果任务完成，附带输出文件列表
    work_dir = WORK_DIR_ROOT / task_id
    if task["status"] == "completed" and work_dir.exists():
        files = _list_output_files(work_dir)
        result = task.get("result", {})
        response["files"] = files
        if result.get("review_result"):
            response["review_score"] = result["review_result"].get("overall_score")
            response["review_passed"] = result["review_result"].get("passed")
        if os.path.exists(work_dir / "res.docx"):
            response["paper_docx"] = f"/task/{task_id}/files/res.docx"
        if os.path.exists(work_dir / "res.md"):
            response["paper_md"] = f"/task/{task_id}/files/res.md"

    return response


@router.get("/task/{task_id}/files/{filename:path}")
async def download_file(task_id: str, filename: str):
    """下载任务输出文件（论文、图表、notebook 等）。

    Args:
        task_id: 任务 ID。
        filename: 文件名（支持子目录，如 figures/figure1.png）。
    """
    work_dir = WORK_DIR_ROOT / task_id
    file_path = work_dir / filename

    # 安全检查：防止路径穿越
    file_path = file_path.resolve()
    work_dir_resolved = work_dir.resolve()
    if not str(file_path).startswith(str(work_dir_resolved)):
        raise HTTPException(status_code=403, detail="禁止访问")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")

    return FileResponse(
        path=str(file_path),
        filename=file_path.name,
        media_type="application/octet-stream",
    )


@router.get("/tasks")
async def list_tasks():
    """列出最近的任务。"""
    tasks = task_store.list_recent(limit=20)
    return {
        "total": len(tasks),
        "tasks": [
            {
                "task_id": t["task_id"],
                "status": t["status"],
                "stage": t.get("stage", ""),
                "created_at": t.get("created_at", 0),
            }
            for t in tasks
        ],
    }


def _list_output_files(work_dir: Path) -> list[dict]:
    """列出工作目录中所有输出文件。"""
    files = []
    for entry in os.listdir(work_dir):
        full = work_dir / entry
        if entry.startswith(".") or entry == "checkpoint.json":
            continue
        if full.is_file():
            files.append({
                "name": entry,
                "size": full.stat().st_size,
                "url": f"/task/{work_dir.name}/files/{entry}",
            })
        elif full.is_dir() and entry == "figures":
            for fig in os.listdir(full):
                files.append({
                    "name": f"figures/{fig}",
                    "size": (full / fig).stat().st_size,
                    "url": f"/task/{work_dir.name}/files/figures/{fig}",
                })
    return files
