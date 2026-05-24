"""任务状态存储模块 —— 内存字典 + JSON 文件持久化。"""

import json
import os
import time
from pathlib import Path


class TaskStore:
    """线程安全的任务状态存储。

    内存字典作为主存储，JSON 文件作为持久化备份。
    后续可替换为 Redis。
    """

    def __init__(self) -> None:
        self._tasks: dict[str, dict] = {}
        self._data_dir = Path("logs/tasks")
        self._data_dir.mkdir(parents=True, exist_ok=True)

    def create(self, task_id: str) -> dict:
        """创建新任务记录。"""
        task = {
            "task_id": task_id,
            "status": "pending",
            "stage": "",
            "progress": "0/5",
            "created_at": time.time(),
            "updated_at": time.time(),
            "result": None,
            "error": None,
        }
        self._tasks[task_id] = task
        self._save(task_id)
        return task

    def update(self, task_id: str, **kwargs) -> dict | None:
        """更新任务状态。"""
        task = self._tasks.get(task_id)
        if not task:
            return None
        task.update(kwargs)
        task["updated_at"] = time.time()
        self._save(task_id)
        return task

    def get(self, task_id: str) -> dict | None:
        """获取任务状态。先从内存取，再从文件恢复。"""
        if task_id in self._tasks:
            return self._tasks[task_id]
        return self._load(task_id)

    def list_recent(self, limit: int = 10) -> list[dict]:
        """列出最近的任务。"""
        tasks = sorted(
            self._tasks.values(),
            key=lambda t: t.get("created_at", 0),
            reverse=True,
        )
        return tasks[:limit]

    def _save(self, task_id: str) -> None:
        """持久化到 JSON 文件。"""
        if task_id in self._tasks:
            filepath = self._data_dir / f"{task_id}.json"
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self._tasks[task_id], f, ensure_ascii=False, indent=2)

    def _load(self, task_id: str) -> dict | None:
        """从 JSON 文件恢复任务状态。"""
        filepath = self._data_dir / f"{task_id}.json"
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                task = json.load(f)
            self._tasks[task_id] = task
            return task
        return None


# 全局单例
task_store = TaskStore()
