"""日志初始化模块，配置 loguru 结构化日志和链路追踪。

支持通过 contextvars 注入 trace_id，实现全流水线链路追踪。
"""

import contextvars
import os
import sys
import time
from contextlib import contextmanager
from loguru import logger as _logger  # type: ignore[import-unresolved]

# ---- 链路追踪 ----
# 使用 contextvars 在异步上下文中传播 trace_id，无需侵入函数签名
_trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "trace_id", default="-"
)
_task_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "task_id", default="-"
)


def set_trace_context(task_id: str) -> None:
    """设置当前异步上下文的 trace_id 和 task_id。

    应在 workflow.execute() 开始时调用一次即可，
    contextvars 会自动传播到所有子协程及 asyncio.gather 并发任务。
    """
    import uuid
    _trace_id_var.set(str(uuid.uuid4())[:8])
    _task_id_var.set(task_id)


def get_trace_id() -> str:
    """返回当前上下文的 trace_id。"""
    return _trace_id_var.get()


def get_task_id() -> str:
    """返回当前上下文的 task_id。"""
    return _task_id_var.get()


@contextmanager
def trace_context(task_id: str):
    """临时覆盖 trace_id（用于嵌套子流程）。

    用法:
        with trace_context("sub-task-x"):
            logger.info("子流程日志")  # 带有临时 trace_id
    """
    import uuid
    tid_token = _trace_id_var.set(str(uuid.uuid4())[:8])
    task_token = _task_id_var.set(task_id)
    try:
        yield
    finally:
        _trace_id_var.reset(tid_token)
        _task_id_var.reset(task_token)


class LoggerInitializer:
    """日志初始化器，配置控制台和文件日志输出。"""

    def __init__(self):
        self.log_path = os.path.join(os.getcwd(), "logs")
        self.__ensure_log_directory_exists()
        self.log_path_error = os.path.join(
            self.log_path, f"{time.strftime('%Y-%m-%d')}_error.log"
        )

    def __ensure_log_directory_exists(self):
        """确保日志目录存在，如果不存在则创建。"""
        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)

    @staticmethod
    def __filter(log: dict):
        """日志过滤器：注入 trace_id / task_id 到 extra 字段实现链路追踪。

        log['extra'] 是 loguru 为用户自定义字段保留的字典，
        注入后在 format_str 中可通过 {extra[trace_id]} 访问。
        """
        log["extra"]["trace_id"] = _trace_id_var.get()
        log["extra"]["task_id"] = _task_id_var.get()
        return log

    def init_log(self):
        """初始化日志配置。"""
        # 结构化日志格式：时间 | 级别 | 追踪ID | 位置 - 消息
        format_str = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<yellow>[{extra[task_id]}:{extra[trace_id]}]</yellow> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
        _logger.remove()
        # 移除后重新添加 sys.stderr, 目的: 控制台输出与文件日志内容和结构一致
        _logger.add(sys.stderr, filter=self.__filter, format=format_str, enqueue=False)
        _logger.add(
            self.log_path_error,
            filter=self.__filter,
            format=format_str,
            rotation="50MB",
            encoding="utf-8",
            enqueue=False,
            compression="zip",
        )

        return _logger


# 初始化日志处理器
log_initializer = LoggerInitializer()
logger = log_initializer.init_log()
