"""代码解释器工厂模块，根据配置创建本地、Docker 或远程解释器。"""

from typing import Literal
from app.tools.base_interpreter import BaseCodeInterpreter
from app.tools.local_interpreter import LocalCodeInterpreter
from app.tools.notebook_serializer import NotebookSerializer
from app.config.setting import settings
from app.utils.log_util import logger


async def create_interpreter(
    kind: Literal["local", "docker", "e2b"] | None = None,
    *,
    task_id: str,
    work_dir: str,
    notebook_serializer: NotebookSerializer,
    timeout=3000,
):
    """创建代码解释器实例。

    优先级: 配置 > E2B key 检测 > 本地兜底

    Args:
        kind: 解释器类型。None 时从 settings.CODE_INTERPRETER 读取。
        task_id: 任务 ID。
        work_dir: 工作目录。
        notebook_serializer: Notebook 序列化器。
        timeout: 超时时间（秒）。
    """
    # 确定类型
    if kind is None:
        kind = settings.CODE_INTERPRETER  # type: ignore[assignment]

    # E2B 降级：无 API Key 时退到 local
    if kind == "e2b" and not settings.E2B_API_KEY:
        logger.warning("E2B_API_KEY 未配置，降级为 local 解释器")
        kind = "local"

    logger.info(f"代码解释器: {kind}")

    interp: BaseCodeInterpreter
    if kind == "e2b":
        from app.tools.e2b_interpreter import E2BCodeInterpreter
        interp = await E2BCodeInterpreter.create(
            task_id=task_id,
            work_dir=work_dir,
            notebook_serializer=notebook_serializer,
        )
        await interp.initialize(timeout=timeout)
        return interp
    elif kind == "docker":
        from app.tools.docker_interpreter import DockerCodeInterpreter
        interp = DockerCodeInterpreter(
            task_id=task_id,
            work_dir=work_dir,
            notebook_serializer=notebook_serializer,
            timeout=timeout,
            mem_limit=settings.DOCKER_MEM_LIMIT,
            cpu_limit=settings.DOCKER_CPU_LIMIT,
        )
        await interp.initialize()
        return interp
    else:  # local
        interp = LocalCodeInterpreter(
            task_id=task_id,
            work_dir=work_dir,
            notebook_serializer=notebook_serializer,
        )
        await interp.initialize()
        return interp
