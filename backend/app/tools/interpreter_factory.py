"""代码解释器工厂模块，根据配置创建本地、Docker 或远程解释器。"""

import os
from typing import Literal
from app.tools.base_interpreter import BaseCodeInterpreter
from app.tools.local_interpreter import LocalCodeInterpreter
from app.tools.notebook_serializer import NotebookSerializer
from app.config.setting import settings
from app.utils.log_util import logger


def _local_exec_allowed() -> bool:
    """是否允许在宿主机上直接跑模型生成的代码。

    审计项：原来只要 E2B key 缺失就静默降级到本机 Jupyter，
    多用户/生产环境下等于把宿主机执行权交出去。现在默认 fail-closed，
    必须显式设置 MODELING_ALLOW_LOCAL_EXEC=1 才允许本地执行。
    """
    return os.getenv("MODELING_ALLOW_LOCAL_EXEC", "").strip() in {"1", "true", "True", "yes"}


async def create_interpreter(
    kind: Literal["local", "docker", "e2b"] | None = None,
    *,
    task_id: str,
    work_dir: str,
    notebook_serializer: NotebookSerializer,
    timeout=3000,
    exec_timeout: int | None = None,
):
    """创建代码解释器实例。

    优先级: 显式 kind > settings.CODE_INTERPRETER。
    `e2b` 未配 key 时**不再静默降级**：要么报错，要么显式允许本地执行。

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

    # E2B 缺 key：fail-closed，除非显式允许本地执行
    if kind == "e2b" and not settings.E2B_API_KEY:
        if not _local_exec_allowed():
            raise RuntimeError(
                "CODE_INTERPRETER=e2b 但未配置 E2B_API_KEY。"
                "生产/多用户环境请补上 key 或改用 docker；"
                "仅在受信任的单机开发环境，可设置 MODELING_ALLOW_LOCAL_EXEC=1 允许回退到本地 Jupyter。"
            )
        logger.warning("E2B_API_KEY 未配置，且已显式允许本地执行 → 回退 local（风险由使用者承担）")
        kind = "local"

    # 本地解释器同样受开关约束：显式选择 local 也要能解释清楚这是危险动作
    if kind == "local" and not _local_exec_allowed():
        logger.warning(
            "正在使用本地解释器执行生成的代码；如需在其他环境禁用，请设置 CODE_INTERPRETER=docker/e2b "
            "或保持 MODELING_ALLOW_LOCAL_EXEC 未设置"
        )

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
            exec_timeout=exec_timeout,
        )
        await interp.initialize()
        return interp
