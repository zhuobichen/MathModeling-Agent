"""Docker 代码执行沙箱模块。

在隔离容器中执行 Python 代码，支持:
- 内存/CPU限制 (mem_limit, cpu_quota)
- 网络隔离 (network_disabled)
- 文件共享 (work_dir 挂载)
- 超时控制
- 自动镜像构建
"""

import os
import tempfile
import uuid

import docker
from docker.errors import ImageNotFound, DockerException

from app.config.setting import settings
from app.tools.base_interpreter import BaseCodeInterpreter
from app.tools.notebook_serializer import NotebookSerializer
from app.utils.log_util import logger


class DockerCodeInterpreter(BaseCodeInterpreter):
    """在 Docker 容器中隔离执行 Python 代码。"""

    def __init__(
        self,
        task_id: str,
        work_dir: str,
        notebook_serializer: NotebookSerializer,
        *,
        timeout: int = 300,
        mem_limit: str = "2g",
        cpu_limit: float = 1.0,
    ):
        super().__init__(task_id, work_dir, notebook_serializer)
        self.timeout = timeout
        self.mem_limit = mem_limit
        self.cpu_limit = cpu_limit
        self._client: docker.DockerClient | None = None
        self._container: docker.models.containers.Container | None = None
        self._image_tag = "mathmodel-sandbox:latest"

    # ── 公开 API ──

    async def initialize(self) -> None:
        """构建/加载 Docker 镜像。"""
        try:
            self._client = docker.from_env()
        except DockerException as e:
            raise RuntimeError(f"Docker 不可用: {e}")

        # 检查镜像是否存在
        try:
            self._client.images.get(self._image_tag)
            logger.info(f"Docker 镜像已存在: {self._image_tag}")
        except ImageNotFound:
            logger.info(f"Docker 镜像构建中: {self._image_tag}...")
            dockerfile_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "Dockerfile"
            )
            if not os.path.exists(dockerfile_path):
                raise FileNotFoundError(f"Dockerfile 不存在: {dockerfile_path}")
            try:
                image, _logs = self._client.images.build(
                    path=os.path.dirname(dockerfile_path),
                    dockerfile=dockerfile_path,
                    tag=self._image_tag,
                    rm=True,
                )
                logger.info(f"Docker 镜像构建完成: {self._image_tag}")
            except docker.errors.BuildError as e:
                raise RuntimeError(f"Docker 镜像构建失败: {e}")

    async def _pre_execute_code(self) -> None:
        """执行初始化代码（非必须）。"""
        init_code = (
            "import matplotlib; matplotlib.use('Agg'); "
            "import matplotlib.pyplot as plt; "
            "import numpy as np; import pandas as pd; "
            f"import os; os.chdir('/workspace'); "
            "print('Docker sandbox ready')"
        )
        await self.execute_code(init_code)

    async def execute_code(self, code: str) -> tuple[str, bool, str]:
        """在 Docker 容器中执行代码。"""
        assert self._client is not None

        # 防止空代码
        if not code.strip():
            return "", False, ""

        # 用临时文件传递代码（避免命令行转义问题）
        script_name = f"_tmp_{uuid.uuid4().hex[:8]}.py"
        script_path = os.path.join(self.work_dir, script_name)
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)

        try:
            cpu_period = 100000
            cpu_quota = int(self.cpu_limit * cpu_period)

            container = self._client.containers.run(
                self._image_tag,
                f"python /workspace/{script_name}",
                detach=True,
                remove=False,
                mem_limit=self.mem_limit,
                cpu_period=cpu_period,
                cpu_quota=cpu_quota,
                network_disabled=True,
                volumes={
                    os.path.abspath(self.work_dir): {
                        "bind": "/workspace", "mode": "rw"
                    }
                },
                working_dir="/workspace",
                # 不允许提权
                security_opt=["no-new-privileges:true"],
            )

            try:
                exit_info = container.wait(timeout=self.timeout)
                exit_code = exit_info.get("StatusCode", -1)
                logs = container.logs(stdout=True, stderr=True).decode("utf-8", errors="replace")

                # 截断过长输出
                max_log = 8000
                if len(logs) > max_log:
                    logs = logs[:max_log // 2] + "\n...(已截断)...\n" + logs[-max_log // 2:]

                error_occurred = exit_code != 0
                error_msg = logs[-500:] if error_occurred else ""
                return logs, error_occurred, error_msg
            except Exception:
                # timeout 或其他异常
                container.kill()
                return "", True, f"代码执行超时 (>{self.timeout}s)"
            finally:
                try:
                    container.remove(force=True)
                except Exception:
                    pass
        except DockerException as e:
            return "", True, f"Docker 执行失败: {e}"
        finally:
            # 清理临时脚本
            try:
                os.remove(script_path)
            except OSError:
                pass

    async def cleanup(self) -> None:
        """清理：停止所有残留容器。"""
        if self._client:
            try:
                containers = self._client.containers.list(
                    filters={"ancestor": self._image_tag}
                )
                for c in containers:
                    try:
                        c.kill()
                        c.remove(force=True)
                    except Exception:
                        pass
                logger.info("Docker 沙箱已清理")
            except Exception:
                pass

    async def get_created_images(self, section: str) -> list[str]:
        """扫描 work_dir 获取新创建的图片。"""
        current = set()
        for root, _dirs, files in os.walk(self.work_dir):
            for f in files:
                if f.endswith((".png", ".jpg", ".jpeg")):
                    rel = os.path.relpath(os.path.join(root, f), self.work_dir)
                    current.add(rel.replace("\\", "/"))
        new = list(current - self.last_created_images)
        self.last_created_images = current.copy()
        logger.info(f"新创建的图片列表: {new}")
        return new
