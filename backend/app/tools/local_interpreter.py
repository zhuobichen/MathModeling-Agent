"""本地代码解释器模块，通过本地 Jupyter 内核执行 Python 代码。"""

from app.tools.base_interpreter import BaseCodeInterpreter
from app.tools.notebook_serializer import NotebookSerializer
import jupyter_client
from app.utils.log_util import logger
import os
import time
from app.services.redis_manager import redis_manager
from app.schemas.response import (
    OutputItem,
    ResultModel,
    StdErrModel,
    SystemMessage,
)


class LocalCodeInterpreter(BaseCodeInterpreter):
    """基于本地 Jupyter 内核的代码解释器。"""
    def __init__(
        self,
        task_id: str,
        work_dir: str,
        notebook_serializer: NotebookSerializer,
    ):
        super().__init__(task_id, work_dir, notebook_serializer)
        self.km, self.kc = None, None
        self.interrupt_signal = False

    async def initialize(self):
        logger.info("初始化本地内核")
        self.km, self.kc = await self._start_kernel_with_retry()
        self._pre_execute_code()

    async def _start_kernel_with_retry(self, max_retries: int = 3):
        """启动 Jupyter 内核，失败时自动清理残留进程和运行时文件后重试。

        ZeroMQ 在 Windows 上可能在异常退出后残留 socket 资源，
        通过重试 + 运行时文件清理来绕过。
        """
        import asyncio, glob, shutil

        last_error = None
        for attempt in range(max_retries):
            try:
                self._cleanup_stale_runtime_files()
                km, kc = jupyter_client.manager.start_new_kernel(
                    kernel_name="python3"
                )
                # 等待内核就绪，超时则抛出
                kc.wait_for_ready(timeout=60)
                logger.info(f"Jupyter 内核启动成功 (attempt {attempt + 1})")
                return km, kc
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Jupyter 内核启动失败 (attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    # 等待 ZMQ 资源释放
                    await asyncio.sleep(3 * (attempt + 1))
                    self._kill_orphan_kernels()

        raise RuntimeError(
            f"Jupyter 内核启动失败，已重试 {max_retries} 次。"
            f"最后错误: {last_error}。建议重启系统。"
        )

    @staticmethod
    def _cleanup_stale_runtime_files():
        """清理 Jupyter 残留的运行时文件（连接文件、pid 等）。"""
        import glob, os
        for pattern in [
            os.path.expanduser("~/.jupyter/runtime/*.json"),
            os.path.join(os.environ.get("TEMP", "/tmp"), "jupyter", "*"),
        ]:
            for f in glob.glob(pattern):
                try:
                    os.remove(f)
                except OSError:
                    pass

    @staticmethod
    def _kill_orphan_kernels():
        """杀掉残留的 ipykernel 进程（不碰主进程）。"""
        import subprocess
        try:
            # 只杀 ipykernel 相关进程，不杀所有 python
            result = subprocess.run(
                ["wmic", "process", "where",
                 "name='python.exe' and commandline like '%ipykernel%'",
                 "get", "processid"],
                capture_output=True, text=True, timeout=10,
            )
            for line in result.stdout.splitlines():
                pid = line.strip()
                if pid.isdigit():
                    subprocess.run(
                        ["taskkill", "/F", "/PID", pid],
                        capture_output=True,
                    )
        except Exception:
            pass  # wmic/taskkill 不可用时静默跳过

    def _pre_execute_code(self):  # type: ignore[reportIncompatibleMethodOverride]
        init_code = (
            f"import os\n"
            f"work_dir = r'{self.work_dir}'\n"
            f"os.makedirs(work_dir, exist_ok=True)\n"
            f"os.chdir(work_dir)\n"
            f"print('当前工作目录:', os.getcwd())\n"
            # f"import matplotlib.pyplot as plt\n"
            # f"import matplotlib as mpl\n"
            # # 更完整的中文字体配置
            # f"plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'PingFang SC', 'Hiragino Sans GB', 'Heiti SC', 'DejaVu Sans', 'sans-serif']\n"
            # f"plt.rcParams['axes.unicode_minus'] = False\n"
            # f"plt.rcParams['font.family'] = 'sans-serif'\n"
            # f"mpl.rcParams['font.size'] = 12\n"
            # f"mpl.rcParams['axes.labelsize'] = 12\n"
            # f"mpl.rcParams['xtick.labelsize'] = 10\n"
            # f"mpl.rcParams['ytick.labelsize'] = 10\n"
            # # 设置DPI以获得更清晰的显示
        )
        self.execute_code_(init_code)

    async def execute_code(self, code: str) -> tuple[str, bool, str]:
        logger.info(f"执行代码: {code}")
        #  添加代码到notebook
        self.notebook_serializer.add_code_cell_to_notebook(code)

        text_to_gpt: list[str] = []
        content_to_display: list[OutputItem] | None = []
        error_occurred: bool = False
        error_message: str = ""

        await redis_manager.publish_message(
            self.task_id,
            SystemMessage(content="开始执行代码"),
        )
        # 执行 Python 代码
        logger.info("开始在本地执行代码...")
        execution = self.execute_code_(code)
        logger.info("代码执行完成，开始处理结果...")

        await redis_manager.publish_message(
            self.task_id,
            SystemMessage(content="代码执行完成"),
        )

        for mark, out_str in execution:
            if mark in ("stdout", "execute_result_text", "display_text"):
                text_to_gpt.append(self._truncate_text(f"[{mark}]\n{out_str}"))
                #  添加text到notebook
                content_to_display.append(
                    ResultModel(res_type="result", format="text", msg=out_str)
                )
                self.notebook_serializer.add_code_cell_output_to_notebook(out_str)

            elif mark in (
                "execute_result_png",
                "execute_result_jpeg",
                "display_png",
                "display_jpeg",
            ):
                # TODO: 视觉模型解释图像
                text_to_gpt.append(f"[{mark} 图片已生成，内容为 base64，未展示]")

                #  添加image到notebook
                if "png" in mark:
                    self.notebook_serializer.add_image_to_notebook(out_str, "image/png")
                    content_to_display.append(
                        ResultModel(res_type="result", format="png", msg=out_str)
                    )
                else:
                    self.notebook_serializer.add_image_to_notebook(
                        out_str, "image/jpeg"
                    )
                    content_to_display.append(
                        ResultModel(res_type="result", format="jpeg", msg=out_str)
                    )

            elif mark == "error":
                error_occurred = True
                error_message = self.delete_color_control_char(out_str)
                error_message = self._truncate_text(error_message)
                logger.error(f"执行错误: {error_message}")
                text_to_gpt.append(error_message)
                #  添加error到notebook
                self.notebook_serializer.add_code_cell_error_to_notebook(out_str)
                content_to_display.append(StdErrModel(msg=out_str))

        logger.info(f"text_to_gpt: {text_to_gpt}")
        combined_text = "\n".join(text_to_gpt)

        await self._push_to_websocket(content_to_display)

        return (
            combined_text,
            error_occurred,
            error_message,
        )

    def execute_code_(self, code, total_timeout: int = 300) -> list[tuple[str, str]]:
        """执行代码并收集输出。

        Args:
            code: 要执行的 Python 代码。
            total_timeout: 总超时秒数（默认 300 秒），超时后强制中断内核。

        Returns:
            (mark, content) 元组列表。
        """
        assert self.kc is not None
        assert self.km is not None
        self.kc.execute(code)
        logger.info(f"执行代码: {code}")
        start_time = time.time()
        msg_list = []
        while True:
            try:
                iopub_msg = self.kc.get_iopub_msg(timeout=1)
                msg_list.append(iopub_msg)
                if (
                    iopub_msg["msg_type"] == "status"
                    and iopub_msg["content"].get("execution_state") == "idle"
                ):
                    break
            except Exception:
                if self.interrupt_signal:
                    self.km.interrupt_kernel()
                    self.interrupt_signal = False
                if time.time() - start_time > total_timeout:
                    logger.error(f"代码执行超时（{total_timeout}秒），强制中断内核")
                    self.km.interrupt_kernel()
                    msg_list.append({
                        "msg_type": "stream",
                        "content": {"name": "stdout", "text": f"\n[错误] 代码执行超时（{total_timeout}秒），已强制中断\n"}
                    })
                    break
                continue

        all_output: list[tuple[str, str]] = []
        for iopub_msg in msg_list:
            if iopub_msg["msg_type"] == "stream":
                if iopub_msg["content"].get("name") == "stdout":
                    output = iopub_msg["content"]["text"]
                    all_output.append(("stdout", output))
            elif iopub_msg["msg_type"] == "execute_result":
                if "data" in iopub_msg["content"]:
                    if "text/plain" in iopub_msg["content"]["data"]:
                        output = iopub_msg["content"]["data"]["text/plain"]
                        all_output.append(("execute_result_text", output))
                    if "text/html" in iopub_msg["content"]["data"]:
                        output = iopub_msg["content"]["data"]["text/html"]
                        all_output.append(("execute_result_html", output))
                    if "image/png" in iopub_msg["content"]["data"]:
                        output = iopub_msg["content"]["data"]["image/png"]
                        all_output.append(("execute_result_png", output))
                    if "image/jpeg" in iopub_msg["content"]["data"]:
                        output = iopub_msg["content"]["data"]["image/jpeg"]
                        all_output.append(("execute_result_jpeg", output))
            elif iopub_msg["msg_type"] == "display_data":
                if "data" in iopub_msg["content"]:
                    if "text/plain" in iopub_msg["content"]["data"]:
                        output = iopub_msg["content"]["data"]["text/plain"]
                        all_output.append(("display_text", output))
                    if "text/html" in iopub_msg["content"]["data"]:
                        output = iopub_msg["content"]["data"]["text/html"]
                        all_output.append(("display_html", output))
                    if "image/png" in iopub_msg["content"]["data"]:
                        output = iopub_msg["content"]["data"]["image/png"]
                        all_output.append(("display_png", output))
                    if "image/jpeg" in iopub_msg["content"]["data"]:
                        output = iopub_msg["content"]["data"]["image/jpeg"]
                        all_output.append(("display_jpeg", output))
            elif iopub_msg["msg_type"] == "error":
                # TODO: 正确返回格式
                if "traceback" in iopub_msg["content"]:
                    output = "\n".join(iopub_msg["content"]["traceback"])
                    cleaned_output = self.delete_color_control_char(output)
                    all_output.append(("error", cleaned_output))
        return all_output

    async def get_created_images(self, section: str) -> list[str]:
        """获取新创建的图片列表，递归扫描 work_dir 及其子目录（如 figures/）。"""
        current_images = set()
        for root, _dirs, files in os.walk(self.work_dir):
            for file in files:
                if file.endswith((".png", ".jpg", ".jpeg")):
                    # 返回相对于 work_dir 的路径，如 "figures/figure1.png"
                    rel_path = os.path.relpath(os.path.join(root, file), self.work_dir)
                    current_images.add(rel_path)

        # 计算新增的图片
        new_images = current_images - self.last_created_images

        # 更新last_created_images为当前的图片集合
        self.last_created_images = current_images

        logger.info(f"新创建的图片列表: {new_images}")
        return list(new_images)

    async def cleanup(self):
        # 关闭内核
        assert self.kc is not None
        assert self.km is not None
        self.kc.shutdown()
        logger.info("关闭内核")
        self.km.shutdown_kernel()

    def send_interrupt_signal(self):
        self.interrupt_signal = True

    def restart_jupyter_kernel(self):
        """Restart the Jupyter kernel and recreate the work directory."""
        assert self.kc is not None
        self.kc.shutdown()
        self.km, self.kc = jupyter_client.manager.start_new_kernel(
            kernel_name="python3"
        )
        self.interrupt_signal = False
        self._create_work_dir()

    def _create_work_dir(self):
        """Ensure the working directory exists after a restart."""
        os.makedirs(self.work_dir, exist_ok=True)
