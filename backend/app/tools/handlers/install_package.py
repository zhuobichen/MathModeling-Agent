"""install_package 工具处理器。"""

from app.utils.log_util import logger


async def _handle_install_package(args: dict) -> str:
    """install_package 工具处理器。"""
    import subprocess, sys
    package = args.get("package", "")
    if not package:
        return "错误: 未指定包名"
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            logger.info(f"pip install {package} 成功")
            return f"成功安装 {package}\n{result.stdout[-500:]}"
        else:
            return f"安装失败: {result.stderr[-500:]}"
    except Exception as e:
        return f"安装异常: {e}"
