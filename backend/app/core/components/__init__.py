"""内置组件包。

导入本包即完成组件注册（幂等）。
"""

from app.core.components.builtin import register_builtins

__all__ = ["register_builtins"]
