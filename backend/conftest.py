"""pytest 根目录配置：确保 backend/ 在 sys.path 中，使测试可 import `app` 包。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
