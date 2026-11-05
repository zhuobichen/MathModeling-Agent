"""read_file 路径越界回归测试。

对应审计项：`backend/app/tools/handlers/read_file.py` 直接拼接 work_dir + filename，
可被 `..`、绝对路径、符号链接逃出工作目录。这里锁定修复后的边界行为。
"""

import os

import pytest

from app.tools.handlers.read_file import _resolve_within


def test_normal_file_is_allowed(tmp_path):
    target = tmp_path / "in.txt"
    target.write_text("hello", encoding="utf-8")
    resolved = _resolve_within(str(tmp_path), "in.txt")
    assert resolved is not None
    assert resolved == target.resolve()


def test_nested_file_is_allowed(tmp_path):
    nested = tmp_path / "sub" / "deep"
    nested.mkdir(parents=True)
    (nested / "data.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    assert _resolve_within(str(tmp_path), "sub/deep/data.csv") is not None


@pytest.mark.parametrize(
    "name",
    [
        "../outside.txt",
        "../../etc/passwd",
        "sub/../../outside.txt",
        "/etc/passwd",
        "C:\\Windows\\win.ini",
    ],
)
def test_escape_attempts_are_rejected(tmp_path, name):
    assert _resolve_within(str(tmp_path), name) is None


@pytest.mark.parametrize("name", ["", "   "])
def test_empty_name_is_rejected(tmp_path, name):
    assert _resolve_within(str(tmp_path), name) is None


def test_symlink_escape_is_rejected(tmp_path):
    outside = tmp_path.parent / "outside_secret.txt"
    outside.write_text("secret", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError, AttributeError):
        pytest.skip("当前平台/权限不允许创建符号链接")
    assert _resolve_within(str(tmp_path), "link.txt") is None
