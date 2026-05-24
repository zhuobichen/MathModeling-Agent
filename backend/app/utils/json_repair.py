"""JSON 修复工具 —— 尝试修复 LLM 输出的格式错误的 JSON。

提供多种修复策略，按成功率从高到低尝试：
1. 直接解析（去除 ```json 标记后）
2. 转义字符串值中的未转义引号
3. 正则提取键值对
"""

import json
import re


def repair_json(json_str: str) -> dict | None:
    """尝试修复 LLM 输出的格式错误的 JSON 字符串。

    Args:
        json_str: 可能格式有误的 JSON 字符串。

    Returns:
        解析成功返回 dict，否则返回 None。
    """
    json_str = json_str.replace("```json", "").replace("```", "").strip()

    # 策略 1：直接解析
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # 策略 2：转义字符串值内部的未转义引号
    try:
        fixed = re.sub(
            r'(?<=: ")(.*?)(?=",\s*\n\s*"|"\s*\n\s*})',
            lambda m: m.group(0).replace('"', '\\"'),
            json_str,
            flags=re.DOTALL,
        )
        return json.loads(fixed)
    except (json.JSONDecodeError, re.error):
        pass

    # 策略 3：正则提取键值对（兜底方案）
    try:
        pattern = r'"(\w+)"\s*:\s*"((?:[^"\\]|\\.|"(?!,\s*\n)|"(?!\s*\n\s*}))*)"'
        matches = re.findall(pattern, json_str, re.DOTALL)
        if matches:
            return {k: v.replace('\\"', '"') for k, v in matches}
    except re.error:
        pass

    return None
