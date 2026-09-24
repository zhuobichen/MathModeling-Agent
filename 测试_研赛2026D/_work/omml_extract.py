# -*- coding: utf-8 -*-
"""把 .docx 里的 OMML 公式抽成可读文本，插回正文的对应位置。

`python-docx` 的 `paragraph.text` 只取 `w:t`，不取 `m:oMath` 里的数学，
所以题目里所有公式都会静默消失——正文看起来是"其中， 表示运输无人机机型 m"
这种带空档的句子。数模题没有公式就没法做，必须先把它们捞出来。

做法：直接解压 docx 读 `word/document.xml`，**递归**遍历段落子树：
遇到 `m:oMath` 就把整棵子树转成文本并且不再往下走（否则公式里的 `w:t`
会被重复取一遍），遇到 `w:t` 就取文本。
"""
from __future__ import annotations

import re
import sys
import zipfile

import xml.etree.ElementTree as ET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"

SYM = {
    "⋅": "·", "×": "×", "÷": "÷", "−": "-", "≤": "≤", "≥": "≥",
    "≠": "≠", "∞": "∞", "∑": "∑", "∫": "∫", "∂": "∂", "√": "√",
    "α": "α", "β": "β", "γ": "γ", "δ": "δ", "ε": "ε", "η": "η",
    "θ": "θ", "κ": "κ", "λ": "λ", "μ": "μ", "π": "π", "ρ": "ρ",
    "σ": "σ", "τ": "τ", "φ": "φ", "χ": "χ", "ψ": "ψ", "ω": "ω",
    "Δ": "Δ", "∇": "∇", "∈": "∈", "→": "→", "⇒": "⇒",
}


def _kid(e, tag):
    return e.find(tag) if e is not None else None


def _raw(e):
    return "".join(t.text or "" for t in e.iter(W + "t")) if e is not None else ""


def omml(e):
    """OMML 元素 → 近似 LaTeX 的文本。只求可读，不求可编译。"""
    tag = e.tag
    if tag == M + "f":
        return f"({omml(_kid(e, M+'num'))})/({omml(_kid(e, M+'den'))})"
    if tag == M + "sSub":
        return f"{omml(_kid(e, M+'e'))}_{{{omml(_kid(e, M+'sub'))}}}"
    if tag == M + "sSup":
        return f"{omml(_kid(e, M+'e'))}^{{{omml(_kid(e, M+'sup'))}}}"
    if tag == M + "sSubSup":
        return (f"{omml(_kid(e, M+'e'))}_{{{omml(_kid(e, M+'sub'))}}}"
                f"^{{{omml(_kid(e, M+'sup'))}}}")
    if tag == M + "rad":
        d = omml(_kid(e, M + "deg"))
        return (f"sqrt[{d}]" if d else "sqrt") + f"({omml(_kid(e, M+'e'))})"
    if tag == M + "d":
        return "(" + omml(_kid(e, M + "e")) + ")"
    if tag == M + "bar":
        return "bar(" + omml(_kid(e, M + "e")) + ")"
    if tag == M + "nary":
        return _raw(e)
    if tag == M + "t":
        return e.text or ""
    return "".join(omml(c) for c in e)


def omath_text(e):
    s = omml(e)
    for a, b in SYM.items():
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s).strip()


def _walk(node, out):
    """递归收集文本；碰到 oMath 整棵取走、不再下探。"""
    for ch in node:
        if ch.tag == M + "oMath":
            out.append(" $" + omath_text(ch) + "$ ")
        elif ch.tag == W + "t":
            out.append(ch.text or "")
        elif ch.tag in (W + "br", W + "cr"):
            out.append("\n")
        else:
            _walk(ch, out)


def para_text(p):
    out = []
    _walk(p, out)
    return re.sub(r"[ \t]+", " ", "".join(out)).strip()


def extract(path):
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("word/document.xml"))
    body = root.find(W + "body")
    lines = []
    for p in body.iter(W + "p"):
        t = para_text(p)
        if t:
            lines.append(t)
    return "\n".join(lines)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(extract(sys.argv[1]))
