"""postprocess.py 纯函数单元测试。

用简单假数据验证输入输出，不依赖 LLM / Redis / 外部服务。
"""

import json
import types

from app.core.postprocess import (
    _assign_images_to_sections,
    _extract_figure_metadata,
    _extract_method_contract,
    _fix_figure_captions,
    _remove_duplicate_headings,
    _remove_duplicate_images,
    _renumber_tables_figures,
    _scan_work_dir_images,
)


def test_remove_duplicate_images_keeps_first_only():
    content = "![a](figures/x.png)\n![b](figures/x.png)\n![c](figures/y.png)"
    result = _remove_duplicate_images(content)
    assert "![a](figures/x.png)" in result
    assert result.count("figures/x.png") == 1
    assert "![c](figures/y.png)" in result


def test_remove_duplicate_images_no_duplicate():
    content = "![a](figures/x.png)\n![b](figures/y.png)"
    assert _remove_duplicate_images(content) == content


def test_renumber_tables_figures():
    content = "**图2: 第二张**\n**表3: 第三表**\n**图1: 第一张**"
    result = _renumber_tables_figures(content)
    assert "**图1: 第二张**" in result
    assert "**表1: 第三表**" in result
    assert "**图2: 第一张**" in result


def test_remove_duplicate_headings():
    content = "# 四、问题重述\n## 4.1 背景\n## 4.1 背景\n正文"
    result = _remove_duplicate_headings(content)
    assert result == "# 四、问题重述\n## 4.1 背景\n正文"


def test_extract_method_contract_found():
    modeler = types.SimpleNamespace(
        questions_solution={"ques1": "使用GMM进行聚类分析"}
    )
    result = _extract_method_contract(modeler)
    assert "高斯混合模型(GMM)" in result
    assert result.startswith("全文统一方法名:")


def test_extract_method_contract_empty():
    modeler = types.SimpleNamespace(questions_solution={})
    result = _extract_method_contract(modeler)
    assert result == "(未从建模方案中提取到方法契约)"


def test_scan_work_dir_images(tmp_path):
    (tmp_path / "a.png").write_text("")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.jpg").write_text("")
    (tmp_path / "c.txt").write_text("")
    result = _scan_work_dir_images(str(tmp_path))
    assert result == ["a.png", "sub/b.jpg"]


def test_assign_images_to_sections():
    section_order = ["eda", "ques1", "ques2", "sensitivity_analysis"]
    all_images = [
        "figures/eda_overview.png",
        "figures/figure1_x.png",
        "figures/figure2_y.png",
        "figures/other.png",
    ]
    result = _assign_images_to_sections(section_order, all_images)
    assert result["eda"] == ["figures/eda_overview.png"]
    assert result["ques1"] == ["figures/figure1_x.png"]
    assert result["ques2"] == ["figures/figure2_y.png"]
    assert result["sensitivity_analysis"] == ["figures/other.png"]


def test_extract_figure_metadata_from_notebook(tmp_path):
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "source": [
                    "fig.suptitle('各因素权重分布')\n",
                    "plt.savefig('figures/weights.png')\n",
                ],
                "outputs": [],
            }
        ]
    }
    (tmp_path / "notebook.ipynb").write_text(
        json.dumps(nb, ensure_ascii=False), encoding="utf-8"
    )
    metadata = _extract_figure_metadata(str(tmp_path))
    assert metadata["weights.png"] == "各因素权重分布"


def test_fix_figure_captions_uses_vl_cache(tmp_path):
    (tmp_path / "vl_verification.json").write_text(
        json.dumps({"figures/x.png": "真实内容描述"}, ensure_ascii=False),
        encoding="utf-8",
    )
    content = "![图](figures/x.png)\n**图1: 编造的描述**"
    result = _fix_figure_captions(content, str(tmp_path))
    assert "**图1: 真实内容描述**" in result
    assert "编造的描述" not in result


def test_fix_figure_captions_no_metadata_is_noop(tmp_path):
    content = "![图](figures/x.png)\n**图1: 编造的描述**"
    assert _fix_figure_captions(content, str(tmp_path)) == content
