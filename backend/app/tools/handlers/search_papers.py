"""search_papers 工具处理器。"""


async def _handle_search_papers(args: dict, scholar) -> str:
    """search_papers 工具处理器。"""
    query = args.get("query", "")
    if not query:
        return "错误: 未指定搜索查询"
    try:
        papers = await scholar.search_papers(query)
        return scholar.papers_to_str(papers) if papers else "未找到相关文献"
    except Exception as e:
        return f"文献搜索失败: {e}"
