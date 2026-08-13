"""search_knowledge 工具处理器。"""


async def _handle_search_knowledge(args: dict, retriever) -> str:
    """search_knowledge 工具处理器。"""
    query = args.get("query", "")
    scope = args.get("scope", "method")
    method_name = args.get("method_name", "")

    results = await retriever.retrieve(
        query=query,
        top_k=5,
        source_type=scope if scope != "all" else None,
        method_name=method_name if method_name else None,
    )
    if not results:
        return "未找到相关知识"

    return "\n\n".join(
        f"### {r.method_name}\n{r.content[:800]}" for r in results
    )
