"""工具函数定义模块，为各 Agent 提供可用的工具 schema。"""

coder_tools = [
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": "This function allows you to execute Python code and retrieve the terminal output. If the code "
            "generates image output, the function will return the text '[image]'. The code is sent to a "
            "Jupyter kernel for execution. The kernel will remain active after execution, retaining all "
            "variables in memory."
            "You cannot show rich outputs like plots or images, but you can store them in the working directory and point the user to them. ",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "The code text"}
                },
                "required": ["code"],
                "additionalProperties": False,
            },
        },
    },
]

# 读取文件工具 schema
read_file_tool = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "读取工作目录中数据文件的前N行（默认20行），用于理解数据结构、列名和内容格式。支持CSV/Excel/TXT文件。",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "文件名，如 '附件.xlsx'"},
                "n_rows": {"type": "integer", "description": "读取行数，默认20"},
                "sheet_name": {"type": "string", "description": "Excel的sheet名称或序号，默认第一个sheet"},
            },
            "required": ["filename"],
            "additionalProperties": False,
        },
    },
}

# 安装Python包工具 schema
install_package_tool = {
    "type": "function",
    "function": {
        "name": "install_package",
        "description": "安装缺失的Python包。仅当import失败且确认包名正确时使用。安装后需要重新import。",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "package": {"type": "string", "description": "包名，如 'scikit-learn'"},
            },
            "required": ["package"],
            "additionalProperties": False,
        },
    },
}

# 文献搜索工具 schema (独立定义，供 tool_registry 注册)
search_papers_tool = {
    "type": "function",
    "function": {
        "name": "search_papers",
        "description": "搜索真实学术论文和参考文献，用于论文中引用。",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索查询"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

# ── Agent 工具配置 ──
AGENT_TOOL_CONFIG = {
    "ParserAgent": {
        "always": ["read_file"],
        "optional": ["search_knowledge"],
    },
    "ModelerAgent": {
        "always": ["search_knowledge"],
        "optional": ["search_web", "read_file"],
    },
    "CoderAgent": {
        "always": ["execute_code"],
        "optional": ["search_knowledge", "install_package", "read_file"],
    },
    "WriterAgent": {
        "always": ["search_papers"],
        "optional": ["search_knowledge", "search_web"],
    },
    "ReviewerAgent": {
        "always": [],
        "optional": ["search_papers", "search_knowledge"],
    },
}


# Web 搜索工具 schema
search_web_tool = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Search the web for real-world data, statistics, and facts. Returns structured data evidence with source URLs, units, time ranges, and regions.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query, be specific about what data you need",
                },
                "data_type": {
                    "type": "string",
                    "description": "Type of data expected: 'statistical', 'timeseries', 'categorical', or 'general'",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (1-10)",
                },
            },
            "required": ["query", "data_type", "max_results"],
            "additionalProperties": False,
        },
    },
}

# RAG 知识检索工具 schema
search_knowledge_tool = {
    "type": "function",
    "function": {
        "name": "search_knowledge",
        "description": "Search the knowledge base for mathematical modeling methods, code templates, and paper writing references.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query describing what knowledge you need",
                },
                "scope": {
                    "type": "string",
                    "description": "Knowledge scope: 'method' for modeling methods, 'code' for code templates, 'paper' for writing references",
                },
                "method_name": {
                    "type": "string",
                    "description": "Specific method name to search for (e.g. 'TOPSIS', 'AHP'), or empty string for general search",
                },
            },
            "required": ["query", "scope", "method_name"],
            "additionalProperties": False,
        },
    },
}

## writeragent tools
writer_tools = [
    {
        "type": "function",
        "function": {
            "name": "search_papers",
            "description": "Search for papers using a query string.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The query string"}
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
]
