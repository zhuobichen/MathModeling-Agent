"""FastAPI 应用入口 —— 数学建模自动化系统 Web 服务。

启动方式:
    cd backend && uv run uvicorn app.server:app --host 0.0.0.0 --port 8000 --reload
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 离线模式：HuggingFace 网络受限
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理。"""
    from app.utils.log_util import logger
    logger.info("MathModel 服务启动")
    # 确保工作目录存在
    os.makedirs("project/work_dir", exist_ok=True)
    yield
    logger.info("MathModel 服务关闭")


app = FastAPI(
    title="MathModel 数学建模自动化系统",
    description="5-Agent 流水线: Parser → Modeler → Coder → Writer → Reviewer",
    version="0.2.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 路由注册
from app.routers.task_router import router as task_router
from app.routers.modeling_router import router as modeling_router

app.include_router(task_router)
app.include_router(modeling_router)


@app.get("/")
async def root():
    """服务首页。"""
    return {
        "service": "MathModel 数学建模数据分析自动化系统",
        "version": "0.2.0",
        "docs": "/docs",
        "endpoints": {
            "submit": "POST /modeling",
            "task_status": "GET /task/{task_id}",
            "list_tasks": "GET /tasks",
            "health": "GET /health",
        },
    }
