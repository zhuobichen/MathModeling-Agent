"""应用配置模块，基于 pydantic-settings 管理环境变量和全局配置。"""

from pydantic import BeforeValidator
from pydantic_settings import BaseSettings, SettingsConfigDict
import os
from typing import Annotated, Optional

# 优先加载私密配置（.env.private），如果存在的话
_private_env = os.path.join(os.path.dirname(__file__), "..", "..", ".env.private")
_private_env = os.path.abspath(_private_env)
if os.path.exists(_private_env):
    with open(_private_env, "r", encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _, _val = _line.partition("=")
                _key, _val = _key.strip(), _val.strip()
                if _key not in os.environ:
                    os.environ[_key] = _val


def parse_cors(value: str) -> list[str]:
    """将 CORS 配置字符串解析为 URL 列表。"""
    if value == "*":
        return ["*"]
    if "," in value:
        return [url.strip() for url in value.split(",")]
    return [value]


class Settings(BaseSettings):
    """全局应用配置，从环境变量和 .env 文件加载。"""

    ENV: str = "dev"

    # ----- 5 Agent 主要 LLM 配置 -----
    PARSER_API_KEY: Optional[str] = None
    PARSER_MODEL: Optional[str] = None
    PARSER_BASE_URL: Optional[str] = None
    PARSER_MAX_TOKENS: Optional[int] = None

    MODELER_API_KEY: Optional[str] = None
    MODELER_MODEL: Optional[str] = None
    MODELER_BASE_URL: Optional[str] = None
    MODELER_MAX_TOKENS: Optional[int] = None

    CODER_API_KEY: Optional[str] = None
    CODER_MODEL: Optional[str] = None
    CODER_BASE_URL: Optional[str] = None
    CODER_MAX_TOKENS: Optional[int] = None

    WRITER_API_KEY: Optional[str] = None
    WRITER_MODEL: Optional[str] = None
    WRITER_BASE_URL: Optional[str] = None
    WRITER_MAX_TOKENS: Optional[int] = None

    REVIEWER_API_KEY: Optional[str] = None
    REVIEWER_MODEL: Optional[str] = None
    REVIEWER_BASE_URL: Optional[str] = None
    REVIEWER_MAX_TOKENS: Optional[int] = None

    # ----- Fallback LLM 配置 -----
    FALLBACK_PARSER_API_KEY: Optional[str] = None
    FALLBACK_PARSER_MODEL: Optional[str] = None
    FALLBACK_PARSER_BASE_URL: Optional[str] = None
    FALLBACK_PARSER_MAX_TOKENS: Optional[int] = None

    FALLBACK_MODELER_API_KEY: Optional[str] = None
    FALLBACK_MODELER_MODEL: Optional[str] = None
    FALLBACK_MODELER_BASE_URL: Optional[str] = None
    FALLBACK_MODELER_MAX_TOKENS: Optional[int] = None

    FALLBACK_CODER_API_KEY: Optional[str] = None
    FALLBACK_CODER_MODEL: Optional[str] = None
    FALLBACK_CODER_BASE_URL: Optional[str] = None
    FALLBACK_CODER_MAX_TOKENS: Optional[int] = None

    FALLBACK_WRITER_API_KEY: Optional[str] = None
    FALLBACK_WRITER_MODEL: Optional[str] = None
    FALLBACK_WRITER_BASE_URL: Optional[str] = None
    FALLBACK_WRITER_MAX_TOKENS: Optional[int] = None

    # ----- 评估器配置 -----
    EVALUATOR_API_KEY: Optional[str] = None
    EVALUATOR_MODEL: Optional[str] = None
    EVALUATOR_BASE_URL: Optional[str] = None

    # ----- Feedback Rerun 配置 -----
    MAX_FEEDBACK_ROUNDS: int = 2
    EVALUATION_THRESHOLD: float = 0.6

    MAX_CHAT_TURNS: Optional[int] = None
    MAX_RETRIES: int = 3
    MAX_MODELER_RETRIES: int = 3
    E2B_API_KEY: Optional[str] = None
    LOG_LEVEL: str = "DEBUG"
    DEBUG: bool = True
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_MAX_CONNECTIONS: int = 10
    CORS_ALLOW_ORIGINS: Annotated[list[str] | str, BeforeValidator(parse_cors)] = "*"
    SERVER_HOST: str = "http://localhost:8000"
    OPENALEX_EMAIL: Optional[str] = None
    OPENALEX_API_KEY: Optional[str] = None

    # ----- Web Search 配置 -----
    TAVILY_API_KEY: Optional[str] = None
    SEARCH_CACHE_TTL: int = 86400
    SEARCH_ENABLED: bool = True

    # ----- RAG 知识库配置 -----
    RAG_ENABLED: bool = False
    RAG_DB_PATH: str = "data/chromadb"
    RAG_TOP_K: int = 5
    RAG_EMBEDDING_MODEL: str = "BAAI/bge-m3"
    RAG_RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"

    # ----- 识图模型配置（PDF 图片识别）-----
    VISION_API_KEY: Optional[str] = None
    VISION_MODEL: Optional[str] = None
    VISION_BASE_URL: Optional[str] = None

    # ----- HIL 人机协作配置 -----
    HIL_ENABLED: bool = True
    HIL_TIMEOUT: int = 300
    HIL_CHECKPOINTS: dict = {
        "model_selection": True,
        "paper_review": True,
    }

    model_config = SettingsConfigDict(
        env_file=".env.dev",
        env_file_encoding="utf-8",
        extra="allow",
    )

    @classmethod
    def from_env(cls, env: str | None = None):
        """根据环境名称加载对应配置。"""
        env = env or os.getenv("ENV", "dev")
        env_file = f".env.{env.lower()}"
        return cls(_env_file=env_file, _env_file_encoding="utf-8")  # type: ignore[call-arg]


settings = Settings()
