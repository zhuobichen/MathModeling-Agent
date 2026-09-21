"""LLM 工厂模块，根据配置创建 5 个 Agent 使用的 LLM 实例。"""

from app.config.setting import settings
from app.core.llm.llm import LLM
from app.utils.log_util import logger


class LLMFactory:
    """LLM 工厂类，创建 Parser、Modeler、Coder、Writer、Reviewer 的 LLM 实例。"""

    task_id: str

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id

    def get_all_llms(self) -> tuple[LLM, LLM, LLM, LLM, LLM]:
        """创建所有 5 个 Agent 的 LLM 实例。

        Returns:
            (parser_llm, modeler_llm, coder_llm, writer_llm, reviewer_llm) 元组。
        """
        parser_llm = LLM(
            api_key=settings.PARSER_API_KEY,
            model=settings.PARSER_MODEL,
            base_url=settings.PARSER_BASE_URL,
            task_id=self.task_id,
            max_tokens=settings.PARSER_MAX_TOKENS,
        )

        modeler_llm = LLM(
            api_key=settings.MODELER_API_KEY,
            model=settings.MODELER_MODEL,
            base_url=settings.MODELER_BASE_URL,
            task_id=self.task_id,
            max_tokens=settings.MODELER_MAX_TOKENS,
        )

        coder_llm = LLM(
            api_key=settings.CODER_API_KEY,
            model=settings.CODER_MODEL,
            base_url=settings.CODER_BASE_URL,
            task_id=self.task_id,
            max_tokens=settings.CODER_MAX_TOKENS,
        )

        writer_llm = LLM(
            api_key=settings.WRITER_API_KEY,
            model=settings.WRITER_MODEL,
            base_url=settings.WRITER_BASE_URL,
            task_id=self.task_id,
            max_tokens=settings.WRITER_MAX_TOKENS,
        )

        reviewer_llm = LLM(
            api_key=settings.REVIEWER_API_KEY,
            model=settings.REVIEWER_MODEL,
            base_url=settings.REVIEWER_BASE_URL,
            task_id=self.task_id,
            max_tokens=settings.REVIEWER_MAX_TOKENS,
        )

        return parser_llm, modeler_llm, coder_llm, writer_llm, reviewer_llm

    def get_fallback_llms(self) -> dict[str, LLM | None]:
        """创建所有 Agent 的 Fallback LLM 实例。

        Returns:
            以 Agent 名称为键、LLM 实例或 None 为值的字典。
        """
        fallbacks: dict[str, LLM | None] = {}

        agent_configs = [
            ("parser", settings.FALLBACK_PARSER_API_KEY, settings.FALLBACK_PARSER_MODEL,
             settings.FALLBACK_PARSER_BASE_URL, settings.FALLBACK_PARSER_MAX_TOKENS),
            ("modeler", settings.FALLBACK_MODELER_API_KEY, settings.FALLBACK_MODELER_MODEL,
             settings.FALLBACK_MODELER_BASE_URL, settings.FALLBACK_MODELER_MAX_TOKENS),
            ("coder", settings.FALLBACK_CODER_API_KEY, settings.FALLBACK_CODER_MODEL,
             settings.FALLBACK_CODER_BASE_URL, settings.FALLBACK_CODER_MAX_TOKENS),
            ("writer", settings.FALLBACK_WRITER_API_KEY, settings.FALLBACK_WRITER_MODEL,
             settings.FALLBACK_WRITER_BASE_URL, settings.FALLBACK_WRITER_MAX_TOKENS),
        ]

        for name, api_key, model, base_url, max_tokens in agent_configs:
            if api_key and model:
                fallbacks[name] = LLM(
                    api_key=api_key,
                    model=model,
                    base_url=base_url,
                    task_id=self.task_id,
                    max_tokens=max_tokens,
                )
            else:
                fallbacks[name] = None

        return fallbacks

    def get_evaluator_llm(self) -> LLM | None:
        """创建评估器的 LLM 实例。"""
        if settings.EVALUATOR_API_KEY and settings.EVALUATOR_MODEL:
            return LLM(
                api_key=settings.EVALUATOR_API_KEY,
                model=settings.EVALUATOR_MODEL,
                base_url=settings.EVALUATOR_BASE_URL,
                task_id=self.task_id,
            )
        return None

    # ── 按角色取模型（组件化后的统一入口）──

    _ROLE_SETTINGS = {
        "parser": ("PARSER_API_KEY", "PARSER_MODEL", "PARSER_BASE_URL",
                   "PARSER_MAX_TOKENS"),
        "modeler": ("MODELER_API_KEY", "MODELER_MODEL", "MODELER_BASE_URL",
                    "MODELER_MAX_TOKENS"),
        "coder": ("CODER_API_KEY", "CODER_MODEL", "CODER_BASE_URL",
                  "CODER_MAX_TOKENS"),
        "writer": ("WRITER_API_KEY", "WRITER_MODEL", "WRITER_BASE_URL",
                   "WRITER_MAX_TOKENS"),
        "reviewer": ("REVIEWER_API_KEY", "REVIEWER_MODEL", "REVIEWER_BASE_URL",
                     "REVIEWER_MAX_TOKENS"),
    }

    def get_llm_by_role(self, role: str | None) -> LLM | None:
        """按 `Component.model_role` 取对应的 LLM。

        与 `get_all_llms()` 的区别：**未配置 API Key 时返回 None**，
        使上层能给出"该角色的模型未配置"这种可诊断的错误，
        而不是发出一个必然失败的请求。

        role 为 None 时退回到 writer（最通用的角色），全无配置则返回 None。
        """
        role = role or "writer"
        cfg = self._ROLE_SETTINGS.get(role)
        if cfg is None:
            logger.warning(f"未知的 model_role: {role!r}，回退到 writer")
            cfg = self._ROLE_SETTINGS["writer"]
        key_name, model_name, url_name, tokens_name = cfg
        api_key = getattr(settings, key_name, None)
        model = getattr(settings, model_name, None)
        if not api_key or not model:
            logger.warning(
                f"model_role={role!r} 未配置（{key_name} / {model_name} 缺失）"
            )
            return None
        return LLM(
            api_key=api_key,
            model=model,
            base_url=getattr(settings, url_name, None),
            task_id=self.task_id,
            max_tokens=getattr(settings, tokens_name, None),
        )
