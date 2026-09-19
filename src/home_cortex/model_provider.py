"""Provider-neutral language-model contract and deployment factory."""
from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, Protocol


class ModelMessage(Protocol):
    """Response message shape used by the bounded model loop."""

    content: str | None
    tool_calls: Sequence[Any] | None

    def model_dump(self, **kwargs: Any) -> dict[str, Any]: ...


class ModelResponse(Protocol):
    """Provider-neutral response envelope consumed by application code."""

    message: ModelMessage


class ModelProvider(Protocol):
    """The model behavior consumed by planners and the ordinary model loop."""

    model: str
    last_planner_runtime: dict[str, Any]

    async def chat(
        self, messages: Sequence[Mapping[str, Any]]
    ) -> ModelResponse: ...

    def stream_chat(
        self, messages: Sequence[Mapping[str, Any]]
    ) -> AsyncIterator[str]: ...

    async def chat_with_tools(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> ModelResponse: ...

    def stream_chat_with_tools(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> AsyncIterator[ModelResponse]: ...

    async def plan_item_mutation(self, messages: Sequence[Mapping[str, Any]]): ...

    async def plan_semantic_fact(
        self,
        messages: Sequence[Mapping[str, Any]],
        capabilities: Mapping[str, Any],
        output_schema: Mapping[str, Any],
        *,
        household_now: str,
    ) -> Mapping[str, Any]: ...

    async def plan_unified_semantic(
        self,
        messages: Sequence[Mapping[str, Any]],
        output_schema: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    async def close(self) -> None: ...


def model_provider_from_settings(
    settings: Any,
    model_name: str | None = None,
) -> ModelProvider:
    """Construct the configured provider without coupling either adapter."""
    if getattr(settings, "llm_provider", "ollama") == "openrouter":
        from .openrouter import OpenRouterService

        name = model_name or settings.openrouter_model
        secret = settings.openrouter_api_key
        if not name or secret is None:
            raise ValueError(
                "OPENROUTER_API_KEY and OPENROUTER_MODEL are required "
                "when LLM_PROVIDER=openrouter"
            )
        return OpenRouterService(
            settings.openrouter_base_url,
            name,
            api_key=secret.get_secret_value(),
            http_referer=settings.openrouter_http_referer,
            app_title=settings.openrouter_app_title,
        )

    from .ollama import OllamaService

    name = model_name or settings.ollama_model
    if not name:
        raise ValueError("OLLAMA_MODEL is required when LLM_PROVIDER=ollama")
    return OllamaService(settings.ollama_url, name)
