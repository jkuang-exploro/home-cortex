"""Hosted OpenRouter adapter over the shared OpenAI-compatible transport."""

import httpx

from .openai_compatible import OpenAICompatibleService


class OpenRouterService(OpenAICompatibleService):
    """OpenRouter configuration over the shared OpenAI-compatible transport."""

    def __init__(
        self, base_url: str, model: str, *, api_key: str,
        http_referer: str | None = None, app_title: str = "home-cortex",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("OpenRouter API key cannot be empty")
        super().__init__(
            base_url, model, api_key=api_key, http_referer=http_referer,
            app_title=app_title, provider_parameters=True, client=client,
        )
