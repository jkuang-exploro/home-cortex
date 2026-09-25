"""Model discovery and cached bare-provider construction."""
from __future__ import annotations

from typing import Any

import httpx
from fastapi import Request

from ..providers.base import ModelProvider, model_provider_from_settings
from .dependencies import request_settings
from .errors import APIError


async def list_bare_models(request: Request) -> list[dict[str, str]]:
    listed = getattr(request.app.state, "bare_models", None)
    if listed is not None:
        return [dict(item) for item in listed]
    settings = request_settings(request)
    provider = getattr(settings, "llm_provider", "ollama")
    if provider == "openrouter":
        name = getattr(settings, "openrouter_model", None)
        return [{"id": name, "owned_by": "openrouter"}] if name else []
    if provider == "llamacpp":
        name = getattr(settings, "local_llm_model", None)
        return [{"id": name, "owned_by": "llamacpp"}] if name else []
    configured = getattr(settings, "ollama_model", None)
    url = getattr(settings, "ollama_url", None)
    models: list[dict[str, str]] = []
    if url:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(f"{str(url).rstrip('/')}/api/tags")
                response.raise_for_status()
                for item in response.json().get("models") or []:
                    name = item.get("name") if isinstance(item, dict) else None
                    if name:
                        models.append({"id": str(name), "owned_by": "ollama"})
        except Exception:
            models = []
    if configured and not any(item["id"] == configured for item in models):
        models.insert(0, {"id": configured, "owned_by": "ollama"})
    return models


async def require_bare_model(request: Request, model_id: str) -> None:
    names = {item["id"] for item in await list_bare_models(request)}
    if model_id not in names:
        raise APIError(404, "model_not_found", f"Model {model_id!r} was not found")


def bare_model_provider(request: Request, model_id: str) -> ModelProvider:
    cached = getattr(request.app.state, "bare_language_models", None)
    if not isinstance(cached, dict):
        cached = {}
        request.app.state.bare_language_models = cached
    if model_id not in cached:
        cached[model_id] = model_provider_from_settings(
            request_settings(request), model_id
        )
    return cached[model_id]
