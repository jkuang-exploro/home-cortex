"""Local OpenAI-compatible provider contract without a live model server."""

import json

import httpx
import pytest

from home_cortex.config import Settings
from home_cortex.providers.base import model_provider_from_settings
from home_cortex.providers.base import ModelProviderError
from home_cortex.providers.llamacpp import LlamaCppService


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_llamacpp_factory_and_structured_request():
    settings = Settings(_env_file=None, llm_provider="llamacpp", local_llm_model="local-qwen")
    provider = model_provider_from_settings(settings)
    assert isinstance(provider, LlamaCppService)
    assert provider.base_url == "http://llama-server:8080/v1"
    await provider.close()

    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": '{"requires_fact":false,"request":null}'}}],
            "usage": {"prompt_tokens": 30, "completion_tokens": 12},
        })

    provider = LlamaCppService("http://llama-server:8080/v1", "local-qwen", client=_client(handler))
    result = await provider.plan_semantic_fact(
        [{"role": "user", "content": "hello"}], {}, {"type": "object"},
        household_now="2026-09-24T12:00:00-07:00",
    )
    assert result == {"requires_fact": False, "request": None}
    assert captured["body"]["model"] == "local-qwen"
    assert captured["body"]["response_format"]["type"] == "json_schema"
    assert "provider" not in captured["body"]
    assert captured["auth"] is None
    assert provider.last_planner_runtime["prompt_eval_count"] == 30
    await provider.close()


@pytest.mark.asyncio
async def test_llamacpp_stream_and_unavailable_server():
    def handler(request):
        if request.method != "POST":
            raise httpx.ConnectError("unavailable")
        return httpx.Response(200, text=(
            'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":" there"}}]}\n\n'
            'data: [DONE]\n\n'
        ))

    provider = LlamaCppService("http://llama-server:8080/v1", "local-qwen", client=_client(handler))
    chunks = [part async for part in provider.stream_chat([{"role": "user", "content": "hello"}])]
    assert chunks == ["Hello", " there"]
    await provider.close()

    def unavailable(_request):
        raise httpx.ConnectError("unavailable")

    provider = LlamaCppService("http://llama-server:8080/v1", "local-qwen", client=_client(unavailable))
    with pytest.raises(ModelProviderError, match="unavailable"):
        await provider.chat([{"role": "user", "content": "hello"}])
    await provider.close()


def test_llamacpp_requires_logical_model():
    with pytest.raises(ValueError, match="LOCAL_LLM_MODEL"):
        Settings(_env_file=None, llm_provider="llamacpp", local_llm_model=None)


@pytest.mark.asyncio
async def test_llamacpp_timeout_and_malformed_response():
    def timeout(_request):
        raise httpx.ReadTimeout("late")

    provider = LlamaCppService("http://llama-server:8080/v1", "local-qwen", client=_client(timeout))
    with pytest.raises(ModelProviderError, match="timed out") as captured:
        await provider.chat([{"role": "user", "content": "hello"}])
    assert captured.value.status_code == 504
    await provider.close()

    provider = LlamaCppService(
        "http://llama-server:8080/v1", "local-qwen",
        client=_client(lambda _request: httpx.Response(200, json={"choices": []})),
    )
    with pytest.raises(ValueError, match="choice"):
        await provider.chat([{"role": "user", "content": "hello"}])
    await provider.close()
