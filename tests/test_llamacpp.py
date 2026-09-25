"""Local OpenAI-compatible provider contract without a live model server."""

import hashlib
import json

import httpx
import pytest

from home_cortex.config import Settings
from home_cortex.providers.base import model_provider_from_settings
from home_cortex.providers.base import ModelProviderError
from home_cortex.providers.llamacpp import LlamaCppService
from home_cortex.benchmark import environment


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


def test_benchmark_uses_active_llamacpp_container_identity(monkeypatch, tmp_path):
    model_path = tmp_path / "actual.gguf"
    model_path.write_bytes(b"model bytes")
    active = {
        "Config": {
            "Image": "local/llama.cpp:pinned",
            "Cmd": ["--model", "/models/actual.gguf", "--ctx-size", "16384",
                    "--n-gpu-layers", "99", "--parallel", "1"],
        },
        "Image": "sha256:active-image",
        "Mounts": [{"Source": str(tmp_path), "Destination": "/models"}],
        "NetworkSettings": {"Networks": {"cortex_default": {"IPAddress": "172.21.0.7"}}},
    }

    def docker_output(args):
        if args[1] == "ps":
            return "cortex-llama-server-1"
        if args[1] == "inspect":
            return json.dumps([active])
        if "org.opencontainers.image.revision" in args[-1]:
            return "pinned-commit"
        if "org.opencontainers.image.version" in args[-1]:
            return "pinned-version"
        raise AssertionError(args)

    monkeypatch.setattr(environment, "_docker_output", docker_output)
    monkeypatch.setattr(environment, "deployment_value", lambda _name: "stale-config")
    metadata = environment.llamacpp_metadata("http://172.21.0.7:8080/v1", "logical")

    assert metadata["image"] == "local/llama.cpp:pinned"
    assert metadata["image_id"] == "sha256:active-image"
    assert metadata["commit"] == "pinned-commit"
    assert metadata["model"]["sha256"] == hashlib.sha256(b"model bytes").hexdigest()
    assert metadata["model"]["filename"] == "actual.gguf"
    assert metadata["context_length"] == "16384"
    assert metadata["gpu_layers"] == "99"


def test_benchmark_does_not_invent_llamacpp_identity(monkeypatch):
    monkeypatch.setattr(environment, "_docker_output", lambda _args: "")
    monkeypatch.setattr(environment, "deployment_value", lambda _name: None)

    metadata = environment.llamacpp_metadata("http://127.0.0.1:8080/v1", "logical")

    assert metadata["image"] == "unavailable"
    assert metadata["image_id"] == "unavailable"
    assert metadata["model"]["sha256"] is None
