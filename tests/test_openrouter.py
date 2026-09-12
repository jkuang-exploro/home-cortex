from home_cortex.mutation_ir import read_plan_schema
import json
from typing import Any

import httpx
import pytest

from home_cortex.config import Settings
from home_cortex.ollama import language_model_from_settings
from home_cortex.openrouter import OpenRouterService
from home_cortex.tools import TOOLS


def _completion(
    content: str = "",
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "choices": [{"message": message}],
        "usage": usage or {"prompt_tokens": 11, "completion_tokens": 5},
    }


def _client(handler: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_language_model_factory_selects_openrouter() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider="openrouter",
        openrouter_api_key="sk-or-test",
        openrouter_model="anthropic/claude-sonnet-4",
    )
    service = language_model_from_settings(settings)
    try:
        assert isinstance(service, OpenRouterService)
        assert service.model == "anthropic/claude-sonnet-4"
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_openrouter_chat_sends_bearer_auth_and_model() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("Authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_completion("Hello from OpenRouter"))

    service = OpenRouterService(
        "https://openrouter.ai/api/v1/",
        "anthropic/claude-sonnet-4",
        api_key="sk-or-test",
        http_referer="https://example.local",
        client=_client(handler),
    )
    response = await service.chat([{"role": "user", "content": "Say hello"}])

    assert response.message.content == "Hello from OpenRouter"
    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert captured["authorization"] == "Bearer sk-or-test"
    assert captured["body"]["model"] == "anthropic/claude-sonnet-4"
    assert captured["body"]["messages"] == [
        {"role": "user", "content": "Say hello"}
    ]
    await service.close()


@pytest.mark.asyncio
async def test_openrouter_tool_calls_are_adapted_to_the_cortex_loop() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["tools"] == TOOLS
        return httpx.Response(
            200,
            json=_completion(
                "",
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "calculate",
                            "arguments": '{"expression":"2 + 2"}',
                        },
                    }
                ],
            ),
        )

    service = OpenRouterService(
        "https://openrouter.ai/api/v1",
        "openai/gpt-4.1",
        api_key="sk-or-test",
        client=_client(handler),
    )
    response = await service.chat_with_tools(
        [
            {"role": "user", "content": "Calculate 2 + 2"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "calculate",
                            "arguments": {"expression": "1 + 1"},
                        }
                    }
                ],
            },
            {"role": "tool", "tool_name": "calculate", "content": '{"result":2}'},
        ],
        TOOLS,
    )

    call = response.message.tool_calls[0]
    assert call.function.name == "calculate"
    assert call.function.arguments == {"expression": "2 + 2"}
    await service.close()


@pytest.mark.asyncio
async def test_openrouter_planner_uses_json_schema_and_parses_content() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json=_completion(
                json.dumps({"requires_fact": False, "request": None}),
                usage={"prompt_tokens": 20, "completion_tokens": 8},
            ),
        )

    service = OpenRouterService(
        "https://openrouter.ai/api/v1",
        "anthropic/claude-sonnet-4",
        api_key="sk-or-test",
        client=_client(handler),
    )
    plan = await service.plan_semantic_fact(
        [{"role": "user", "content": "hello"}],
        {"relations": ["spouse"]},
        {"type": "object"},
        household_now="2026-09-03T12:00:00-07:00",
    )

    body = captured["body"]
    assert body["temperature"] == 0
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["name"] == "SemanticPlan"
    assert body["response_format"]["json_schema"]["strict"] is False
    assert body["provider"] == {"require_parameters": True}
    assert plan == {"requires_fact": False, "request": None}
    assert service.last_planner_runtime["prompt_eval_count"] == 20
    await service.close()


@pytest.mark.asyncio
async def test_openrouter_streaming_yields_content_then_completed_tool_calls() -> None:
    chunks = [
        'data: {"choices":[{"delta":{"content":"Hello "}}]}\n\n',
        'data: {"choices":[{"delta":{"content":"there"}}]}\n\n',
        (
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
            '"id":"call_1","function":{"name":"calculate","arguments":"{\\"e\\""}}]}}]}\n\n'
        ),
        (
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
            '"function":{"arguments":":\\"2\\"}"}}]}}]}\n\n'
        ),
        "data: [DONE]\n\n",
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(
            200,
            content="".join(chunks).encode(),
            headers={"Content-Type": "text/event-stream"},
        )

    service = OpenRouterService(
        "https://openrouter.ai/api/v1",
        "openai/gpt-4.1",
        api_key="sk-or-test",
        client=_client(handler),
    )
    seen = [
        (chunk.message.content, list(chunk.message.tool_calls or []))
        async for chunk in service.stream_chat_with_tools(
            [{"role": "user", "content": "hi"}],
            TOOLS,
        )
    ]

    assert seen[0][0] == "Hello "
    assert seen[1][0] == "there"
    assert seen[2][0] == ""
    assert seen[2][1][0].function.name == "calculate"
    assert seen[2][1][0].function.arguments == {"e": "2"}
    await service.close()


@pytest.mark.asyncio
async def test_openrouter_http_error_does_not_include_the_api_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "sk-or-test leaked"}})

    service = OpenRouterService(
        "https://openrouter.ai/api/v1",
        "openai/gpt-4.1",
        api_key="sk-or-test",
        client=_client(handler),
    )
    with pytest.raises(RuntimeError, match="HTTP 401") as captured:
        await service.chat([{"role": "user", "content": "hi"}])
    assert "sk-or-test" not in str(captured.value)
    await service.close()


@pytest.mark.asyncio
async def test_openrouter_serving_fact_contract_is_expanded():
    from home_cortex.semantic_ir import SemanticPlan
    plan = {'requires_fact': True, 'request': {
        'operation': 'count', 'subject': {'kind': 'current_household'},
        'property': None, 'property_source': 'entity',
    }}
    captured = {}
    def handler(request):
        captured.update(json.loads(request.content))
        return httpx.Response(200, json=_completion(json.dumps(plan)))
    service = OpenRouterService('https://openrouter.ai/api/v1', 'fake', api_key='test', client=_client(handler))
    schema = SemanticPlan.model_json_schema()
    result = await service.plan_semantic_fact(
        [{'role': 'user', 'content': 'How many members are in this home?'}], {}, schema,
        household_now='2026-09-09T12:00:00Z')
    assert result == plan
    assert captured['response_format']['json_schema']['schema'] == read_plan_schema(schema)
    assert 'Transport v1:' not in captured['messages'][0]['content']
    for message in captured['messages']:
        if message['role'] == 'assistant':
            assert isinstance(json.loads(message['content']), dict)
    await service.close()
