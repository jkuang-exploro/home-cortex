from collections.abc import AsyncIterator
from typing import Any

import pytest
from ollama import ChatResponse

from home_cortex.ollama import OllamaService
from home_cortex.tools import TOOLS


class FakeOllamaClient:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    async def chat(self, **request: Any) -> Any:
        self.calls.append(request)
        return self.responses.pop(0)

    async def close(self) -> None:
        self.closed = True


class FakeResponseStream:
    def __init__(self, chunks: list[ChatResponse]) -> None:
        self._chunks = iter(chunks)
        self.closed = False

    def __aiter__(self) -> AsyncIterator[ChatResponse]:
        return self

    async def __anext__(self) -> ChatResponse:
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration from None

    async def aclose(self) -> None:
        self.closed = True


def _chat_response(message: dict[str, Any]) -> ChatResponse:
    return ChatResponse.model_validate(
        {
            "model": "qwen3:8b",
            "created_at": "2026-08-16T00:00:00Z",
            "done": True,
            "message": message,
        }
    )


@pytest.mark.asyncio
async def test_ordinary_chat_call() -> None:
    client = FakeOllamaClient(
        [_chat_response({"role": "assistant", "content": "Hello from Ollama"})]
    )
    service = OllamaService(
        "http://ollama:11434/",
        "qwen3:8b",
        client=client,  # type: ignore[arg-type]
    )

    response = await service.chat([{"role": "user", "content": "Say hello"}])

    assert response.message.content == "Hello from Ollama"
    assert service.base_url == "http://ollama:11434"
    assert client.calls == [
        {
            "model": "qwen3:8b",
            "messages": [{"role": "user", "content": "Say hello"}],
            "stream": False,
            "think": False,
        }
    ]


@pytest.mark.asyncio
async def test_tool_call_response() -> None:
    client = FakeOllamaClient(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "search_entities",
                                "arguments": {
                                    "text": "Test House",
                                    "entity_type": "location",
                                },
                            }
                        }
                    ],
                }
            )
        ]
    )
    service = OllamaService(
        "http://ollama:11434",
        "qwen3:8b",
        client=client,  # type: ignore[arg-type]
    )

    response = await service.chat_with_tools(
        [{"role": "user", "content": "Find Test House"}]
    )

    call = response.message.tool_calls[0]
    assert call.function.name == "search_entities"
    assert call.function.arguments == {
        "text": "Test House",
        "entity_type": "location",
    }
    assert client.calls[0]["tools"] == TOOLS
    assert client.calls[0]["stream"] is False
    assert client.calls[0]["think"] is False


@pytest.mark.asyncio
async def test_streaming_tool_chat_yields_chunks_and_closes_stream() -> None:
    response_stream = FakeResponseStream(
        [
            _chat_response({"role": "assistant", "content": "Hello "}),
            _chat_response({"role": "assistant", "content": "there"}),
        ]
    )
    client = FakeOllamaClient([response_stream])
    service = OllamaService(
        "http://ollama:11434",
        "qwen3:8b",
        client=client,  # type: ignore[arg-type]
    )

    chunks = [
        chunk.message.content
        async for chunk in service.stream_chat_with_tools(
            [{"role": "user", "content": "Say hello"}]
        )
    ]

    assert chunks == ["Hello ", "there"]
    assert response_stream.closed is True
    assert client.calls == [
        {
            "model": "qwen3:8b",
            "messages": [{"role": "user", "content": "Say hello"}],
            "tools": TOOLS,
            "stream": True,
            "think": False,
        }
    ]


@pytest.mark.asyncio
async def test_injected_client_is_not_closed_by_service() -> None:
    client = FakeOllamaClient([])
    service = OllamaService(
        "http://ollama:11434",
        "qwen3:8b",
        client=client,  # type: ignore[arg-type]
    )

    await service.close()

    assert client.closed is False
