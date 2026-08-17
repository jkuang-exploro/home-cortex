import asyncio
import json
from typing import Any

import pytest
from ollama import ChatResponse

from home_cortex.agent import (
    MAX_AGENT_STEPS,
    AgentLimitError,
    AgentService,
)


class FakeOllamaService:
    def __init__(self, responses: list[ChatResponse]) -> None:
        self.responses = responses
        self.calls: list[list[dict[str, Any]]] = []

    async def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
    ) -> ChatResponse:
        self.calls.append([dict(message) for message in messages])
        return self.responses.pop(0)


class FakeDispatcher:
    def __init__(
        self,
        result: dict[str, Any] | None = None,
        *,
        delay: float = 0,
    ) -> None:
        self.result = result or {
            "ok": True,
            "tool": "search_entities",
            "result": [{"id": "home:test_home", "name": "Test House"}],
        }
        self.delay = delay
        self.calls: list[tuple[str, Any]] = []

    async def dispatch(self, tool_name: str, arguments: Any) -> dict[str, Any]:
        self.calls.append((tool_name, arguments))
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.result


def _chat_response(
    content: str = "",
    *,
    tool_calls: list[dict[str, Any]] | None = None,
) -> ChatResponse:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return ChatResponse.model_validate(
        {
            "model": "qwen3:8b",
            "created_at": "2026-08-16T00:00:00Z",
            "done": True,
            "message": message,
        }
    )


def _tool_call(
    name: str = "search_entities",
    arguments: Any = None,
) -> dict[str, Any]:
    return {
        "function": {
            "name": name,
            "arguments": arguments if arguments is not None else {"text": "Test"},
        }
    }


@pytest.mark.asyncio
async def test_returns_first_normal_answer_without_dispatching_tools() -> None:
    ollama = FakeOllamaService([_chat_response("The answer is ready.")])
    dispatcher = FakeDispatcher()
    agent = AgentService(ollama, dispatcher)  # type: ignore[arg-type]

    result = await agent.answer("What is known?")

    assert result.answer == "The answer is ready."
    assert result.steps == 1
    assert result.tool_calls == 0
    assert dispatcher.calls == []
    assert ollama.calls[0][0]["role"] == "system"
    assert ollama.calls[0][-1] == {"role": "user", "content": "What is known?"}


@pytest.mark.asyncio
async def test_dispatches_tool_result_and_calls_ollama_again() -> None:
    ollama = FakeOllamaService(
        [
            _chat_response(tool_calls=[_tool_call(arguments={"text": "Test"})]),
            _chat_response("Alex lives at Test House."),
        ]
    )
    dispatcher = FakeDispatcher()
    agent = AgentService(ollama, dispatcher)  # type: ignore[arg-type]

    result = await agent.answer("Where does Alex live?")

    assert result.answer == "Alex lives at Test House."
    assert result.steps == 2
    assert result.tool_calls == 1
    assert dispatcher.calls == [
        ("search_entities", {"text": "Test", "limit": 25})
    ]
    second_request = ollama.calls[1]
    assert second_request[-2]["role"] == "assistant"
    assert second_request[-1]["role"] == "tool"
    assert second_request[-1]["tool_name"] == "search_entities"
    assert json.loads(second_request[-1]["content"])["ok"] is True


@pytest.mark.asyncio
async def test_clamps_requested_limit_and_returned_record_count() -> None:
    records = [{"id": f"person:{number}"} for number in range(10)]
    dispatcher = FakeDispatcher(
        {"ok": True, "tool": "search_entities", "result": records}
    )
    ollama = FakeOllamaService(
        [
            _chat_response(
                tool_calls=[_tool_call(arguments={"text": "person", "limit": 100})]
            ),
            _chat_response("Done"),
        ]
    )
    agent = AgentService(
        ollama,
        dispatcher,  # type: ignore[arg-type]
        max_tool_records=3,
    )

    await agent.answer("Find people")

    assert dispatcher.calls[0][1]["limit"] == 3
    tool_result = json.loads(ollama.calls[1][-1]["content"])
    assert len(tool_result["result"]) == 3
    assert tool_result["meta"] == {
        "truncated": True,
        "records_available": 10,
        "records_returned": 3,
    }


@pytest.mark.asyncio
async def test_rejects_too_many_tool_calls_in_one_step() -> None:
    dispatcher = FakeDispatcher()
    ollama = FakeOllamaService(
        [_chat_response(tool_calls=[_tool_call() for _ in range(3)])]
    )
    agent = AgentService(
        ollama,
        dispatcher,  # type: ignore[arg-type]
        max_tool_calls_per_step=2,
    )

    with pytest.raises(AgentLimitError, match="3 tools in one step"):
        await agent.answer("Use too many tools")

    assert dispatcher.calls == []


@pytest.mark.asyncio
async def test_stops_after_hard_agent_step_limit() -> None:
    dispatcher = FakeDispatcher()
    ollama = FakeOllamaService(
        [_chat_response(tool_calls=[_tool_call()]) for _ in range(MAX_AGENT_STEPS)]
    )
    agent = AgentService(ollama, dispatcher)  # type: ignore[arg-type]

    with pytest.raises(AgentLimitError, match="within 4 steps"):
        await agent.answer("Never-ending lookup")

    assert len(ollama.calls) == MAX_AGENT_STEPS
    assert len(dispatcher.calls) == MAX_AGENT_STEPS - 1


@pytest.mark.asyncio
async def test_tool_timeout_becomes_a_bounded_tool_error() -> None:
    dispatcher = FakeDispatcher(delay=0.05)
    ollama = FakeOllamaService(
        [
            _chat_response(tool_calls=[_tool_call()]),
            _chat_response("I could not retrieve the data in time."),
        ]
    )
    agent = AgentService(
        ollama,
        dispatcher,  # type: ignore[arg-type]
        tool_timeout_seconds=0.001,
    )

    result = await agent.answer("Find the home")

    assert result.steps == 2
    tool_result = json.loads(ollama.calls[1][-1]["content"])
    assert tool_result["error"]["code"] == "tool_timeout"


@pytest.mark.asyncio
async def test_oversized_tool_record_is_truncated_before_sending_to_ollama() -> None:
    dispatcher = FakeDispatcher(
        {
            "ok": True,
            "tool": "search_entities",
            "result": [{"id": "home:test", "description": "x" * 2_000}],
        }
    )
    ollama = FakeOllamaService(
        [
            _chat_response(tool_calls=[_tool_call()]),
            _chat_response("The result was too large."),
        ]
    )
    agent = AgentService(
        ollama,
        dispatcher,  # type: ignore[arg-type]
        max_tool_result_bytes=256,
    )

    await agent.answer("Find a large record")

    content = ollama.calls[1][-1]["content"]
    assert len(content.encode("utf-8")) <= 256
    parsed = json.loads(content)
    assert parsed["result"] == []
    assert parsed["meta"] == {
        "truncated": True,
        "records_available": 1,
        "records_returned": 0,
    }
    assert "x" * 100 not in content


def test_cannot_configure_limits_above_hard_caps() -> None:
    ollama = FakeOllamaService([])
    dispatcher = FakeDispatcher()

    with pytest.raises(ValueError, match="max_steps"):
        AgentService(
            ollama,
            dispatcher,  # type: ignore[arg-type]
            max_steps=MAX_AGENT_STEPS + 1,
        )
