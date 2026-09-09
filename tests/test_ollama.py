import json
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
            "keep_alive": "24h",
            "options": {"num_ctx": 8192},
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
                                "name": "calculate",
                                "arguments": {"expression": "2 + 2"},
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
        [{"role": "user", "content": "Calculate 2 + 2"}],
        TOOLS,
    )

    call = response.message.tool_calls[0]
    assert call.function.name == "calculate"
    assert call.function.arguments == {"expression": "2 + 2"}
    assert client.calls[0]["tools"] == TOOLS
    assert client.calls[0]["stream"] is False
    assert client.calls[0]["think"] is False
    assert client.calls[0]["keep_alive"] == "24h"
    assert client.calls[0]["options"] == {"num_ctx": 8192}


@pytest.mark.asyncio
async def test_semantic_planner_prompt_preserves_speaker_resolver_boundary() -> None:
    client = FakeOllamaClient(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {"requires_fact": False, "request": None}
                    ),
                }
            )
        ]
    )
    service = OllamaService(
        "http://ollama:11434",
        "qwen3:8b",
        client=client,  # type: ignore[arg-type]
    )

    await service.plan_semantic_fact(
        [
            {"role": "assistant", "content": "prior household chatter"},
            {"role": "user", "content": "earlier user wording"},
            {"role": "assistant", "content": "invented assistant fact"},
            {"role": "user", "content": "a speaker-relative relation"},
        ],
        {"concepts": {}, "relations": ["spouse"]},
        {"type": "object"},
        household_now="2026-09-03T12:00:00-07:00",
    )

    call = client.calls[0]
    prompt = call["messages"][0]["content"]
    clock = [
        message["content"]
        for message in call["messages"]
        if message["role"] == "system" and str(message["content"]).startswith("Household now:")
    ]
    user_contents = [
        message["content"]
        for message in call["messages"]
        if message["role"] == "user"
    ]
    assert call["options"] == {
        "temperature": 0,
        "num_ctx": 8192,
        "num_predict": 384,
        "seed": 0,
    }
    assert call["keep_alive"] == "24h"
    assert call["think"] is False
    assert "Household now:" not in prompt
    assert clock == [
        "Household now: 2026-09-03T12:00:00-07:00\n"
        "Person deixis for identity: first person → kind=self; "
        "second person addressing this helper → kind=assistant. "
        "Do not add path, filters, or amount unless the latest "
        "utterance requires them. Do not copy filters from earlier turns. "
        "Age-at-least N is birth_date transform=date_difference mode=years operator=gte value=N. "
        "以上/满=gte; 以下/未满=lt; do not invert. "
        "Household rooms use path concept room from current_household."
    ]
    assert sum(m["role"] == "system" for m in call["messages"]) == 2
    assert "second person addressing this helper" in json.dumps(call["messages"])
    assert "self 是已认证的当前说话人" in prompt
    assert "kind=assistant" in prompt
    assert "请介绍一下你自己。" in json.dumps(call["messages"], ensure_ascii=False)
    assert "逐字复制用户说出的姓名或称呼" in prompt
    assert "property=null" in prompt
    assert "不计算或表述答案" in prompt
    assert "只编译最后一条 user 消息" in prompt
    assert "person:dylan_kuang" not in prompt
    assert "prior household chatter" not in json.dumps(call["messages"])
    assert "invented assistant fact" not in json.dumps(call["messages"])
    assert user_contents[-1] == "a speaker-relative relation"
    assert call["messages"][-3:] == [
        {"role": "user", "content": "earlier user wording"},
        {
            "role": "assistant",
            "content": (
                "[End of earlier user turn. Its assistant answer is omitted. "
                "Compile only the following user message; use this earlier "
                "turn solely for discourse antecedents.]"
            ),
        },
        {"role": "user", "content": "a speaker-relative relation"},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("utterance", "kind_note"),
    (
        ("你是谁", "subject.kind must be assistant"),
        ("我是谁", "subject.kind must be self"),
    ),
)
async def test_planner_prompt_keeps_stable_prefix_and_identity_note(
    utterance: str,
    kind_note: str,
) -> None:
    client = FakeOllamaClient(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {"requires_fact": False, "request": None}
                    ),
                }
            )
        ]
    )
    service = OllamaService(
        "http://ollama:11434",
        "qwen3:8b",
        client=client,  # type: ignore[arg-type]
    )
    await service.plan_semantic_fact(
        [{"role": "user", "content": utterance}],
        {"relations": ["spouse"]},
        {"type": "object"},
        household_now="2026-09-07T15:00:00-07:00",
    )

    messages = client.calls[0]["messages"]
    assert messages[0]["role"] == "system"
    assert "Household now:" not in messages[0]["content"]
    assert messages[-2] == {"role": "user", "content": utterance}
    assert messages[-1]["role"] == "system"
    assert messages[-1]["content"].startswith("Household now:") is False
    assert kind_note in messages[-1]["content"]


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
            [{"role": "user", "content": "Say hello"}],
            TOOLS,
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
            "keep_alive": "24h",
            "options": {"num_ctx": 8192},
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


def test_capability_serialization_ignores_mapping_order_preserves_arrays() -> None:
    from home_cortex.ollama import planner_system_prompt
    first = {'relations': ['parent', 'spouse'], 'ownership': {'entity': ['age'], 'relationship': ['start']}}
    reordered = {'ownership': {'relationship': ['start'], 'entity': ['age']}, 'relations': ['parent', 'spouse']}
    assert planner_system_prompt(first) == planner_system_prompt(reordered)
    serialized = planner_system_prompt(first).split('\nCapabilities:\n', 1)[1]
    assert json.loads(serialized) == first
    assert planner_system_prompt({**first, 'relations': ['spouse', 'parent']}) != planner_system_prompt(first)


def test_static_prefix_survives_clock_history_and_identity_notes() -> None:
    from home_cortex.ollama import planner_chat_messages, _semantic_planner_examples
    count = 1 + len(_semantic_planner_examples())
    first = planner_chat_messages([{'role': 'user', 'content': 'Who am I?'}], {}, household_now='2026-09-08T10:00:00Z')
    other = planner_chat_messages([
        {'role': 'user', 'content': 'Who is my daughter?'},
        {'role': 'assistant', 'content': 'Untrusted answer'},
        {'role': 'user', 'content': 'When is her birthday?'},
    ], {}, household_now='2026-09-08T11:00:00Z')
    assert first[:count] == other[:count]
    assert first[count] != other[count]
    assert 'Untrusted answer' not in json.dumps(other)
