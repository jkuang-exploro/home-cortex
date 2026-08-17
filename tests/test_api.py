from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from home_cortex.api import VIRTUAL_MODEL, app


class FakeAgent:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, Any]]] = []

    async def answer_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> SimpleNamespace:
        self.calls.append(messages)
        return SimpleNamespace(answer="Jian and Pu reside at Fort Cerritos.")


@pytest.fixture
def api_client() -> Iterator[tuple[TestClient, FakeAgent]]:
    agent = FakeAgent()
    app.state.agent = agent
    client = TestClient(app, raise_server_exceptions=True)
    try:
        yield client, agent
    finally:
        client.close()


def test_models_advertises_only_home_cortex(
    api_client: tuple[TestClient, FakeAgent],
) -> None:
    client, _ = api_client

    response = client.get("/v1/models")

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert [model["id"] for model in body["data"]] == [VIRTUAL_MODEL]
    assert body["data"][0]["object"] == "model"
    assert body["data"][0]["owned_by"] == "home-cortex"


def test_chat_completions_invokes_agent_and_returns_openai_shape(
    api_client: tuple[TestClient, FakeAgent],
) -> None:
    client, agent = api_client
    messages = [
        {"role": "system", "content": "Answer briefly."},
        {"role": "user", "content": "Who resides at Fort Cerritos?"},
    ]

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "home-cortex",
            "stream": False,
            "messages": messages,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["id"].startswith("chatcmpl-")
    assert body["object"] == "chat.completion"
    assert body["model"] == "home-cortex"
    assert body["choices"] == [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Jian and Pu reside at Fort Cerritos.",
            },
            "finish_reason": "stop",
        }
    ]
    assert agent.calls == [messages]


@pytest.mark.parametrize(
    ("request_body", "status_code"),
    [
        (
            {
                "model": "qwen3:8b",
                "stream": False,
                "messages": [{"role": "user", "content": "Hello"}],
            },
            404,
        ),
        (
            {
                "model": "home-cortex",
                "stream": True,
                "messages": [{"role": "user", "content": "Hello"}],
            },
            400,
        ),
        (
            {
                "model": "home-cortex",
                "stream": False,
                "messages": [],
            },
            422,
        ),
    ],
)
def test_chat_completions_rejects_unsupported_requests(
    api_client: tuple[TestClient, FakeAgent],
    request_body: dict[str, Any],
    status_code: int,
) -> None:
    client, agent = api_client

    response = client.post("/v1/chat/completions", json=request_body)

    assert response.status_code == status_code
    assert agent.calls == []
