import json
import logging
import re
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from home_cortex.api import VIRTUAL_MODEL, app


class FakeAgent:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, Any]]] = []
        self.request_ids: list[str] = []

    async def answer_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        request_id: str = "-",
    ) -> SimpleNamespace:
        self.calls.append(messages)
        self.request_ids.append(request_id)
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
    _assert_request_id(response)


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
    assert agent.request_ids == [response.headers["X-Request-ID"]]
    _assert_request_id(response)


def test_chat_completions_returns_openai_sse_stream(
    api_client: tuple[TestClient, FakeAgent],
) -> None:
    client, agent = api_client
    messages = [{"role": "user", "content": "Who resides at Fort Cerritos?"}]

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "home-cortex",
            "stream": True,
            "messages": messages,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [
        line.removeprefix("data: ")
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(event) for event in events[:-1]]
    assert {chunk["id"] for chunk in chunks} == {chunks[0]["id"]}
    assert all(chunk["object"] == "chat.completion.chunk" for chunk in chunks)
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert chunks[1]["choices"][0]["delta"] == {
        "content": "Jian and Pu reside at Fort Cerritos."
    }
    assert chunks[2]["choices"][0]["finish_reason"] == "stop"
    assert agent.calls == [messages]
    assert agent.request_ids == [response.headers["X-Request-ID"]]
    _assert_request_id(response)


@pytest.mark.parametrize(
    ("request_body", "status_code", "error_code"),
    [
        (
            {
                "model": "qwen3:8b",
                "stream": False,
                "messages": [{"role": "user", "content": "Hello"}],
            },
            404,
            "model_not_found",
        ),
        (
            {
                "model": "home-cortex",
                "stream": False,
                "messages": [],
            },
            422,
            "request_validation_error",
        ),
    ],
)
def test_chat_completions_rejects_unsupported_requests(
    api_client: tuple[TestClient, FakeAgent],
    request_body: dict[str, Any],
    status_code: int,
    error_code: str,
) -> None:
    client, agent = api_client

    response = client.post("/v1/chat/completions", json=request_body)

    assert response.status_code == status_code
    error = response.json()["error"]
    assert error["code"] == error_code
    assert isinstance(error["message"], str)
    assert error["request_id"] == response.headers["X-Request-ID"]
    if error_code == "request_validation_error":
        assert error["details"][0]["field"] == "body.messages"
    _assert_request_id(response)
    assert agent.calls == []


def test_unknown_route_uses_consistent_json_error(
    api_client: tuple[TestClient, FakeAgent],
) -> None:
    client, _ = api_client

    response = client.get("/does-not-exist")

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "code": "http_404",
            "message": "Not Found",
            "request_id": response.headers["X-Request-ID"],
        }
    }
    _assert_request_id(response)


def test_unexpected_error_is_json_and_does_not_log_private_message(
    caplog: pytest.LogCaptureFixture,
) -> None:
    private_value = "private-dob-1988-11-11"

    class ExplodingAgent:
        async def answer_messages(
            self,
            messages: list[dict[str, Any]],
            *,
            request_id: str = "-",
        ) -> None:
            raise RuntimeError(private_value)

    previous_agent = getattr(app.state, "agent", None)
    app.state.agent = ExplodingAgent()
    client = TestClient(app, raise_server_exceptions=False)
    try:
        with caplog.at_level(logging.INFO):
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "home-cortex",
                    "stream": False,
                    "messages": [{"role": "user", "content": private_value}],
                },
            )
    finally:
        client.close()
        if previous_agent is None:
            del app.state.agent
        else:
            app.state.agent = previous_agent

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "code": "internal_server_error",
            "message": "An unexpected server error occurred",
            "request_id": response.headers["X-Request-ID"],
        }
    }
    assert "exception_type=RuntimeError" in caplog.text
    assert private_value not in caplog.text
    _assert_request_id(response)


def _assert_request_id(response: Any) -> None:
    request_id = response.headers["X-Request-ID"]
    assert re.fullmatch(r"[0-9a-f]{32}", request_id)
