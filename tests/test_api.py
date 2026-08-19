import asyncio
import json
import logging
import re
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from unittest.mock import AsyncMock, patch

from home_cortex.api import (
    ConversationStore,
    VIRTUAL_MODEL,
    _stream_chat_completion,
    app,
)
from home_cortex.greetings import GreetingService
from home_cortex.ingestion import IngestionResult
from home_cortex.retrieval import RetrievedContext


class FakeAgent:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, Any]]] = []
        self.request_ids: list[str] = []
        self.user_entity_ids: list[str | None] = []
        self.user_entities: list[dict[str, Any] | None] = []

    async def answer(
        self,
        question: str,
        *,
        request_id: str = "-",
        user_entity_id: str | None = None,
        user_entity: dict[str, Any] | None = None,
    ) -> SimpleNamespace:
        return await self.answer_messages(
            [{"role": "user", "content": question}],
            request_id=request_id,
            user_entity_id=user_entity_id,
            user_entity=user_entity,
        )

    async def answer_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        request_id: str = "-",
        user_entity_id: str | None = None,
        user_entity: dict[str, Any] | None = None,
    ) -> SimpleNamespace:
        self.calls.append(messages)
        self.request_ids.append(request_id)
        self.user_entity_ids.append(user_entity_id)
        self.user_entities.append(user_entity)
        return SimpleNamespace(
            answer="Jian and Pu reside at Fort Cerritos.",
            steps=3,
            tool_calls=2,
            stop_reason="answer",
        )

    async def stream_answer_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        request_id: str = "-",
        user_entity_id: str | None = None,
        user_entity: dict[str, Any] | None = None,
    ):
        self.calls.append(messages)
        self.request_ids.append(request_id)
        self.user_entity_ids.append(user_entity_id)
        self.user_entities.append(user_entity)
        yield "Jian and Pu "
        yield "reside at Fort Cerritos."


class FakeHealthDatabase:
    def __init__(self) -> None:
        self.fail = False

    async def version(self) -> str:
        if self.fail:
            raise RuntimeError("surreal unavailable")
        return "2.0.0-test"


class FakeIdentityRetrieval:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = [
            {
                "id": "person:jian_kuang",
                "name": ["Jian Kuang", "匡健"],
                "address_as": {"en": "Mr. Kuang", "zh": "先生"},
                "dob": "1988-11-11",
            }
        ]
        self.locations: list[dict[str, Any]] = [
            {
                "id": "location:fort_cerritos",
                "name": ["Fort Cerritos", "喜瑞匡家"],
            }
        ]
        self.calls: list[tuple[str, str | None, int | None]] = []
        self.entity_calls: list[str] = []
        self.relationship_calls: list[str] = []
        self.retrieve_calls: list[str] = []

    async def get_entity(self, record_id: str) -> dict[str, Any] | None:
        self.entity_calls.append(record_id)
        for record in (*self.records, *self.locations):
            if record.get("id") == record_id:
                return record
        return None

    async def search_entities(
        self,
        text: str,
        entity_type: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        self.calls.append((text, entity_type, limit))
        if entity_type == "location":
            return list(self.locations)
        return list(self.records)

    async def retrieve(self, question: str) -> RetrievedContext:
        self.retrieve_calls.append(question)
        return RetrievedContext(
            question=question,
            nodes={"person": list(self.records), "location": list(self.locations)},
            edges={"lives_in": []},
            text="{}",
        )

    async def get_relationships(
        self,
        entity_id: str,
        relation: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        self.relationship_calls.append(entity_id)
        return [
            {
                "id": "lives_in:jian",
                "in": entity_id,
                "out": "location:fort_cerritos",
                "household_role": "owner",
                "related_entity": {
                    "id": "location:fort_cerritos",
                    "name": ["Fort Cerritos", "喜瑞匡家"],
                },
            }
        ]


@pytest.fixture
def api_client() -> Iterator[tuple[TestClient, FakeAgent]]:
    previous_agent = getattr(app.state, "agent", None)
    previous_agents = getattr(app.state, "agents", None)
    previous_settings = getattr(app.state, "settings", None)
    previous_retrieval = getattr(app.state, "retrieval", None)
    previous_greetings = getattr(app.state, "greetings", None)
    previous_conversations = getattr(app.state, "conversations", None)
    previous_database = getattr(app.state, "database", None)
    agent = FakeAgent()
    app.state.agent = agent
    app.state.agents = {"steward": agent}
    app.state.settings = SimpleNamespace(
        cortex_api_key=None,
        cortex_identity_map={},
    )
    retrieval = FakeIdentityRetrieval()
    app.state.retrieval = retrieval
    app.state.greetings = GreetingService(retrieval)
    app.state.conversations = ConversationStore()
    app.state.database = FakeHealthDatabase()
    client = TestClient(app, raise_server_exceptions=True)
    try:
        yield client, agent
    finally:
        client.close()
        if previous_agent is None:
            del app.state.agent
        else:
            app.state.agent = previous_agent
        if previous_agents is None:
            del app.state.agents
        else:
            app.state.agents = previous_agents
        if previous_settings is None:
            del app.state.settings
        else:
            app.state.settings = previous_settings
        if previous_retrieval is None:
            del app.state.retrieval
        else:
            app.state.retrieval = previous_retrieval
        if previous_greetings is None:
            del app.state.greetings
        else:
            app.state.greetings = previous_greetings
        if previous_conversations is None:
            del app.state.conversations
        else:
            app.state.conversations = previous_conversations
        if previous_database is None:
            del app.state.database
        else:
            app.state.database = previous_database


def test_new_steward_conversation_returns_owner_greeting_once(
    api_client: tuple[TestClient, FakeAgent],
) -> None:
    client, agent = api_client
    app.state.settings = SimpleNamespace(
        cortex_api_key="test-cortex-key",
        cortex_identity_map={"id:webui-user-123": "person:jian_kuang"},
    )
    headers = {
        "Authorization": "Bearer test-cortex-key",
        "X-OpenWebUI-User-Id": "webui-user-123",
    }

    created = client.post(
        "/agent/steward/conversations",
        headers=headers,
        json={"language": "zh-CN"},
    )

    assert created.status_code == 201
    body = created.json()
    assert body["object"] == "agent.conversation"
    assert body["agent"] == {"id": "steward", "display_name": "老管家"}
    assert body["language"] == "zh"
    assert body["greeting"] == "先生，您回来了。老管家在此，今日有什么需要吩咐？"
    assert "person:" not in body["greeting"]
    assert agent.calls == []
    assert app.state.retrieval.relationship_calls == ["person:jian_kuang"]

    refreshed = client.get(
        f"/agent/steward/conversations/{body['id']}",
        headers=headers,
    )

    assert refreshed.status_code == 200
    assert refreshed.json() == body
    assert app.state.retrieval.relationship_calls == ["person:jian_kuang"]
    assert agent.calls == []


def test_new_conversation_without_identity_uses_neutral_greeting(
    api_client: tuple[TestClient, FakeAgent],
) -> None:
    client, agent = api_client

    response = client.post(
        "/agent/steward/conversations",
        json={"language": "zh"},
    )

    assert response.status_code == 201
    assert response.json()["greeting"] == "您好，有什么可以帮您？"
    assert app.state.retrieval.relationship_calls == []
    assert agent.calls == []


def test_models_advertises_steward_display_name(
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
    assert VIRTUAL_MODEL == "老管家"
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
            "model": VIRTUAL_MODEL,
            "stream": False,
            "messages": messages,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["id"].startswith("chatcmpl-")
    assert body["object"] == "chat.completion"
    assert body["model"] == VIRTUAL_MODEL
    assert body["choices"] == [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": (
                    "Hello. How may I help you?\n\n"
                    "Jian and Pu reside at Fort Cerritos."
                ),
            },
            "finish_reason": "stop",
        }
    ]
    assert agent.calls[0] == [messages[-1]]
    assert agent.request_ids == [response.headers["X-Request-ID"]]
    _assert_request_id(response)


def test_chat_completions_passes_mapped_openwebui_identity(
    api_client: tuple[TestClient, FakeAgent],
) -> None:
    client, agent = api_client
    app.state.settings = SimpleNamespace(
        cortex_api_key="test-cortex-key",
        cortex_identity_map={
            "id:webui-user-123": "person:jian_kuang",
        },
    )

    response = client.post(
        "/v1/chat/completions",
        headers={
            "Authorization": "Bearer test-cortex-key",
            "X-OpenWebUI-User-Id": "webui-user-123",
        },
        json={
            "model": VIRTUAL_MODEL,
            "stream": False,
            "messages": [{"role": "user", "content": "Where do I live?"}],
        },
    )

    assert response.status_code == 200
    assert agent.user_entity_ids == [None]
    assert agent.user_entities == [
        {
            "id": "person:jian_kuang",
            "name": ["Jian Kuang", "匡健"],
            "address_as": {"en": "Mr. Kuang", "zh": "先生"},
        }
    ]
    assert response.json()["choices"][0]["message"]["content"].startswith(
        "Mr. Kuang, welcome home. The butler is here."
    )


def test_existing_openai_conversation_does_not_repeat_greeting(
    api_client: tuple[TestClient, FakeAgent],
) -> None:
    client, _ = api_client
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Earlier greeting"},
        {"role": "user", "content": "Who lives here?"},
    ]

    response = client.post(
        "/v1/chat/completions",
        json={"model": VIRTUAL_MODEL, "stream": False, "messages": messages},
    )

    assert response.status_code == 200
    assert (
        response.json()["choices"][0]["message"]["content"]
        == "Jian and Pu reside at Fort Cerritos."
    )
    assert app.state.retrieval.relationship_calls == []


@pytest.mark.parametrize("empty_content", [None, "", "   "])
def test_chat_completions_ignores_empty_assistant_placeholder(
    api_client: tuple[TestClient, FakeAgent],
    empty_content: str | None,
) -> None:
    client, agent = api_client
    messages = [
        {"role": "user", "content": "Earlier question"},
        {"role": "assistant", "content": empty_content},
        {"role": "user", "content": "What is my wife's birthday?"},
    ]

    response = client.post(
        "/v1/chat/completions",
        json={"model": VIRTUAL_MODEL, "stream": False, "messages": messages},
    )

    assert response.status_code == 200
    assert agent.calls == [
        [messages[0], messages[2]],
    ]


@pytest.mark.parametrize("empty_content", [None, "", "   "])
def test_chat_completions_rejects_empty_user_message(
    api_client: tuple[TestClient, FakeAgent],
    empty_content: str | None,
) -> None:
    client, agent = api_client

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": VIRTUAL_MODEL,
            "stream": False,
            "messages": [{"role": "user", "content": empty_content}],
        },
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"
    assert agent.calls == []


def test_standalone_first_turn_greeting_does_not_call_ollama(
    api_client: tuple[TestClient, FakeAgent],
) -> None:
    client, agent = api_client
    app.state.settings = SimpleNamespace(
        cortex_api_key="test-cortex-key",
        cortex_identity_map={"id:webui-user-123": "person:jian_kuang"},
    )

    response = client.post(
        "/v1/chat/completions",
        headers={
            "Authorization": "Bearer test-cortex-key",
            "X-OpenWebUI-User-Id": "webui-user-123",
        },
        json={
            "model": VIRTUAL_MODEL,
            "stream": False,
            "messages": [{"role": "user", "content": "您好！"}],
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == (
        "先生，您回来了。老管家在此，今日有什么需要吩咐？"
    )
    assert agent.calls == []


@pytest.mark.parametrize(
    ("headers", "status_code", "error_code"),
    [
        (
            {"X-OpenWebUI-User-Id": "webui-user-123"},
            401,
            "authentication_required",
        ),
        (
            {
                "Authorization": "Bearer test-cortex-key",
                "X-OpenWebUI-User-Id": "unmapped-user",
            },
            403,
            "identity_not_mapped",
        ),
    ],
)
def test_chat_completions_rejects_untrusted_or_unmapped_identity(
    api_client: tuple[TestClient, FakeAgent],
    headers: dict[str, str],
    status_code: int,
    error_code: str,
) -> None:
    client, agent = api_client
    app.state.settings = SimpleNamespace(
        cortex_api_key="test-cortex-key",
        cortex_identity_map={
            "id:webui-user-123": "person:jian_kuang",
        },
    )

    response = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": VIRTUAL_MODEL,
            "messages": [{"role": "user", "content": "Where do I live?"}],
        },
    )

    assert response.status_code == status_code
    assert response.json()["error"]["code"] == error_code
    assert agent.calls == []


def test_chat_completions_rejects_missing_mapped_person_record(
    api_client: tuple[TestClient, FakeAgent],
) -> None:
    client, agent = api_client
    app.state.settings = SimpleNamespace(
        cortex_api_key="test-cortex-key",
        cortex_identity_map={
            "id:webui-user-123": "person:jian_kuang",
        },
    )
    app.state.retrieval.records = []

    response = client.post(
        "/v1/chat/completions",
        headers={
            "Authorization": "Bearer test-cortex-key",
            "X-OpenWebUI-User-Id": "webui-user-123",
        },
        json={
            "model": VIRTUAL_MODEL,
            "messages": [{"role": "user", "content": "Who am I?"}],
        },
    )

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "identity_record_not_found"
    assert agent.calls == []


def test_chat_completions_returns_openai_sse_stream(
    api_client: tuple[TestClient, FakeAgent],
) -> None:
    client, agent = api_client
    messages = [{"role": "user", "content": "Who resides at Fort Cerritos?"}]

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": VIRTUAL_MODEL,
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
    assert all(chunk["model"] == "老管家" for chunk in chunks)
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert chunks[1]["choices"][0]["delta"] == {
        "content": "Hello. How may I help you?\n\n"
    }
    assert chunks[2]["choices"][0]["delta"] == {"content": "Jian and Pu "}
    assert chunks[3]["choices"][0]["delta"] == {
        "content": "reside at Fort Cerritos."
    }
    assert chunks[4]["choices"][0]["finish_reason"] == "stop"
    assert agent.calls[0][-len(messages) :] == messages
    assert agent.request_ids == [response.headers["X-Request-ID"]]
    _assert_request_id(response)


def test_named_steward_chat_route(
    api_client: tuple[TestClient, FakeAgent],
) -> None:
    client, agent = api_client

    response = client.post(
        "/agent/steward/chat",
        json={"message": "Who resides at Fort Cerritos?"},
    )

    assert response.status_code == 200
    assert response.json()["agent"] == {
        "id": "steward",
        "display_name": "老管家",
    }
    assert response.json()["answer"] == "Jian and Pu reside at Fort Cerritos."
    assert len(agent.calls) == 1


def test_named_agent_route_rejects_unknown_agent(
    api_client: tuple[TestClient, FakeAgent],
) -> None:
    client, agent = api_client

    response = client.post(
        "/agent/accountant/chat",
        json={"message": "Show this month's spending"},
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "agent_not_found"
    assert response.json()["error"]["request_id"] == response.headers[
        "X-Request-ID"
    ]
    assert agent.calls == []


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
                "model": VIRTUAL_MODEL,
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
            user_entity_id: str | None = None,
            user_entity: dict[str, Any] | None = None,
        ) -> None:
            raise RuntimeError(private_value)

    previous_agent = getattr(app.state, "agent", None)
    previous_agents = getattr(app.state, "agents", None)
    exploding_agent = ExplodingAgent()
    app.state.agent = exploding_agent
    app.state.agents = {"steward": exploding_agent}
    client = TestClient(app, raise_server_exceptions=False)
    try:
        with caplog.at_level(logging.INFO):
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": VIRTUAL_MODEL,
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
        if previous_agents is None:
            del app.state.agents
        else:
            app.state.agents = previous_agents

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


@pytest.mark.asyncio
async def test_sse_generator_cancels_active_answer_stream(
    caplog: pytest.LogCaptureFixture,
) -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()
    closed = asyncio.Event()

    async def slow_answer():
        started.set()
        try:
            await asyncio.Event().wait()
            yield "never reached"
        except asyncio.CancelledError:
            cancelled.set()
            raise
        finally:
            closed.set()

    request = SimpleNamespace(
        state=SimpleNamespace(request_id="request-sse-cancel")
    )
    stream = _stream_chat_completion(
        "chatcmpl-test",
        123,
        slow_answer(),
        request,  # type: ignore[arg-type]
    )

    first_event = await anext(stream)
    with caplog.at_level(
        logging.INFO,
        logger="uvicorn.error.home_cortex.api",
    ):
        pending_event = asyncio.create_task(anext(stream))
        await started.wait()
        pending_event.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending_event

    assert '"role": "assistant"' in first_event
    assert cancelled.is_set()
    assert closed.is_set()
    assert (
        "stream_cancelled request_id=request-sse-cancel phase=ollama_stream"
        in caplog.text
    )


IDENTITY_MAP = {
    "id:webui-user-a": "person:user_a",
    "id:webui-user-b": "person:user_b",
    "id:webui-user-123": "person:jian_kuang",
}


def _protect_api() -> None:
    app.state.settings = SimpleNamespace(
        cortex_api_key="test-cortex-key",
        cortex_identity_map=IDENTITY_MAP,
    )


def _auth_headers(user_id: str, *, api_key: str = "test-cortex-key") -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "X-OpenWebUI-User-Id": user_id,
    }


def _seed_people() -> None:
    app.state.retrieval.records.extend(
        [
            {
                "id": "person:user_a",
                "name": ["User A"],
                "address_as": {"en": "A"},
            },
            {
                "id": "person:user_b",
                "name": ["User B"],
                "address_as": {"en": "B"},
            },
        ]
    )


def test_health_is_public(
    api_client: tuple[TestClient, FakeAgent],
) -> None:
    client, _ = api_client

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "surrealdb": "2.0.0-test"}
    _assert_request_id(response)


def test_health_maps_database_failure_to_503(
    api_client: tuple[TestClient, FakeAgent],
) -> None:
    client, _ = api_client
    app.state.database.fail = True

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "database_unavailable"


@pytest.mark.parametrize(
    ("method", "path", "json_body"),
    [
        ("POST", "/admin/ingest", None),
        ("POST", "/v1/retrieve", {"query": "Fort Cerritos"}),
        ("POST", "/v1/chat", {"message": "Who lives here?"}),
        (
            "POST",
            "/v1/chat/completions",
            {
                "model": VIRTUAL_MODEL,
                "messages": [{"role": "user", "content": "Who lives here?"}],
            },
        ),
    ],
)
@pytest.mark.parametrize(
    "headers",
    [{}, {"Authorization": "Bearer wrong-key"}],
)
def test_protected_routes_reject_missing_or_invalid_credentials(
    api_client: tuple[TestClient, FakeAgent],
    method: str,
    path: str,
    json_body: dict[str, Any] | None,
    headers: dict[str, str],
) -> None:
    client, agent = api_client
    _protect_api()

    response = client.request(method, path, headers=headers, json=json_body)

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "authentication_required"
    assert agent.calls == []
    assert app.state.retrieval.retrieve_calls == []
    assert app.state.retrieval.entity_calls == []


def test_admin_ingest_accepts_household_api_key(
    api_client: tuple[TestClient, FakeAgent],
) -> None:
    """V1 authorizes ingest with the household API key only; no admin role yet."""
    client, _ = api_client
    _protect_api()
    ingested = IngestionResult(
        node_files=1,
        edge_files=1,
        nodes_upserted=2,
        edges_upserted=1,
    )

    with patch(
        "home_cortex.api.ingest_directory",
        new_callable=AsyncMock,
        return_value=ingested,
    ) as ingest:
        response = client.post(
            "/admin/ingest",
            headers={"Authorization": "Bearer test-cortex-key"},
        )

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["nodes_upserted"] == 2
    ingest.assert_awaited_once()


def test_retrieve_accepts_household_api_key(
    api_client: tuple[TestClient, FakeAgent],
) -> None:
    client, _ = api_client
    _protect_api()

    response = client.post(
        "/v1/retrieve",
        headers={"Authorization": "Bearer test-cortex-key"},
        json={"query": "Fort Cerritos"},
    )

    assert response.status_code == 200
    assert response.json()["query"] == "Fort Cerritos"
    assert app.state.retrieval.retrieve_calls == ["Fort Cerritos"]


def test_chat_accepts_mapped_identity(
    api_client: tuple[TestClient, FakeAgent],
) -> None:
    client, agent = api_client
    _protect_api()

    response = client.post(
        "/v1/chat",
        headers=_auth_headers("webui-user-123"),
        json={"message": "Where do I live?"},
    )

    assert response.status_code == 200
    assert agent.user_entities == [
        {
            "id": "person:jian_kuang",
            "name": ["Jian Kuang", "匡健"],
            "address_as": {"en": "Mr. Kuang", "zh": "先生"},
        }
    ]
    assert app.state.retrieval.entity_calls == ["person:jian_kuang"]


def test_chat_rejects_missing_identity_record(
    api_client: tuple[TestClient, FakeAgent],
) -> None:
    client, agent = api_client
    _protect_api()
    app.state.retrieval.records = []

    response = client.post(
        "/v1/chat",
        headers=_auth_headers("webui-user-123"),
        json={"message": "Who am I?"},
    )

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "identity_record_not_found"
    assert agent.calls == []


def test_chat_ignores_spoofed_person_id_headers(
    api_client: tuple[TestClient, FakeAgent],
) -> None:
    client, agent = api_client
    _protect_api()
    _seed_people()

    response = client.post(
        "/v1/chat",
        headers={
            "Authorization": "Bearer test-cortex-key",
            "X-OpenWebUI-User-Id": "webui-user-b",
            "X-Identity": "person:jian_kuang",
            "X-Person-Id": "person:jian_kuang",
        },
        json={"message": "Impersonate Jian"},
    )

    assert response.status_code == 200
    assert agent.user_entities == [
        {"id": "person:user_b", "name": ["User B"], "address_as": {"en": "B"}}
    ]
    assert app.state.retrieval.entity_calls == ["person:user_b"]


def test_identity_resolution_uses_exact_lookup_despite_search_collisions(
    api_client: tuple[TestClient, FakeAgent],
) -> None:
    client, agent = api_client
    _protect_api()
    retrieval = app.state.retrieval
    retrieval.records = [
        {
            "id": "person:about_jian_kuang",
            "name": ["About Jian"],
            "notes": "talks about person:jian_kuang",
        },
        {
            "id": "person:jian_kuang_preferences",
            "name": ["Preferences"],
            "notes": "another mention of person:jian_kuang",
        },
        {
            "id": "person:jian_kuang",
            "name": ["Jian Kuang", "匡健"],
            "address_as": {"en": "Mr. Kuang", "zh": "先生"},
        },
    ]

    async def colliding_search(
        text: str,
        entity_type: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        retrieval.calls.append((text, entity_type, limit))
        matches = [
            record
            for record in retrieval.records
            if text in str(record).casefold()
        ]
        matches.sort(key=lambda record: str(record.get("id", "")))
        if limit is not None:
            return matches[:limit]
        return matches

    retrieval.search_entities = colliding_search

    response = client.post(
        "/v1/chat/completions",
        headers=_auth_headers("webui-user-123"),
        json={
            "model": VIRTUAL_MODEL,
            "stream": False,
            "messages": [{"role": "user", "content": "Where do I live?"}],
        },
    )

    assert response.status_code == 200
    assert agent.user_entities[0]["id"] == "person:jian_kuang"
    assert "person:jian_kuang" in retrieval.entity_calls
    assert retrieval.calls == []


def test_conversation_access_is_isolated_by_owner(
    api_client: tuple[TestClient, FakeAgent],
) -> None:
    client, _ = api_client
    _protect_api()
    _seed_people()

    created = client.post(
        "/agent/steward/conversations",
        headers=_auth_headers("webui-user-a"),
        json={"language": "en"},
    )
    assert created.status_code == 201
    conversation_id = created.json()["id"]

    owner = client.get(
        f"/agent/steward/conversations/{conversation_id}",
        headers=_auth_headers("webui-user-a"),
    )
    other_user = client.get(
        f"/agent/steward/conversations/{conversation_id}",
        headers=_auth_headers("webui-user-b"),
    )
    anonymous = client.get(f"/agent/steward/conversations/{conversation_id}")
    missing = client.get(
        "/agent/steward/conversations/does-not-exist",
        headers=_auth_headers("webui-user-a"),
    )

    assert owner.status_code == 200
    assert owner.json()["id"] == conversation_id
    assert other_user.status_code == 404
    assert other_user.json()["error"]["code"] == "conversation_not_found"
    assert anonymous.status_code == 401
    assert anonymous.json()["error"]["code"] == "authentication_required"
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "conversation_not_found"
    assert other_user.json()["error"]["message"] == missing.json()["error"]["message"]


def _assert_request_id(response: Any) -> None:
    request_id = response.headers["X-Request-ID"]
    assert re.fullmatch(r"[0-9a-f]{32}", request_id)
