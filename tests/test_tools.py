import json
from typing import Any

import pytest

from home_cortex.tools import TOOLS, ToolDispatcher


class FakeRetrievalService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.error: Exception | None = None

    async def search_entities(
        self,
        text: str,
        entity_type: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        self.calls.append(
            (
                "search_entities",
                {"text": text, "entity_type": entity_type, "limit": limit},
            )
        )
        if self.error:
            raise self.error
        return [{"id": "location:test_house", "name": "Test House"}]

    async def get_entity(self, record_id: str) -> dict[str, Any] | None:
        self.calls.append(("get_entity", {"entity_id": record_id}))
        if self.error:
            raise self.error
        if record_id == "person:alex_example":
            return {
                "id": "person:alex_example",
                "name": ["Alex Example"],
                "dob": "1980-01-02",
            }
        return None

    async def get_relationships(
        self,
        entity_id: str,
        relation: str | None = None,
        direction: str | None = None,
        limit: int | None = None,
        *,
        include_ended: bool = False,
    ) -> list[dict[str, Any]]:
        self.calls.append(
            (
                "get_relationships",
                {
                    "entity_id": entity_id,
                    "relation": relation,
                    "direction": direction,
                    "limit": limit,
                    "include_ended": include_ended,
                },
            )
        )
        if self.error:
            raise self.error
        return [
            {
                "id": "lives_in:alex_location",
                "in": "person:alex_example",
                "out": "location:test_house",
                "related_entity": {
                    "id": "person:alex_example",
                    "first_name": "Alex",
                    "last_name": "Example",
                },
            }
        ]


def _dispatcher() -> tuple[ToolDispatcher, FakeRetrievalService]:
    retrieval = FakeRetrievalService()
    return ToolDispatcher(retrieval), retrieval  # type: ignore[arg-type]


def test_tool_definitions_are_json_serializable_and_read_only() -> None:
    serialized = json.dumps(TOOLS)
    names = {tool["function"]["name"] for tool in TOOLS}

    assert names == {"get_entity", "search_entities", "get_relationships"}
    assert "Person record stores date of birth in dob" in serialized
    assert "surrealql" not in serialized.lower()
    assert "execute" not in names
    assert "person, location, or space" in serialized
    assert "semantic_relation" in serialized
    assert "includes residents" in serialized
    assert all(
        tool["function"]["parameters"]["additionalProperties"] is False
        for tool in TOOLS
    )


@pytest.mark.asyncio
async def test_dispatches_entity_search_with_validated_arguments() -> None:
    dispatcher, retrieval = _dispatcher()

    response = await dispatcher.dispatch(
        "search_entities",
        {"text": "  Test House  ", "entity_type": "location", "limit": 5},
    )

    assert response["ok"] is True
    assert response["result"][0]["id"] == "location:test_house"
    assert retrieval.calls == [
        (
            "search_entities",
            {"text": "Test House", "entity_type": "location", "limit": 5},
        )
    ]


@pytest.mark.asyncio
async def test_dispatches_exact_entity_lookup() -> None:
    dispatcher, retrieval = _dispatcher()

    found = await dispatcher.dispatch(
        "get_entity",
        {"entity_id": "person:alex_example"},
    )
    missing = await dispatcher.dispatch(
        "get_entity",
        {"entity_id": "person:missing"},
    )

    assert found == {
        "ok": True,
        "tool": "get_entity",
        "result": [
            {
                "id": "person:alex_example",
                "name": ["Alex Example"],
                "dob": "1980-01-02",
            }
        ],
    }
    assert missing == {"ok": True, "tool": "get_entity", "result": []}
    assert retrieval.calls == [
        ("get_entity", {"entity_id": "person:alex_example"}),
        ("get_entity", {"entity_id": "person:missing"}),
    ]


@pytest.mark.asyncio
async def test_dispatches_relationship_lookup() -> None:
    dispatcher, retrieval = _dispatcher()

    response = await dispatcher.dispatch(
        "get_relationships",
        {"entity_id": "location:test_house", "relation": "lives_in"},
    )

    assert response["ok"] is True
    assert response["result"][0]["related_entity"]["first_name"] == "Alex"
    assert retrieval.calls[0] == (
        "get_relationships",
        {
            "entity_id": "location:test_house",
            "relation": "lives_in",
            "direction": None,
            "limit": None,
            "include_ended": False,
        },
    )


@pytest.mark.asyncio
async def test_rejects_unknown_tool_without_calling_retrieval() -> None:
    dispatcher, retrieval = _dispatcher()

    response = await dispatcher.dispatch("execute_surrealql", {"query": "DELETE *"})

    assert response == {
        "ok": False,
        "tool": "execute_surrealql",
        "error": {
            "code": "unknown_tool",
            "message": "Tool 'execute_surrealql' is not available",
            "available_tools": [
                "get_entity",
                "get_relationships",
                "search_entities",
            ],
        },
    }
    assert retrieval.calls == []


@pytest.mark.asyncio
async def test_dispatcher_rejects_tools_outside_agent_policy() -> None:
    retrieval = FakeRetrievalService()
    dispatcher = ToolDispatcher(  # type: ignore[arg-type]
        retrieval,
        allowed_tools=("search_entities",),
    )

    response = await dispatcher.dispatch(
        "get_relationships",
        {"entity_id": "location:test_house"},
    )

    assert response["ok"] is False
    assert response["error"]["code"] == "unknown_tool"
    assert response["error"]["available_tools"] == ["search_entities"]
    assert retrieval.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        None,
        {},
        {"text": "   "},
        {"text": "home", "limit": True},
        {"text": "home", "limit": 101},
        {"text": "home", "unexpected": "value"},
    ],
)
async def test_rejects_invalid_search_arguments(arguments: Any) -> None:
    dispatcher, retrieval = _dispatcher()

    response = await dispatcher.dispatch("search_entities", arguments)

    assert response["ok"] is False
    assert response["error"]["code"] == "invalid_arguments"
    assert retrieval.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "entity_id",
    ["location", "location:", "bad-table:one", "location:test:extra"],
)
async def test_rejects_invalid_record_ids(entity_id: str) -> None:
    dispatcher, retrieval = _dispatcher()

    response = await dispatcher.dispatch(
        "get_relationships",
        {"entity_id": entity_id},
    )

    assert response["ok"] is False
    assert response["error"]["code"] == "invalid_arguments"
    assert retrieval.calls == []


@pytest.mark.asyncio
async def test_returns_safe_error_when_retrieval_rejects_arguments() -> None:
    dispatcher, retrieval = _dispatcher()
    retrieval.error = ValueError("Unknown entity type 'vehicle'")

    response = await dispatcher.dispatch(
        "search_entities",
        {"text": "car", "entity_type": "vehicle"},
    )

    assert response["ok"] is False
    assert response["error"] == {
        "code": "invalid_arguments",
        "message": "Unknown entity type 'vehicle'",
    }


@pytest.mark.asyncio
async def test_does_not_expose_internal_execution_errors() -> None:
    dispatcher, retrieval = _dispatcher()
    retrieval.error = RuntimeError("database password should not be returned")

    response = await dispatcher.dispatch(
        "search_entities",
        {"text": "home"},
    )

    assert response["ok"] is False
    assert response["error"] == {
        "code": "tool_execution_failed",
        "message": "The tool could not complete its read operation",
    }
