import json
from typing import Any

import pytest

from home_cortex.tools import TOOLS, ToolDispatcher, get_tool_definitions


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
        return [{"id": "address:test_house", "name": "Test House"}]

    async def resolve_entity_alias(
        self,
        text: str,
        entity_type: str | None = None,
        limit: int | None = None,
        *,
        speaker_id: str | None = None,
        household_id: str | None = None,
    ) -> list[dict[str, Any]]:
        self.calls.append(
            (
                "resolve_entity_alias",
                {
                    "text": text,
                    "entity_type": entity_type,
                    "limit": limit,
                    "speaker_id": speaker_id,
                    "household_id": household_id,
                },
            )
        )
        return [{"id": "person:alex_example", "name": ["Alex Example"]}]

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
        include_residents: bool = True,
    ) -> list[dict[str, Any]]:
        options = {
            "entity_id": entity_id,
            "relation": relation,
            "direction": direction,
            "limit": limit,
            "include_ended": include_ended,
        }
        if not include_residents:
            options["include_residents"] = False
        self.calls.append(
            (
                "get_relationships",
                options,
            )
        )
        if self.error:
            raise self.error
        return [
            {
                "id": "lives_in:alex_address",
                "in": "person:alex_example",
                "out": "address:test_house",
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

    assert names == {
        "calculate",
        "calendar.check_availability",
        "calendar.list_events",
        "get_entity",
        "get_relationships",
        "search_entities",
    }
    assert "date of birth in dob" not in serialized
    assert "Semantic property names" in serialized
    assert "surrealql" not in serialized.lower()
    assert "execute" not in names
    assert "person, address, space, or item" in serialized
    assert "hosts_space" in serialized
    assert "semantic_relation" in serialized
    assert "includes residents" in serialized
    assert "If complete is false" in serialized
    assert "date-only end equal to start" in serialized
    assert all(
        tool["function"]["parameters"]["additionalProperties"] is False
        for tool in TOOLS
    )


@pytest.mark.asyncio
async def test_dispatches_entity_search_with_validated_arguments() -> None:
    dispatcher, retrieval = _dispatcher()

    response = await dispatcher.dispatch(
        "search_entities",
        {"text": "  Test House  ", "entity_type": "address", "limit": 5},
    )

    assert response["ok"] is True
    assert response["result"][0]["id"] == "address:test_house"
    assert retrieval.calls == [
        (
            "search_entities",
            {"text": "Test House", "entity_type": "address", "limit": 5},
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
        {"entity_id": "address:test_house", "relation": "lives_in"},
    )

    assert response["ok"] is True
    assert response["result"][0]["related_entity"]["first_name"] == "Alex"
    assert retrieval.calls[0] == (
        "get_relationships",
        {
            "entity_id": "address:test_house",
            "relation": "lives_in",
            "direction": None,
            "limit": None,
            "include_ended": False,
        },
    )


@pytest.mark.asyncio
async def test_relationship_lookup_can_skip_resident_roster_expansion() -> None:
    dispatcher, retrieval = _dispatcher()

    response = await dispatcher.dispatch(
        "get_relationships",
        {
            "entity_id": "person:alex_example",
            "relation": "lives_in",
            "include_residents": False,
        },
    )

    assert response["ok"] is True
    assert retrieval.calls[0][1]["include_residents"] is False


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
                "calculate",
                "calendar.check_availability",
                "calendar.list_events",
                "get_entity",
                "get_relationships",
                "search_entities",
            ],
        },
    }
    assert retrieval.calls == []


@pytest.mark.asyncio
async def test_alias_resolution_is_resolver_only_not_model_facing() -> None:
    dispatcher, retrieval = _dispatcher()

    public = await dispatcher.dispatch(
        "resolve_entity_alias",
        {"text": "Alex", "entity_type": "person"},
    )
    internal = await dispatcher.dispatch_internal(
        "resolve_entity_alias",
        {"text": "Alex", "entity_type": "person"},
    )

    assert public["ok"] is False
    assert public["error"]["code"] == "unknown_tool"
    assert "resolve_entity_alias" not in public["error"]["available_tools"]
    assert internal["ok"] is True
    assert internal["result"][0]["id"] == "person:alex_example"
    assert retrieval.calls == [
        (
            "resolve_entity_alias",
            {
                "text": "Alex",
                "entity_type": "person",
                "limit": None,
                "speaker_id": None,
                "household_id": None,
            },
        )
    ]


@pytest.mark.asyncio
async def test_dispatcher_rejects_tools_outside_agent_policy() -> None:
    retrieval = FakeRetrievalService()
    dispatcher = ToolDispatcher(  # type: ignore[arg-type]
        retrieval,
        allowed_tools=("search_entities",),
    )

    response = await dispatcher.dispatch(
        "get_relationships",
        {"entity_id": "address:test_house"},
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
    ["address", "address:", "bad-table:one", "address:test::extra"],
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
async def test_accepts_multi_segment_space_ids_for_inverse_traversal() -> None:
    dispatcher, retrieval = _dispatcher()

    response = await dispatcher.dispatch(
        "get_relationships",
        {
            "entity_id": "space:home:kitchen:fridge_01:interior",
            "relation": "hosted_by",
        },
    )

    assert response["ok"] is True
    assert retrieval.calls == [
        (
            "get_relationships",
            {
                "entity_id": "space:home:kitchen:fridge_01:interior",
                "relation": "hosted_by",
                "direction": None,
                "limit": None,
                "include_ended": False,
            },
        )
    ]


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


def test_calculate_and_calendar_can_be_granted_without_graph_tools() -> None:
    definitions = get_tool_definitions(
        ("calculate", "calendar.list_events", "calendar.check_availability")
    )

    assert tuple(tool["function"]["name"] for tool in definitions) == (
        "calculate",
        "calendar.list_events",
        "calendar.check_availability",
    )


@pytest.mark.asyncio
async def test_calculate_returns_structured_numeric_result() -> None:
    dispatcher, retrieval = _dispatcher()

    response = await dispatcher.dispatch("calculate", {"expression": "2 + 3 * 4"})

    assert response == {
        "ok": True,
        "tool": "calculate",
        "result": {"result": 14},
    }
    assert retrieval.calls == []


@pytest.mark.asyncio
async def test_calculate_rejects_arbitrary_code() -> None:
    dispatcher, retrieval = _dispatcher()

    response = await dispatcher.dispatch(
        "calculate",
        {"expression": "__import__('os').system('echo pwned')"},
    )

    assert response["ok"] is False
    assert response["error"]["code"] == "invalid_arguments"
    assert retrieval.calls == []
