import json
import asyncio
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from surrealdb import AsyncSurreal, RecordID

from home_cortex.edge_schema import EdgeSchemaRegistry
from home_cortex.ingestion import ingest_directory
from home_cortex.record_ids import canonical_record_id
from home_cortex.schema_catalog import RuntimeSchemaCatalog
from home_cortex.tools import ToolDispatcher, get_tool_definitions
from home_cortex.writing import ItemWritingService, MutationResult


STATIC_TEST_DATA = Path(__file__).parent / "static_test_data"


class MemoryDatabase:
    def __init__(self) -> None:
        self.client = AsyncSurreal("mem://")
        self.queries: list[tuple[str, dict[str, Any]]] = []

    async def connect(self) -> None:
        await self.client.connect()
        await self.client.use("writing", "writing")

    async def close(self) -> None:
        await self.client.close()

    async def query(
        self, statement: str, variables: dict[str, Any] | None = None
    ) -> Any:
        values = variables or {}
        self.queries.append((statement, values))
        return await self.client.query(statement, values)

    async def upsert(self, record: Any, data: dict[str, Any]) -> Any:
        return await self.client.upsert(record, data)


class FailingTransactionDatabase:
    """Insert a database-side THROW after the first destructive statement."""

    def __init__(self, database: MemoryDatabase) -> None:
        self.database = database

    async def query(
        self, statement: str, variables: dict[str, Any] | None = None
    ) -> Any:
        if statement.lstrip().startswith("BEGIN TRANSACTION"):
            if "RELATE" in statement:
                statement = statement.replace(
                    "RELATE", 'THROW "FORCED_WRITE_FAILURE";\nRELATE', 1
                )
            else:
                statement = statement.replace(
                    "DELETE $item;",
                    'THROW "FORCED_WRITE_FAILURE";\nDELETE $item;',
                    1,
                )
        return await self.database.query(statement, variables)


class DummyRetrieval:
    pass


@pytest_asyncio.fixture
async def writing_service():
    database = MemoryDatabase()
    await database.connect()
    registry = EdgeSchemaRegistry.load_default(STATIC_TEST_DATA)
    await ingest_directory(database, STATIC_TEST_DATA, registry)  # type: ignore[arg-type]
    catalog = RuntimeSchemaCatalog.from_data_dir(STATIC_TEST_DATA, registry)
    service = ItemWritingService(database, catalog, registry)  # type: ignore[arg-type]
    try:
        yield service, database, registry, catalog
    finally:
        await database.close()


def create_request(mode="commit"):
    return {
        "operation": "create",
        "item": {
            "id": "item:test_drill",
            "type": "item",
            "properties": {
                "name": {"en": "Test drill", "zh": "测试电钻"},
                "item_type": "tool",
            },
        },
        "location_id": "space:kitchen",
        "mode": mode,
        "source": {"type": "vision", "session_id": "perception:test_001"},
    }


async def record(database, entity_id):
    value = await database.query(
        "SELECT * FROM ONLY $record;",
        {"record": RecordID(*entity_id.split(":", 1))},
    )
    return value


async def locations(database, item_id="test_drill"):
    rows = await database.query(
        "SELECT VALUE out FROM located_in WHERE in = $item ORDER BY out;",
        {"item": RecordID("item", item_id)},
    )
    return [canonical_record_id(value) for value in rows]


def transaction_count(database):
    return sum(
        statement.lstrip().startswith("BEGIN TRANSACTION")
        for statement, _ in database.queries
    )


@pytest.mark.asyncio
async def test_create_item_and_location_atomically(writing_service):
    service, database, *_ = writing_service
    result = await service.mutate(create_request())

    assert result.status == "APPLIED"
    assert result.operation == "create"
    assert result.entity_id == "item:test_drill"
    assert result.affected_nodes == result.affected_edges == 1
    assert result.source.session_id == "perception:test_001"
    assert (await record(database, "item:test_drill"))["item_type"] == "tool"
    assert "source" not in (await record(database, "item:test_drill"))
    assert await locations(database) == ["space:kitchen"]


@pytest.mark.asyncio
async def test_create_conflict_missing_location_and_invalid_schema(writing_service):
    service, database, *_ = writing_service
    assert (await service.mutate(create_request())).status == "APPLIED"
    conflict = await service.mutate(create_request())
    assert conflict.status == "CONFLICT"
    assert conflict.reason == "ENTITY_ALREADY_EXISTS"

    missing = create_request()
    missing["item"]["id"] = "item:missing_location_item"
    missing["location_id"] = "space:missing"
    rejected = await service.mutate(missing)
    assert (rejected.status, rejected.reason) == ("REJECTED", "LOCATION_NOT_FOUND")

    invalid = create_request()
    invalid["item"]["id"] = "item:invalid_schema_item"
    invalid["location_id"] = "person:alex_example"
    invalid["item"]["properties"]["arbitrary"] = True
    rejected = await service.mutate(invalid)
    assert (rejected.status, rejected.reason) == (
        "REJECTED", "UNKNOWN_ITEM_PROPERTY"
    )
    assert await record(database, "item:missing_location_item") is None
    assert await record(database, "item:invalid_schema_item") is None


@pytest.mark.asyncio
async def test_create_preview_has_no_persistent_state(writing_service):
    service, database, *_ = writing_service
    before = transaction_count(database)
    result = await service.mutate(create_request("preview"))
    assert result.status == "PROPOSED"
    assert result.resulting_state["location"] == "space:kitchen"
    assert await record(database, "item:test_drill") is None
    assert await locations(database) == []
    assert transaction_count(database) == before


@pytest.mark.asyncio
async def test_create_rollback_removes_node_when_edge_write_fails(writing_service):
    _, database, registry, catalog = writing_service
    service = ItemWritingService(
        FailingTransactionDatabase(database), catalog, registry  # type: ignore[arg-type]
    )
    result = await service.mutate(create_request())
    assert (result.status, result.reason) == ("REJECTED", "TRANSACTION_FAILED")
    assert await record(database, "item:test_drill") is None
    assert await locations(database) == []


@pytest.mark.asyncio
async def test_update_location_replaces_old_edge_and_is_idempotent(writing_service):
    service, database, *_ = writing_service
    await service.mutate(create_request())
    request = {
        "operation": "update_location",
        "item_id": "item:test_drill",
        "location_id": "space:drawer_1:interior",
        "mode": "commit",
    }
    result = await service.mutate(request)
    assert result.status == "APPLIED"
    assert result.previous_state == {"location": "space:kitchen"}
    assert result.resulting_state == {"location": "space:drawer_1:interior"}
    assert await locations(database) == ["space:drawer_1:interior"]

    before = transaction_count(database)
    unchanged = await service.mutate(request)
    assert (unchanged.status, unchanged.reason) == (
        "NO_CHANGE", "LOCATION_UNCHANGED"
    )
    assert unchanged.affected_edges == 0
    assert await locations(database) == ["space:drawer_1:interior"]
    assert transaction_count(database) == before


@pytest.mark.asyncio
async def test_update_assigns_first_location_when_edge_table_is_absent(writing_service):
    _, _, registry, catalog = writing_service
    database = MemoryDatabase()
    await database.connect()
    try:
        await database.query(
            "CREATE $item CONTENT $item_properties;"
            "CREATE $space CONTENT $space_properties;",
            {
                "item": RecordID("item", "unlocated_drill"),
                "item_properties": {"name": "Unlocated drill", "item_type": "tool"},
                "space": RecordID("space", "new_workbench"),
                "space_properties": {"name": "New workbench", "space_type": "surface"},
            },
        )
        service = ItemWritingService(database, catalog, registry)  # type: ignore[arg-type]

        result = await service.mutate({
            "operation": "update_location",
            "item_id": "item:unlocated_drill",
            "location_id": "space:new_workbench",
        })

        assert result.status == "APPLIED"
        assert result.previous_state == {"location": None}
        assert await locations(database, "unlocated_drill") == ["space:new_workbench"]
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_update_validation_preview_and_rollback(writing_service):
    service, database, registry, catalog = writing_service
    missing = await service.mutate({
        "operation": "update_location", "item_id": "item:absent",
        "location_id": "space:kitchen",
    })
    assert (missing.status, missing.reason) == ("NOT_FOUND", "ITEM_NOT_FOUND")

    await service.mutate(create_request())
    before = transaction_count(database)
    preview = await service.mutate({
        "operation": "update_location", "item_id": "item:test_drill",
        "location_id": "space:drawer_1:interior", "mode": "preview",
    })
    assert preview.status == "PROPOSED"
    assert await locations(database) == ["space:kitchen"]
    assert transaction_count(database) == before

    invalid = await service.mutate({
        "operation": "update_location", "item_id": "item:test_drill",
        "location_id": "person:alex_example",
    })
    assert (invalid.status, invalid.reason) == ("REJECTED", "INVALID_LOCATION_TYPE")

    missing_location = await service.mutate({
        "operation": "update_location", "item_id": "item:test_drill",
        "location_id": "space:missing",
    })
    assert (missing_location.status, missing_location.reason) == (
        "REJECTED", "LOCATION_NOT_FOUND"
    )

    failing = ItemWritingService(
        FailingTransactionDatabase(database), catalog, registry  # type: ignore[arg-type]
    )
    failed = await failing.mutate({
        "operation": "update_location", "item_id": "item:test_drill",
        "location_id": "space:drawer_1:interior",
    })
    assert (failed.status, failed.reason) == ("REJECTED", "TRANSACTION_FAILED")
    assert await locations(database) == ["space:kitchen"]


@pytest.mark.asyncio
async def test_multiple_current_locations_are_a_conflict(writing_service):
    service, database, *_ = writing_service
    await service.mutate(create_request())
    await database.query(
        "RELATE $item->$edge->$location CONTENT {};",
        {
            "item": RecordID("item", "test_drill"),
            "edge": RecordID("located_in", "deliberate_conflict"),
            "location": RecordID("space", "drawer_1:interior"),
        },
    )
    result = await service.mutate({
        "operation": "update_location", "item_id": "item:test_drill",
        "location_id": "space:test_house:kitchen:fridge_01:interior",
    })
    assert (result.status, result.reason) == (
        "CONFLICT", "MULTIPLE_CURRENT_LOCATIONS"
    )
    assert len(await locations(database)) == 2


@pytest.mark.asyncio
async def test_concurrent_canonical_moves_preserve_one_location(writing_service):
    service, database, *_ = writing_service
    await service.mutate(create_request())
    results = await asyncio.gather(
        service.mutate({
            "operation": "update_location", "item_id": "item:test_drill",
            "location_id": "space:drawer_1:interior",
        }),
        service.mutate({
            "operation": "update_location", "item_id": "item:test_drill",
            "location_id": "space:test_house:kitchen:fridge_01:interior",
        }),
    )
    assert [result.status for result in results] == ["APPLIED", "APPLIED"]
    assert await locations(database) == [
        "space:test_house:kitchen:fridge_01:interior"
    ]


@pytest.mark.asyncio
async def test_delete_removes_all_schema_applicable_incident_edges(writing_service):
    service, database, *_ = writing_service
    await service.mutate(create_request())
    await database.query(
        "CREATE $space CONTENT { name: { en: 'Drill case' }, space_type: 'storage' };"
        "RELATE $space->$edge->$item CONTENT {};",
        {
            "space": RecordID("space", "test_drill_case"),
            "edge": RecordID("hosted_by", "test_drill_case"),
            "item": RecordID("item", "test_drill"),
        },
    )
    result = await service.mutate({
        "operation": "delete", "item_id": "item:test_drill", "mode": "commit"
    })
    assert result.status == "APPLIED"
    assert result.affected_nodes == 1
    assert result.affected_edges == 2
    assert await record(database, "item:test_drill") is None
    dangling = await database.query(
        "SELECT * FROM located_in, hosted_by WHERE in = $item OR out = $item;",
        {"item": RecordID("item", "test_drill")},
    )
    assert dangling == []
    absent = await service.mutate({"operation": "delete", "item_id": "item:test_drill"})
    assert (absent.status, absent.reason) == ("NOT_FOUND", "ITEM_NOT_FOUND")


@pytest.mark.asyncio
async def test_delete_preview_and_rollback_preserve_node_and_edges(writing_service):
    service, database, registry, catalog = writing_service
    await service.mutate(create_request())
    before = transaction_count(database)
    preview = await service.mutate({
        "operation": "delete", "item_id": "item:test_drill", "mode": "preview"
    })
    assert preview.status == "PROPOSED"
    assert await record(database, "item:test_drill") is not None
    assert await locations(database) == ["space:kitchen"]
    assert transaction_count(database) == before

    failing = ItemWritingService(
        FailingTransactionDatabase(database), catalog, registry  # type: ignore[arg-type]
    )
    failed = await failing.mutate({"operation": "delete", "item_id": "item:test_drill"})
    assert (failed.status, failed.reason) == ("REJECTED", "TRANSACTION_FAILED")
    assert await record(database, "item:test_drill") is not None
    assert await locations(database) == ["space:kitchen"]


@pytest.mark.asyncio
async def test_malformed_and_arbitrary_query_fields_are_cleanly_rejected(writing_service):
    service, database, *_ = writing_service
    before = len(database.queries)
    for request in (
        {"operation": "drop_table", "surrealql": "DELETE item"},
        {"operation": "delete", "item_id": "item:test_drill", "where": "true"},
        {"operation": "create", "item": {"id": "person:x", "type": "item", "properties": {"name": "x"}}, "location_id": "space:kitchen"},
    ):
        result = await service.mutate(request)
        assert result.status == "REJECTED"
        assert result.reason == "INVALID_REQUEST"
        assert result.errors[0].message == (
            "The semantic item mutation request failed validation"
        )
    assert len(database.queries) == before
    assert not any(
        "DELETE item" in statement for statement, _ in database.queries
    )


@pytest.mark.asyncio
async def test_item_property_types_and_required_name_are_schema_validated(writing_service):
    service, database, *_ = writing_service
    wrong_type = create_request()
    wrong_type["item"]["id"] = "item:wrong_type"
    wrong_type["item"]["properties"]["item_type"] = ["tool"]
    result = await service.mutate(wrong_type)
    assert (result.status, result.reason) == (
        "REJECTED", "INVALID_ITEM_PROPERTY_TYPE"
    )

    missing_name = create_request()
    missing_name["item"]["id"] = "item:missing_name"
    del missing_name["item"]["properties"]["name"]
    result = await service.mutate(missing_name)
    assert (result.status, result.reason) == ("REJECTED", "ITEM_NAME_REQUIRED")
    assert await record(database, "item:wrong_type") is None
    assert await record(database, "item:missing_name") is None


def test_tool_contract_is_closed_and_has_exactly_three_intents():
    definition = get_tool_definitions(["write_item"])[0]
    schema = definition["function"]["parameters"]
    serialized = json.dumps(schema)
    assert definition["function"]["name"] == "write_item"
    branches = [schema["$defs"][ref["$ref"].rsplit("/", 1)[-1]] for ref in schema["oneOf"]]
    assert {branch["properties"]["operation"]["const"] for branch in branches} == {
        "create", "update_location", "delete"
    }
    assert "surrealql" not in serialized.lower()
    assert "where" not in serialized.lower()
    assert all(branch["additionalProperties"] is False for branch in branches)


@pytest.mark.asyncio
async def test_tool_adapter_uses_canonical_writing_service(writing_service):
    service, *_ = writing_service
    dispatcher = ToolDispatcher(
        DummyRetrieval(), ["write_item"], writing=service  # type: ignore[arg-type]
    )
    response = await dispatcher.dispatch("write_item", create_request("preview"))
    assert response["ok"] is True
    assert MutationResult.model_validate_json(
        json.dumps(response["result"])
    ).status == "PROPOSED"


def test_write_tool_must_be_explicitly_allowlisted_and_configured():
    dispatcher = ToolDispatcher(DummyRetrieval())  # type: ignore[arg-type]
    assert "write_item" not in dispatcher._public_tools
    with pytest.raises(ValueError, match="requires an ItemWritingService"):
        ToolDispatcher(DummyRetrieval(), ["write_item"])  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_result_shape_is_stable_across_all_operations(writing_service):
    service, *_ = writing_service
    requests = [
        create_request("preview"),
        {
            "operation": "update_location", "item_id": "item:missing",
            "location_id": "space:kitchen", "mode": "preview",
        },
        {"operation": "delete", "item_id": "item:missing", "mode": "preview"},
    ]
    expected_keys = set(MutationResult.model_fields)
    for request in requests:
        result = await service.mutate(request)
        assert set(result.model_dump(mode="json")) == expected_keys
