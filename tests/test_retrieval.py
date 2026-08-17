from pathlib import Path
from typing import Any

import pytest
from surrealdb import AsyncSurreal, RecordID

from home_cortex.ingestion import ingest_directory
from home_cortex.retrieval import RetrievalService, to_json_value

STATIC_TEST_DATA = Path(__file__).parent / "static_test_data"


class FakeDatabase:
    def __init__(self, results: dict[str, list[dict[str, Any]]]) -> None:
        self.results = results
        self.queries: list[tuple[str, dict[str, Any]]] = []

    async def query(
        self,
        statement: str,
        variables: dict[str, Any],
    ) -> list[dict[str, Any]]:
        self.queries.append((statement, variables))
        table = variables.get("table") or variables.get("relation")
        return self.results.get(str(table), [])


class MemoryDatabase:
    def __init__(self) -> None:
        self.client = AsyncSurreal("mem://")

    async def connect(self) -> None:
        await self.client.connect()
        await self.client.use("test", "test")

    async def close(self) -> None:
        await self.client.close()

    async def query(
        self,
        statement: str,
        variables: dict[str, Any] | None = None,
    ) -> Any:
        return await self.client.query(statement, variables or {})

    async def upsert(self, record: Any, data: dict[str, Any]) -> Any:
        return await self.client.upsert(record, data)


@pytest.mark.asyncio
async def test_search_entities_queries_known_tables_and_sorts_globally() -> None:
    database = FakeDatabase(
        {
            "location": [
                {
                    "id": RecordID("location", "fort_cerritos"),
                    "name": "Fort Cerritos",
                }
            ],
            "person": [{"id": RecordID("person", "jian"), "first_name": "Jian"}],
        }
    )
    service = RetrievalService(database, limit=10)  # type: ignore[arg-type]

    result = await service.search_entities("  CERRITOS  ")

    assert [record["id"] for record in result] == [
        "location:fort_cerritos",
        "person:jian",
    ]
    assert [variables["table"] for _, variables in database.queries] == [
        "location",
        "person",
    ]
    assert all(variables["text"] == "cerritos" for _, variables in database.queries)
    assert "type::table($table)" in database.queries[0][0]


@pytest.mark.asyncio
async def test_search_entities_can_restrict_entity_type_and_limit() -> None:
    database = FakeDatabase(
        {
            "person": [
                {"id": RecordID("person", "b")},
                {"id": RecordID("person", "a")},
            ]
        }
    )
    service = RetrievalService(database, limit=10)  # type: ignore[arg-type]

    result = await service.search_entities("person", entity_type="person", limit=1)

    assert result == [{"id": "person:a"}]
    assert len(database.queries) == 1
    assert database.queries[0][1]["limit"] == 1


@pytest.mark.asyncio
async def test_search_entities_rejects_unknown_type_empty_text_and_bad_limit() -> None:
    database = FakeDatabase({})
    service = RetrievalService(database, limit=10)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="cannot be empty"):
        await service.search_entities("  ")
    with pytest.raises(ValueError, match="Unknown entity type"):
        await service.search_entities("test", entity_type="secret_table")
    with pytest.raises(ValueError, match="between 1 and 10"):
        await service.search_entities("test", limit=11)

    assert database.queries == []


@pytest.mark.asyncio
async def test_get_relationships_returns_direction_and_filters_relation() -> None:
    database = FakeDatabase(
        {
            "resides_in": [
                {
                    "id": RecordID("resides_in", "person_a__location_main"),
                    "in": RecordID("person", "a"),
                    "out": RecordID("location", "main"),
                    "source_entity": {
                        "id": RecordID("person", "a"),
                        "first_name": "Alex",
                    },
                    "target_entity": {
                        "id": RecordID("location", "main"),
                        "name": "Main Home",
                    },
                }
            ]
        }
    )
    service = RetrievalService(database, limit=10)  # type: ignore[arg-type]

    incoming = await service.get_relationships(
        "location:main",
        relation="resides_in",
    )
    outgoing = await service.get_relationships(
        "person:a",
        relation="resides_in",
    )

    assert incoming[0]["direction"] == "incoming"
    assert outgoing[0]["direction"] == "outgoing"
    assert incoming[0]["relation"] == "resides_in"
    assert incoming[0]["related_entity"] == {
        "id": "person:a",
        "first_name": "Alex",
    }
    assert outgoing[0]["related_entity"] == {
        "id": "location:main",
        "name": "Main Home",
    }
    assert "source_entity" not in incoming[0]
    assert "target_entity" not in incoming[0]
    assert database.queries[0][1]["entity"] == RecordID("location", "main")
    assert "in = $entity OR out = $entity" in database.queries[0][0]
    assert "in.* AS source_entity" in database.queries[0][0]


@pytest.mark.asyncio
async def test_get_relationships_rejects_unapproved_inputs() -> None:
    database = FakeDatabase({})
    service = RetrievalService(database)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="table:record_id"):
        await service.get_relationships("not-a-record")
    with pytest.raises(ValueError, match="Unknown relation"):
        await service.get_relationships("person:a", relation="secret_relation")

    assert database.queries == []


def test_table_names_come_from_static_test_data() -> None:
    service = RetrievalService(  # type: ignore[arg-type]
        FakeDatabase({}),
        data_dir=STATIC_TEST_DATA,
    )

    assert service.node_tables == ("location", "person")
    assert service.edge_tables == ("resides_in",)


def test_record_id_is_serialized() -> None:
    assert to_json_value(RecordID("person", "alice")) == "person:alice"


@pytest.mark.asyncio
async def test_queries_execute_against_embedded_surrealdb() -> None:
    database = MemoryDatabase()
    await database.connect()
    try:
        await ingest_directory(  # type: ignore[arg-type]
            database,
            STATIC_TEST_DATA,
        )
        service = RetrievalService(  # type: ignore[arg-type]
            database,
            limit=10,
            data_dir=STATIC_TEST_DATA,
        )

        entities = await service.search_entities(
            "test house",
            entity_type="location",
        )
        relationships = await service.get_relationships(
            "location:test_house",
            relation="resides_in",
        )
        context = await service.retrieve("test house")
    finally:
        await database.close()

    assert [entity["id"] for entity in entities] == ["location:test_house"]
    assert sorted(edge["in"] for edge in relationships) == [
        "person:alex_example",
        "person:blair_example",
    ]
    assert all(edge["out"] == "location:test_house" for edge in relationships)
    assert all(edge["direction"] == "incoming" for edge in relationships)
    assert sorted(
        edge["related_entity"]["first_name"] for edge in relationships
    ) == ["Alex", "Blair"]
    assert sorted(person["first_name"] for person in context.nodes["person"]) == [
        "Alex",
        "Blair",
    ]
    assert all(
        "related_entity" not in edge for edge in context.edges["resides_in"]
    )
