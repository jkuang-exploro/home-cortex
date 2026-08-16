from pathlib import Path
from typing import Any

import pytest
from surrealdb import AsyncSurreal, RecordID

from home_cortex.retrieval import RetrievalService, to_json_value


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


@pytest.mark.asyncio
async def test_search_entities_queries_known_tables_and_sorts_globally() -> None:
    database = FakeDatabase(
        {
            "home": [{"id": RecordID("home", "cerritos"), "name": "Fort Cerritos"}],
            "person": [{"id": RecordID("person", "jian"), "first_name": "Jian"}],
        }
    )
    service = RetrievalService(database, limit=10)  # type: ignore[arg-type]

    result = await service.search_entities("  CERRITOS  ")

    assert [record["id"] for record in result] == ["home:cerritos", "person:jian"]
    assert [variables["table"] for _, variables in database.queries] == [
        "home",
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
                    "id": RecordID("resides_in", "person_a__home_main"),
                    "in": RecordID("person", "a"),
                    "out": RecordID("home", "main"),
                }
            ]
        }
    )
    service = RetrievalService(database, limit=10)  # type: ignore[arg-type]

    incoming = await service.get_relationships(
        "home:main",
        relation="resides_in",
    )
    outgoing = await service.get_relationships(
        "person:a",
        relation="resides_in",
    )

    assert incoming[0]["direction"] == "incoming"
    assert outgoing[0]["direction"] == "outgoing"
    assert incoming[0]["relation"] == "resides_in"
    assert database.queries[0][1]["entity"] == RecordID("home", "main")
    assert "in = $entity OR out = $entity" in database.queries[0][0]


@pytest.mark.asyncio
async def test_get_relationships_rejects_unapproved_inputs() -> None:
    database = FakeDatabase({})
    service = RetrievalService(database)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="table:record_id"):
        await service.get_relationships("not-a-record")
    with pytest.raises(ValueError, match="Unknown relation"):
        await service.get_relationships("person:a", relation="secret_relation")

    assert database.queries == []


def test_table_names_come_from_json_file_names(tmp_path: Path) -> None:
    (tmp_path / "nodes").mkdir()
    (tmp_path / "edges").mkdir()
    (tmp_path / "nodes" / "object.json").touch()
    (tmp_path / "edges" / "located_in.json").touch()

    service = RetrievalService(FakeDatabase({}), data_dir=tmp_path)  # type: ignore[arg-type]

    assert service.node_tables == ("object",)
    assert service.edge_tables == ("located_in",)


def test_record_id_is_serialized() -> None:
    assert to_json_value(RecordID("person", "alice")) == "person:alice"


@pytest.mark.asyncio
async def test_queries_execute_against_embedded_surrealdb() -> None:
    database = MemoryDatabase()
    await database.connect()
    try:
        await database.query(
            """
            CREATE person:alice CONTENT { name: 'Alice' };
            CREATE home:main CONTENT { name: 'Main Home' };
            RELATE person:alice->resides_in->home:main;
            """
        )
        service = RetrievalService(database, limit=10)  # type: ignore[arg-type]

        entities = await service.search_entities("main home", entity_type="home")
        relationships = await service.get_relationships(
            "home:main",
            relation="resides_in",
        )
    finally:
        await database.close()

    assert [entity["id"] for entity in entities] == ["home:main"]
    assert relationships[0]["in"] == "person:alice"
    assert relationships[0]["out"] == "home:main"
    assert relationships[0]["direction"] == "incoming"
