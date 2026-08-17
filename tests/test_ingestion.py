import json
from pathlib import Path
from shutil import copytree
from typing import Any

import pytest
from surrealdb import AsyncSurreal

from home_cortex.ingestion import ingest_directory

STATIC_TEST_DATA = Path(__file__).parent / "static_test_data"


class MemoryDatabase:
    def __init__(self) -> None:
        self.client = AsyncSurreal("mem://")

    async def connect(self) -> None:
        await self.client.connect()
        await self.client.use("test", "test")

    async def close(self) -> None:
        await self.client.close()

    async def upsert(self, record: Any, data: dict[str, Any]) -> Any:
        return await self.client.upsert(record, data)

    async def query(
        self,
        statement: str,
        variables: dict[str, Any] | None = None,
    ) -> Any:
        return await self.client.query(statement, variables or {})


@pytest.mark.asyncio
async def test_repeated_ingestion_does_not_duplicate_relationships() -> None:
    database = MemoryDatabase()
    await database.connect()
    try:
        first = await ingest_directory(  # type: ignore[arg-type]
            database,
            STATIC_TEST_DATA,
        )
        second = await ingest_directory(  # type: ignore[arg-type]
            database,
            STATIC_TEST_DATA,
        )
        edges = await database.query("SELECT * FROM resides_in;")
    finally:
        await database.close()

    assert first.nodes_upserted == second.nodes_upserted == 3
    assert first.edges_upserted == second.edges_upserted == 2
    assert sorted(str(edge["id"]) for edge in edges) == [
        "resides_in:blair_primary",
        "resides_in:person_alex_example__location_test_house",
    ]


@pytest.mark.asyncio
async def test_explicit_relationship_id_is_stable() -> None:
    database = MemoryDatabase()
    await database.connect()
    try:
        await ingest_directory(  # type: ignore[arg-type]
            database,
            STATIC_TEST_DATA,
        )
        edges = await database.query("SELECT * FROM resides_in;")
    finally:
        await database.close()

    assert "resides_in:blair_primary" in [str(edge["id"]) for edge in edges]


@pytest.mark.asyncio
async def test_duplicate_implicit_relationships_require_ids(tmp_path: Path) -> None:
    data_dir = tmp_path / "static_test_data"
    copytree(STATIC_TEST_DATA, data_dir)
    edge = {
        "from": "person:alex_example",
        "to": "location:test_house",
    }
    (data_dir / "edges" / "resides_in.json").write_text(
        json.dumps([edge, {**edge, "residence_type": "historical"}]),
        encoding="utf-8",
    )
    database = MemoryDatabase()
    await database.connect()
    try:
        with pytest.raises(ValueError, match="unique 'id'"):
            await ingest_directory(database, data_dir)  # type: ignore[arg-type]
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_node_name_must_be_a_non_empty_string_list(tmp_path: Path) -> None:
    data_dir = tmp_path / "static_test_data"
    copytree(STATIC_TEST_DATA, data_dir)
    location_path = data_dir / "nodes" / "location.json"
    location = json.loads(location_path.read_text(encoding="utf-8"))
    location["name"] = "Test House"
    location_path.write_text(json.dumps(location), encoding="utf-8")

    database = MemoryDatabase()
    await database.connect()
    try:
        with pytest.raises(ValueError, match="non-empty list of strings"):
            await ingest_directory(database, data_dir)  # type: ignore[arg-type]
    finally:
        await database.close()
