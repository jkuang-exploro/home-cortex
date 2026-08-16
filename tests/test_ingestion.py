import json
from pathlib import Path
from typing import Any

import pytest
from surrealdb import AsyncSurreal

from home_cortex.ingestion import ingest_directory


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


def _write_data(data_dir: Path, edges: list[dict[str, Any]]) -> None:
    nodes_dir = data_dir / "nodes"
    edges_dir = data_dir / "edges"
    nodes_dir.mkdir()
    edges_dir.mkdir()
    (nodes_dir / "person.json").write_text(
        json.dumps({"id": "person:alice", "name": "Alice"}),
        encoding="utf-8",
    )
    (nodes_dir / "home.json").write_text(
        json.dumps({"id": "home:main", "name": "Main Home"}),
        encoding="utf-8",
    )
    (edges_dir / "resides_in.json").write_text(
        json.dumps(edges),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_repeated_ingestion_does_not_duplicate_relationships(
    tmp_path: Path,
) -> None:
    _write_data(
        tmp_path,
        [
            {
                "from": "person:alice",
                "to": "home:main",
                "residence_type": "primary",
            }
        ],
    )
    database = MemoryDatabase()
    await database.connect()
    try:
        first = await ingest_directory(database, tmp_path)  # type: ignore[arg-type]
        second = await ingest_directory(database, tmp_path)  # type: ignore[arg-type]
        edges = await database.query("SELECT * FROM resides_in;")
    finally:
        await database.close()

    assert first.nodes_upserted == second.nodes_upserted == 2
    assert first.edges_upserted == second.edges_upserted == 1
    assert len(edges) == 1
    assert str(edges[0]["id"]) == "resides_in:person_alice__home_main"


@pytest.mark.asyncio
async def test_explicit_relationship_id_is_stable(tmp_path: Path) -> None:
    _write_data(
        tmp_path,
        [
            {
                "id": "resides_in:alice_primary",
                "from": "person:alice",
                "to": "home:main",
            }
        ],
    )
    database = MemoryDatabase()
    await database.connect()
    try:
        await ingest_directory(database, tmp_path)  # type: ignore[arg-type]
        edges = await database.query("SELECT * FROM resides_in;")
    finally:
        await database.close()

    assert [str(edge["id"]) for edge in edges] == ["resides_in:alice_primary"]


@pytest.mark.asyncio
async def test_duplicate_implicit_relationships_require_ids(tmp_path: Path) -> None:
    edge = {"from": "person:alice", "to": "home:main"}
    _write_data(tmp_path, [edge, {**edge, "residence_type": "historical"}])
    database = MemoryDatabase()
    await database.connect()
    try:
        with pytest.raises(ValueError, match="unique 'id'"):
            await ingest_directory(database, tmp_path)  # type: ignore[arg-type]
    finally:
        await database.close()
