import json
from pathlib import Path
from typing import Any

import pytest

from home_cortex.ingestion import ingest_directory


class FakeDatabase:
    def __init__(self) -> None:
        self.upserts: list[tuple[Any, dict[str, Any]]] = []
        self.queries: list[tuple[str, dict[str, Any]]] = []

    async def upsert(self, record: Any, data: dict[str, Any]) -> None:
        self.upserts.append((record, data))

    async def query(self, statement: str, variables: dict[str, Any]) -> None:
        self.queries.append((statement, variables))


@pytest.mark.asyncio
async def test_ingests_nodes_before_edges(tmp_path: Path) -> None:
    nodes = tmp_path / "nodes"
    edges = tmp_path / "edges"
    nodes.mkdir()
    edges.mkdir()
    (nodes / "person.json").write_text(
        json.dumps({"id": "person:alice", "name": "Alice"}),
        encoding="utf-8",
    )
    (edges / "resides_in.json").write_text(
        json.dumps(
            {
                "from": "person:alice",
                "to": "home:main",
                "residence_type": "primary",
            }
        ),
        encoding="utf-8",
    )
    database = FakeDatabase()

    result = await ingest_directory(database, tmp_path)  # type: ignore[arg-type]

    assert result.nodes_upserted == 1
    assert result.edges_replaced == 1
    assert database.upserts[0][1] == {"name": "Alice"}
    assert database.queries[0][1]["content"] == {"residence_type": "primary"}
