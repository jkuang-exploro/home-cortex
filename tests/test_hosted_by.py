import json
from pathlib import Path
from shutil import copytree
from typing import Any

import pytest
from surrealdb import AsyncSurreal

from home_cortex.ingestion import ingest_directory
from home_cortex.retrieval import RetrievalService


STATIC_TEST_DATA = Path(__file__).parent / "static_test_data"


class MemoryDatabase:
    def __init__(self) -> None:
        self.client = AsyncSurreal("mem://")

    async def connect(self) -> None:
        await self.client.connect()
        await self.client.use("test", "hosted_by")

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
async def test_hosted_by_ingests_multi_segment_ids_once() -> None:
    database = MemoryDatabase()
    await database.connect()
    try:
        await ingest_directory(database, STATIC_TEST_DATA)  # type: ignore[arg-type]
        stored = await database.query("SELECT * FROM hosted_by ORDER BY id;")
        inverse_table = await database.query(
            "SELECT * FROM type::table($table);", {"table": "hosts_space"}
        )
    finally:
        await database.close()

    assert len(stored) == 4
    assert inverse_table == []
    assert {
        f"{edge['in'].table_name}:{edge['in'].id}" for edge in stored
    } == {
        "space:test_house:kitchen:fridge_01:interior",
        "space:test_house:kitchen:fridge_01:freezer",
        "space:drawer_1:interior",
        "space:kitchen",
    }
    assert all(edge["in"].table_name == "space" for edge in stored)
    assert all(edge["out"].table_name == "item" for edge in stored)


@pytest.mark.asyncio
async def test_repeated_ingestion_keeps_one_canonical_hosted_by_edge_per_pair() -> None:
    database = MemoryDatabase()
    await database.connect()
    try:
        await ingest_directory(database, STATIC_TEST_DATA)  # type: ignore[arg-type]
        first = await database.query("SELECT id FROM hosted_by ORDER BY id;")
        await ingest_directory(database, STATIC_TEST_DATA)  # type: ignore[arg-type]
        second = await database.query("SELECT id FROM hosted_by ORDER BY id;")
    finally:
        await database.close()

    assert [str(edge["id"]) for edge in first] == [
        str(edge["id"]) for edge in second
    ]
    assert len(second) == 4


@pytest.mark.asyncio
async def test_duplicate_hosted_by_pair_is_rejected(tmp_path: Path) -> None:
    data_dir = tmp_path / "static_test_data"
    copytree(STATIC_TEST_DATA, data_dir)
    path = data_dir / "edges" / "hosted_by.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps([records[0], records[0]]), encoding="utf-8")
    database = MemoryDatabase()
    await database.connect()
    try:
        with pytest.raises(ValueError, match="Duplicate hosted_by relationship"):
            await ingest_directory(database, data_dir)  # type: ignore[arg-type]
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_hosted_space_rejects_multiple_host_items(tmp_path: Path) -> None:
    data_dir = tmp_path / "static_test_data"
    copytree(STATIC_TEST_DATA, data_dir)
    item_path = data_dir / "nodes" / "item.json"
    items = json.loads(item_path.read_text(encoding="utf-8"))
    items.append({"id": "item:fridge_02", "name": ["Second refrigerator"]})
    item_path.write_text(json.dumps(items), encoding="utf-8")
    edge_path = data_dir / "edges" / "hosted_by.json"
    edges = json.loads(edge_path.read_text(encoding="utf-8"))
    edges.append(
        {
            "from": "space:test_house:kitchen:fridge_01:interior",
            "to": "item:fridge_02",
        }
    )
    edge_path.write_text(json.dumps(edges), encoding="utf-8")
    database = MemoryDatabase()
    await database.connect()
    try:
        with pytest.raises(ValueError, match="allows only one target"):
            await ingest_directory(database, data_dir)  # type: ignore[arg-type]
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_empty_hosted_by_file_prunes_previously_stored_facts(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "static_test_data"
    copytree(STATIC_TEST_DATA, data_dir)
    database = MemoryDatabase()
    await database.connect()
    try:
        await ingest_directory(database, data_dir)  # type: ignore[arg-type]
        before = await database.query("SELECT id FROM hosted_by;")
        (data_dir / "edges" / "hosted_by.json").write_text(
            "[]",
            encoding="utf-8",
        )
        await ingest_directory(database, data_dir)  # type: ignore[arg-type]
        after = await database.query("SELECT id FROM hosted_by;")
    finally:
        await database.close()

    assert len(before) == 4
    assert after == []


@pytest.mark.asyncio
async def test_hosted_by_rejects_reversed_endpoints_and_temporal_fields(
    tmp_path: Path,
) -> None:
    for record, message in (
        (
            {"from": "item:fridge_01", "to": "space:kitchen"},
            "Invalid hosted_by endpoints",
        ),
        (
            {
                "from": "space:test_house:kitchen:fridge_01:interior",
                "to": "item:fridge_01",
                "start": "2026-01-01",
            },
            "Non-temporal relationship",
        ),
    ):
        data_dir = tmp_path / message.replace(" ", "_")
        copytree(STATIC_TEST_DATA, data_dir)
        (data_dir / "edges" / "hosted_by.json").write_text(
            json.dumps([record]), encoding="utf-8"
        )
        database = MemoryDatabase()
        await database.connect()
        try:
            with pytest.raises(ValueError, match=message):
                await ingest_directory(database, data_dir)  # type: ignore[arg-type]
        finally:
            await database.close()


@pytest.mark.asyncio
async def test_forward_and_inverse_hosted_space_traversal_is_deterministic() -> None:
    database = MemoryDatabase()
    await database.connect()
    try:
        await ingest_directory(database, STATIC_TEST_DATA)  # type: ignore[arg-type]
        service = RetrievalService(  # type: ignore[arg-type]
            database, limit=20, data_dir=STATIC_TEST_DATA
        )
        interior_id = "space:test_house:kitchen:fridge_01:interior"
        interior = await service.get_entity(interior_id)
        host = await service.get_relationships(interior_id, relation="hosted_by")
        hosted_spaces = await service.get_relationships(
            "item:fridge_01", relation="hosts_space"
        )
        drawer_spaces = await service.get_relationships(
            "item:drawer_1", relation="hosts_space"
        )
        lamp_spaces = await service.get_relationships(
            "item:lamp", relation="hosts_space"
        )
    finally:
        await database.close()

    assert interior is not None and interior["id"] == interior_id
    assert [edge["related_entity"]["id"] for edge in host] == ["item:fridge_01"]
    assert host[0]["relation"] == host[0]["semantic_relation"] == "hosted_by"
    assert {
        edge["related_entity"]["id"] for edge in hosted_spaces
    } == {
        "space:test_house:kitchen:fridge_01:interior",
        "space:test_house:kitchen:fridge_01:freezer",
    }
    assert all(edge["relation"] == "hosted_by" for edge in hosted_spaces)
    assert all(edge["semantic_relation"] == "hosts_space" for edge in hosted_spaces)
    assert [edge["related_entity"]["id"] for edge in drawer_spaces] == [
        "space:drawer_1:interior"
    ]
    assert lamp_spaces == []


@pytest.mark.asyncio
async def test_spatial_relationships_remain_distinct_for_nested_contents() -> None:
    database = MemoryDatabase()
    await database.connect()
    try:
        await ingest_directory(database, STATIC_TEST_DATA)  # type: ignore[arg-type]
        service = RetrievalService(  # type: ignore[arg-type]
            database, limit=20, data_dir=STATIC_TEST_DATA
        )
        fridge_location = await service.get_relationships(
            "item:fridge_01", relation="located_in"
        )
        milk_location = await service.get_relationships(
            "item:milk", relation="located_in"
        )
        interior_host = await service.get_relationships(
            "space:test_house:kitchen:fridge_01:interior",
            relation="hosted_by",
        )
        kitchen_host = await service.get_relationships(
            "space:kitchen", relation="hosted_by"
        )
        house_location = await service.get_relationships(
            "item:test_house", relation="located_in"
        )
    finally:
        await database.close()

    assert fridge_location[0]["related_entity"]["id"] == "space:kitchen"
    assert (
        milk_location[0]["related_entity"]["id"]
        == "space:test_house:kitchen:fridge_01:interior"
    )
    assert interior_host[0]["related_entity"]["id"] == "item:fridge_01"
    assert kitchen_host[0]["related_entity"]["id"] == "item:test_house"
    assert house_location[0]["related_entity"]["id"] == "location:test_house"
    assert {
        fridge_location[0]["relation"],
        milk_location[0]["relation"],
        interior_host[0]["relation"],
        kitchen_host[0]["relation"],
        house_location[0]["relation"],
    } == {"located_in", "hosted_by"}


@pytest.mark.asyncio
async def test_item_rejects_multiple_current_locations(tmp_path: Path) -> None:
    data_dir = tmp_path / "static_test_data"
    copytree(STATIC_TEST_DATA, data_dir)
    space_path = data_dir / "nodes" / "space.json"
    spaces = json.loads(space_path.read_text(encoding="utf-8"))
    spaces.append(
        {
            "id": "space:pantry",
            "space_type": "storage",
            "name": ["Pantry"],
        }
    )
    space_path.write_text(json.dumps(spaces), encoding="utf-8")
    located_path = data_dir / "edges" / "located_in.json"
    located = json.loads(located_path.read_text(encoding="utf-8"))
    located.append({"from": "item:fridge_01", "to": "space:pantry"})
    located_path.write_text(json.dumps(located), encoding="utf-8")
    database = MemoryDatabase()
    await database.connect()
    try:
        with pytest.raises(ValueError, match="allows only one target"):
            await ingest_directory(database, data_dir)  # type: ignore[arg-type]
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_item_container_lookup_chain_returns_stored_contents() -> None:
    database = MemoryDatabase()
    await database.connect()
    try:
        await ingest_directory(database, STATIC_TEST_DATA)  # type: ignore[arg-type]
        service = RetrievalService(  # type: ignore[arg-type]
            database, limit=20, data_dir=STATIC_TEST_DATA
        )
        containers = await service.search_entities("冰箱", entity_type="item")
        hosted_spaces = await service.get_relationships(
            containers[0]["id"], relation="hosts_space"
        )
        contents = []
        for edge in hosted_spaces:
            contents.extend(
                await service.get_relationships(
                    edge["related_entity"]["id"],
                    relation="located_in",
                )
            )
    finally:
        await database.close()

    assert [entity["id"] for entity in containers] == ["item:fridge_01"]
    assert {
        edge["related_entity"]["id"] for edge in hosted_spaces
    } == {
        "space:test_house:kitchen:fridge_01:interior",
        "space:test_house:kitchen:fridge_01:freezer",
    }
    assert [edge["related_entity"]["id"] for edge in contents] == ["item:milk"]
