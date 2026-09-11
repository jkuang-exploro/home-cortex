import json
from datetime import date, datetime
from pathlib import Path
from shutil import copytree
from typing import Any

import pytest
from surrealdb import AsyncSurreal, RecordID

from home_cortex.export import (
    canonical_json_value,
    export_directory,
)
from home_cortex.ingestion import ingest_directory
from home_cortex.record_ids import canonical_record_id

STATIC_TEST_DATA = Path(__file__).parent / "static_test_data"
REGISTERED_EDGES = (
    "hosted_by",
    "lives_in",
    "located_in",
    "parent_of",
    "spouse_of",
)


class MemoryDatabase:
    def __init__(self) -> None:
        self.client = AsyncSurreal("mem://")

    async def connect(self) -> None:
        await self.client.connect()
        await self.client.use("test", "export")

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


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _comparable(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _records_by_id(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {record["id"]: record for record in records}


async def _factual_state(database: MemoryDatabase) -> dict[str, list[dict[str, Any]]]:
    info = await database.query("INFO FOR DB;")
    tables = sorted(info["tables"])
    state: dict[str, list[dict[str, Any]]] = {}
    for table in tables:
        rows = await database.query(
            "SELECT * FROM type::table($table);",
            {"table": table},
        )
        normalized = [canonical_json_value(row) for row in rows or []]
        normalized.sort(key=lambda row: json.dumps(row, sort_keys=True))
        if normalized:
            state[table] = normalized
    return state


def _json_tree(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*.json"))
    }


@pytest.mark.asyncio
async def test_object_round_trip_preserves_canonical_node_content(
    tmp_path: Path,
) -> None:
    database = MemoryDatabase()
    await database.connect()
    try:
        await ingest_directory(database, STATIC_TEST_DATA)  # type: ignore[arg-type]
        target = tmp_path / "export"
        await export_directory(database, target)  # type: ignore[arg-type]
    finally:
        await database.close()

    for table in ("person", "address", "space", "item"):
        source = _load_json(STATIC_TEST_DATA / "nodes" / f"{table}.json")
        exported = _load_json(target / "nodes" / f"{table}.json")
        assert _records_by_id(_comparable(exported)) == _records_by_id(
            _comparable(source)
        )


@pytest.mark.asyncio
async def test_edge_round_trip_preserves_properties_and_null_end(
    tmp_path: Path,
) -> None:
    database = MemoryDatabase()
    await database.connect()
    try:
        await ingest_directory(database, STATIC_TEST_DATA)  # type: ignore[arg-type]
        target = tmp_path / "export"
        await export_directory(database, target)  # type: ignore[arg-type]
    finally:
        await database.close()

    lives_in = _load_json(target / "edges" / "lives_in.json")
    by_from = {edge["from"]: edge for edge in lives_in}
    assert by_from["person:alex_example"]["to"] == "address:test_house"
    assert by_from["person:alex_example"]["start"] == "2020-01-01"
    assert by_from["person:alex_example"]["end"] is None
    assert by_from["person:alex_example"]["household_role"] == "owner"
    assert "id" not in by_from["person:alex_example"]
    assert by_from["person:blair_example"]["id"] == "lives_in:blair_primary"
    assert by_from["person:blair_example"]["end"] is None

    spouse_of = _load_json(target / "edges" / "spouse_of.json")
    assert spouse_of == [
        {
            "from": "person:alex_example",
            "to": "person:blair_example",
            "end": None,
            "start": "2011-03-15",
        }
    ]


@pytest.mark.asyncio
async def test_multiple_object_types_write_distinct_files(tmp_path: Path) -> None:
    database = MemoryDatabase()
    await database.connect()
    try:
        await ingest_directory(database, STATIC_TEST_DATA)  # type: ignore[arg-type]
        target = tmp_path / "export"
        result = await export_directory(database, target)  # type: ignore[arg-type]
    finally:
        await database.close()

    assert {path.name for path in (target / "nodes").glob("*.json")} == {
        "address.json",
        "item.json",
        "person.json",
        "space.json",
    }
    assert {path.name for path in (target / "edges").glob("*.json")} == {
        f"{name}.json" for name in REGISTERED_EDGES
    }
    assert result.node_files == 4
    assert result.edge_files == len(REGISTERED_EDGES)
    people = _load_json(target / "nodes" / "person.json")
    addresses = _load_json(target / "nodes" / "address.json")
    assert all(record["id"].startswith("person:") for record in people)
    assert all(record["id"].startswith("address:") for record in addresses)
    assert not (target / "nodes" / "item").exists()


@pytest.mark.asyncio
async def test_repeated_export_is_byte_identical(tmp_path: Path) -> None:
    database = MemoryDatabase()
    await database.connect()
    try:
        await ingest_directory(database, STATIC_TEST_DATA)  # type: ignore[arg-type]
        first = tmp_path / "first"
        second = tmp_path / "second"
        await export_directory(database, first)  # type: ignore[arg-type]
        await export_directory(database, second)  # type: ignore[arg-type]
    finally:
        await database.close()

    assert _json_tree(first) == _json_tree(second)


@pytest.mark.asyncio
async def test_export_does_not_leak_surrealdb_identity_fields(
    tmp_path: Path,
) -> None:
    database = MemoryDatabase()
    await database.connect()
    try:
        await ingest_directory(database, STATIC_TEST_DATA)  # type: ignore[arg-type]
        target = tmp_path / "export"
        await export_directory(database, target)  # type: ignore[arg-type]
    finally:
        await database.close()

    leaked = ("in", "out", "table_name", "record_id")
    for path in target.rglob("*.json"):
        payload = _load_json(path)
        assert isinstance(payload, list)
        for record in payload:
            assert isinstance(record, dict)
            for field in leaked:
                assert field not in record
            serialized = json.dumps(record)
            assert "RecordID" not in serialized
            assert "⟨" not in serialized
            if "id" in record:
                assert isinstance(record["id"], str)
            if path.parent.name == "edges":
                assert isinstance(record["from"], str)
                assert isinstance(record["to"], str)


@pytest.mark.asyncio
async def test_export_then_reingest_preserves_factual_state(tmp_path: Path) -> None:
    source = MemoryDatabase()
    await source.connect()
    try:
        await ingest_directory(source, STATIC_TEST_DATA)  # type: ignore[arg-type]
        exported = tmp_path / "export"
        await export_directory(source, exported)  # type: ignore[arg-type]
        original = await _factual_state(source)
    finally:
        await source.close()

    restored = MemoryDatabase()
    await restored.connect()
    try:
        await ingest_directory(restored, exported)  # type: ignore[arg-type]
        round_tripped = await _factual_state(restored)
    finally:
        await restored.close()

    assert round_tripped == original


@pytest.mark.asyncio
async def test_sharded_item_table_exports_as_one_file(tmp_path: Path) -> None:
    data_dir = tmp_path / "source"
    copytree(STATIC_TEST_DATA, data_dir)
    items = _load_json(data_dir / "nodes" / "item.json")
    (data_dir / "nodes" / "item.json").unlink()
    shard_dir = data_dir / "nodes" / "item"
    shard_dir.mkdir()
    (shard_dir / "appliance.json").write_text(
        json.dumps([item for item in items if item["item_type"] == "appliance"]),
        encoding="utf-8",
    )
    (shard_dir / "other.json").write_text(
        json.dumps([item for item in items if item["item_type"] != "appliance"]),
        encoding="utf-8",
    )

    database = MemoryDatabase()
    await database.connect()
    try:
        await ingest_directory(database, data_dir)  # type: ignore[arg-type]
        target = tmp_path / "export"
        await export_directory(database, target)  # type: ignore[arg-type]
    finally:
        await database.close()

    exported_items = _load_json(target / "nodes" / "item.json")
    assert not (target / "nodes" / "item").is_dir()
    assert _records_by_id(_comparable(exported_items)) == _records_by_id(
        _comparable(items)
    )


@pytest.mark.asyncio
async def test_empty_registered_edges_are_written_empty_node_tables_omitted(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "source"
    copytree(STATIC_TEST_DATA, data_dir)
    (data_dir / "nodes" / "person.json").write_text("[]", encoding="utf-8")
    for name in REGISTERED_EDGES:
        (data_dir / "edges" / f"{name}.json").write_text("[]", encoding="utf-8")

    database = MemoryDatabase()
    await database.connect()
    try:
        await ingest_directory(database, data_dir)  # type: ignore[arg-type]
        target = tmp_path / "export"
        await export_directory(database, target)  # type: ignore[arg-type]
    finally:
        await database.close()

    assert not (target / "nodes" / "person.json").exists()
    assert (target / "nodes" / "address.json").exists()
    for name in REGISTERED_EDGES:
        assert _load_json(target / "edges" / f"{name}.json") == []


@pytest.mark.asyncio
async def test_runtime_records_are_exported_and_readme_is_preserved(
    tmp_path: Path,
) -> None:
    target = tmp_path / "export"
    target.mkdir()
    (target / "Readme.md").write_text("keep me\n", encoding="utf-8")
    database = MemoryDatabase()
    await database.connect()
    try:
        await ingest_directory(database, STATIC_TEST_DATA)  # type: ignore[arg-type]
        await database.upsert(
            RecordID("person", "drew_example"),
            {"name": ["Drew Example"]},
        )
        await export_directory(database, target)  # type: ignore[arg-type]
    finally:
        await database.close()

    people = _records_by_id(_load_json(target / "nodes" / "person.json"))
    assert people["person:drew_example"]["name"] == ["Drew Example"]
    assert (target / "Readme.md").read_text(encoding="utf-8") == "keep me\n"


@pytest.mark.asyncio
async def test_failed_export_does_not_clobber_existing_target(tmp_path: Path) -> None:
    target = tmp_path / "export"
    (target / "nodes").mkdir(parents=True)
    original = target / "nodes" / "person.json"
    original.write_text('[{"id":"person:kept"}]\n', encoding="utf-8")

    database = MemoryDatabase()
    await database.connect()
    try:
        await ingest_directory(database, STATIC_TEST_DATA)  # type: ignore[arg-type]
        await database.query(
            "RELATE $source->$edge->$target CONTENT $content;",
            {
                "source": RecordID("person", "alex_example"),
                "edge": RecordID("unregistered_rel", "alex_guest"),
                "target": RecordID("address", "test_house"),
                "content": {"role": "guest"},
            },
        )
        with pytest.raises(ValueError, match="unregistered relationship"):
            await export_directory(database, target)  # type: ignore[arg-type]
    finally:
        await database.close()

    assert original.read_text(encoding="utf-8") == '[{"id":"person:kept"}]\n'
    assert not list(target.glob(".*.export-tmp-*"))
    assert not list(target.glob(".*.export-backup-*"))


@pytest.mark.asyncio
async def test_retired_table_with_records_fails_export(tmp_path: Path) -> None:
    database = MemoryDatabase()
    await database.connect()
    try:
        await ingest_directory(database, STATIC_TEST_DATA)  # type: ignore[arg-type]
        await database.upsert(RecordID("location", "legacy_home"), {"name": "Legacy"})
        with pytest.raises(ValueError, match="retired tables"):
            await export_directory(database, tmp_path / "export")  # type: ignore[arg-type]
    finally:
        await database.close()


def test_canonical_json_value_converts_record_ids_and_dates() -> None:
    record_id = RecordID("person", "alex_example")
    assert canonical_json_value(record_id) == "person:alex_example"
    assert canonical_json_value(date(2020, 1, 1)) == "2020-01-01"
    assert canonical_json_value(datetime(2020, 1, 1, 12, 0, 0)) == "2020-01-01T12:00:00"
    nested = {"id": record_id, "name": ["Alex"]}
    assert canonical_json_value(nested) == {
        "id": "person:alex_example",
        "name": ["Alex"],
    }
    with pytest.raises(ValueError, match="Cannot losslessly represent"):
        canonical_json_value({1, 2, 3})


def test_canonical_record_id_matches_export_mapping() -> None:
    assert canonical_record_id(RecordID("space", "test_house:kitchen:fridge_01:interior")) == (
        "space:test_house:kitchen:fridge_01:interior"
    )
