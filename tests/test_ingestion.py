import json
from pathlib import Path
from shutil import copytree
from typing import Any

import pytest
from surrealdb import AsyncSurreal, RecordID
from surrealdb.errors import NotFoundError

from home_cortex.ingestion import _prune_table, ingest_directory

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
async def test_pruning_a_never_created_empty_table_is_idempotent() -> None:
    class MissingTableDatabase:
        async def query(self, *_: Any, **__: Any) -> Any:
            raise NotFoundError(
                "Table",
                "The table 'contained_in' does not exist",
                details={"Table": {"name": "contained_in"}},
            )

    await _prune_table(  # type: ignore[arg-type]
        MissingTableDatabase(),
        "contained_in",
        [],
    )


@pytest.mark.asyncio
async def test_pruning_nonempty_or_unrelated_missing_table_still_fails() -> None:
    class MissingTableDatabase:
        async def query(self, *_: Any, **__: Any) -> Any:
            raise NotFoundError(
                "Table",
                "The table 'other' does not exist",
                details={"Table": {"name": "other"}},
            )

    with pytest.raises(NotFoundError):
        await _prune_table(  # type: ignore[arg-type]
            MissingTableDatabase(),
            "contained_in",
            [],
        )
    with pytest.raises(NotFoundError):
        await _prune_table(  # type: ignore[arg-type]
            MissingTableDatabase(),
            "other",
            [RecordID("other", "one")],
        )


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
        edges = await database.query("SELECT * FROM lives_in;")
    finally:
        await database.close()

    assert first.nodes_upserted == second.nodes_upserted == 13
    assert first.edges_upserted == second.edges_upserted == 13
    assert sorted(str(edge["id"]) for edge in edges) == [
        "lives_in:blair_primary",
        "lives_in:person_alex_example__address_test_house",
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
        edges = await database.query("SELECT * FROM lives_in;")
    finally:
        await database.close()

    assert "lives_in:blair_primary" in [str(edge["id"]) for edge in edges]


@pytest.mark.asyncio
async def test_duplicate_implicit_relationships_require_ids(tmp_path: Path) -> None:
    data_dir = tmp_path / "static_test_data"
    copytree(STATIC_TEST_DATA, data_dir)
    edge = {
        "from": "person:alex_example",
        "to": "address:test_house",
    }
    (data_dir / "edges" / "lives_in.json").write_text(
        json.dumps([edge, {**edge, "residence_type": "historical"}]),
        encoding="utf-8",
    )
    database = MemoryDatabase()
    await database.connect()
    try:
        with pytest.raises(ValueError, match="Duplicate lives_in relationship"):
            await ingest_directory(database, data_dir)  # type: ignore[arg-type]
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_registered_relationship_requires_a_source_file(tmp_path: Path) -> None:
    data_dir = tmp_path / "static_test_data"
    copytree(STATIC_TEST_DATA, data_dir)
    (data_dir / "edges" / "hosted_by.json").unlink()
    database = MemoryDatabase()
    await database.connect()
    try:
        with pytest.raises(
            ValueError,
            match=r"Missing relationship data files.*hosted_by\.json",
        ):
            await ingest_directory(database, data_dir)  # type: ignore[arg-type]
        nodes = await database.query("SELECT * FROM person;")
    finally:
        await database.close()

    assert nodes == []


@pytest.mark.asyncio
async def test_node_name_must_be_a_non_empty_string_list(tmp_path: Path) -> None:
    data_dir = tmp_path / "static_test_data"
    copytree(STATIC_TEST_DATA, data_dir)
    address_path = data_dir / "nodes" / "address.json"
    address = json.loads(address_path.read_text(encoding="utf-8"))[0]
    address["name"] = "Test House"
    address_path.write_text(json.dumps([address]), encoding="utf-8")

    database = MemoryDatabase()
    await database.connect()
    try:
        with pytest.raises(ValueError, match="non-empty list of strings"):
            await ingest_directory(database, data_dir)  # type: ignore[arg-type]
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_person_address_as_is_optional_and_preserved() -> None:
    database = MemoryDatabase()
    await database.connect()
    try:
        await ingest_directory(database, STATIC_TEST_DATA)  # type: ignore[arg-type]
        people = await database.query("SELECT * FROM person ORDER BY id;")
    finally:
        await database.close()

    alex = next(
        person
        for person in people
        if str(person["id"]) == "person:alex_example"
    )
    blair = next(
        person for person in people if str(person["id"]) == "person:blair_example"
    )
    assert alex["address_as"] == {"en": "Mr. Example", "zh": "先生"}
    assert "address_as" not in blair


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "intrinsic_status",
    [{"is_guest": True}, {"person_type": "guest"}],
)
async def test_guest_status_must_be_stored_on_a_relationship(
    tmp_path: Path,
    intrinsic_status: dict[str, Any],
) -> None:
    data_dir = tmp_path / "static_test_data"
    copytree(STATIC_TEST_DATA, data_dir)
    person_path = data_dir / "nodes" / "person.json"
    people = json.loads(person_path.read_text(encoding="utf-8"))
    people[0].update(intrinsic_status)
    person_path.write_text(json.dumps(people), encoding="utf-8")
    database = MemoryDatabase()
    await database.connect()
    try:
        with pytest.raises(ValueError, match="household relationship"):
            await ingest_directory(database, data_dir)  # type: ignore[arg-type]
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_localized_name_object_is_accepted(tmp_path: Path) -> None:
    data_dir = tmp_path / "static_test_data"
    copytree(STATIC_TEST_DATA, data_dir)
    address_path = data_dir / "nodes" / "address.json"
    address = json.loads(address_path.read_text(encoding="utf-8"))[0]
    address["name"] = {"en": "Test House", "zh": "测试之家"}
    address_path.write_text(json.dumps([address]), encoding="utf-8")

    database = MemoryDatabase()
    await database.connect()
    try:
        await ingest_directory(database, data_dir)  # type: ignore[arg-type]
        addresses = await database.query("SELECT * FROM address;")
    finally:
        await database.close()

    assert addresses[0]["name"] == {"en": "Test House", "zh": "测试之家"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "address_as",
    ["Sir", {}, {"zh": ""}, {"": "先生"}],
)
async def test_person_address_as_must_be_a_localized_object(
    tmp_path: Path,
    address_as: object,
) -> None:
    data_dir = tmp_path / "static_test_data"
    copytree(STATIC_TEST_DATA, data_dir)
    person_path = data_dir / "nodes" / "person.json"
    people = json.loads(person_path.read_text(encoding="utf-8"))
    people[0]["address_as"] = address_as
    person_path.write_text(json.dumps(people), encoding="utf-8")

    database = MemoryDatabase()
    await database.connect()
    try:
        with pytest.raises(ValueError, match="non-empty localized object"):
            await ingest_directory(database, data_dir)  # type: ignore[arg-type]
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_non_person_nodes_cannot_define_address_as(tmp_path: Path) -> None:
    data_dir = tmp_path / "static_test_data"
    copytree(STATIC_TEST_DATA, data_dir)
    address_path = data_dir / "nodes" / "address.json"
    address = json.loads(address_path.read_text(encoding="utf-8"))[0]
    address["address_as"] = {"en": "Home"}
    address_path.write_text(json.dumps([address]), encoding="utf-8")

    database = MemoryDatabase()
    await database.connect()
    try:
        with pytest.raises(ValueError, match="Only Person nodes"):
            await ingest_directory(database, data_dir)  # type: ignore[arg-type]
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_lives_in_rejects_reversed_endpoint_types(tmp_path: Path) -> None:
    data_dir = tmp_path / "static_test_data"
    copytree(STATIC_TEST_DATA, data_dir)
    path = data_dir / "edges" / "lives_in.json"
    path.write_text(
        json.dumps(
            [{"from": "address:test_house", "to": "person:alex_example"}]
        ),
        encoding="utf-8",
    )
    database = MemoryDatabase()
    await database.connect()
    try:
        with pytest.raises(ValueError, match="Invalid lives_in endpoints"):
            await ingest_directory(database, data_dir)  # type: ignore[arg-type]
    finally:
        await database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["from", "to"])
async def test_relationships_reject_missing_endpoint_nodes(
    tmp_path: Path,
    role: str,
) -> None:
    data_dir = tmp_path / "static_test_data"
    copytree(STATIC_TEST_DATA, data_dir)
    path = data_dir / "edges" / "hosted_by.json"
    edge = {
        "from": "space:test_house:kitchen:fridge_01:interior",
        "to": "item:fridge_01",
    }
    edge[role] = (
        "space:missing:interior" if role == "from" else "item:missing"
    )
    path.write_text(json.dumps([edge]), encoding="utf-8")
    database = MemoryDatabase()
    await database.connect()
    try:
        with pytest.raises(ValueError, match=rf"unknown {role} node"):
            await ingest_directory(database, data_dir)  # type: ignore[arg-type]
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_non_temporal_parent_of_rejects_dates(tmp_path: Path) -> None:
    data_dir = tmp_path / "static_test_data"
    copytree(STATIC_TEST_DATA, data_dir)
    path = data_dir / "edges" / "parent_of.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    records[0]["start"] = "2020-01-01"
    path.write_text(json.dumps(records), encoding="utf-8")
    database = MemoryDatabase()
    await database.connect()
    try:
        with pytest.raises(ValueError, match="Non-temporal relationship"):
            await ingest_directory(database, data_dir)  # type: ignore[arg-type]
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_symmetric_reverse_duplicate_is_rejected(tmp_path: Path) -> None:
    data_dir = tmp_path / "static_test_data"
    copytree(STATIC_TEST_DATA, data_dir)
    path = data_dir / "edges" / "spouse_of.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    records.append(
        {
            "from": "person:blair_example",
            "to": "person:alex_example",
            "start": "2011-03-15",
            "end": None,
        }
    )
    path.write_text(json.dumps(records), encoding="utf-8")
    database = MemoryDatabase()
    await database.connect()
    try:
        with pytest.raises(ValueError, match="Duplicate symmetric spouse_of"):
            await ingest_directory(database, data_dir)  # type: ignore[arg-type]
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_reingestion_prunes_records_removed_from_json(tmp_path: Path) -> None:
    data_dir = tmp_path / "static_test_data"
    copytree(STATIC_TEST_DATA, data_dir)
    database = MemoryDatabase()
    await database.connect()
    try:
        await ingest_directory(database, data_dir)  # type: ignore[arg-type]
        path = data_dir / "edges" / "lives_in.json"
        records = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps(records[:1]), encoding="utf-8")
        await ingest_directory(database, data_dir)  # type: ignore[arg-type]
        edges = await database.query("SELECT * FROM lives_in;")
    finally:
        await database.close()

    assert [str(edge["id"]) for edge in edges] == [
        "lives_in:person_alex_example__address_test_house"
    ]


@pytest.mark.asyncio
async def test_ingestion_prunes_retired_relationship_tables() -> None:
    database = MemoryDatabase()
    await database.connect()
    try:
        await ingest_directory(database, STATIC_TEST_DATA)  # type: ignore[arg-type]
        await database.query(
            "RELATE $source->contained_in->$target;",
            {
                "source": RecordID("space", "kitchen"),
                "target": RecordID("address", "test_house"),
            },
        )
        legacy = await database.query("SELECT * FROM contained_in;")
        await ingest_directory(database, STATIC_TEST_DATA)  # type: ignore[arg-type]
        remaining = await database.query("SELECT * FROM contained_in;")
    finally:
        await database.close()

    assert len(legacy) == 1
    assert remaining == []


@pytest.mark.asyncio
async def test_ingestion_prunes_retired_location_node_table() -> None:
    database = MemoryDatabase()
    await database.connect()
    try:
        await database.upsert(
            RecordID("location", "legacy_home"),
            {"name": "Legacy Home"},
        )
        legacy = await database.query("SELECT * FROM location;")
        await ingest_directory(database, STATIC_TEST_DATA)  # type: ignore[arg-type]
        remaining = await database.query("SELECT * FROM location;")
        addresses = await database.query("SELECT * FROM address;")
    finally:
        await database.close()

    assert len(legacy) == 1
    assert remaining == []
    assert [str(record["id"]) for record in addresses] == ["address:test_house"]
