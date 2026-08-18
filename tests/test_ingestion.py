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
    assert first.edges_upserted == second.edges_upserted == 3
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
    location_path = data_dir / "nodes" / "location.json"
    location = json.loads(location_path.read_text(encoding="utf-8"))
    location["name"] = {"en": "Test House", "zh": "测试之家"}
    location_path.write_text(json.dumps(location), encoding="utf-8")

    database = MemoryDatabase()
    await database.connect()
    try:
        await ingest_directory(database, data_dir)  # type: ignore[arg-type]
        locations = await database.query("SELECT * FROM location;")
    finally:
        await database.close()

    assert locations[0]["name"] == {"en": "Test House", "zh": "测试之家"}


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
    location_path = data_dir / "nodes" / "location.json"
    location = json.loads(location_path.read_text(encoding="utf-8"))
    location["address_as"] = {"en": "Home"}
    location_path.write_text(json.dumps(location), encoding="utf-8")

    database = MemoryDatabase()
    await database.connect()
    try:
        with pytest.raises(ValueError, match="Only Person nodes"):
            await ingest_directory(database, data_dir)  # type: ignore[arg-type]
    finally:
        await database.close()
