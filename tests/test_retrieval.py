from pathlib import Path
from typing import Any

import pytest
from surrealdb import AsyncSurreal, RecordID

from home_cortex.ingestion import ingest_directory
from home_cortex.retrieval import RetrievalService, to_json_value
from home_cortex.schema_catalog import normalize_entity_alias

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
        record_id = variables.get("id")
        if record_id is not None:
            return self.results.get(str(record_id), [])
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
async def test_resolve_entity_alias_is_exact_normalized_and_database_backed() -> None:
    database = FakeDatabase(
        {
            "person": [
                {
                    "id": RecordID("person", "dylan"),
                    "name": ["Dylan Kuang", "匡德伦"],
                    "aliases": ["Dylan", "德伦"],
                },
                {
                    "id": RecordID("person", "dylan_sr"),
                    "name": ["Dylan Senior"],
                },
            ]
        }
    )
    service = RetrievalService(database, limit=10)  # type: ignore[arg-type]

    result = await service.resolve_entity_alias(
        "  ＤＹＬＡＮ!!!  ",
        entity_type="person",
    )

    assert [record["id"] for record in result] == ["person:dylan"]
    assert len(database.queries) == 1
    assert "SELECT * FROM type::table($table)" in database.queries[0][0]
    assert "LIMIT $limit" not in database.queries[0][0]
    assert "limit" not in database.queries[0][1]
    assert normalize_entity_alias(" ＤＹＬＡＮ!!! ") == "dylan"


@pytest.mark.asyncio
async def test_resolve_entity_alias_preserves_ambiguity() -> None:
    database = FakeDatabase(
        {
            "person": [
                {"id": RecordID("person", "first"), "aliases": ["David"]},
                {"id": RecordID("person", "second"), "aliases": ["david"]},
            ]
        }
    )
    service = RetrievalService(database, limit=10)  # type: ignore[arg-type]

    result = await service.resolve_entity_alias("DAVID", entity_type="person")

    assert [record["id"] for record in result] == [
        "person:first",
        "person:second",
    ]


@pytest.mark.asyncio
async def test_scoped_appellation_requires_matching_speaker_and_household() -> None:
    database = FakeDatabase(
        {
            "person": [
                {
                    "id": RecordID("person", "parent"),
                    "name": ["Household Parent"],
                    "appellations": [
                        {
                            "value": "Papa",
                            "household_id": "address:test_house",
                            "speaker_ids": ["person:child"],
                        }
                    ],
                }
            ]
        }
    )
    service = RetrievalService(database, limit=10)  # type: ignore[arg-type]

    unscoped = await service.resolve_entity_alias("Papa", entity_type="person")
    wrong_speaker = await service.resolve_entity_alias(
        "Papa",
        entity_type="person",
        speaker_id="person:other",
        household_id="address:test_house",
    )
    resolved = await service.resolve_entity_alias(
        "PAPA!",
        entity_type="person",
        speaker_id="person:child",
        household_id="address:test_house",
    )

    assert unscoped == []
    assert wrong_speaker == []
    assert [record["id"] for record in resolved] == ["person:parent"]


@pytest.mark.asyncio
async def test_get_entity_selects_the_canonical_record() -> None:
    database = FakeDatabase(
        {
            "person:jian_kuang": [
                {
                    "id": RecordID("person", "jian_kuang"),
                    "name": ["Jian Kuang"],
                }
            ]
        }
    )
    service = RetrievalService(database, limit=10)  # type: ignore[arg-type]

    result = await service.get_entity("person:jian_kuang")

    assert result == {"id": "person:jian_kuang", "name": ["Jian Kuang"]}
    assert len(database.queries) == 1
    assert "SELECT * FROM $id" in database.queries[0][0]
    assert database.queries[0][1]["id"] == RecordID("person", "jian_kuang")


@pytest.mark.asyncio
async def test_get_entity_returns_none_for_a_missing_record() -> None:
    database = FakeDatabase({})
    service = RetrievalService(database, limit=10)  # type: ignore[arg-type]

    assert await service.get_entity("person:missing") is None
    assert database.queries[0][1]["id"] == RecordID("person", "missing")


@pytest.mark.asyncio
async def test_get_entity_rejects_unknown_type_and_malformed_id() -> None:
    database = FakeDatabase({})
    service = RetrievalService(database, limit=10)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="table:record_id"):
        await service.get_entity("not-a-record")
    with pytest.raises(ValueError, match="Unknown entity type"):
        await service.get_entity("secret:jian_kuang")

    assert database.queries == []


@pytest.mark.asyncio
async def test_get_relationships_returns_direction_and_filters_relation() -> None:
    database = FakeDatabase(
        {
            "lives_in": [
                {
                    "id": RecordID("lives_in", "person_a__address_main"),
                    "in": RecordID("person", "a"),
                    "out": RecordID("address", "main"),
                    "source_entity": {
                        "id": RecordID("person", "a"),
                        "first_name": "Alex",
                    },
                    "target_entity": {
                        "id": RecordID("address", "main"),
                        "name": ["Main Home"],
                    },
                }
            ]
        }
    )
    service = RetrievalService(database, limit=10)  # type: ignore[arg-type]

    incoming = await service.get_relationships(
        "address:main",
        relation="lives_in",
    )
    outgoing = await service.get_relationships(
        "person:a",
        relation="lives_in",
    )

    assert incoming[0]["direction"] == "incoming"
    assert outgoing[0]["direction"] == "outgoing"
    assert incoming[0]["relation"] == "lives_in"
    assert incoming[0]["related_entity"] == {
        "id": "person:a",
        "first_name": "Alex",
    }
    assert incoming[0]["entity"] == {
        "id": "address:main",
        "name": ["Main Home"],
    }
    assert outgoing[0]["related_entity"] == {
        "id": "address:main",
        "name": ["Main Home"],
    }
    assert outgoing[0]["entity"] == {
        "id": "person:a",
        "first_name": "Alex",
    }
    assert "source_entity" not in incoming[0]
    assert "target_entity" not in incoming[0]
    assert database.queries[0][1]["entity"] == RecordID("address", "main")
    assert "out = $entity" in database.queries[0][0]
    assert "in.* AS source_entity" in database.queries[0][0]


@pytest.mark.asyncio
async def test_typed_relation_overrides_impossible_requested_direction() -> None:
    database = FakeDatabase({"lives_in": []})
    service = RetrievalService(database, limit=10)  # type: ignore[arg-type]

    await service.get_relationships(
        "address:main",
        relation="lives_in",
        direction="out",
    )
    await service.get_relationships(
        "person:alex",
        relation="lives_in",
        direction="in",
    )

    assert "out = $entity" in database.queries[0][0]
    assert "in = $entity" in database.queries[1][0]


@pytest.mark.asyncio
async def test_relationship_entity_summaries_exclude_dob_and_address() -> None:
    database = FakeDatabase(
        {
            "lives_in": [
                {
                    "id": RecordID("lives_in", "alex_home"),
                    "in": RecordID("person", "alex"),
                    "out": RecordID("address", "main"),
                    "source_entity": {
                        "id": RecordID("person", "alex"),
                        "name": ["Alex", "艾力克斯"],
                        "gender": "male",
                        "dob": "1980-01-02",
                        "address": {"street": "123 Private Street"},
                    },
                    "target_entity": {
                        "id": RecordID("address", "main"),
                        "name": ["Main Home", "主宅"],
                        "address": {"street": "123 Private Street"},
                    },
                }
            ]
        }
    )
    service = RetrievalService(database, limit=10)  # type: ignore[arg-type]

    result = await service.get_relationships(
        "address:main",
        relation="lives_in",
    )

    assert result[0]["related_entity"] == {
        "id": "person:alex",
        "name": ["Alex", "艾力克斯"],
        "gender": "male",
    }
    assert result[0]["entity"] == {
        "id": "address:main",
        "name": ["Main Home", "主宅"],
    }


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

    assert service.node_tables == ("address", "item", "person", "space")
    assert service.edge_tables == (
        "hosted_by",
        "lives_in",
        "located_in",
        "parent_of",
        "spouse_of",
    )


def test_record_id_is_serialized() -> None:
    assert to_json_value(RecordID("person", "alice")) == "person:alice"
    assert (
        to_json_value(RecordID("space", "fridge_01:interior"))
        == "space:fridge_01:interior"
    )


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

        chinese_people = await service.resolve_entity_alias(
            "艾力克斯", entity_type="person"
        )
        kitchens = await service.resolve_entity_alias("厨房", entity_type="space")
        relationships = await service.get_relationships(
            "address:test_house",
            relation="lives_in",
        )
        alex_home = await service.get_relationships(
            "person:alex_example",
            relation="lives_in",
        )
        marriage = await service.get_relationships(
            "person:alex_example",
            relation="spouse_of",
        )
    finally:
        await database.close()

    assert [entity["id"] for entity in chinese_people] == [
        "person:alex_example"
    ]
    assert [entity["id"] for entity in kitchens] == ["space:kitchen"]
    assert kitchens[0]["space_type"] == "room"
    assert chinese_people[0]["name"] == ["Alex Example", "艾力克斯"]
    assert sorted(edge["in"] for edge in relationships) == [
        "person:alex_example",
        "person:blair_example",
    ]
    assert all(edge["out"] == "address:test_house" for edge in relationships)
    assert all(edge["direction"] == "incoming" for edge in relationships)
    assert alex_home[0]["related_entity"]["id"] == "address:test_house"
    assert [edge["out"] for edge in marriage] == ["person:blair_example"]
    assert marriage[0]["relation"] == "spouse_of"
    assert marriage[0]["start"] == "2011-03-15"
    assert marriage[0].get("end") in {None}
    assert sorted(
        edge["related_entity"]["first_name"] for edge in relationships
    ) == ["Alex", "Blair"]
@pytest.mark.asyncio
async def test_registry_drives_symmetric_directed_and_inverse_traversal() -> None:
    database = MemoryDatabase()
    await database.connect()
    try:
        await ingest_directory(database, STATIC_TEST_DATA)  # type: ignore[arg-type]
        service = RetrievalService(  # type: ignore[arg-type]
            database,
            limit=10,
            data_dir=STATIC_TEST_DATA,
        )

        blair_spouse = await service.get_relationships(
            "person:blair_example",
            relation="spouse_of",
        )
        alex_children = await service.get_relationships(
            "person:alex_example",
            relation="parent_of",
            direction="out",
        )
        casey_parents = await service.get_relationships(
            "person:casey_example",
            relation="parent_of",
            direction="in",
        )
        casey_inverse = await service.get_relationships(
            "person:casey_example",
            relation="child_of",
        )
        kitchen_host = await service.get_relationships(
            "space:kitchen",
            relation="hosted_by",
        )
        house_spaces = await service.get_relationships(
            "item:test_house",
            relation="hosts_space",
        )
        house_address = await service.get_relationships(
            "item:test_house",
            relation="located_in",
        )
    finally:
        await database.close()

    assert [edge["related_entity"]["id"] for edge in blair_spouse] == [
        "person:alex_example"
    ]
    assert blair_spouse[0]["semantic_relation"] == "spouse_of"
    assert [edge["related_entity"]["id"] for edge in alex_children] == [
        "person:casey_example"
    ]
    assert sorted(edge["related_entity"]["id"] for edge in casey_parents) == [
        "person:alex_example",
        "person:blair_example",
    ]
    assert sorted(edge["related_entity"]["id"] for edge in casey_inverse) == [
        "person:alex_example",
        "person:blair_example",
    ]
    assert all(edge["relation"] == "parent_of" for edge in casey_inverse)
    assert all(edge["semantic_relation"] == "child_of" for edge in casey_inverse)
    assert [edge["related_entity"]["id"] for edge in kitchen_host] == [
        "item:test_house"
    ]
    assert kitchen_host[0]["relation"] == "hosted_by"
    assert kitchen_host[0]["semantic_relation"] == "hosted_by"
    assert [edge["related_entity"]["id"] for edge in house_spaces] == [
        "space:kitchen"
    ]
    assert house_spaces[0]["relation"] == "hosted_by"
    assert house_spaces[0]["semantic_relation"] == "hosts_space"
    assert house_spaces[0]["related_entity"]["space_type"] == "room"
    assert [edge["related_entity"]["id"] for edge in house_address] == [
        "address:test_house"
    ]


@pytest.mark.asyncio
async def test_ended_relationships_are_excluded_unless_requested() -> None:
    ended = {
        "id": RecordID("spouse_of", "old_marriage"),
        "in": RecordID("person", "a"),
        "out": RecordID("person", "b"),
        "end": "2020-01-01",
    }
    service = RetrievalService(  # type: ignore[arg-type]
        FakeDatabase({"spouse_of": [ended]}),
        limit=10,
    )

    current = await service.get_relationships("person:a", relation="spouse_of")
    historical = await service.get_relationships(
        "person:a",
        relation="spouse_of",
        include_ended=True,
    )

    assert current == []
    assert historical[0]["end"] == "2020-01-01"


@pytest.mark.asyncio
async def test_get_entity_ignores_colliding_searchable_records() -> None:
    database = MemoryDatabase()
    await database.connect()
    try:
        await database.upsert(
            RecordID("person", "about_jian_kuang"),
            {
                "name": ["About Jian"],
                "notes": "preferences for person:jian_kuang",
            },
        )
        await database.upsert(
            RecordID("person", "jian_kuang_preferences"),
            {
                "name": ["Jian preferences"],
                "notes": "another mention of person:jian_kuang",
            },
        )
        await database.upsert(
            RecordID("person", "jian_kuang"),
            {"name": ["Jian Kuang"], "address_as": {"en": "Mr. Kuang"}},
        )
        service = RetrievalService(database, limit=10)  # type: ignore[arg-type]

        exact = await service.get_entity("person:jian_kuang")
        missing = await service.get_entity("person:does_not_exist")
    finally:
        await database.close()

    assert exact is not None
    assert exact["id"] == "person:jian_kuang"
    assert exact["name"] == ["Jian Kuang"]
    assert missing is None
