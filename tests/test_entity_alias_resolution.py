"""Deterministic stored-name / appellation resolution without an LLM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from surrealdb import RecordID

from home_cortex.edge_schema import EdgeSchemaRegistry
from home_cortex.fact_benchmark import _JsonGraphDispatcher
from home_cortex.retrieval import RetrievalService
from home_cortex.schema_catalog import record_aliases
from test_retrieval import FakeDatabase

ROOT = Path(__file__).parents[1]
SCHEMA_DIR = ROOT / "schemas" / "edge"

DYLAN = {
    "id": "person:dylan_kuang",
    "first_name": "Dylan",
    "last_name": "Kuang",
    "name": ["Dylan Kuang", "匡德伦"],
    "aliases": ["Dylan", "德伦"],
}


def _people_dispatcher(tmp_path: Path, people: list[dict[str, Any]]) -> _JsonGraphDispatcher:
    nodes = tmp_path / "nodes"
    edges = tmp_path / "edges"
    nodes.mkdir()
    edges.mkdir()
    (nodes / "person.json").write_text(
        json.dumps(people, ensure_ascii=False),
        encoding="utf-8",
    )
    return _JsonGraphDispatcher(tmp_path, EdgeSchemaRegistry.from_directory(SCHEMA_DIR))


async def _resolve(
    dispatcher: _JsonGraphDispatcher,
    text: str,
    *,
    speaker_id: str | None = None,
    household_id: str | None = None,
) -> list[dict[str, Any]]:
    payload = await dispatcher.dispatch_internal(
        "resolve_entity_alias",
        {
            "text": text,
            "entity_type": "person",
            "limit": 25,
            "speaker_id": speaker_id,
            "household_id": household_id,
        },
    )
    return list(payload["result"])


def test_record_aliases_does_not_invent_given_name_from_full_chinese_name() -> None:
    with_alias = record_aliases(DYLAN)
    production_shaped = record_aliases(
        {
            "id": "person:dylan_kuang",
            "first_name": "Dylan",
            "last_name": "Kuang",
            "name": ["Dylan Kuang", "匡德伦"],
        }
    )

    assert "匡德伦" in with_alias
    assert "德伦" in with_alias
    assert "匡德伦" in production_shaped
    assert "德伦" not in production_shaped


@pytest.mark.asyncio
async def test_full_name_and_stored_alias_resolve_to_the_same_entity(
    tmp_path: Path,
) -> None:
    dispatcher = _people_dispatcher(tmp_path, [DYLAN])

    full = await _resolve(dispatcher, "匡德伦")
    alias = await _resolve(dispatcher, "德伦")

    assert [row["id"] for row in full] == ["person:dylan_kuang"]
    assert [row["id"] for row in alias] == ["person:dylan_kuang"]


@pytest.mark.asyncio
async def test_production_shaped_record_does_not_resolve_德伦(tmp_path: Path) -> None:
    production_shaped = {
        "id": "person:dylan_kuang",
        "first_name": "Dylan",
        "last_name": "Kuang",
        "name": ["Dylan Kuang", "匡德伦"],
    }
    dispatcher = _people_dispatcher(tmp_path, [production_shaped])

    full = await _resolve(dispatcher, "匡德伦")
    alias = await _resolve(dispatcher, "德伦")

    assert [row["id"] for row in full] == ["person:dylan_kuang"]
    assert alias == []


@pytest.mark.asyncio
async def test_unknown_aliases_remain_unresolved(tmp_path: Path) -> None:
    dispatcher = _people_dispatcher(tmp_path, [DYLAN])

    assert await _resolve(dispatcher, "不存在的小名") == []
    assert await _resolve(dispatcher, "伦") == []


@pytest.mark.asyncio
async def test_duplicate_aliases_remain_ambiguous(tmp_path: Path) -> None:
    people = [
        DYLAN,
        {
            "id": "person:other",
            "name": ["Other Person"],
            "aliases": ["德伦"],
        },
    ]
    dispatcher = _people_dispatcher(tmp_path, people)

    matches = await _resolve(dispatcher, "德伦")

    assert [row["id"] for row in matches] == ["person:dylan_kuang", "person:other"]


@pytest.mark.asyncio
async def test_speaker_and_household_appellation_scoping_is_unchanged(
    tmp_path: Path,
) -> None:
    people = [
        {
            "id": "person:dylan_kuang",
            "name": ["Dylan Kuang", "匡德伦"],
            "appellations": [
                {
                    "value": "大宝",
                    "household_id": "address:fort_cerritos",
                    "speaker_ids": ["person:jian_kuang"],
                }
            ],
        }
    ]
    dispatcher = _people_dispatcher(tmp_path, people)

    resolved = await _resolve(
        dispatcher,
        "大宝",
        speaker_id="person:jian_kuang",
        household_id="address:fort_cerritos",
    )
    wrong_speaker = await _resolve(
        dispatcher,
        "大宝",
        speaker_id="person:pu_ba",
        household_id="address:fort_cerritos",
    )
    unscoped = await _resolve(dispatcher, "大宝")

    assert [row["id"] for row in resolved] == ["person:dylan_kuang"]
    assert wrong_speaker == []
    assert unscoped == []


@pytest.mark.asyncio
async def test_retrieval_service_full_name_and_alias_agree() -> None:
    database = FakeDatabase(
        {
            "person": [
                {
                    "id": RecordID("person", "dylan_kuang"),
                    "name": ["Dylan Kuang", "匡德伦"],
                    "aliases": ["Dylan", "德伦"],
                }
            ]
        }
    )
    service = RetrievalService(database, limit=25)  # type: ignore[arg-type]

    full = await service.resolve_entity_alias("匡德伦", entity_type="person")
    alias = await service.resolve_entity_alias("德伦", entity_type="person")
    unknown = await service.resolve_entity_alias("不存在的小名", entity_type="person")

    assert [row["id"] for row in full] == [row["id"] for row in alias] == [
        "person:dylan_kuang"
    ]
    assert unknown == []
