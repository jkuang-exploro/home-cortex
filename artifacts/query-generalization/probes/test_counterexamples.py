"""G2: desired execution vs current executor. Not on pytest testpaths.

Run:
  PYTHONPATH=src python -m pytest -q artifacts/query-generalization/probes/test_counterexamples.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from home_cortex.edge_schema import EdgeSchemaRegistry
from home_cortex.fact_benchmark import _JsonGraphDispatcher
from home_cortex.schema_catalog import RuntimeSchemaCatalog
from home_cortex.semantic_facts import (
    AgentRequestContext,
    HouseholdFactEngine,
    SemanticFactRequest,
    SemanticFilter,
    SemanticReference,
    SemanticRelationStep,
    SemanticSchemaRegistry,
)
from home_cortex.semantic_planner_benchmark import primary_entity_ids

CLOCK = datetime.fromisoformat("2026-09-03T12:00:00-07:00")
OUT = Path(__file__).resolve().parent / "counterexamples.json"
_RECORDS: list[dict] = []


def _write(kind, root: Path, name: str, rows: list) -> None:
    (root / kind).mkdir(parents=True, exist_ok=True)
    (root / kind / f"{name}.json").write_text(json.dumps(rows), encoding="utf-8")


def engine_for(root: Path, speaker: str, household: str, *, max_records: int = 25):
    registry = EdgeSchemaRegistry.load_default()
    schema = SemanticSchemaRegistry(RuntimeSchemaCatalog.from_data_dir(root, registry))
    engine = HouseholdFactEngine(
        _JsonGraphDispatcher(root, registry), schema, max_records=max_records
    )
    context = AgentRequestContext(
        speaker, "assistant", "Helper", household, CLOCK, "en"
    )
    return engine, context


def members():
    return SemanticReference(
        kind="current_household",
        entity_type="address",
        path=(SemanticRelationStep(relation="member"),),
    )


def record(case_id: str, request: SemanticFactRequest, result, *, expected_population) -> None:
    _RECORDS.append(
        {
            "id": case_id,
            "plan": request.model_dump(mode="json", exclude_none=True),
            "status": result.status,
            "value": result.value if not isinstance(result.value, (list, dict)) else None,
            "shape": result.shape,
            "candidate_ids": [
                str(item.get("id"))
                for item in (result.candidates or ())
                if isinstance(item, dict) and item.get("id")
            ],
            "entity_ids": list(primary_entity_ids(result)),
            "row_ids": [
                row.entity.get("id")
                for row in (result.rows or ())
                if row.entity.get("id")
            ],
            "row_statuses": [row.status for row in (result.rows or ())],
            "expected_population": list(expected_population),
        }
    )


@pytest.fixture(scope="session", autouse=True)
def persist_records():
    yield
    OUT.write_text(json.dumps(_RECORDS, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@pytest.fixture
def age_house(tmp_path):
    people = [
        ("a", "male", "1980-03-02"),
        ("b", "female", "1982-06-01"),
        ("son1", "male", "2008-09-03"),
        ("son2", "male", "2012-10-15"),
        ("daughter", "female", "2010-04-09"),
        ("father", "male", "1955-01-01"),
        ("mother", "female", "1957-02-02"),
        ("grandson", "male", "2025-01-01"),
    ]
    _write("nodes", tmp_path, "person", [
        {"id": f"person:{i}", "name": i, "gender": g, "dob": d} for i, g, d in people
    ])
    _write("nodes", tmp_path, "address", [{"id": "address:fictional", "full_address": "Invented"}])
    _write("edges", tmp_path, "spouse_of", [
        {"from": "person:a", "to": "person:b", "start": "2005-05-06", "end": None}
    ])
    _write("edges", tmp_path, "parent_of", [
        {"from": f"person:{p}", "to": f"person:{c}"}
        for p, c in [
            ("a", "son1"), ("a", "son2"), ("a", "daughter"),
            ("b", "son1"), ("b", "son2"), ("b", "daughter"),
            ("father", "b"), ("mother", "b"), ("daughter", "grandson"),
        ]
    ])
    _write("edges", tmp_path, "lives_in", [
        {"from": f"person:{i}", "to": "address:fictional", "start": "2018-01-01", "end": None}
        for i, _, _ in people
    ])
    _write("edges", tmp_path, "located_in", [])
    _write("edges", tmp_path, "hosted_by", [])
    return engine_for(tmp_path, "person:a", "address:fictional")


@pytest.mark.asyncio
async def test_c1_scalar_age_plus_minor_filters_before_cardinality(age_house):
    engine, context = age_house
    minors = {"person:son2", "person:daughter", "person:grandson"}
    request = SemanticFactRequest(
        operation="date_difference",
        subject=members(),
        property="birth_date",
        mode="years",
        filters=(SemanticFilter(predicate="minor"),),
    )
    each = request.model_copy(update={"projection": "each"})
    scalar, *_ = await engine.execute(request, context)
    rows, *_ = await engine.execute(each, context)
    record("C1-scalar", request, scalar, expected_population=sorted(minors))
    record("C1-each", each, rows, expected_population=sorted(minors))
    assert rows.status == "found" and rows.shape == "rows"
    assert {row.entity["id"] for row in rows.rows} == minors
    # Filters precede scalar cardinality. Three filtered entities remain, so a
    # scalar projection must ask for clarification rather than inventing rows.
    assert scalar.status == "ambiguous"
    assert {
        item["id"] for item in scalar.candidates
    } == minors


@pytest.mark.asyncio
async def test_c2_exactly_one_filtered_scalar_age(age_house):
    engine, context = age_house
    request = SemanticFactRequest(
        operation="date_difference",
        subject=members(),
        property="birth_date",
        mode="years",
        filters=(
            SemanticFilter(predicate="minor"),
            SemanticFilter(property="gender", value="female"),
        ),
    )
    result, *_ = await engine.execute(request, context)
    record("C2", request, result, expected_population=["person:daughter"])
    assert result.status == "found"
    assert primary_entity_ids(result) == ("person:daughter",)


@pytest.mark.asyncio
async def test_c3_zero_count_is_found_zero(age_house):
    engine, context = age_house
    request = SemanticFactRequest(
        operation="count",
        subject=members(),
        filters=(
            SemanticFilter(
                property="birth_date",
                operator="date_range",
                value=("1999-01-01", "2000-01-01"),
            ),
        ),
    )
    result, *_ = await engine.execute(request, context)
    record("C3", request, result, expected_population=[])
    assert result.status == "found" and result.value == 0


@pytest.mark.asyncio
async def test_c4_each_missing_dob_keeps_row(age_house):
    engine, context, dispatcher = (*age_house[:2], age_house[0].dispatcher)
    dispatcher.entities["person:son2"].pop("dob")
    request = SemanticFactRequest(
        operation="date_difference",
        subject=members(),
        property="birth_date",
        mode="years",
        projection="each",
        filters=(SemanticFilter(predicate="minor"),),
    )
    result, *_ = await engine.execute(request, context)
    record("C4", request, result, expected_population=["person:son2", "person:daughter", "person:grandson"])
    by_id = {row.entity["id"]: row for row in result.rows}
    assert "person:son2" in by_id
    assert by_id["person:son2"].status in {"filter_input_missing", "property_unavailable", "computation_input_missing"}


@pytest.mark.asyncio
async def test_c5_ambiguous_named_root_is_not_a_bulk_query(age_house):
    engine, context, dispatcher = (*age_house[:2], age_house[0].dispatcher)
    dispatcher.entities["person:b"]["name"] = "son1"
    request = SemanticFactRequest(
        operation="date_difference",
        subject=SemanticReference(kind="named_entity", entity_type="person", value="son1"),
        property="birth_date",
        mode="years",
    )
    result, *_ = await engine.execute(request, context)
    record("C5", request, result, expected_population=["person:b", "person:son1"])
    assert result.status == "ambiguous"
    assert request.subject.kind == "named_entity"


@pytest.mark.asyncio
async def test_c6_ambiguous_final_daughters(age_house):
    engine, context = age_house
    dispatcher = engine.dispatcher
    dispatcher.edges["parent_of"].append({"from": "person:a", "to": "person:mother"})
    dispatcher.entities["person:mother"]["gender"] = "female"
    request = SemanticFactRequest(
        operation="resolve_reference",
        subject=SemanticReference(
            kind="self",
            entity_type="person",
            path=(SemanticRelationStep(
                relation="child",
                filters=(SemanticFilter(property="gender", value="female"),),
            ),),
        ),
    )
    result, *_ = await engine.execute(request, context)
    record("C6", request, result, expected_population=["person:daughter", "person:mother"])
    assert result.status == "ambiguous"


@pytest.mark.asyncio
async def test_c7_relation_filter_is_existential_across_edges(age_house):
    engine, context = age_house
    dispatcher = engine.dispatcher
    dispatcher.edges["lives_in"].append(
        {"from": "person:a", "to": "address:fictional", "start": "2020-08-01", "end": None}
    )
    request = SemanticFactRequest(
        operation="count",
        subject=members(),
        filters=(
            SemanticFilter(
                property="start_date",
                source="relation",
                operator="date_range",
                value=("2020-01-01", "2021-01-01"),
            ),
        ),
    )
    result, *_ = await engine.execute(request, context)
    record("C7", request, result, expected_population=["person:a"])
    assert result.status == "found"
    assert result.value == 1
    assert primary_entity_ids(result) == ("person:a",)


@pytest.mark.asyncio
async def test_c8_count_of_30_members_is_not_silently_25(tmp_path):
    people = [{"id": f"person:p{i:02d}", "name": f"p{i:02d}", "gender": "male", "dob": "1990-01-01"} for i in range(30)]
    _write("nodes", tmp_path, "person", people)
    _write("nodes", tmp_path, "address", [{"id": "address:big", "full_address": "Big"}])
    _write("edges", tmp_path, "lives_in", [
        {"from": p["id"], "to": "address:big", "start": "2018-01-01", "end": None} for p in people
    ])
    _write("edges", tmp_path, "spouse_of", [])
    _write("edges", tmp_path, "parent_of", [])
    _write("edges", tmp_path, "located_in", [])
    _write("edges", tmp_path, "hosted_by", [])
    engine, context = engine_for(tmp_path, "person:p00", "address:big", max_records=25)
    request = SemanticFactRequest(operation="count", subject=members())
    result, *_ = await engine.execute(request, context)
    record("C8", request, result, expected_population=[p["id"] for p in people])
    assert result.status != "found" or result.value == 30
    assert result.value != 25


@pytest.mark.asyncio
async def test_c9_same_named_items_are_scoped_to_speaker_household(tmp_path):
    _write("nodes", tmp_path, "person", [
        {"id": "person:alpha", "name": "Alpha", "gender": "male", "dob": "1980-01-01"},
        {"id": "person:beta", "name": "Beta", "gender": "female", "dob": "1981-01-01"},
    ])
    _write("nodes", tmp_path, "address", [
        {"id": "address:a", "full_address": "A"},
        {"id": "address:b", "full_address": "B"},
    ])
    _write("nodes", tmp_path, "item", [
        {"id": "item:milk_a", "name": "Milk", "item_type": "food"},
        {"id": "item:milk_b", "name": "Milk", "item_type": "food"},
    ])
    _write("nodes", tmp_path, "space", [])
    _write("edges", tmp_path, "lives_in", [
        {"from": "person:alpha", "to": "address:a", "start": "2018-01-01", "end": None},
        {"from": "person:beta", "to": "address:b", "start": "2018-01-01", "end": None},
    ])
    _write("edges", tmp_path, "located_in", [
        {"from": "item:milk_a", "to": "address:a"},
        {"from": "item:milk_b", "to": "address:b"},
    ])
    _write("edges", tmp_path, "hosted_by", [])
    _write("edges", tmp_path, "spouse_of", [])
    _write("edges", tmp_path, "parent_of", [])
    engine, context = engine_for(tmp_path, "person:alpha", "address:a")
    request = SemanticFactRequest(
        operation="resolve_reference",
        subject=SemanticReference(kind="named_entity", entity_type="item", value="Milk"),
    )
    result, *_ = await engine.execute(request, context)
    record("C9", request, result, expected_population=["item:milk_a"])
    ids = set(primary_entity_ids(result)) | {
        str(item.get("id")) for item in (result.candidates or ()) if isinstance(item, dict)
    }
    assert "item:milk_b" not in ids
    assert result.status in {"found", "entity_not_found"}
    if result.status == "found":
        assert primary_entity_ids(result) == ("item:milk_a",)
