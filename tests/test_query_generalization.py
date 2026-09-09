"""Cross-domain execution rules over invented household graphs."""

import json
from datetime import datetime
from pathlib import Path

import pytest

from home_cortex.edge_schema import EdgeSchemaRegistry
from home_cortex.fact_benchmark import _JsonGraphDispatcher
from home_cortex.schema_catalog import RuntimeSchemaCatalog
from home_cortex.semantic_facts import (
    AgentRequestContext,
    FactRenderer,
    HouseholdFactEngine,
    SemanticFactRequest,
    SemanticFilter,
    SemanticReference,
    SemanticSchemaRegistry,
)
from home_cortex.semantic_ontology import SemanticOntology


V2_ONTOLOGY = Path(__file__).parents[1] / "schemas/semantic/ontology-v2.yaml"


def _write(root, kind: str, name: str, rows: list[dict]) -> None:
    (root / kind).mkdir(exist_ok=True)
    (root / kind / f"{name}.json").write_text(json.dumps(rows))


def _engine(root, *, max_records: int = 25):
    registry = EdgeSchemaRegistry.load_default()
    schema = SemanticSchemaRegistry(RuntimeSchemaCatalog.from_data_dir(root, registry))
    engine = HouseholdFactEngine(
        _JsonGraphDispatcher(root, registry), schema, max_records=max_records
    )
    context = AgentRequestContext(
        "person:alpha",
        "assistant",
        "Helper",
        "address:a",
        datetime.fromisoformat("2026-09-03T12:00:00-07:00"),
        "zh",
    )
    return engine, context


def _contained_households(root) -> None:
    _write(root, "nodes", "person", [
        {"id": "person:alpha", "name": "Alpha"},
        {"id": "person:beta", "name": "Beta"},
    ])
    _write(root, "nodes", "address", [
        {"id": "address:a", "name": "Home A"},
        {"id": "address:b", "name": "Home B"},
    ])
    _write(root, "nodes", "item", [
        {"id": "item:house_a", "name": "House A", "item_type": "house"},
        {"id": "item:house_b", "name": "House B", "item_type": "house"},
        {"id": "item:milk_a", "name": "Milk", "item_type": "food"},
        {"id": "item:milk_b", "name": "Milk", "item_type": "food"},
    ])
    _write(root, "nodes", "space", [
        {"id": "space:kitchen_a", "name": "Kitchen A", "space_type": "room"},
        {"id": "space:kitchen_b", "name": "Kitchen B", "space_type": "room"},
    ])
    _write(root, "edges", "lives_in", [
        {"from": "person:alpha", "to": "address:a", "end": None},
        {"from": "person:beta", "to": "address:b", "end": None},
    ])
    _write(root, "edges", "located_in", [
        {"from": "item:house_a", "to": "address:a"},
        {"from": "item:house_b", "to": "address:b"},
        {"from": "item:milk_a", "to": "space:kitchen_a"},
        {"from": "item:milk_b", "to": "space:kitchen_b"},
    ])
    _write(root, "edges", "hosted_by", [
        {"from": "space:kitchen_a", "to": "item:house_a"},
        {"from": "space:kitchen_b", "to": "item:house_b"},
    ])
    _write(root, "edges", "parent_of", [])
    _write(root, "edges", "spouse_of", [])


@pytest.mark.asyncio
async def test_declared_room_path_and_named_item_location_are_household_scoped(tmp_path):
    _contained_households(tmp_path)
    engine, context = _engine(tmp_path)

    room_payload = engine.schema.expand_planner_concepts({
        "request": {
            "operation": "count",
            "subject": {
                "kind": "current_household",
                "entity_type": "address",
                "path": [{"concept": "room"}],
            },
        }
    })
    rooms = SemanticFactRequest.model_validate(room_payload["request"])
    assert engine.schema.validates(rooms)
    room_result, *_ = await engine.execute(rooms, context)
    assert room_result.status == "found" and room_result.value == 1
    assert FactRenderer().render(rooms, room_result, context) == "家里有1个房间。"

    location_payload = engine.schema.expand_planner_concepts({
        "request": {
            "operation": "resolve_reference",
            "subject": {
                "kind": "named_entity",
                "entity_type": "item",
                "value": "Milk",
                "path": [{"concept": "location"}],
            },
        }
    })
    location = SemanticFactRequest.model_validate(location_payload["request"])
    location_result, *_ = await engine.execute(location, context)
    assert location_result.status == "found"
    assert location_result.value["id"] == "space:kitchen_a"
    assert "Kitchen A" in FactRenderer().render(location, location_result, context)

    v2_schema = SemanticSchemaRegistry(
        RuntimeSchemaCatalog.from_data_dir(tmp_path, engine.schema.edge_registry),
        SemanticOntology.from_file(V2_ONTOLOGY),
    )
    v2_rooms = SemanticFactRequest.model_validate(
        v2_schema.expand_planner_concepts({
            "request": {
                "operation": "count",
                "subject": {
                    "kind": "current_household",
                    "entity_type": "address",
                    "path": [{"concept": "room"}],
                },
            }
        })["request"]
    )
    assert v2_schema.validates(v2_rooms)


def test_active_contract_rejects_declared_disjoint_predicates(tmp_path):
    _contained_households(tmp_path)
    engine, _ = _engine(tmp_path)
    payload = engine.schema.expand_planner_concepts({
        "request": {
            "operation": "count",
            "subject": {
                "kind": "current_household",
                "entity_type": "address",
                "path": [{"concept": "member"}],
            },
            "filters": [{"predicate": "adult"}, {"predicate": "minor"}],
        }
    })
    request = SemanticFactRequest.model_validate(payload["request"])
    assert engine.schema.contract_error(request) == "CONTRADICTORY_PREDICATES"
    assert engine.schema.validation_code(request) == "INVALID_PLAN"
    assert engine.schema.planner_capability_payload()["predicate_disjointness"] == {
        "adult": ["minor"],
        "minor": ["adult"],
    }


@pytest.mark.asyncio
async def test_large_collection_is_not_reported_as_an_exact_truncated_count(tmp_path):
    people = [{"id": f"person:p{i:02d}", "name": f"P{i:02d}"} for i in range(30)]
    _write(tmp_path, "nodes", "person", people)
    _write(tmp_path, "nodes", "address", [{"id": "address:a", "name": "Home"}])
    for name in ("item", "space"):
        _write(tmp_path, "nodes", name, [])
    _write(tmp_path, "edges", "lives_in", [
        {"from": person["id"], "to": "address:a", "end": None}
        for person in people
    ])
    for name in ("located_in", "hosted_by", "parent_of", "spouse_of"):
        _write(tmp_path, "edges", name, [])
    engine, context = _engine(tmp_path, max_records=25)
    payload = engine.schema.expand_planner_concepts({
        "request": {
            "operation": "count",
            "subject": {
                "kind": "current_household",
                "entity_type": "address",
                "path": [{"concept": "member"}],
            },
        }
    })
    result, *_ = await engine.execute(
        SemanticFactRequest.model_validate(payload["request"]), context
    )
    assert result.status == "collection_incomplete"
    assert result.value is None
