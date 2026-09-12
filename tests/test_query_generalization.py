"""Cross-domain execution rules over invented household graphs."""

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest

from home_cortex.edge_schema import EdgeSchemaRegistry
from home_cortex.fact_benchmark import _JsonGraphDispatcher
from home_cortex.schema_catalog import RuntimeSchemaCatalog
from home_cortex.semantic_facts import (
    AgentRequestContext,
    DiscourseContext,
    FactRenderer,
    HouseholdFactEngine,
    SemanticFactPlanner,
    SemanticPlannerFailure,
    SemanticFactRequest,
    SemanticFilter,
    SemanticReference,
    SemanticRelationStep,
    SemanticSchemaRegistry,
    _invalid_plan_retry_hint,
    _object_location_hint,
    _object_location_mismatch,
)
from home_cortex.semantic_ontology import SemanticOntology
from test_semantic_contract import household


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
        {"id": "item:fridge_a", "name": {"zh": "冰箱", "en": "Fridge"}, "item_type": "appliance"},
    ])
    _write(root, "nodes", "space", [
        {"id": "space:kitchen_a", "name": {"en": "Kitchen A", "zh": "厨房"}, "space_type": "room"},
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
        {"from": "item:fridge_a", "to": "space:kitchen_a"},
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
    assert "厨房" in FactRenderer().render(location, location_result, context)

    fridge_payload = engine.schema.expand_planner_concepts({
        "request": {
            "operation": "select",
            "subject": {
                "kind": "named_entity",
                "entity_type": "item",
                "value": "冰箱",
                "path": [{"concept": "location"}],
            },
        }
    })
    fridge = SemanticFactRequest.model_validate(fridge_payload["request"])
    assert engine.schema.validates(fridge)
    fridge_result, *_ = await engine.execute(fridge, context)
    assert fridge_result.status == "found"
    located = fridge_result.value if isinstance(fridge_result.value, list) else [fridge_result.value]
    assert [item["id"] for item in located] == ["space:kitchen_a"]

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


@pytest.mark.asyncio
async def test_named_space_contents_filter_by_stored_item_type(tmp_path):
    _contained_households(tmp_path)
    engine, context = _engine(tmp_path)
    payload = engine.schema.expand_planner_concepts({
        "request": {
            "operation": "select",
            "subject": {
                "kind": "named_entity",
                "entity_type": "space",
                "value": "厨房",
                "path": [{"concept": "contents"}],
            },
            "filters": [{"property": "item_type", "value": "appliance"}],
        }
    })
    request = SemanticFactRequest.model_validate(payload["request"])

    assert engine.schema.validation_code(request) == "VALID"
    result, *_ = await engine.execute(request, context)

    assert result.status == "found"
    assert [item["id"] for item in result.value] == ["item:fridge_a"]

    location = SemanticFactRequest(
        operation="resolve_reference",
        subject=SemanticReference(
            kind="named_entity",
            entity_type="item",
            value="冰箱",
            path=(SemanticRelationStep(relation="location"),),
        ),
    )
    located, *_ = await engine.execute(location, context)
    assert located.focus_entity_ids == ("item:fridge_a", "space:kitchen_a")

    other = request.model_copy(update={
        "exclude": (
            SemanticReference(
                kind="discourse",
                entity_type="item",
                turn_offset=1,
            ),
        )
    })
    discourse = DiscourseContext(
        "conversation",
        context.caller_entity_id,
        context.household_id,
        context.assistant_id,
        (located.focus_entity_ids,),
    )
    other_result, *_ = await engine.execute(
        other,
        replace(context, conversation_id="conversation", discourse=discourse),
    )
    assert other_result.status == "found"
    assert other_result.value == []


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


def test_retry_hint_covers_self_member_and_disjoint_predicates(tmp_path):
    _contained_households(tmp_path)
    engine, _ = _engine(tmp_path)
    self_member = SemanticFactRequest(
        operation="argmin",
        subject=SemanticReference(
            kind="self",
            entity_type="person",
            path=(SemanticRelationStep(relation="member"),),
        ),
        property="birth_date",
    )
    hint = _invalid_plan_retry_hint(engine.schema, self_member)
    assert hint and "current_household" in hint and "member" in hint
    assert engine.schema.validation_code(self_member) == "INVALID_PLAN"

    both = SemanticFactRequest(
        operation="count",
        subject=SemanticReference(
            kind="current_household",
            entity_type="address",
            path=(SemanticRelationStep(relation="member"),),
        ),
        filters=(
            SemanticFilter(predicate="adult"),
            SemanticFilter(predicate="minor"),
        ),
    )
    assert engine.schema.contract_error(both) == "CONTRADICTORY_PREDICATES"
    hint = _invalid_plan_retry_hint(engine.schema, both)
    assert hint and "gender" in hint
    space_on_people = SemanticFactRequest(
        operation="select",
        subject=SemanticReference(
            kind="current_household",
            entity_type="address",
            path=(SemanticRelationStep(relation="member"),),
        ),
        filters=(SemanticFilter(property="space_type", value="room"),),
    )
    hint = _invalid_plan_retry_hint(engine.schema, space_on_people)
    assert hint and "room" in hint
    identity = SemanticFactRequest(
        operation="same_entity",
        subject=SemanticReference(kind="self", entity_type="person"),
    )
    assert engine.schema.validation_code(identity) == "INVALID_PLAN"
    hint = _invalid_plan_retry_hint(engine.schema, identity)
    assert hint and "same_entity" in hint and "other" in hint
    household_location = SemanticFactRequest(
        operation="select",
        subject=SemanticReference(
            kind="current_household",
            entity_type="address",
            path=(
                SemanticRelationStep(relation="contents"),
                SemanticRelationStep(relation="location"),
            ),
        ),
    )
    hint = _invalid_plan_retry_hint(engine.schema, household_location)
    assert hint and "named_entity" in hint and "location" in hint
    person_location = SemanticFactRequest(
        operation="select",
        subject=SemanticReference(
            kind="named_entity",
            entity_type="person",
            value="花瓶",
            path=(SemanticRelationStep(relation="location"),),
        ),
    )
    hint = _invalid_plan_retry_hint(engine.schema, person_location)
    assert hint and "entity_type=item" in hint
    named_adult = SemanticFactRequest(
        operation="select",
        subject=SemanticReference(
            kind="named_entity",
            entity_type="person",
            value="冰箱",
        ),
        filters=(SemanticFilter(predicate="adult"),),
    )
    hint = _invalid_plan_retry_hint(engine.schema, named_adult)
    assert hint and "location" in hint and "item" in hint


@pytest.mark.parametrize("utterance", ["冰箱在哪里", "牛奶在哪里", "洗衣机在哪里", "Where is the kettle?"])
def test_object_location_hint_compiles_named_item_not_person(utterance):
    hint = _object_location_hint(utterance)
    assert hint and "entity_type=item" in hint and "location" in hint
    wrong = SemanticFactRequest(
        operation="resolve_reference",
        subject=SemanticReference(kind="named_entity", value="冰箱", entity_type="person"),
    )
    assert _object_location_mismatch(utterance, wrong)
    right = SemanticFactRequest(
        operation="resolve_reference",
        subject=SemanticReference(
            kind="named_entity",
            value="冰箱" if "在哪里" in utterance else "kettle",
            entity_type="item",
            path=(SemanticRelationStep(relation="location"),),
        ),
    )
    assert _object_location_mismatch(utterance, right) is None


@pytest.mark.parametrize("utterance", ["我家在哪里", "Where is my son?", "家里有几个房间", "Who am I?"])
def test_object_location_hint_ignores_residence_and_identity(utterance):
    assert _object_location_hint(utterance) is None


@pytest.mark.asyncio
async def test_object_where_question_never_repairs_interpreter_semantics(tmp_path):
    _contained_households(tmp_path)
    engine, context = _engine(tmp_path)

    class Interpreter:
        async def plan_semantic_fact(self, *_args, **_kwargs):
            return {
                "requires_fact": True,
                "request": {
                    "operation": "select",
                    "subject": {
                        "kind": "named_entity",
                        "value": "冰箱",
                        "entity_type": "person",
                    },
                    "property": None,
                    "property_source": "entity",
                    "filters": [{"predicate": "adult"}],
                },
            }

    with pytest.raises(SemanticPlannerFailure):
        await SemanticFactPlanner(Interpreter(), engine.schema).plan(
            [{"role": "user", "content": "冰箱在哪里"}], context,
        )


@pytest.mark.asyncio
async def test_gender_year_and_age_floor_filters_do_not_need_adult(household):
    engine, context, _ = household
    members = SemanticReference(
        kind="current_household",
        path=(SemanticRelationStep(relation="member"),),
    )
    male = SemanticFactRequest(
        operation="count",
        subject=members,
        filters=(SemanticFilter(property="gender", value="male"),),
    )
    year = SemanticFactRequest(
        operation="count",
        subject=members,
        filters=(
            SemanticFilter(
                property="birth_date",
                operator="date_range",
                value=("2012-01-01", "2013-01-01"),
            ),
        ),
    )
    age_floor = SemanticFactRequest(
        operation="count",
        subject=members,
        filters=(
            SemanticFilter(property="birth_date", operator="lte", value="1971-09-03"),
        ),
    )
    oldest = SemanticFactRequest(
        operation="argmin",
        subject=members,
        property="birth_date",
    )
    for request in (male, year, age_floor, oldest):
        assert engine.schema.validates(request)
        assert engine.schema.contract_error(request) is None
    male_result, *_ = await engine.execute(male, context)
    year_result, *_ = await engine.execute(year, context)
    age_result, *_ = await engine.execute(age_floor, context)
    oldest_result, *_ = await engine.execute(oldest, context)
    assert male_result.status == "found" and male_result.value == 5
    assert year_result.status == "found" and year_result.value == 1
    assert age_result.status == "found" and age_result.value == 2
    assert oldest_result.status == "found"
    assert oldest_result.value["id"] == "person:father"


@pytest.mark.asyncio
async def test_derived_age_threshold_is_computed_not_a_guessed_date_range(household):
    engine, context, _ = household
    members = SemanticReference(
        kind="current_household",
        path=(SemanticRelationStep(relation="member"),),
    )
    request = SemanticFactRequest(
        operation="select",
        subject=members,
        filters=(
            SemanticFilter(
                property="birth_date",
                transform="date_difference",
                mode="years",
                operator="gte",
                value=35,
            ),
        ),
    )
    assert engine.schema.validates(request)
    result, *_ = await engine.execute(request, context)
    ids = {item["id"] for item in result.value}
    assert ids == {"person:a", "person:b", "person:father", "person:mother"}
    text = FactRenderer(engine.schema.ontology).render(
        request, result, replace(context, locale="zh")
    )
    assert "匡" not in text
    assert "年龄 ≥ 35岁" in text
    assert "2010-09-08" not in text
    inverted = SemanticFactRequest(
        operation="select",
        subject=members,
        filters=(
            SemanticFilter(
                property="birth_date",
                operator="date_range",
                value=("2010-09-08", "2026-09-08"),
            ),
        ),
    )
    wrong, *_ = await engine.execute(inverted, context)
    wrong_ids = {item["id"] for item in wrong.value}
    assert wrong_ids != ids
    assert engine.schema.validates(inverted)
    older = SemanticFactRequest(
        operation="count",
        subject=members,
        filters=(
            SemanticFilter(
                property="birth_date",
                transform="date_difference",
                mode="years",
                operator="gte",
                value=30,
            ),
        ),
    )
    younger = older.model_copy(
        update={
            "filters": (
                SemanticFilter(
                    property="birth_date",
                    transform="date_difference",
                    mode="years",
                    operator="lt",
                    value=30,
                ),
            )
        }
    )
    older_result, *_ = await engine.execute(older, context)
    younger_result, *_ = await engine.execute(younger, context)
    assert older_result.value == 4 and younger_result.value == 4
    assert older_result.value + younger_result.value == 8
    hop = SemanticFactRequest(
        operation="select",
        subject=SemanticReference(
            kind="current_household",
            path=(
                SemanticRelationStep(
                    relation="member",
                    filters=(
                        SemanticFilter(
                            property="birth_date",
                            transform="date_difference",
                            mode="years",
                            operator="gte",
                            value=35,
                        ),
                    ),
                ),
            ),
        ),
    )
    hop_result, *_ = await engine.execute(hop, context)
    assert {item["id"] for item in hop_result.value} == ids
