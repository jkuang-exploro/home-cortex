from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from ollama import ChatResponse
from pydantic import ValidationError

from home_cortex.agent_service import AgentService
from home_cortex.agents import get_agent
from home_cortex.edge_schema import EdgeSchemaRegistry
from home_cortex.grounding import (
    FreshnessRequirement,
    GroundingEvidence,
    GroundingExecutor,
    GroundingPlan,
    GroundingPlanner,
    GroundingSubject,
    OpenWorldGroundingService,
    QueryFilter,
    QuerySort,
    RequiredEvidence,
    TransformSpec,
    TraversalStep,
    _apply_sort,
    _apply_transform,
    _complete_evidence_requirements,
    _deterministic_evidence_answer,
    _planner_output_schema,
)
from home_cortex.schema_catalog import (
    EntityTypeSchema,
    RelationTypeSchema,
    RuntimeSchemaCatalog,
)


CATALOG = RuntimeSchemaCatalog(
    {
        "person": EntityTypeSchema(
            "person",
            (
                "id",
                "name",
                "dob",
                "occupation",
                "shoe_size_us",
                "income",
            ),
        ),
        "measurement": EntityTypeSchema(
            "measurement",
            ("id", "name", "temperature_c", "observed_at"),
        ),
        "transaction": EntityTypeSchema(
            "transaction",
            ("id", "name", "amount", "occurred_at"),
        ),
    },
    {
        "spouse_of": RelationTypeSchema(
            "spouse_of",
            ("person",),
            ("person",),
            ("start", "end"),
            True,
            True,
            None,
        ),
        "parent_of": RelationTypeSchema(
            "parent_of",
            ("person",),
            ("person",),
            ("type",),
            False,
            False,
            "child_of",
        ),
        "has_measurement": RelationTypeSchema(
            "has_measurement",
            ("person",),
            ("measurement",),
            (),
            False,
            False,
            None,
        ),
        "has_transaction": RelationTypeSchema(
            "has_transaction",
            ("person",),
            ("transaction",),
            (),
            False,
            False,
            None,
        ),
    },
    frozenset({"Test Person", "Test Spouse", "Test Father-in-law"}),
)


class Dispatcher:
    def __init__(
        self,
        entities: dict[str, dict[str, Any]],
        relationships: dict[tuple[str, str], list[dict[str, Any]]] | None = None,
    ) -> None:
        self.entities = entities
        self.relationships = relationships or {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def dispatch(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        **_: Any,
    ) -> dict[str, Any]:
        self.calls.append((tool_name, arguments))
        if tool_name == "search_entities":
            query = arguments["text"].casefold()
            entity_type = arguments["entity_type"]
            records = [
                {key: entity[key] for key in ("id", "name") if key in entity}
                for entity in self.entities.values()
                if entity["id"].startswith(f"{entity_type}:")
                and query
                in {
                    alias.casefold()
                    for alias in (
                        entity["name"]
                        if isinstance(entity.get("name"), list)
                        else [entity.get("name")]
                    )
                    if isinstance(alias, str)
                }
            ]
        elif tool_name == "get_relationships":
            records = self.relationships.get(
                (arguments["entity_id"], arguments["relation"]),
                [],
            )
        else:
            entity = self.entities.get(arguments["entity_id"])
            records = [entity] if entity is not None else []
        return {"ok": True, "tool": tool_name, "result": records}


def _named_property_plan(field: str) -> GroundingPlan:
    return GroundingPlan(
        requires_grounding=True,
        grounding_domain="household",
        goal=f"Test Person's {field}",
        subject=GroundingSubject(
            anchor="named_entity",
            reference="Test Person",
            expected_type="person",
        ),
        fields=(field,),
        required_evidence=(RequiredEvidence(field=field),),
    )


def test_planner_schema_requires_every_top_level_plan_slot() -> None:
    schema = _planner_output_schema()

    assert set(schema["required"]) == set(schema["properties"])


def test_planner_compiler_adds_declared_query_dependencies_to_evidence() -> None:
    payload = {
        "requires_grounding": True,
        "grounding_domain": "household",
        "goal": "my parent's date of birth",
        "subject": {
            "anchor": "authenticated_user",
            "reference": "me",
            "expected_type": "person",
        },
        "traversal": [{"relation": "parent_of", "direction": "in"}],
        "fields": ["dob"],
        "required_evidence": [],
    }

    completed = _complete_evidence_requirements(payload)

    assert {item.get("field") for item in completed["required_evidence"]} == {
        None,
        "dob",
    }
    assert {item.get("relation") for item in completed["required_evidence"]} == {
        None,
        "parent_of",
    }


@pytest.mark.asyncio
async def test_arbitrary_new_entity_property_needs_no_code_change() -> None:
    dispatcher = Dispatcher(
        {
            "person:test": {
                "id": "person:test",
                "name": ["Test Person"],
                "shoe_size_us": 10,
                "income": 999_999,
            }
        }
    )

    evidence = await GroundingExecutor(
        dispatcher,
        CATALOG,
        home_entity_id=None,
    ).execute(
        _named_property_plan("shoe_size_us"),
        caller_entity_id=None,
        household_now=datetime.fromisoformat("2026-08-31T12:00:00-07:00"),
    )

    assert evidence.status == "sufficient"
    assert evidence.records == (
        {"id": "person:test", "name": ["Test Person"], "shoe_size_us": 10},
    )
    assert "income" not in evidence.records[0]


@pytest.mark.asyncio
async def test_new_favorite_temperature_property_is_immediately_queryable() -> None:
    catalog = RuntimeSchemaCatalog(
        {
            "person": EntityTypeSchema(
                "person",
                ("id", "name", "favorite_temperature_c"),
            )
        },
        {},
        frozenset({"Test Person"}),
    )
    plan = _named_property_plan("favorite_temperature_c")

    evidence = await GroundingExecutor(
        Dispatcher(
            {
                "person:test": {
                    "id": "person:test",
                    "name": ["Test Person"],
                    "favorite_temperature_c": 21,
                }
            }
        ),
        catalog,
        home_entity_id=None,
    ).execute(
        plan,
        caller_entity_id=None,
        household_now=datetime.fromisoformat("2026-08-31T12:00:00-07:00"),
    )

    assert evidence.status == "sufficient"
    assert evidence.records[0]["favorite_temperature_c"] == 21


@pytest.mark.asyncio
async def test_derived_age_uses_generic_date_difference_operator() -> None:
    dispatcher = Dispatcher(
        {
            "person:test": {
                "id": "person:test",
                "name": ["Test Person"],
                "dob": "1988-01-01",
            }
        }
    )
    plan = _named_property_plan("dob").model_copy(
        update={
            "goal": "Test Person's age",
            "transform": TransformSpec(
                operator="date_difference",
                field="dob",
                mode="completed_years",
                reference="household_today",
            ),
        }
    )

    evidence = await GroundingExecutor(
        dispatcher,
        CATALOG,
        home_entity_id=None,
    ).execute(
        plan,
        caller_entity_id=None,
        household_now=datetime.fromisoformat("2026-08-31T12:00:00-07:00"),
    )

    assert evidence.status == "sufficient"
    assert evidence.value == 38


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "stored_date", "mode", "expected"),
    (
        ("dob", "2016-10-30", "days", 60),
        ("start", "2010-09-15", None, "2026-09-15"),
    ),
)
async def test_recurring_dates_use_generic_annual_occurrence_operator(
    field: str,
    stored_date: str,
    mode: str | None,
    expected: Any,
) -> None:
    records = [{field: stored_date}]
    transform = TransformSpec(
        operator="annual_occurrence",
        source="entity" if field == "dob" else "edge",
        field=field,
        reference="household_today",
        mode=mode,
    )

    value = _apply_transform(
        records if field == "dob" else [],
        records if field == "start" else [],
        transform,
        datetime.fromisoformat("2026-08-31T12:00:00-07:00"),
    )

    assert value == expected


@pytest.mark.asyncio
async def test_father_in_law_age_uses_generic_traversal_and_date_operator() -> None:
    spouse = {
        "id": "person:spouse",
        "name": ["Test Spouse"],
        "gender": "female",
    }
    father = {
        "id": "person:father_in_law",
        "name": ["Test Father-in-law"],
        "gender": "male",
        "dob": "1961-10-10",
    }
    dispatcher = Dispatcher(
        {spouse["id"]: spouse, father["id"]: father},
        {
            ("person:test", "spouse_of"): [
                {
                    "relation": "spouse_of",
                    "semantic_relation": "spouse_of",
                    "related_entity": spouse,
                }
            ],
            ("person:spouse", "parent_of"): [
                {
                    "relation": "parent_of",
                    "semantic_relation": "child_of",
                    "related_entity": father,
                }
            ],
        },
    )
    plan = GroundingPlan(
        requires_grounding=True,
        grounding_domain="household",
        goal="my father-in-law's age",
        subject=GroundingSubject(
            anchor="authenticated_user",
            reference="me",
            expected_type="person",
        ),
        traversal=(
            TraversalStep(relation="spouse_of", related_type="person"),
            TraversalStep(
                relation="parent_of",
                direction="in",
                related_type="person",
                field_equals={"gender": "male"},
            ),
        ),
        fields=("dob",),
        transform=TransformSpec(
            operator="date_difference",
            field="dob",
            mode="completed_years",
            reference="household_today",
        ),
        required_evidence=(
            RequiredEvidence(relation="spouse_of"),
            RequiredEvidence(relation="parent_of"),
            RequiredEvidence(field="gender"),
            RequiredEvidence(field="dob"),
        ),
    )

    evidence = await GroundingExecutor(
        dispatcher,
        CATALOG,
        home_entity_id=None,
    ).execute(
        plan,
        caller_entity_id="person:test",
        household_now=datetime.fromisoformat("2026-08-31T12:00:00-07:00"),
    )

    assert evidence.status == "sufficient"
    assert evidence.value == 64
    assert evidence.records == (
        {
            "id": "person:father_in_law",
            "name": ["Test Father-in-law"],
            "gender": "male",
            "dob": "1961-10-10",
        },
    )


@pytest.mark.asyncio
async def test_existing_entity_without_requested_field_is_distinguished() -> None:
    dispatcher = Dispatcher(
        {
            "person:test": {
                "id": "person:test",
                "name": ["Test Person"],
                "occupation": "engineer",
            }
        }
    )

    evidence = await GroundingExecutor(
        dispatcher,
        CATALOG,
        home_entity_id=None,
    ).execute(
        _named_property_plan("income"),
        caller_entity_id=None,
        household_now=datetime.fromisoformat("2026-08-31T12:00:00-07:00"),
    )

    assert evidence.status == "field_not_available"
    assert evidence.missing_fields == ("income",)
    assert evidence.records == ()


@pytest.mark.asyncio
async def test_missing_field_never_reaches_answer_generation() -> None:
    plan = _named_property_plan("income")

    class Ollama:
        async def plan_grounding(self, *_: Any, **__: Any) -> dict[str, Any]:
            return plan.model_dump(mode="json")

        async def render_grounded_answer(self, **_: Any) -> str:
            raise AssertionError("renderer must not receive insufficient evidence")

    service = OpenWorldGroundingService(
        GroundingPlanner(Ollama(), CATALOG),
        GroundingExecutor(
            Dispatcher(
                {
                    "person:test": {
                        "id": "person:test",
                        "name": ["Test Person"],
                        "occupation": "engineer",
                    }
                }
            ),
            CATALOG,
            home_entity_id=None,
        ),
    )

    answer = await service.try_answer(
        [{"role": "user", "content": "What is Test Person's income?"}],
        caller_entity_id=None,
        household_now=datetime.fromisoformat("2026-08-31T12:00:00-07:00"),
        language="en",
    )

    assert answer is not None
    assert "required fields are absent: income" in answer.text


def test_plan_rejects_fields_not_declared_as_required_evidence() -> None:
    with pytest.raises(ValidationError, match="every used field"):
        GroundingPlan(
            requires_grounding=True,
            grounding_domain="household",
            goal="private income",
            subject=GroundingSubject(
                anchor="named_entity",
                reference="Test Person",
                expected_type="person",
            ),
            fields=("income",),
            required_evidence=(RequiredEvidence(field="occupation"),),
        )


@pytest.mark.asyncio
async def test_authenticated_anchor_must_match_planned_entity_type() -> None:
    plan = GroundingPlan(
        requires_grounding=True,
        grounding_domain="household",
        goal="my temperature",
        subject=GroundingSubject(
            anchor="authenticated_user",
            reference="me",
            expected_type="measurement",
        ),
        fields=("temperature_c",),
        required_evidence=(RequiredEvidence(field="temperature_c"),),
    )

    evidence = await GroundingExecutor(
        Dispatcher({}),
        CATALOG,
        home_entity_id=None,
    ).execute(
        plan,
        caller_entity_id="person:test",
        household_now=datetime.fromisoformat("2026-08-31T12:00:00-07:00"),
    )

    assert evidence.status == "entity_not_found"


@pytest.mark.asyncio
async def test_missing_authenticated_entity_is_entity_not_found() -> None:
    evidence = await GroundingExecutor(
        Dispatcher({}),
        CATALOG,
        home_entity_id=None,
    ).execute(
        GroundingPlan(
            requires_grounding=True,
            grounding_domain="household",
            goal="my occupation",
            subject=GroundingSubject(
                anchor="authenticated_user",
                reference="me",
                expected_type="person",
            ),
            fields=("occupation",),
            required_evidence=(RequiredEvidence(field="occupation"),),
        ),
        caller_entity_id="person:missing",
        household_now=datetime.fromisoformat("2026-08-31T12:00:00-07:00"),
    )

    assert evidence.status == "entity_not_found"


@pytest.mark.asyncio
async def test_unknown_subject_is_entity_not_found() -> None:
    evidence = await GroundingExecutor(
        Dispatcher({}),
        CATALOG,
        home_entity_id=None,
    ).execute(
        _named_property_plan("occupation"),
        caller_entity_id=None,
        household_now=datetime.fromisoformat("2026-08-31T12:00:00-07:00"),
    )

    assert evidence.status == "entity_not_found"


def _temperature_plan(max_age_seconds: int) -> GroundingPlan:
    return GroundingPlan(
        requires_grounding=True,
        grounding_domain="household",
        goal="my latest body temperature",
        subject=GroundingSubject(
            anchor="authenticated_user",
            reference="me",
            expected_type="person",
        ),
        traversal=(
            TraversalStep(
                relation="has_measurement",
                direction="out",
                related_type="measurement",
            ),
        ),
        fields=("temperature_c", "observed_at"),
        sort=(QuerySort(field="observed_at", direction="desc"),),
        transform=TransformSpec(
            operator="latest",
            field="temperature_c",
            order_by="observed_at",
        ),
        required_evidence=(
            RequiredEvidence(
                field="temperature_c",
                freshness=FreshnessRequirement(
                    timestamp_field="observed_at",
                    max_age_seconds=max_age_seconds,
                ),
            ),
            RequiredEvidence(relation="has_measurement"),
        ),
    )


@pytest.mark.asyncio
async def test_latest_measurement_is_selected_generically() -> None:
    entities = {
        "measurement:old": {
            "id": "measurement:old",
            "name": "older reading",
            "temperature_c": 37.1,
            "observed_at": "2026-08-31T10:00:00-07:00",
        },
        "measurement:new": {
            "id": "measurement:new",
            "name": "newer reading",
            "temperature_c": 36.8,
            "observed_at": "2026-08-31T11:30:00-07:00",
        },
    }
    relationships = {
        ("person:test", "has_measurement"): [
            {
                "relation": "has_measurement",
                "related_entity": {"id": entity["id"], "name": entity["name"]},
            }
            for entity in entities.values()
        ]
    }

    evidence = await GroundingExecutor(
        Dispatcher(entities, relationships),
        CATALOG,
        home_entity_id=None,
    ).execute(
        _temperature_plan(7_200),
        caller_entity_id="person:test",
        household_now=datetime.fromisoformat("2026-08-31T12:00:00-07:00"),
    )

    assert evidence.status == "sufficient"
    assert evidence.value == 36.8
    assert evidence.records == (
        {
            "id": "measurement:new",
            "name": "newer reading",
            "temperature_c": 36.8,
            "observed_at": "2026-08-31T11:30:00-07:00",
        },
    )


@pytest.mark.asyncio
async def test_latest_sort_keeps_missing_timestamps_out_of_first_place() -> None:
    entities = {
        "measurement:undated": {
            "id": "measurement:undated",
            "name": "undated reading",
            "temperature_c": 39.9,
        },
        "measurement:dated": {
            "id": "measurement:dated",
            "name": "dated reading",
            "temperature_c": 36.8,
            "observed_at": "2026-08-31T11:30:00-07:00",
        },
    }
    dispatcher = Dispatcher(
        entities,
        {
            ("person:test", "has_measurement"): [
                {
                    "relation": "has_measurement",
                    "related_entity": {
                        "id": entity["id"],
                        "name": entity["name"],
                    },
                }
                for entity in entities.values()
            ]
        },
    )

    evidence = await GroundingExecutor(
        dispatcher,
        CATALOG,
        home_entity_id=None,
    ).execute(
        _temperature_plan(7_200),
        caller_entity_id="person:test",
        household_now=datetime.fromisoformat("2026-08-31T12:00:00-07:00"),
    )

    assert evidence.status == "sufficient"
    assert evidence.value == 36.8
    assert evidence.records == (
        {
            "id": "measurement:dated",
            "name": "dated reading",
            "temperature_c": 36.8,
            "observed_at": "2026-08-31T11:30:00-07:00",
        },
    )


@pytest.mark.asyncio
async def test_stale_measurement_is_rejected_generically() -> None:
    entity = {
        "id": "measurement:old",
        "name": "old reading",
        "temperature_c": 37.1,
        "observed_at": "2026-08-30T12:00:00-07:00",
    }
    dispatcher = Dispatcher(
        {entity["id"]: entity},
        {
            ("person:test", "has_measurement"): [
                {
                    "relation": "has_measurement",
                    "related_entity": {"id": entity["id"], "name": entity["name"]},
                }
            ]
        },
    )

    evidence = await GroundingExecutor(
        dispatcher,
        CATALOG,
        home_entity_id=None,
    ).execute(
        _temperature_plan(7_200),
        caller_entity_id="person:test",
        household_now=datetime.fromisoformat("2026-08-31T12:00:00-07:00"),
    )

    assert evidence.status == "evidence_stale"
    assert evidence.records == ()


@pytest.mark.asyncio
async def test_filter_and_sum_support_period_aggregation() -> None:
    entities = {
        "transaction:august": {
            "id": "transaction:august",
            "name": "August groceries",
            "amount": 30,
            "occurred_at": "2026-08-15",
        },
        "transaction:july": {
            "id": "transaction:july",
            "name": "July groceries",
            "amount": 20,
            "occurred_at": "2026-07-15",
        },
    }
    dispatcher = Dispatcher(
        entities,
        {
            ("person:test", "has_transaction"): [
                {
                    "relation": "has_transaction",
                    "related_entity": {"id": entity["id"], "name": entity["name"]},
                }
                for entity in entities.values()
            ]
        },
    )
    plan = GroundingPlan(
        requires_grounding=True,
        grounding_domain="household",
        goal="my spending this month",
        subject=GroundingSubject(
            anchor="authenticated_user",
            reference="me",
            expected_type="person",
        ),
        traversal=(
            TraversalStep(
                relation="has_transaction",
                direction="out",
                related_type="transaction",
            ),
        ),
        fields=("amount",),
        filters=(
            QueryFilter(field="occurred_at", operator="gte", value="2026-08-01"),
            QueryFilter(field="occurred_at", operator="lt", value="2026-09-01"),
        ),
        transform=TransformSpec(operator="sum", field="amount"),
        required_evidence=(
            RequiredEvidence(field="amount"),
            RequiredEvidence(field="occurred_at"),
            RequiredEvidence(relation="has_transaction"),
        ),
    )

    evidence = await GroundingExecutor(
        dispatcher,
        CATALOG,
        home_entity_id=None,
    ).execute(
        plan,
        caller_entity_id="person:test",
        household_now=datetime.fromisoformat("2026-08-31T12:00:00-07:00"),
    )

    assert evidence.status == "sufficient"
    assert evidence.value == 30


@pytest.mark.asyncio
async def test_edge_property_is_retrieved_without_a_fact_handler() -> None:
    spouse = {
        "id": "person:spouse",
        "name": ["Test Spouse"],
    }
    dispatcher = Dispatcher(
        {spouse["id"]: spouse},
        {
            ("person:test", "spouse_of"): [
                {
                    "id": "spouse_of:test_spouse",
                    "relation": "spouse_of",
                    "start": "2014-05-04",
                    "end": None,
                    "related_entity": spouse,
                }
            ]
        },
    )
    plan = GroundingPlan(
        requires_grounding=True,
        grounding_domain="household",
        goal="our marriage date",
        subject=GroundingSubject(
            anchor="authenticated_user",
            reference="me",
            expected_type="person",
        ),
        traversal=(TraversalStep(relation="spouse_of"),),
        edge_fields=("start",),
        required_evidence=(
            RequiredEvidence(source="edge", field="start"),
            RequiredEvidence(source="edge", relation="spouse_of"),
        ),
    )

    evidence = await GroundingExecutor(
        dispatcher,
        CATALOG,
        home_entity_id=None,
    ).execute(
        plan,
        caller_entity_id="person:test",
        household_now=datetime.fromisoformat("2026-08-31T12:00:00-07:00"),
    )

    assert evidence.status == "sufficient"
    assert evidence.edges == (
        {
            "id": "spouse_of:test_spouse",
            "relation": "spouse_of",
            "start": "2014-05-04",
        },
    )


@pytest.mark.asyncio
async def test_relation_count_uses_edge_evidence_without_loading_entities() -> None:
    children = [
        {"id": "person:child_one", "name": ["Child One"]},
        {"id": "person:child_two", "name": ["Child Two"]},
    ]
    dispatcher = Dispatcher(
        {},
        {
            ("person:test", "parent_of"): [
                {
                    "relation": "parent_of",
                    "semantic_relation": "parent_of",
                    "related_entity": child,
                }
                for child in children
            ]
        },
    )
    plan = GroundingPlan(
        requires_grounding=True,
        grounding_domain="household",
        goal="how many children I have",
        subject=GroundingSubject(
            anchor="authenticated_user",
            reference="me",
            expected_type="person",
        ),
        traversal=(TraversalStep(relation="parent_of", direction="out"),),
        transform=TransformSpec(operator="count", source="edge"),
        required_evidence=(RequiredEvidence(relation="parent_of"),),
    )

    evidence = await GroundingExecutor(
        dispatcher,
        CATALOG,
        home_entity_id=None,
    ).execute(
        plan,
        caller_entity_id="person:test",
        household_now=datetime.fromisoformat("2026-08-31T12:00:00-07:00"),
    )

    assert evidence.status == "sufficient"
    assert evidence.value == 2
    assert evidence.records == ()
    assert [name for name, _ in dispatcher.calls] == ["get_relationships"]


def test_runtime_schema_catalog_discovers_new_fields_without_code_changes(
    tmp_path: Path,
) -> None:
    nodes = tmp_path / "data" / "nodes"
    edges = tmp_path / "data" / "edges"
    nodes.mkdir(parents=True)
    edges.mkdir(parents=True)
    (nodes / "person.json").write_text(
        '[{"id":"person:test","name":"Test Person","annual_income":1}]',
        encoding="utf-8",
    )

    catalog = RuntimeSchemaCatalog.from_data_dir(
        tmp_path / "data",
        EdgeSchemaRegistry.load_default(),
    )

    assert catalog.has_entity_field("person", "annual_income")
    assert "Test Person" in catalog.entity_aliases


@pytest.mark.asyncio
async def test_structured_llm_plan_drives_open_world_service() -> None:
    plan = _named_property_plan("occupation")

    class Ollama:
        async def plan_grounding(self, *_: Any, **__: Any) -> dict[str, Any]:
            return plan.model_dump(mode="json")

    planner = GroundingPlanner(Ollama(), CATALOG)
    service = OpenWorldGroundingService(
        planner,
        GroundingExecutor(
            Dispatcher(
                {
                    "person:test": {
                        "id": "person:test",
                        "name": ["Test Person"],
                        "occupation": "engineer",
                    }
                }
            ),
            CATALOG,
            home_entity_id=None,
        ),
    )

    answer = await service.try_answer(
        [{"role": "user", "content": "What does Test Person do for work?"}],
        caller_entity_id=None,
        household_now=datetime.fromisoformat("2026-08-31T12:00:00-07:00"),
        language="en",
    )

    assert answer is not None
    assert answer.text == (
        'According to the home graph, the result is: {"occupation": "engineer"}.'
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("question", "plan", "expected"),
    (
        (
            "我是谁",
            GroundingPlan(
                requires_grounding=True,
                grounding_domain="household",
                goal="authenticated speaker identity",
                subject=GroundingSubject(
                    anchor="authenticated_user",
                    reference="我",
                    expected_type="person",
                ),
                fields=("name",),
                required_evidence=(RequiredEvidence(field="name"),),
            ),
            '"name": "匡健"',
        ),
        (
            "家里都有谁",
            GroundingPlan(
                requires_grounding=True,
                grounding_domain="household",
                goal="current home residents",
                subject=GroundingSubject(
                    anchor="configured_home",
                    reference="家里",
                    expected_type="address",
                ),
                traversal=(
                    TraversalStep(
                        relation="lives_in",
                        direction="in",
                        related_type="person",
                    ),
                ),
                fields=("name",),
                required_evidence=(
                    RequiredEvidence(relation="lives_in"),
                    RequiredEvidence(field="name", minimum_records=2),
                ),
            ),
            "匡健",
        ),
        (
            "家里地址是哪里",
            GroundingPlan(
                requires_grounding=True,
                grounding_domain="household",
                goal="home address",
                subject=GroundingSubject(
                    anchor="configured_home",
                    reference="家里",
                    expected_type="address",
                ),
                fields=("address",),
                required_evidence=(RequiredEvidence(field="address"),),
            ),
            "12745 Droxford St",
        ),
        (
            "我的出生日期是什么时候？",
            GroundingPlan(
                requires_grounding=True,
                grounding_domain="household",
                goal="authenticated speaker date of birth",
                subject=GroundingSubject(
                    anchor="authenticated_user",
                    reference="我",
                    expected_type="person",
                ),
                fields=("dob",),
                required_evidence=(RequiredEvidence(field="dob"),),
            ),
            "1988-11-11",
        ),
    ),
)
async def test_reported_chinese_household_queries_use_grounding_pipeline(
    question: str,
    plan: GroundingPlan,
    expected: str,
) -> None:
    catalog = RuntimeSchemaCatalog(
        {
            "person": EntityTypeSchema("person", ("id", "name", "dob")),
            "address": EntityTypeSchema("address", ("id", "name", "address")),
        },
        {
            "lives_in": RelationTypeSchema(
                "lives_in",
                ("person",),
                ("address",),
                ("start", "end"),
                False,
                True,
                None,
            )
        },
    )
    people = {
        "person:jian": {
            "id": "person:jian",
            "name": ["Jian Kuang", "匡健"],
            "dob": "1988-11-11",
        },
        "person:pu": {
            "id": "person:pu",
            "name": ["Pu Ba", "巴璞"],
            "dob": "1988-02-26",
        },
        "address:home": {
            "id": "address:home",
            "name": ["Test Home", "测试之家"],
            "address": {"street": "12745 Droxford St"},
        },
    }
    relationships = {
        ("address:home", "lives_in"): [
            {"relation": "lives_in", "related_entity": people["person:jian"]},
            {"relation": "lives_in", "related_entity": people["person:pu"]},
        ]
    }

    class Ollama:
        async def plan_grounding(self, *_: Any, **__: Any) -> dict[str, Any]:
            payload = plan.model_dump(mode="json")
            payload["required_evidence"] = []
            return payload

    answer = await OpenWorldGroundingService(
        GroundingPlanner(Ollama(), catalog),
        GroundingExecutor(
            Dispatcher(people, relationships),
            catalog,
            home_entity_id="address:home",
        ),
    ).try_answer(
        [{"role": "user", "content": question}],
        caller_entity_id="person:jian",
        household_now=datetime.fromisoformat("2026-09-01T12:00:00-07:00"),
        language="zh",
    )

    assert answer is not None
    assert answer.stop_reason == "answer"
    assert expected in answer.text


@pytest.mark.asyncio
async def test_agent_service_uses_open_world_grounding_as_authoritative_path() -> None:
    plan = _named_property_plan("shoe_size_us")

    class Ollama:
        def __init__(self) -> None:
            self.render_calls = 0

        async def plan_grounding(self, *_: Any, **__: Any) -> dict[str, Any]:
            return plan.model_dump(mode="json")

    ollama = Ollama()
    steward = get_agent("steward")
    agent = AgentService(
        ollama,  # type: ignore[arg-type]
        Dispatcher(
            {
                "person:test": {
                    "id": "person:test",
                    "name": ["Test Person"],
                    "shoe_size_us": 10,
                }
            }
        ),
        system_prompt=steward.prompt,
        tools=steward.tool_definitions,
        home_entity_id=steward.settings["home_entity_id"],
        schema_catalog=CATALOG,
    )

    result = await agent.answer("What is Test Person's shoe size?")

    assert result.answer == (
        'According to the home graph, the result is: {"shoe_size_us": 10}.'
    )
    assert result.tool_calls == 2
    assert ollama.render_calls == 0


@pytest.mark.asyncio
async def test_partial_aggregation_is_evidence_insufficient() -> None:
    entities = {
        "transaction:complete": {
            "id": "transaction:complete",
            "name": "complete transaction",
            "amount": 30,
            "occurred_at": "2026-08-15",
        },
        "transaction:partial": {
            "id": "transaction:partial",
            "name": "transaction without amount",
            "occurred_at": "2026-08-20",
        },
    }
    dispatcher = Dispatcher(
        entities,
        {
            ("person:test", "has_transaction"): [
                {
                    "relation": "has_transaction",
                    "related_entity": {"id": item["id"], "name": item["name"]},
                }
                for item in entities.values()
            ]
        },
    )
    plan = GroundingPlan(
        requires_grounding=True,
        grounding_domain="household",
        goal="my total spending",
        subject=GroundingSubject(
            anchor="authenticated_user",
            reference="me",
            expected_type="person",
        ),
        traversal=(TraversalStep(relation="has_transaction"),),
        fields=("amount",),
        transform=TransformSpec(operator="sum", field="amount"),
        required_evidence=(
            RequiredEvidence(field="amount"),
            RequiredEvidence(relation="has_transaction"),
        ),
    )

    evidence = await GroundingExecutor(
        dispatcher,
        CATALOG,
        home_entity_id=None,
    ).execute(
        plan,
        caller_entity_id="person:test",
        household_now=datetime.fromisoformat("2026-08-31T12:00:00-07:00"),
    )

    assert evidence.status == "evidence_insufficient"


@pytest.mark.asyncio
async def test_missing_filter_field_makes_aggregation_insufficient() -> None:
    entities = {
        "transaction:dated": {
            "id": "transaction:dated",
            "name": "dated transaction",
            "amount": 30,
            "occurred_at": "2026-08-15",
        },
        "transaction:undated": {
            "id": "transaction:undated",
            "name": "undated transaction",
            "amount": 20,
        },
    }
    dispatcher = Dispatcher(
        entities,
        {
            ("person:test", "has_transaction"): [
                {
                    "relation": "has_transaction",
                    "related_entity": {"id": item["id"], "name": item["name"]},
                }
                for item in entities.values()
            ]
        },
    )
    plan = GroundingPlan(
        requires_grounding=True,
        grounding_domain="household",
        goal="my August spending",
        subject=GroundingSubject(
            anchor="authenticated_user",
            reference="me",
            expected_type="person",
        ),
        traversal=(TraversalStep(relation="has_transaction"),),
        fields=("amount",),
        filters=(
            QueryFilter(field="occurred_at", operator="gte", value="2026-08-01"),
            QueryFilter(field="occurred_at", operator="lt", value="2026-09-01"),
        ),
        transform=TransformSpec(operator="sum", field="amount"),
        required_evidence=(
            RequiredEvidence(field="amount"),
            RequiredEvidence(field="occurred_at"),
            RequiredEvidence(relation="has_transaction"),
        ),
    )

    evidence = await GroundingExecutor(
        dispatcher,
        CATALOG,
        home_entity_id=None,
    ).execute(
        plan,
        caller_entity_id="person:test",
        household_now=datetime.fromisoformat("2026-08-31T12:00:00-07:00"),
    )

    assert evidence.status == "evidence_insufficient"


@pytest.mark.asyncio
async def test_truncated_aggregation_input_is_insufficient() -> None:
    entities = {
        "transaction:one": {
            "id": "transaction:one",
            "name": "one",
            "amount": 10,
        },
        "transaction:two": {
            "id": "transaction:two",
            "name": "two",
            "amount": 20,
        },
    }
    dispatcher = Dispatcher(
        entities,
        {
            ("person:test", "has_transaction"): [
                {
                    "relation": "has_transaction",
                    "related_entity": {"id": item["id"], "name": item["name"]},
                }
                for item in entities.values()
            ]
        },
    )
    plan = GroundingPlan(
        requires_grounding=True,
        grounding_domain="household",
        goal="my total spending",
        subject=GroundingSubject(
            anchor="authenticated_user",
            reference="me",
            expected_type="person",
        ),
        traversal=(TraversalStep(relation="has_transaction"),),
        fields=("amount",),
        transform=TransformSpec(operator="sum", field="amount"),
        required_evidence=(
            RequiredEvidence(field="amount"),
            RequiredEvidence(relation="has_transaction"),
        ),
    )

    evidence = await GroundingExecutor(
        dispatcher,
        CATALOG,
        home_entity_id=None,
        max_records=1,
    ).execute(
        plan,
        caller_entity_id="person:test",
        household_now=datetime.fromisoformat("2026-08-31T12:00:00-07:00"),
    )

    assert evidence.status == "evidence_insufficient"


@pytest.mark.asyncio
async def test_latest_record_missing_value_is_evidence_insufficient() -> None:
    entities = {
        "measurement:new": {
            "id": "measurement:new",
            "name": "new reading without a value",
            "observed_at": "2026-08-31T11:50:00-07:00",
        },
        "measurement:old": {
            "id": "measurement:old",
            "name": "older complete reading",
            "temperature_c": 36.8,
            "observed_at": "2026-08-31T11:30:00-07:00",
        },
    }
    dispatcher = Dispatcher(
        entities,
        {
            ("person:test", "has_measurement"): [
                {
                    "relation": "has_measurement",
                    "related_entity": {"id": item["id"], "name": item["name"]},
                }
                for item in entities.values()
            ]
        },
    )

    evidence = await GroundingExecutor(
        dispatcher,
        CATALOG,
        home_entity_id=None,
    ).execute(
        _temperature_plan(7_200),
        caller_entity_id="person:test",
        household_now=datetime.fromisoformat("2026-08-31T12:00:00-07:00"),
    )

    assert evidence.status == "evidence_insufficient"


@pytest.mark.asyncio
async def test_future_measurement_timestamp_is_insufficient() -> None:
    entity = {
        "id": "measurement:future",
        "name": "future reading",
        "temperature_c": 36.8,
        "observed_at": "2026-08-31T12:30:00-07:00",
    }
    dispatcher = Dispatcher(
        {entity["id"]: entity},
        {
            ("person:test", "has_measurement"): [
                {
                    "relation": "has_measurement",
                    "related_entity": {"id": entity["id"], "name": entity["name"]},
                }
            ]
        },
    )

    evidence = await GroundingExecutor(
        dispatcher,
        CATALOG,
        home_entity_id=None,
    ).execute(
        _temperature_plan(7_200),
        caller_entity_id="person:test",
        household_now=datetime.fromisoformat("2026-08-31T12:00:00-07:00"),
    )

    assert evidence.status == "evidence_insufficient"


@pytest.mark.asyncio
async def test_hallucinating_renderer_cannot_add_unsupported_fact() -> None:
    plan = _named_property_plan("occupation")

    class Ollama:
        async def plan_grounding(self, *_: Any, **__: Any) -> dict[str, Any]:
            return plan.model_dump(mode="json")

        async def render_grounded_answer(self, **_: Any) -> str:
            return "Test Person is an engineer and earns $1,000,000."

    service = OpenWorldGroundingService(
        GroundingPlanner(Ollama(), CATALOG),
        GroundingExecutor(
            Dispatcher(
                {
                    "person:test": {
                        "id": "person:test",
                        "name": ["Test Person"],
                        "occupation": "engineer",
                    }
                }
            ),
            CATALOG,
            home_entity_id=None,
        ),
    )

    answer = await service.try_answer(
        [{"role": "user", "content": "What does Test Person do?"}],
        caller_entity_id=None,
        household_now=datetime.fromisoformat("2026-08-31T12:00:00-07:00"),
        language="en",
    )

    assert answer is not None
    assert "$1,000,000" not in answer.text


@pytest.mark.asyncio
async def test_multi_step_edge_count_counts_only_terminal_relation() -> None:
    spouse = {"id": "person:spouse", "name": ["Test Spouse"]}
    parents = [
        {"id": "person:parent_one", "name": ["Parent One"]},
        {"id": "person:parent_two", "name": ["Parent Two"]},
    ]
    dispatcher = Dispatcher(
        {},
        {
            ("person:test", "spouse_of"): [
                {
                    "relation": "spouse_of",
                    "semantic_relation": "spouse_of",
                    "related_entity": spouse,
                }
            ],
            ("person:spouse", "parent_of"): [
                {
                    "relation": "parent_of",
                    "semantic_relation": "child_of",
                    "related_entity": parent,
                }
                for parent in parents
            ],
        },
    )
    plan = GroundingPlan(
        requires_grounding=True,
        grounding_domain="household",
        goal="how many parents my spouse has",
        subject=GroundingSubject(
            anchor="authenticated_user",
            reference="me",
            expected_type="person",
        ),
        traversal=(
            TraversalStep(relation="spouse_of"),
            TraversalStep(relation="parent_of", direction="in"),
        ),
        transform=TransformSpec(operator="count", source="edge"),
        required_evidence=(
            RequiredEvidence(relation="spouse_of"),
            RequiredEvidence(relation="parent_of", minimum_records=2),
        ),
    )

    evidence = await GroundingExecutor(
        dispatcher,
        CATALOG,
        home_entity_id=None,
    ).execute(
        plan,
        caller_entity_id="person:test",
        household_now=datetime.fromisoformat("2026-08-31T12:00:00-07:00"),
    )

    assert evidence.status == "sufficient"
    assert evidence.value == 2


@pytest.mark.parametrize(
    ("transform", "records", "expected"),
    (
        (TransformSpec(operator="average", field="value"), [{"value": 2}, {"value": 4}], 3),
        (TransformSpec(operator="min", field="value"), [{"value": 2}, {"value": 4}], 2),
        (TransformSpec(operator="max", field="value"), [{"value": 2}, {"value": 4}], 4),
        (
            TransformSpec(operator="difference", field="left", other_field="right"),
            [{"left": 10, "right": 3}],
            7,
        ),
        (
            TransformSpec(operator="ratio", field="left", other_field="right"),
            [{"left": 10, "right": 2}],
            5,
        ),
        (
            TransformSpec(
                operator="duration",
                field="started_at",
                mode="seconds",
                reference="household_now",
            ),
            [{"started_at": "2026-08-31T10:00:00-07:00"}],
            7_200,
        ),
        (
            TransformSpec(
                operator="unit_conversion",
                field="temperature_c",
                from_unit="c",
                to_unit="f",
            ),
            [{"temperature_c": 0}],
            32,
        ),
    ),
)
def test_generic_deterministic_operators(
    transform: TransformSpec,
    records: list[dict[str, Any]],
    expected: float,
) -> None:
    value = _apply_transform(
        records,
        [],
        transform,
        datetime.fromisoformat("2026-08-31T12:00:00-07:00"),
    )

    assert value == pytest.approx(expected)


def test_earliest_operator_uses_ascending_order() -> None:
    records = _apply_sort(
        [
            {"value": 2, "observed_at": "2026-08-31T11:00:00-07:00"},
            {"value": 1, "observed_at": "2026-08-31T10:00:00-07:00"},
        ],
        [QuerySort(field="observed_at", direction="asc")],
    )

    value = _apply_transform(
        records,
        [],
        TransformSpec(operator="earliest", field="value", order_by="observed_at"),
        datetime.fromisoformat("2026-08-31T12:00:00-07:00"),
    )

    assert value == 1


def test_latest_plan_rejects_non_primary_order_by_sort() -> None:
    with pytest.raises(ValidationError, match="primary desc sort"):
        GroundingPlan(
            requires_grounding=True,
            grounding_domain="household",
            goal="latest temperature",
            subject=GroundingSubject(
                anchor="authenticated_user",
                reference="me",
                expected_type="person",
            ),
            fields=("temperature_c", "observed_at"),
            sort=(
                QuerySort(field="temperature_c", direction="desc"),
                QuerySort(field="observed_at", direction="desc"),
            ),
            transform=TransformSpec(
                operator="latest",
                field="temperature_c",
                order_by="observed_at",
            ),
            required_evidence=(
                RequiredEvidence(field="temperature_c"),
                RequiredEvidence(field="observed_at"),
            ),
        )


def test_deterministic_renderer_preserves_multiple_requested_records() -> None:
    plan = GroundingPlan(
        requires_grounding=True,
        grounding_domain="household",
        goal="parent birthdays",
        subject=GroundingSubject(
            anchor="authenticated_user",
            reference="me",
            expected_type="person",
        ),
        fields=("dob",),
        required_evidence=(RequiredEvidence(field="dob", minimum_records=2),),
    )
    evidence = GroundingEvidence(
        "sufficient",
        records=(
            {"id": "person:one", "name": ["Parent One"], "dob": "1960-01-01"},
            {"id": "person:two", "name": ["Parent Two"], "dob": "1965-01-01"},
        ),
    )

    answer = _deterministic_evidence_answer(plan, evidence, "en")

    assert "Parent One" in answer
    assert "1960-01-01" in answer
    assert "Parent Two" in answer
    assert "1965-01-01" in answer
    assert "null" not in answer


@pytest.mark.asyncio
async def test_ambiguous_multi_record_scalar_derivation_is_insufficient() -> None:
    parents = {
        "person:parent_one": {
            "id": "person:parent_one",
            "name": ["Parent One"],
            "dob": "1960-01-01",
        },
        "person:parent_two": {
            "id": "person:parent_two",
            "name": ["Parent Two"],
            "dob": "1965-01-01",
        },
    }
    dispatcher = Dispatcher(
        parents,
        {
            ("person:test", "parent_of"): [
                {
                    "relation": "parent_of",
                    "semantic_relation": "child_of",
                    "related_entity": parent,
                }
                for parent in parents.values()
            ]
        },
    )
    plan = GroundingPlan(
        requires_grounding=True,
        grounding_domain="household",
        goal="my parent's age",
        subject=GroundingSubject(
            anchor="authenticated_user",
            reference="me",
            expected_type="person",
        ),
        traversal=(TraversalStep(relation="parent_of", direction="in"),),
        fields=("dob",),
        transform=TransformSpec(
            operator="date_difference",
            field="dob",
            mode="completed_years",
            reference="household_today",
        ),
        required_evidence=(
            RequiredEvidence(relation="parent_of"),
            RequiredEvidence(field="dob"),
        ),
    )

    evidence = await GroundingExecutor(
        dispatcher,
        CATALOG,
        home_entity_id=None,
    ).execute(
        plan,
        caller_entity_id="person:test",
        household_now=datetime.fromisoformat("2026-08-31T12:00:00-07:00"),
    )

    assert evidence.status == "evidence_insufficient"


@pytest.mark.asyncio
async def test_malformed_planner_output_fails_closed() -> None:
    class Ollama:
        async def plan_grounding(self, *_: Any, **__: Any) -> dict[str, Any]:
            return {"requires_grounding": True}

    service = OpenWorldGroundingService(
        GroundingPlanner(Ollama(), CATALOG),
        GroundingExecutor(Dispatcher({}), CATALOG, home_entity_id=None),
    )

    answer = await service.try_answer(
        [{"role": "user", "content": "What is my income?"}],
        caller_entity_id="person:test",
        household_now=datetime.fromisoformat("2026-08-31T12:00:00-07:00"),
        language="en",
    )

    assert answer is not None
    assert answer.stop_reason == "tool_error"
    assert "could not determine" in answer.text


@pytest.mark.asyncio
async def test_planner_retries_once_after_invalid_structured_output() -> None:
    class Ollama:
        def __init__(self) -> None:
            self.calls = 0

        async def plan_grounding(self, *_: Any, **__: Any) -> dict[str, Any]:
            self.calls += 1
            if self.calls == 1:
                return {"requires_grounding": True}
            return _named_property_plan("occupation").model_dump(mode="json")

    ollama = Ollama()
    plan = await GroundingPlanner(ollama, CATALOG).plan(
        [{"role": "user", "content": "What does Test Person do for work?"}],
        household_now=datetime.fromisoformat("2026-08-31T12:00:00-07:00"),
    )

    assert ollama.calls == 2
    assert plan.fields == ("occupation",)


@pytest.mark.asyncio
async def test_planner_compiles_omitted_evidence_bookkeeping_without_retry() -> None:
    payload = _named_property_plan("occupation").model_dump(mode="json")
    payload["required_evidence"] = []

    class Ollama:
        def __init__(self) -> None:
            self.calls = 0

        async def plan_grounding(self, *_: Any, **__: Any) -> dict[str, Any]:
            self.calls += 1
            return payload

    ollama = Ollama()
    plan = await GroundingPlanner(ollama, CATALOG).plan(
        [{"role": "user", "content": "What does Test Person do for work?"}],
        household_now=datetime.fromisoformat("2026-08-31T12:00:00-07:00"),
    )

    assert ollama.calls == 1
    assert plan.required_evidence == (RequiredEvidence(field="occupation"),)


@pytest.mark.asyncio
async def test_grounding_executor_reports_timeout() -> None:
    class SlowDispatcher:
        async def dispatch(self, *_: Any, **__: Any) -> dict[str, Any]:
            await asyncio.sleep(0.05)
            return {"ok": True, "result": []}

    evidence = await GroundingExecutor(
        SlowDispatcher(),
        CATALOG,
        home_entity_id=None,
        timeout_seconds=0.001,
    ).execute(
        _named_property_plan("occupation"),
        caller_entity_id=None,
        household_now=datetime.fromisoformat("2026-08-31T12:00:00-07:00"),
    )

    assert evidence.status == "timeout"


@pytest.mark.asyncio
async def test_grounding_executor_reports_tool_failure() -> None:
    class FailingDispatcher:
        async def dispatch(self, tool_name: str, *_: Any, **__: Any) -> dict[str, Any]:
            return {
                "ok": False,
                "tool": tool_name,
                "error": {"code": "tool_execution_failed"},
            }

    evidence = await GroundingExecutor(
        FailingDispatcher(),
        CATALOG,
        home_entity_id=None,
    ).execute(
        _named_property_plan("occupation"),
        caller_entity_id=None,
        household_now=datetime.fromisoformat("2026-08-31T12:00:00-07:00"),
    )

    assert evidence.status == "tool_error"


def _chat_response(content: str) -> ChatResponse:
    return ChatResponse.model_validate(
        {
            "model": "qwen3:8b",
            "created_at": "2026-08-31T00:00:00Z",
            "done": True,
            "message": {"role": "assistant", "content": content},
        }
    )


class _NonGroundingOllama:
    def __init__(self, answer: str, *, domain: str = "none") -> None:
        self.answer = answer
        self.domain = domain
        self.chat_calls = 0

    async def plan_grounding(self, *_: Any, **__: Any) -> dict[str, Any]:
        return GroundingPlan(
            requires_grounding=False,
            grounding_domain=self.domain,
            goal="general response",
        ).model_dump(mode="json")

    async def chat_with_tools(self, *_: Any, **__: Any) -> ChatResponse:
        self.chat_calls += 1
        return _chat_response(self.answer)


def _agent_with_open_world(ollama: Any) -> AgentService:
    steward = get_agent("steward")
    return AgentService(
        ollama,  # type: ignore[arg-type]
        Dispatcher({}),
        system_prompt=steward.prompt,
        tools=steward.tool_definitions,
        home_entity_id=steward.settings["home_entity_id"],
        schema_catalog=CATALOG,
    )


@pytest.mark.asyncio
async def test_non_grounded_request_continues_to_normal_model_loop() -> None:
    ollama = _NonGroundingOllama("A normal temperature is context-dependent.")

    result = await _agent_with_open_world(ollama).answer(
        "What is a normal body temperature?"
    )

    assert result.answer == "A normal temperature is context-dependent."
    assert ollama.chat_calls == 1


@pytest.mark.asyncio
async def test_non_grounded_personal_advice_continues_normally() -> None:
    ollama = _NonGroundingOllama("Keep a consistent sleep schedule.")

    result = await _agent_with_open_world(ollama).answer(
        "How can I improve my sleep?"
    )

    assert result.answer == "Keep a consistent sleep schedule."
    assert ollama.chat_calls == 1


@pytest.mark.asyncio
async def test_planner_false_negative_cannot_bypass_grounding() -> None:
    unsupported = "Your shoe size is US 11."
    ollama = _NonGroundingOllama(unsupported)

    result = await _agent_with_open_world(ollama).answer("What is my shoe size?")

    assert result.answer != unsupported
    assert result.stop_reason == "tool_error"
    assert ollama.chat_calls == 0


@pytest.mark.asyncio
async def test_external_tool_label_cannot_bypass_household_grounding() -> None:
    unsupported = "Your shoe size is US 11."
    ollama = _NonGroundingOllama(unsupported, domain="external_tool")

    result = await _agent_with_open_world(ollama).answer("What is my shoe size?")

    assert result.answer != unsupported
    assert result.stop_reason == "tool_error"
    assert ollama.chat_calls == 0


@pytest.mark.asyncio
async def test_chinese_planner_false_negative_cannot_bypass_grounding() -> None:
    unsupported = "您的收入是100万元。"
    ollama = _NonGroundingOllama(unsupported)

    result = await _agent_with_open_world(ollama).answer("我的收入是多少？")

    assert result.answer != unsupported
    assert result.stop_reason == "tool_error"
    assert ollama.chat_calls == 0


@pytest.mark.asyncio
async def test_named_entity_planner_false_negative_cannot_bypass_grounding() -> None:
    unsupported = "Test Person's shoe size is US 11."
    ollama = _NonGroundingOllama(unsupported)

    result = await _agent_with_open_world(ollama).answer(
        "What is Test Person's shoe size?"
    )

    assert result.answer != unsupported
    assert result.stop_reason == "tool_error"
    assert ollama.chat_calls == 0


@pytest.mark.asyncio
async def test_non_household_named_entity_question_continues_normally() -> None:
    ollama = _NonGroundingOllama("Beijing is in northern China.")

    result = await _agent_with_open_world(ollama).answer("Where is Beijing?")

    assert result.answer == "Beijing is in northern China."
    assert ollama.chat_calls == 1
