from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from home_cortex.grounding import (
    GroundingExecutor,
    GroundingPlan,
    GroundingPlanner,
    GroundingSubject,
    RequiredEvidence,
    TransformSpec,
    validate_plan_operators,
)
from home_cortex.operator_registry import (
    OPERATORS,
    TRANSFORM_OPERATORS,
    OperatorExecutionError,
    OperatorInput,
    OperatorValidationError,
    execute_operator,
)
from home_cortex.schema_catalog import EntityTypeSchema, RuntimeSchemaCatalog


TYPED_CATALOG = RuntimeSchemaCatalog(
    {
        "person": EntityTypeSchema(
            "person",
            ("id", "name", "dob", "income", "spouse", "new_numeric_field"),
            {
                "id": "string",
                "name": "collection",
                "dob": "date",
                "income": "number",
                "spouse": "string",
                "new_numeric_field": "number",
            },
        )
    },
    {},
)


def _plan(transform: TransformSpec) -> GroundingPlan:
    fields = tuple(
        field
        for field in (transform.field, transform.other_field, transform.order_by)
        if field is not None
    )
    return GroundingPlan(
        requires_grounding=True,
        grounding_domain="household",
        goal="generic computation",
        subject=GroundingSubject(
            reference_type="speaker",
            expected_type="person",
        ),
        fields=fields,
        transform=transform,
        required_evidence=tuple(
            RequiredEvidence(field=field) for field in fields
        ),
        **(
            {
                "sort": (
                    {
                        "field": transform.order_by,
                        "direction": (
                            "desc" if transform.operator == "latest" else "asc"
                        ),
                    },
                )
            }
            if transform.order_by is not None
            else {}
        ),
    )


def test_registry_is_explicit_generic_and_bounded() -> None:
    assert {
        "select",
        "traverse",
        "resolve_reference",
        "eq",
        "date_range",
        "count",
        "latest",
        "sum",
        "average",
        "argmin",
        "argmax",
        "sort",
        "subtract",
        "divide",
        "date_difference",
        "completed_years",
        "duration",
        "annual_occurrence",
        "unit_conversion",
    }.issubset(OPERATORS)
    assert {
        "get_age",
        "get_income",
        "oldest_member",
        "monthly_spending",
    }.isdisjoint(OPERATORS)
    assert TRANSFORM_OPERATORS == {
        name
        for name, definition in OPERATORS.items()
        if definition.implementation is not None
    }


def test_operator_contracts_are_machine_readable() -> None:
    assert OPERATORS["count"].input_shape == "collection"
    assert OPERATORS["count"].output_kind == "integer"
    assert OPERATORS["average"].field_kinds == {"integer", "number"}
    assert OPERATORS["argmin"].output_kind == "record"
    assert OPERATORS["completed_years"].output_kind == "integer"
    assert OPERATORS["annual_occurrence"].field_kinds == {"date", "datetime"}


@pytest.mark.parametrize(
    ("stored", "now", "mode", "expected"),
    (
        ("2016-10-30", "2026-10-30T00:01:00-07:00", "days", 0),
        ("2016-10-30", "2026-10-31T00:01:00-07:00", None, "2027-10-30"),
        ("2000-01-01", "2026-12-31T23:30:00-08:00", "days", 1),
        ("2000-02-29", "2026-03-01T12:00:00-08:00", None, "2028-02-29"),
    ),
)
def test_annual_occurrence_uses_household_local_calendar_and_leap_day_policy(
    stored: str,
    now: str,
    mode: str | None,
    expected: str | int,
) -> None:
    result = execute_operator(
        "annual_occurrence",
        OperatorInput(
            records=({"birth_date": stored},),
            field="birth_date",
            mode=mode,
            reference="household_now",
            now=datetime.fromisoformat(now),
        ),
    )

    assert result == expected


def test_annual_occurrence_rejects_invalid_temporal_values() -> None:
    with pytest.raises(OperatorExecutionError):
        execute_operator(
            "annual_occurrence",
            OperatorInput(
                records=({"birth_date": "not-a-date"},),
                field="birth_date",
                reference="household_now",
                now=datetime.fromisoformat("2026-09-03T12:00:00-07:00"),
            ),
        )


@pytest.mark.parametrize(
    "transform",
    (
        TransformSpec(operator="average", field="name"),
        TransformSpec(operator="sum", field="dob"),
        TransformSpec(
            operator="completed_years",
            field="income",
            reference="household_today",
        ),
        TransformSpec(operator="argmax", field="spouse"),
    ),
)
def test_type_invalid_computations_fail_plan_validation(
    transform: TransformSpec,
) -> None:
    with pytest.raises(OperatorValidationError):
        validate_plan_operators(_plan(transform), TYPED_CATALOG)


def test_new_numeric_field_immediately_supports_existing_operator() -> None:
    plan = _plan(TransformSpec(operator="average", field="new_numeric_field"))

    validate_plan_operators(plan, TYPED_CATALOG)
    result = execute_operator(
        "average",
        OperatorInput(
            records=(
                {"new_numeric_field": 10},
                {"new_numeric_field": 14},
            ),
            field="new_numeric_field",
            now=datetime.fromisoformat("2026-09-01T12:00:00-07:00"),
        ),
    )

    assert result == 12
    assert "new_numeric_field" not in OPERATORS


def test_argmin_is_generic_over_new_type_compatible_fields() -> None:
    result = execute_operator(
        "argmin",
        OperatorInput(
            records=(
                {"id": "person:a", "new_numeric_field": 10},
                {"id": "person:b", "new_numeric_field": 4},
            ),
            field="new_numeric_field",
        ),
    )

    assert result == {"id": "person:b", "new_numeric_field": 4}


def test_non_allowlisted_model_operation_is_rejected_by_ir_schema() -> None:
    with pytest.raises(ValidationError):
        TransformSpec.model_validate(
            {
                "operator": "GET_AGE",
                "field": "dob",
                "reference": "household_today",
            }
        )


@pytest.mark.asyncio
async def test_planner_repairs_type_invalid_model_plan() -> None:
    invalid = _plan(TransformSpec(operator="average", field="name"))
    valid = _plan(TransformSpec(operator="average", field="new_numeric_field"))

    class Ollama:
        def __init__(self) -> None:
            self.calls: list[object] = []

        async def plan_grounding(self, messages, *_args, **_kwargs):
            self.calls.append(messages)
            selected = invalid if len(self.calls) == 1 else valid
            return selected.model_dump(mode="json")

    ollama = Ollama()
    plan = await GroundingPlanner(ollama, TYPED_CATALOG).plan(
        [{"role": "user", "content": "average the new numeric field"}],
        household_now=datetime.fromisoformat("2026-09-01T12:00:00-07:00"),
    )

    assert plan.transform == valid.transform
    assert len(ollama.calls) == 2
    repair_messages = ollama.calls[1]
    assert isinstance(repair_messages, list)
    assert "operator contract validation" in repair_messages[-1]["content"]


@pytest.mark.asyncio
async def test_executor_rejects_invalid_operator_contract_before_tools() -> None:
    class Dispatcher:
        def __init__(self) -> None:
            self.calls = 0

        async def dispatch(self, *_args, **_kwargs):
            self.calls += 1
            raise AssertionError("invalid plans must not execute tools")

    dispatcher = Dispatcher()
    evidence = await GroundingExecutor(
        dispatcher,
        TYPED_CATALOG,
        home_entity_id=None,
    ).execute(
        _plan(TransformSpec(operator="average", field="name")),
        caller_entity_id="person:test",
        household_now=datetime.fromisoformat("2026-09-01T12:00:00-07:00"),
    )

    assert evidence.status == "evidence_insufficient"
    assert dispatcher.calls == 0
