from __future__ import annotations

from datetime import datetime

import pytest
from home_cortex.operator_registry import (
    OPERATORS,
    TRANSFORM_OPERATORS,
    OperatorExecutionError,
    OperatorInput,
    execute_operator,
)


def test_registry_is_explicit_generic_and_bounded() -> None:
    assert {
        "select",
        "traverse",
        "resolve_reference",
        "same_entity",
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
    assert OPERATORS["same_entity"].output_kind == "boolean"
    assert OPERATORS["same_entity"].input_shape == "plan"
    assert OPERATORS["completed_years"].output_kind == "integer"
    assert OPERATORS["annual_occurrence"].field_kinds == {"date", "datetime"}


@pytest.mark.parametrize('stored,now,unit,expected', [
    ('2014-05-04','2026-05-03','years',11),
    ('2014-05-04','2026-05-04','years',12),
    ('2014-05-04','2026-09-07','years',12),
    ('2014-05-04','2026-09-07','months',148),
    ('2014-05-04','2026-09-07','days',4509),
    ('2014-05-04','2026-09-07','seconds',4509*86400),
    ('2020-02-29','2021-02-28','years',0),
    ('2020-02-29','2021-03-01','years',1),
    ('2026-01-31','2026-02-28','months',0),
    ('2026-01-31','2026-03-01','months',1),
    ('2027-05-04','2026-05-04','years',-1),
    ('2027-05-04','2026-05-05','years',0),
    ('2014-05-04T13:00:00-07:00','2026-05-04','years',11),
])
def test_date_interval_uses_calendar_units_and_explicit_direction(stored,now,unit,expected):
    result=execute_operator('date_difference',OperatorInput(
        records=[{'start':stored}],field='start',mode=unit,
        now=datetime.fromisoformat(now+'T12:00:00-07:00'),
    ))
    assert result==expected


@pytest.mark.parametrize('unit', [None,'weeks','invalid'])
def test_interval_rejects_unspecified_or_unknown_units(unit):
    with pytest.raises(OperatorExecutionError):
        execute_operator('date_difference',OperatorInput(records=[{'start':'2000-01-01'}],field='start',mode=unit,now=datetime.fromisoformat('2026-09-07T12:00:00-07:00')))


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


def test_new_numeric_field_immediately_supports_existing_operator() -> None:
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
