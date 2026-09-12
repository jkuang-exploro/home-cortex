from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from home_cortex.operator_registry import (
    OPERATORS,
    OperatorExecutionError,
    OperatorInput,
    OperatorValidationError,
    evaluate_predicate,
    execute_operator,
)


def test_registry_is_explicit_generic_and_bounded() -> None:
    assert {
        "select",
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
        "date_difference",
        "annual_occurrence",
        "unit_conversion",
    }.issubset(OPERATORS)
    assert {
        "get_age",
        "get_income",
        "oldest_member",
        "monthly_spending",
    }.isdisjoint(OPERATORS)


def test_operator_contracts_are_machine_readable() -> None:
    assert OPERATORS["count"].input_shape == "collection"
    assert OPERATORS["average"].field_kinds == {"integer", "number"}
    assert OPERATORS["same_entity"].input_shape == "plan"
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


@pytest.mark.parametrize(
    ("operator", "options"),
    (
        ("sum", {}),
        ("count", {"field": "value"}),
        ("latest", {"field": "when", "field_kind": "date"}),
        ("count", {"order_by": "when", "order_by_kind": "date"}),
        ("sum", {"field": "value", "field_kind": "string"}),
        ("date_difference", {"field": "when", "field_kind": "date"}),
        ("date_add", {"field": "when", "field_kind": "date", "parameters": {"amount": True, "mode": "days"}}),
        ("annual_occurrence", {"field": "when", "field_kind": "date", "parameters": {"reference": "household_now", "mode": "years"}}),
    ),
)
def test_operator_contracts_reject_invalid_shapes(operator, options) -> None:
    with pytest.raises(OperatorValidationError):
        OPERATORS[operator].validate(**options)


@pytest.mark.parametrize("operator", ["sum", "average", "min", "max"])
def test_numeric_reductions_reject_partial_or_nonnumeric_collections(operator) -> None:
    with pytest.raises(OperatorExecutionError):
        execute_operator(
            operator,
            OperatorInput(records=({"value": 1}, {"value": None}), field="value"),
        )
    with pytest.raises(OperatorExecutionError):
        execute_operator(
            operator,
            OperatorInput(records=({"value": "one"},), field="value"),
        )


def test_selection_and_extrema_fail_closed_on_incomplete_inputs() -> None:
    for operator in ("first", "last"):
        with pytest.raises(OperatorExecutionError):
            execute_operator(operator, OperatorInput(records=()))
    with pytest.raises(OperatorExecutionError):
        execute_operator("argmin", OperatorInput(records=({"value": 1},)))
    with pytest.raises(OperatorExecutionError):
        execute_operator(
            "argmax", OperatorInput(records=({"other": 1},), field="value")
        )
    with pytest.raises(OperatorExecutionError):
        execute_operator(
            "argmin",
            OperatorInput(
                records=({"value": 1}, {"value": "two"}), field="value"
            ),
        )


@pytest.mark.parametrize(
    ("name", "left", "right", "expected"),
    (
        ("eq", 2, 2, True),
        ("ne", 2, 3, True),
        ("lt", 2, 3, True),
        ("lte", 2, 2, True),
        ("gt", 3, 2, True),
        ("gte", 3, 3, True),
        ("in", "a", ("a", "b"), True),
        ("exists", None, False, True),
        ("date_range", "2026-06-01", ("2026-01-01", "2027-01-01"), True),
        ("date_range", "2026-06-01", "invalid", False),
        ("lt", None, 3, False),
    ),
)
def test_predicate_execution_is_bounded(name, left, right, expected) -> None:
    assert evaluate_predicate(name, left, right) is expected


def test_unknown_operator_and_predicate_are_rejected() -> None:
    with pytest.raises(OperatorValidationError):
        execute_operator("invented", OperatorInput(records=()))
    with pytest.raises(OperatorValidationError):
        evaluate_predicate("invented", 1, 1)


def test_unit_conversion_accepts_supported_pairs_and_rejects_bad_inputs() -> None:
    assert execute_operator(
        "unit_conversion",
        OperatorInput(records=({"value": 10},), field="value", from_unit="c", to_unit="f"),
    ) == 50
    assert execute_operator(
        "unit_conversion",
        OperatorInput(records=({"value": 10},), field="value", from_unit="kg", to_unit="kg"),
    ) == 10.0
    for value, source, target in (("ten", "kg", "lb"), (10, "kg", "m")):
        with pytest.raises(OperatorExecutionError):
            execute_operator(
                "unit_conversion",
                OperatorInput(
                    records=({"value": value},),
                    field="value",
                    from_unit=source,
                    to_unit=target,
                ),
            )


def test_date_add_rejects_ambiguous_household_wall_time() -> None:
    with pytest.raises(OperatorExecutionError, match="ambiguous"):
        execute_operator(
            "date_add",
            OperatorInput(
                records=({"value": "2026-10-01T01:30:00-07:00"},),
                field="value",
                amount=1,
                mode="months",
                now=datetime(2026, 9, 1, tzinfo=ZoneInfo("America/Los_Angeles")),
            ),
        )
