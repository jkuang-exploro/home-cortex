"""Bounded generic computation protocol for household query plans."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from types import MappingProxyType
from typing import Any, Literal

OperatorFamily = Literal[
    "retrieval",
    "predicate",
    "collection",
    "aggregation",
    "ordering",
    "transform",
]
ValueKind = Literal[
    "unknown",
    "any",
    "boolean",
    "integer",
    "number",
    "string",
    "date",
    "datetime",
    "object",
    "collection",
    "record",
]

NUMERIC_KINDS = frozenset({"integer", "number"})
TEMPORAL_KINDS = frozenset({"date", "datetime"})
ORDERED_KINDS = frozenset({*NUMERIC_KINDS, *TEMPORAL_KINDS, "string"})
EXTREME_KINDS = frozenset({*NUMERIC_KINDS, *TEMPORAL_KINDS})


class OperatorValidationError(ValueError):
    """A proposed operation violates its deterministic type contract."""


class OperatorExecutionError(RuntimeError):
    """Runtime values do not satisfy a validated operator contract."""


@dataclass(frozen=True)
class OperatorInput:
    records: Sequence[Mapping[str, Any]]
    field: str | None = None
    other_field: str | None = None
    order_by: str | None = None
    mode: str | None = None
    reference: str | None = None
    from_unit: str | None = None
    to_unit: str | None = None
    now: datetime | None = None


OperatorImplementation = Callable[[OperatorInput], Any]


@dataclass(frozen=True)
class OperatorDefinition:
    name: str
    family: OperatorFamily
    input_shape: Literal["plan", "scalar", "collection"]
    output_kind: ValueKind
    field_requirement: Literal["none", "optional", "required"] = "none"
    field_kinds: frozenset[ValueKind] = frozenset({"any"})
    other_field_required: bool = False
    other_field_kinds: frozenset[ValueKind] = frozenset({"any"})
    order_by_required: bool = False
    order_by_kinds: frozenset[ValueKind] = frozenset({"any"})
    required_parameters: frozenset[str] = frozenset()
    implementation: OperatorImplementation | None = None

    def validate(
        self,
        *,
        field: str | None = None,
        field_kind: ValueKind = "unknown",
        other_field: str | None = None,
        other_field_kind: ValueKind = "unknown",
        order_by: str | None = None,
        order_by_kind: ValueKind = "unknown",
        parameters: Mapping[str, Any] | None = None,
    ) -> None:
        parameters = parameters or {}
        if self.field_requirement == "required" and field is None:
            raise OperatorValidationError(f"{self.name} requires field")
        if self.field_requirement == "none" and field is not None:
            raise OperatorValidationError(f"{self.name} does not accept field")
        _validate_kind(self.name, "field", field_kind, self.field_kinds)
        if self.other_field_required and other_field is None:
            raise OperatorValidationError(f"{self.name} requires other_field")
        if not self.other_field_required and other_field is not None:
            raise OperatorValidationError(
                f"{self.name} does not accept other_field"
            )
        _validate_kind(
            self.name,
            "other_field",
            other_field_kind,
            self.other_field_kinds,
        )
        if self.order_by_required and order_by is None:
            raise OperatorValidationError(f"{self.name} requires order_by")
        if not self.order_by_required and order_by is not None:
            raise OperatorValidationError(f"{self.name} does not accept order_by")
        _validate_kind(
            self.name,
            "order_by",
            order_by_kind,
            self.order_by_kinds,
        )
        missing = sorted(
            parameter
            for parameter in self.required_parameters
            if parameters.get(parameter) is None
        )
        if missing:
            raise OperatorValidationError(
                f"{self.name} requires {', '.join(missing)}"
            )

    def execute(self, values: OperatorInput) -> Any:
        if self.implementation is None:
            raise OperatorExecutionError(f"{self.name} is not a transform")
        return self.implementation(values)


def _validate_kind(
    operator: str,
    argument: str,
    actual: ValueKind,
    accepted: frozenset[ValueKind],
) -> None:
    if actual in {"unknown", "any"} or "any" in accepted:
        return
    compatible = actual in accepted or (
        actual == "integer" and "number" in accepted
    )
    if not compatible:
        expected = "|".join(sorted(accepted))
        raise OperatorValidationError(
            f"{operator} requires {argument}:{expected}, got {actual}"
        )


def infer_value_kind(value: Any) -> ValueKind:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number" if math.isfinite(value) else "unknown"
    if isinstance(value, str):
        try:
            date.fromisoformat(value)
            return "date"
        except ValueError:
            pass
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
            return "datetime"
        except ValueError:
            return "string"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return "collection"
    return "unknown"


def infer_field_kind(values: Sequence[Any]) -> ValueKind:
    kinds = {infer_value_kind(value) for value in values if value is not None}
    if not kinds:
        return "unknown"
    if kinds <= NUMERIC_KINDS:
        return "number" if "number" in kinds else "integer"
    return next(iter(kinds)) if len(kinds) == 1 else "unknown"


def execute_operator(name: str, values: OperatorInput) -> Any:
    try:
        definition = OPERATORS[name]
    except KeyError as error:
        raise OperatorValidationError(f"operator is not allowlisted: {name}") from error
    return definition.execute(values)


def operator_prompt_payload() -> dict[str, Any]:
    """Expose contracts without implementation details to the semantic planner."""
    return {
        name: {
            "family": definition.family,
            "input": definition.input_shape,
            "output": definition.output_kind,
            "field": definition.field_requirement,
            "field_types": sorted(definition.field_kinds),
            "other_field": definition.other_field_required,
            "other_field_types": sorted(definition.other_field_kinds),
            "order_by": definition.order_by_required,
            "order_by_types": sorted(definition.order_by_kinds),
            "required_parameters": sorted(definition.required_parameters),
        }
        for name, definition in OPERATORS.items()
    }


def evaluate_predicate(name: str, left: Any, right: Any) -> bool:
    if name not in PREDICATE_OPERATORS:
        raise OperatorValidationError(f"predicate is not allowlisted: {name}")
    try:
        if name == "eq":
            return left == right
        if name == "ne":
            return left != right
        if name == "lt":
            return left < right
        if name == "lte":
            return left <= right
        if name == "gt":
            return left > right
        if name == "gte":
            return left >= right
        if name == "in":
            return left in right
        if name == "exists":
            return (left is not None) is bool(right if right is not None else True)
        if name == "date_range":
            if not isinstance(right, Sequence) or isinstance(right, str):
                return False
            return len(right) == 2 and right[0] <= left < right[1]
    except (TypeError, ValueError):
        return False
    return False


def _source_values(values: OperatorInput) -> list[Any]:
    if values.field is None:
        return []
    return [
        record.get(values.field)
        for record in values.records
        if record.get(values.field) is not None
    ]


def _count(values: OperatorInput) -> int:
    return len(values.records)


def _sum(values: OperatorInput) -> float | int:
    numeric = _numeric_values(_source_values(values))
    if not numeric or len(numeric) != len(values.records):
        raise OperatorExecutionError("sum requires collection<number>")
    return sum(numeric)


def _average(values: OperatorInput) -> float:
    numeric = _numeric_values(_source_values(values))
    if not numeric or len(numeric) != len(values.records):
        raise OperatorExecutionError("average requires collection<number>")
    return sum(numeric) / len(numeric)


def _minimum(values: OperatorInput) -> float | int:
    numeric = _numeric_values(_source_values(values))
    if not numeric or len(numeric) != len(values.records):
        raise OperatorExecutionError("min requires collection<number>")
    return min(numeric)


def _maximum(values: OperatorInput) -> float | int:
    numeric = _numeric_values(_source_values(values))
    if not numeric or len(numeric) != len(values.records):
        raise OperatorExecutionError("max requires collection<number>")
    return max(numeric)


def _first(values: OperatorInput) -> Any:
    if not values.records:
        raise OperatorExecutionError("first requires a non-empty collection")
    return _project(values.records[0], values.field)


def _last(values: OperatorInput) -> Any:
    if not values.records:
        raise OperatorExecutionError("last requires a non-empty collection")
    return _project(values.records[-1], values.field)


def _argmin(values: OperatorInput) -> dict[str, Any]:
    return _arg_extreme(values, minimum=True)


def _argmax(values: OperatorInput) -> dict[str, Any]:
    return _arg_extreme(values, minimum=False)


def _arg_extreme(values: OperatorInput, *, minimum: bool) -> dict[str, Any]:
    if values.field is None or not values.records:
        raise OperatorExecutionError("argmin/argmax require field and records")
    candidates = [
        record for record in values.records if record.get(values.field) is not None
    ]
    if not candidates:
        raise OperatorExecutionError("argmin/argmax field is unavailable")
    function = min if minimum else max
    try:
        selected = function(candidates, key=lambda item: item[values.field])
    except TypeError as error:
        raise OperatorExecutionError("argmin/argmax field is not ordered") from error
    return dict(selected)


def _subtract(values: OperatorInput) -> float | int:
    left, right = _binary_numbers(values)
    return left - right


def _divide(values: OperatorInput) -> float:
    left, right = _binary_numbers(values)
    if right == 0:
        raise OperatorExecutionError("divide denominator cannot be zero")
    return left / right


def _binary_numbers(values: OperatorInput) -> tuple[float | int, float | int]:
    if not values.records or values.field is None or values.other_field is None:
        raise OperatorExecutionError("binary numeric operator requires one record")
    left = values.records[0].get(values.field)
    right = values.records[0].get(values.other_field)
    if not _is_number(left) or not _is_number(right):
        raise OperatorExecutionError("binary operator requires numeric fields")
    return left, right


def _date_difference(values: OperatorInput) -> int | float:
    value = _only_value(values)
    now = _required_now(values)
    parsed_date, parsed_datetime = _temporal_value(value)
    if parsed_date is not None:
        delta = now.date() - parsed_date
        return delta.days if values.mode == "days" else delta.total_seconds()
    if parsed_datetime is None:
        raise OperatorExecutionError("date_difference requires date|datetime")
    delta = now.astimezone(timezone.utc) - parsed_datetime
    return delta.days if values.mode == "days" else delta.total_seconds()


def _completed_years(values: OperatorInput) -> int:
    value = _only_value(values)
    now = _required_now(values)
    parsed_date, parsed_datetime = _temporal_value(value)
    start = parsed_date or (parsed_datetime.date() if parsed_datetime else None)
    if start is None or start > now.date():
        raise OperatorExecutionError("completed_years requires a past date")
    passed = (now.month, now.day) >= (start.month, start.day)
    return now.year - start.year - (not passed)


def _duration(values: OperatorInput) -> int | float:
    return _date_difference(values)


def _annual_occurrence(values: OperatorInput) -> str | int:
    value = _only_value(values)
    now = _required_now(values)
    stored, parsed_datetime = _temporal_value(value)
    stored = stored or (parsed_datetime.date() if parsed_datetime else None)
    if stored is None:
        raise OperatorExecutionError("annual_occurrence requires date|datetime")
    today = now.date()
    for year in range(today.year, today.year + 9):
        try:
            occurrence = date(year, stored.month, stored.day)
        except ValueError:
            continue
        if occurrence >= today:
            return (
                (occurrence - today).days
                if values.mode == "days"
                else occurrence.isoformat()
            )
    raise OperatorExecutionError("annual occurrence is not representable")


def _unit_conversion(values: OperatorInput) -> float:
    value = _only_value(values)
    if not _is_number(value):
        raise OperatorExecutionError("unit_conversion requires number")
    source = (values.from_unit or "").casefold()
    target = (values.to_unit or "").casefold()
    conversions: dict[tuple[str, str], Callable[[float | int], float]] = {
        ("c", "f"): lambda number: number * 9 / 5 + 32,
        ("f", "c"): lambda number: (number - 32) * 5 / 9,
        ("kg", "lb"): lambda number: number * 2.2046226218,
        ("lb", "kg"): lambda number: number / 2.2046226218,
        ("cm", "in"): lambda number: number / 2.54,
        ("in", "cm"): lambda number: number * 2.54,
    }
    if source == target and source:
        return float(value)
    conversion = conversions.get((source, target))
    if conversion is None:
        raise OperatorExecutionError("unsupported unit conversion")
    return conversion(value)


def _project(record: Mapping[str, Any], field: str | None) -> Any:
    if field is None:
        return dict(record)
    if record.get(field) is None:
        raise OperatorExecutionError("projected field is unavailable")
    return record[field]


def _only_value(values: OperatorInput) -> Any:
    source = _source_values(values)
    if len(values.records) != 1 or len(source) != 1:
        raise OperatorExecutionError("operator requires exactly one value")
    return source[0]


def _required_now(values: OperatorInput) -> datetime:
    if values.now is None:
        raise OperatorExecutionError("operator requires current time")
    return values.now


def _temporal_value(value: Any) -> tuple[date | None, datetime | None]:
    if not isinstance(value, str):
        return None, None
    try:
        return date.fromisoformat(value), None
    except ValueError:
        pass
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None, None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return None, parsed.astimezone(timezone.utc)


def _numeric_values(values: Sequence[Any]) -> list[float | int]:
    return [value for value in values if _is_number(value)]


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _definition(
    name: str,
    family: OperatorFamily,
    input_shape: Literal["plan", "scalar", "collection"],
    output_kind: ValueKind,
    **kwargs: Any,
) -> OperatorDefinition:
    return OperatorDefinition(name, family, input_shape, output_kind, **kwargs)


_DEFINITIONS = (
    _definition("select", "retrieval", "plan", "collection"),
    _definition("traverse", "retrieval", "plan", "collection"),
    _definition("resolve_reference", "retrieval", "plan", "record"),
    _definition("filter", "collection", "collection", "collection"),
    *(
        _definition(
            name,
            "predicate",
            "scalar",
            "boolean",
            field_requirement="required",
            field_kinds=field_kinds,
        )
        for name, field_kinds in (
            ("eq", frozenset({"any"})),
            ("ne", frozenset({"any"})),
            ("gt", ORDERED_KINDS),
            ("gte", ORDERED_KINDS),
            ("lt", ORDERED_KINDS),
            ("lte", ORDERED_KINDS),
            ("in", frozenset({"any"})),
            ("exists", frozenset({"any"})),
        )
    ),
    _definition(
        "date_range",
        "predicate",
        "scalar",
        "boolean",
        field_requirement="required",
        field_kinds=TEMPORAL_KINDS,
    ),
    _definition(
        "count",
        "collection",
        "collection",
        "integer",
        implementation=_count,
    ),
    _definition(
        "first",
        "collection",
        "collection",
        "any",
        field_requirement="optional",
        implementation=_first,
    ),
    _definition(
        "last",
        "collection",
        "collection",
        "any",
        field_requirement="optional",
        implementation=_last,
    ),
    _definition(
        "latest",
        "collection",
        "collection",
        "any",
        field_requirement="required",
        order_by_required=True,
        order_by_kinds=TEMPORAL_KINDS,
        implementation=_first,
    ),
    _definition(
        "earliest",
        "collection",
        "collection",
        "any",
        field_requirement="required",
        order_by_required=True,
        order_by_kinds=TEMPORAL_KINDS,
        implementation=_first,
    ),
    *(
        _definition(
            name,
            "aggregation",
            "collection",
            "number",
            field_requirement="required",
            field_kinds=NUMERIC_KINDS,
            implementation=implementation,
        )
        for name, implementation in (
            ("sum", _sum),
            ("average", _average),
            ("min", _minimum),
            ("max", _maximum),
        )
    ),
    *(
        _definition(
            name,
            "aggregation",
            "collection",
            "record",
            field_requirement="required",
            field_kinds=EXTREME_KINDS,
            implementation=implementation,
        )
        for name, implementation in (("argmin", _argmin), ("argmax", _argmax))
    ),
    _definition(
        "sort",
        "ordering",
        "collection",
        "collection",
        field_requirement="required",
        field_kinds=ORDERED_KINDS,
    ),
    *(
        _definition(
            name,
            "transform",
            "scalar",
            "number",
            field_requirement="required",
            field_kinds=NUMERIC_KINDS,
            other_field_required=True,
            other_field_kinds=NUMERIC_KINDS,
            implementation=implementation,
        )
        for name, implementation in (("subtract", _subtract), ("divide", _divide))
    ),
    _definition(
        "date_difference",
        "transform",
        "scalar",
        "number",
        field_requirement="required",
        field_kinds=TEMPORAL_KINDS,
        required_parameters=frozenset({"mode", "reference"}),
        implementation=_date_difference,
    ),
    _definition(
        "completed_years",
        "transform",
        "scalar",
        "integer",
        field_requirement="required",
        field_kinds=TEMPORAL_KINDS,
        required_parameters=frozenset({"reference"}),
        implementation=_completed_years,
    ),
    _definition(
        "duration",
        "transform",
        "scalar",
        "number",
        field_requirement="required",
        field_kinds=TEMPORAL_KINDS,
        required_parameters=frozenset({"mode", "reference"}),
        implementation=_duration,
    ),
    _definition(
        "annual_occurrence",
        "transform",
        "scalar",
        "any",
        field_requirement="required",
        field_kinds=TEMPORAL_KINDS,
        required_parameters=frozenset({"reference"}),
        implementation=_annual_occurrence,
    ),
    _definition(
        "unit_conversion",
        "transform",
        "scalar",
        "number",
        field_requirement="required",
        field_kinds=NUMERIC_KINDS,
        required_parameters=frozenset({"from_unit", "to_unit"}),
        implementation=_unit_conversion,
    ),
)

OPERATORS: Mapping[str, OperatorDefinition] = MappingProxyType(
    {definition.name: definition for definition in _DEFINITIONS}
)
TRANSFORM_OPERATORS = frozenset(
    name for name, definition in OPERATORS.items() if definition.implementation
)
PREDICATE_OPERATORS = frozenset(
    name for name, definition in OPERATORS.items() if definition.family == "predicate"
)
