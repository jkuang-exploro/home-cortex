"""Bounded generic computation protocol for household query plans."""

from __future__ import annotations

import math
from calendar import monthrange
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from types import MappingProxyType
from typing import Any, Literal, get_args

FactOperation = Literal[
    "inspect",
    "resolve_reference",
    "same_entity",
    "select",
    "count",
    "first",
    "last",
    "latest",
    "earliest",
    "sum",
    "average",
    "min",
    "max",
    "argmin",
    "argmax",
    "date_add",
    "date_difference",
    "annual_occurrence",
    "unit_conversion",
]

OperatorFamily = Literal[
    "retrieval",
    "predicate",
    "collection",
    "aggregation",
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
    order_by: str | None = None
    mode: str | None = None
    reference: str | None = None
    from_unit: str | None = None
    to_unit: str | None = None
    now: datetime | None = None
    amount: int | None = None


OperatorImplementation = Callable[[OperatorInput], Any]


@dataclass(frozen=True)
class OperatorDefinition:
    name: str
    family: OperatorFamily
    input_shape: Literal["plan", "scalar", "collection"]
    field_requirement: Literal["none", "optional", "required"] = "none"
    field_kinds: frozenset[ValueKind] = frozenset({"any"})
    order_by_required: bool = False
    order_by_kinds: frozenset[ValueKind] = frozenset({"any"})
    required_parameters: frozenset[str] = frozenset()
    implementation: OperatorImplementation | None = None

    def validate(
        self,
        *,
        field: str | None = None,
        field_kind: ValueKind = "unknown",
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
        if self.name == "date_difference" and parameters.get("mode") not in {"years", "months", "days", "seconds"}:
            raise OperatorValidationError("date interval requires a supported unit")
        if self.name == "date_add" and (
            type(parameters.get("amount")) is not int
            or abs(parameters["amount"]) > 120000
            or parameters.get("mode") not in {"years", "months", "days"}
        ):
            raise OperatorValidationError("date_add requires bounded integer amount and calendar unit")
        if self.name == "annual_occurrence" and parameters.get("mode") not in {None, "days"}:
            raise OperatorValidationError("annual occurrence only supports a date or days")

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


def _date_difference(values: OperatorInput) -> int | float:
    value = _only_value(values)
    now = _required_now(values)
    parsed_date, parsed_datetime = _temporal_value(value)
    if values.mode not in {"years", "months", "days", "seconds"}:
        raise OperatorExecutionError("date interval requires a supported unit")
    if values.mode in {"years", "months"}:
        if parsed_date is not None:
            start, end = parsed_date, now.date()
        elif parsed_datetime is not None:
            start, end = parsed_datetime.astimezone(now.tzinfo), now
        else:
            raise OperatorExecutionError("date_difference requires date|datetime")
        sign = 1 if end >= start else -1
        if sign < 0:
            start, end = end, start
        # Calendar anniversaries, truncated toward zero. A Feb 29 anniversary
        # is not complete on Feb 28; a month starting on the 31st is not
        # complete in a shorter month. No fixed day/year or day/month ratio.
        if values.mode == "years":
            count = end.year - start.year
            start_tail = (start.month, start.day)
            end_tail = (end.month, end.day)
        else:
            count = (end.year - start.year) * 12 + end.month - start.month
            start_tail, end_tail = (start.day,), (end.day,)
        if isinstance(start, datetime):
            start_tail += (start.hour, start.minute, start.second, start.microsecond)
            end_tail += (end.hour, end.minute, end.second, end.microsecond)
        return sign * (count - (end_tail < start_tail))
    if parsed_date is not None:
        delta = now.date() - parsed_date
        return delta.days if values.mode == "days" else delta.total_seconds()
    if parsed_datetime is None:
        raise OperatorExecutionError("date_difference requires date|datetime")
    delta = now.astimezone(timezone.utc) - parsed_datetime
    return delta.days if values.mode == "days" else delta.total_seconds()


def _date_add(values: OperatorInput) -> str:
    amount = values.amount
    if type(amount) is not int or abs(amount) > 120000 or values.mode not in {"years", "months", "days"}:
        raise OperatorExecutionError("date_add requires bounded integer amount and calendar unit")
    stored, instant = _temporal_value(_only_value(values))
    now = _required_now(values)
    if now.tzinfo is None or now.utcoffset() is None:
        raise OperatorExecutionError("calendar offset requires a timezone-aware household clock")
    if stored is None and instant is None:
        raise OperatorExecutionError("date_add requires date|datetime")
    start = stored if stored is not None else instant.astimezone(now.tzinfo)
    try:
        if values.mode == "days":
            target = start + timedelta(days=amount)
        else:
            months = amount * 12 if values.mode == "years" else amount
            year, month = divmod(start.year * 12 + start.month - 1 + months, 12)
            last_day = monthrange(year, month + 1)[1]
            target = start.replace(year=year, month=month + 1, day=min(start.day, last_day))
            if start.day > last_day:
                # The requested month/day does not exist. Its anniversary becomes
                # complete on the next month's first day, as in date_difference.
                target += timedelta(days=1)
        if isinstance(target, datetime):
            # A wall time must identify exactly one instant in the household zone.
            if (target.replace(fold=0).utcoffset() != target.replace(fold=1).utcoffset()
                or target.astimezone(timezone.utc).astimezone(target.tzinfo).replace(tzinfo=None)
                != target.replace(tzinfo=None)):
                raise OperatorExecutionError("calendar offset lands on an ambiguous or nonexistent wall time")
        return target.isoformat()
    except (ValueError, OverflowError) as error:
        raise OperatorExecutionError("calendar offset is out of range") from error


def _annual_occurrence(values: OperatorInput) -> str | int:
    if values.mode not in {None, "days"}:
        raise OperatorExecutionError("annual occurrence only supports a date or days")
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
    **kwargs: Any,
) -> OperatorDefinition:
    return OperatorDefinition(name, family, input_shape, **kwargs)


_DEFINITIONS = (
    _definition("date_add", "transform", "scalar",
                field_requirement="required", field_kinds=TEMPORAL_KINDS,
                required_parameters=frozenset({"amount", "mode"}), implementation=_date_add),
    _definition("select", "retrieval", "plan"),
    _definition("resolve_reference", "retrieval", "plan"),
    _definition("inspect", "retrieval", "plan"),
    _definition("same_entity", "retrieval", "plan"),
    *(
        _definition(
            name,
            "predicate",
            "scalar",
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
        field_requirement="required",
        field_kinds=TEMPORAL_KINDS,
    ),
    _definition(
        "count",
        "collection",
        "collection",
        implementation=_count,
    ),
    _definition(
        "first",
        "collection",
        "collection",
        field_requirement="optional",
        implementation=_first,
    ),
    _definition(
        "last",
        "collection",
        "collection",
        field_requirement="optional",
        implementation=_last,
    ),
    _definition(
        "latest",
        "collection",
        "collection",
        field_requirement="required",
        order_by_required=True,
        order_by_kinds=TEMPORAL_KINDS,
        implementation=_first,
    ),
    _definition(
        "earliest",
        "collection",
        "collection",
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
            field_requirement="required",
            field_kinds=EXTREME_KINDS,
            implementation=implementation,
        )
        for name, implementation in (("argmin", _argmin), ("argmax", _argmax))
    ),
    _definition(
        "date_difference",
        "transform",
        "scalar",
        field_requirement="required",
        field_kinds=TEMPORAL_KINDS,
        required_parameters=frozenset({"mode", "reference"}),
        implementation=_date_difference,
    ),
    _definition(
        "annual_occurrence",
        "transform",
        "scalar",
        field_requirement="required",
        field_kinds=TEMPORAL_KINDS,
        required_parameters=frozenset({"reference"}),
        implementation=_annual_occurrence,
    ),
    _definition(
        "unit_conversion",
        "transform",
        "scalar",
        field_requirement="required",
        field_kinds=NUMERIC_KINDS,
        required_parameters=frozenset({"from_unit", "to_unit"}),
        implementation=_unit_conversion,
    ),
)

OPERATORS: Mapping[str, OperatorDefinition] = MappingProxyType(
    {definition.name: definition for definition in _DEFINITIONS}
)
PREDICATE_OPERATORS = frozenset(
    name for name, definition in OPERATORS.items() if definition.family == "predicate"
)
FACT_OPERATORS: Mapping[str, OperatorDefinition] = MappingProxyType(
    {
        name: definition
        for name, definition in OPERATORS.items()
        if definition.family != "predicate"
    }
)
if set(FACT_OPERATORS) != set(get_args(FactOperation)):
    raise RuntimeError("FactOperation and its registry definitions must match")
