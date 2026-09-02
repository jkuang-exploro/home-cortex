"""Open-world, schema-aware household evidence planning and execution."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .display import resolve_display_name
from .operator_registry import (
    OPERATORS,
    PREDICATE_OPERATORS,
    TRANSFORM_OPERATORS,
    OperatorExecutionError,
    OperatorInput,
    OperatorValidationError,
    ValueKind,
    evaluate_predicate,
    execute_operator,
    infer_value_kind,
)
from .schema_catalog import RuntimeSchemaCatalog, record_aliases
from .text import safe_log_token

GroundingStatus = Literal[
    "sufficient",
    "caller_context_missing",
    "entity_not_found",
    "field_not_available",
    "evidence_insufficient",
    "evidence_stale",
    "tool_error",
    "timeout",
]
GroundingOperator = Literal[
    "count",
    "first",
    "last",
    "sum",
    "average",
    "min",
    "max",
    "argmin",
    "argmax",
    "latest",
    "earliest",
    "subtract",
    "divide",
    "date_difference",
    "completed_years",
    "duration",
    "annual_occurrence",
    "unit_conversion",
]
PredicateOperator = Literal[
    "eq",
    "ne",
    "lt",
    "lte",
    "gt",
    "gte",
    "in",
    "exists",
    "date_range",
]
if frozenset(get_args(GroundingOperator)) != TRANSFORM_OPERATORS:
    raise RuntimeError("GroundingOperator must match the transform registry")
if frozenset(get_args(PredicateOperator)) != PREDICATE_OPERATORS:
    raise RuntimeError("PredicateOperator must match the predicate registry")
GroundedStopReason = Literal["answer", "tool_error", "timeout"]

logger = logging.getLogger("uvicorn.error.home_cortex.grounding")


@dataclass(frozen=True)
class GroundedAnswer:
    text: str
    tool_calls: int
    stop_reason: GroundedStopReason = "answer"


@dataclass(frozen=True)
class AgentRequestContext:
    """Trusted request and runtime identities used to resolve plan references."""

    caller_entity_id: str | None
    assistant_id: str
    assistant_display_name: str
    household_id: str | None
    current_time: datetime
    locale: str | None = None


class _PlanModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
    )


class GroundingSubject(_PlanModel):
    reference_type: Literal[
        "speaker",
        "assistant",
        "named_entity",
        "entity_id",
        "configured_home",
    ]
    reference: str | None = Field(default=None, min_length=1, max_length=256)
    expected_type: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
    )

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_anchor(cls, value: Any) -> Any:
        """Read persisted/test plans from before canonical reference types."""
        if not isinstance(value, Mapping) or "reference_type" in value:
            return value
        normalized = dict(value)
        anchor = normalized.pop("anchor", None)
        normalized["reference_type"] = {
            "authenticated_user": "speaker",
            "configured_home": "configured_home",
            "named_entity": "named_entity",
        }.get(anchor, anchor)
        return normalized

    @model_validator(mode="after")
    def validate_reference(self) -> "GroundingSubject":
        if self.reference_type in {"named_entity", "entity_id"} and not self.reference:
            raise ValueError(f"{self.reference_type} requires reference")
        if self.reference_type != "assistant" and self.expected_type is None:
            raise ValueError(f"{self.reference_type} requires expected_type")
        if self.reference_type == "entity_id" and self.expected_type is not None:
            if not _entity_matches_type(self.reference, self.expected_type):
                raise ValueError("entity_id reference must match expected_type")
        return self

    @property
    def anchor(self) -> str:
        """Compatibility view for callers that still display the old field name."""
        return {
            "speaker": "authenticated_user",
            "configured_home": "configured_home",
        }.get(self.reference_type, self.reference_type)


class TraversalStep(_PlanModel):
    relation: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    direction: Literal["out", "in", "both"] | None = None
    related_type: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
    )
    include_ended: bool = False
    field_equals: Mapping[str, Any] = Field(default_factory=dict, max_length=8)


class QueryFilter(_PlanModel):
    source: Literal["entity", "edge"] = "entity"
    field: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    operator: PredicateOperator
    value: Any = None


class QuerySort(_PlanModel):
    source: Literal["entity", "edge"] = "entity"
    field: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    direction: Literal["asc", "desc"] = "asc"


class FreshnessRequirement(_PlanModel):
    timestamp_field: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    max_age_seconds: int = Field(gt=0, le=31_536_000)


class RequiredEvidence(_PlanModel):
    source: Literal["entity", "edge"] = "entity"
    field: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
    )
    relation: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
    )
    minimum_records: int = Field(default=1, ge=1, le=100)
    freshness: FreshnessRequirement | None = None

    @model_validator(mode="after")
    def require_field_or_relation(self) -> "RequiredEvidence":
        if self.field is None and self.relation is None:
            raise ValueError("required evidence needs a field or relation")
        return self


class TransformSpec(_PlanModel):
    operator: GroundingOperator
    source: Literal["entity", "edge"] = "entity"
    field: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
    )
    other_field: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
    )
    order_by: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
    )
    mode: Literal["days", "seconds"] | None = None
    reference: Literal["household_today", "household_now"] | None = None
    from_unit: str | None = None
    to_unit: str | None = None

    @model_validator(mode="after")
    def require_operator_inputs(self) -> "TransformSpec":
        OPERATORS[self.operator].validate(
            field=self.field,
            other_field=self.other_field,
            order_by=self.order_by,
            parameters={
                "mode": self.mode,
                "reference": self.reference,
                "from_unit": self.from_unit,
                "to_unit": self.to_unit,
            },
        )
        if self.operator == "completed_years":
            if self.reference != "household_today" or self.mode is not None:
                raise ValueError(
                    "completed_years requires household_today and no mode"
                )
        if self.operator == "annual_occurrence":
            if self.reference != "household_today":
                raise ValueError(
                    "annual_occurrence requires field and household_today"
                )
            if self.mode not in {None, "days"}:
                raise ValueError(
                    "annual_occurrence supports a projected date or days"
                )
        return self


class GroundingPlan(_PlanModel):
    requires_grounding: bool
    grounding_domain: Literal["household", "runtime", "external_tool", "none"]
    goal: str = Field(min_length=1, max_length=512)
    subject: GroundingSubject | None = None
    traversal: tuple[TraversalStep, ...] = Field(default=(), max_length=6)
    fields: tuple[str, ...] = Field(default=(), max_length=32)
    edge_fields: tuple[str, ...] = Field(default=(), max_length=32)
    filters: tuple[QueryFilter, ...] = Field(default=(), max_length=16)
    sort: tuple[QuerySort, ...] = Field(default=(), max_length=8)
    transform: TransformSpec | None = None
    required_evidence: tuple[RequiredEvidence, ...] = Field(
        default=(),
        max_length=64,
    )

    @model_validator(mode="after")
    def validate_grounded_plan(self) -> "GroundingPlan":
        if self.requires_grounding:
            if (
                self.grounding_domain not in {"household", "runtime"}
                or self.subject is None
            ):
                raise ValueError("grounded plans require a resolvable subject")
            if not self.required_evidence:
                raise ValueError("grounded plans require explicit evidence")
            if self.grounding_domain == "runtime":
                if self.subject.reference_type != "assistant":
                    raise ValueError("runtime plans require an assistant reference")
                if self.traversal or self.edge_fields:
                    raise ValueError(
                        "runtime plans cannot traverse the household graph"
                    )
            elif self.subject.reference_type == "assistant":
                raise ValueError("assistant references belong to runtime grounding")
            required_relations = {
                item.relation
                for item in self.required_evidence
                if item.relation is not None
            }
            traversal_relations = {item.relation for item in self.traversal}
            if not traversal_relations.issubset(required_relations):
                missing = ", ".join(
                    sorted(traversal_relations - required_relations)
                )
                raise ValueError(
                    "every traversal relation must be required evidence; "
                    f"missing relations: {missing}"
                )
            required_fields = {
                source: {
                    item.field
                    for item in self.required_evidence
                    if item.source == source and item.field is not None
                }
                for source in ("entity", "edge")
            }
            for source in ("entity", "edge"):
                required_fields[source].update(
                    item.freshness.timestamp_field
                    for item in self.required_evidence
                    if item.source == source and item.freshness is not None
                )
            used_fields = {
                "entity": set(self.fields),
                "edge": set(self.edge_fields),
            }
            used_fields["entity"].update(
                field
                for step in self.traversal
                for field in step.field_equals
            )
            for item in self.filters:
                used_fields[item.source].add(item.field)
            for item in self.sort:
                used_fields[item.source].add(item.field)
            if self.transform is not None:
                used_fields[self.transform.source].update(
                    field
                    for field in (
                        self.transform.field,
                        self.transform.other_field,
                        self.transform.order_by,
                    )
                    if field is not None
                )
                if self.transform.operator in {"latest", "earliest"}:
                    expected_direction = (
                        "desc" if self.transform.operator == "latest" else "asc"
                    )
                    source_sorts = [
                        item
                        for item in self.sort
                        if item.source == self.transform.source
                    ]
                    if (
                        not source_sorts
                        or source_sorts[0].field != self.transform.order_by
                        or source_sorts[0].direction != expected_direction
                    ):
                        raise ValueError(
                            f"{self.transform.operator} requires a primary "
                            f"{expected_direction} sort on its order_by field"
                        )
                if self.transform.operator in {"first", "last"} and not self.sort:
                    raise ValueError(
                        f"{self.transform.operator} requires an explicit sort"
                    )
            for source in ("entity", "edge"):
                if not used_fields[source].issubset(required_fields[source]):
                    missing = ", ".join(
                        sorted(used_fields[source] - required_fields[source])
                    )
                    raise ValueError(
                        "every used field must be required evidence; missing "
                        f"{source} fields: {missing}"
                    )
        else:
            if self.grounding_domain not in {"external_tool", "none"}:
                raise ValueError(
                    "non-grounded plans must use the external_tool or none domain"
                )
            if (
                self.subject is not None
                or self.traversal
                or self.fields
                or self.edge_fields
                or self.filters
                or self.sort
                or self.transform is not None
                or self.required_evidence
            ):
                raise ValueError("non-grounded plans cannot contain query operations")
        return self


@dataclass(frozen=True)
class GroundingEvidence:
    status: GroundingStatus
    records: tuple[dict[str, Any], ...] = ()
    edges: tuple[dict[str, Any], ...] = ()
    value: Any = None
    missing_fields: tuple[str, ...] = ()
    tool_calls: int = 0
    resolved_subject: str | None = None
    requires_household_evidence: bool = True

    @property
    def sufficient(self) -> bool:
        return self.status == "sufficient"


class GroundingPlanner:
    """Use strict LLM output to decide and plan household evidence retrieval."""

    def __init__(self, ollama: Any, catalog: RuntimeSchemaCatalog) -> None:
        self.ollama = ollama
        self.catalog = catalog

    async def plan(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        household_now: datetime,
    ) -> GroundingPlan:
        output_schema = _planner_output_schema()
        try:
            payload = await self.ollama.plan_grounding(
                messages,
                self.catalog.prompt_payload(),
                output_schema,
                household_now=household_now.isoformat(),
            )
            return _validate_planner_payload(
                payload,
                messages=messages,
                catalog=self.catalog,
            )
        except (ValidationError, ValueError, TypeError) as error:
            validation_errors = (
                error.errors(include_input=False)
                if isinstance(error, ValidationError)
                else [
                    {
                        "type": type(error).__name__,
                        "message": str(error),
                    }
                ]
            )
            repair_instruction = {
                "role": "system",
                "content": (
                    "The previous grounding plan failed structural or operator "
                    "contract validation. "
                    "Return a corrected complete plan. Every top-level property "
                    "must be present. Grounded plans need a subject and at least "
                    "one field or relation in required_evidence. Every traversal "
                    "relation and every field used by selection, filtering, "
                    "sorting, or transformation must be covered by "
                    "required_evidence. Validation errors: "
                    + json.dumps(
                        validation_errors,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                ),
            }
            repaired = await self.ollama.plan_grounding(
                [*messages, repair_instruction],
                self.catalog.prompt_payload(),
                output_schema,
                household_now=household_now.isoformat(),
            )
            return _validate_planner_payload(
                repaired,
                messages=messages,
                catalog=self.catalog,
            )


def _planner_output_schema() -> dict[str, Any]:
    """Expose every top-level plan slot to constrained model generation.

    Pydantic defaults are useful for trusted Python callers, but JSON-schema
    generators otherwise let a model omit the very fields on which the
    grounded/non-grounded invariant depends.
    """
    schema = GroundingPlan.model_json_schema()
    properties = schema.get("properties")
    if isinstance(properties, Mapping):
        schema["required"] = list(properties)
    return schema


def _validate_planner_payload(
    payload: Mapping[str, Any],
    *,
    messages: Sequence[Mapping[str, Any]] = (),
    catalog: RuntimeSchemaCatalog | None = None,
) -> GroundingPlan:
    completed = _normalize_trusted_plan_roots(payload, messages, catalog)
    if catalog is not None:
        completed = _normalize_schema_backed_plan(completed, catalog)
    completed = _complete_evidence_requirements(completed)
    plan = GroundingPlan.model_validate_json(json.dumps(completed))
    if catalog is not None:
        validate_plan_operators(plan, catalog)
    return plan


def validate_plan_operators(
    plan: GroundingPlan,
    catalog: RuntimeSchemaCatalog,
) -> None:
    """Reject non-allowlisted or type-invalid computation before retrieval."""
    _validate_traversal_contract(plan, catalog)
    for item in plan.filters:
        definition = OPERATORS[item.operator]
        field_kind = _plan_field_kind(plan, catalog, item.source, item.field)
        definition.validate(field=item.field, field_kind=field_kind)
        if item.operator == "in" and (
            not isinstance(item.value, Sequence)
            or isinstance(item.value, (str, bytes, bytearray))
        ):
            raise OperatorValidationError("in requires a collection value")
        if item.operator == "exists" and not (
            item.value is None or isinstance(item.value, bool)
        ):
            raise OperatorValidationError("exists accepts only boolean or null")
        if item.operator == "date_range":
            if (
                not isinstance(item.value, Sequence)
                or isinstance(item.value, (str, bytes, bytearray))
                or len(item.value) != 2
                or any(
                    infer_value_kind(bound) not in {"date", "datetime"}
                    for bound in item.value
                )
            ):
                raise OperatorValidationError(
                    "date_range requires [date|datetime, date|datetime]"
                )
    for item in plan.sort:
        field_kind = _plan_field_kind(plan, catalog, item.source, item.field)
        OPERATORS["sort"].validate(field=item.field, field_kind=field_kind)
    transform = plan.transform
    if transform is None:
        return
    definition = OPERATORS[transform.operator]
    definition.validate(
        field=transform.field,
        field_kind=_plan_field_kind(
            plan,
            catalog,
            transform.source,
            transform.field,
        ),
        other_field=transform.other_field,
        other_field_kind=_plan_field_kind(
            plan,
            catalog,
            transform.source,
            transform.other_field,
        ),
        order_by=transform.order_by,
        order_by_kind=_plan_field_kind(
            plan,
            catalog,
            transform.source,
            transform.order_by,
        ),
        parameters={
            "mode": transform.mode,
            "reference": transform.reference,
            "from_unit": transform.from_unit,
            "to_unit": transform.to_unit,
        },
    )


def _validate_traversal_contract(
    plan: GroundingPlan,
    catalog: RuntimeSchemaCatalog,
) -> None:
    if not plan.requires_grounding or plan.grounding_domain == "runtime":
        return
    entity_type = plan.subject.expected_type if plan.subject is not None else None
    if entity_type is None or not catalog.has_entity_type(entity_type):
        raise OperatorValidationError("subject entity type is not in runtime schema")
    for step in plan.traversal:
        candidates = _related_types_for_step(entity_type, step.relation, catalog)
        if step.related_type is not None and step.related_type in candidates:
            related_type = step.related_type
        elif len(candidates) == 1:
            related_type = candidates[0]
        else:
            raise OperatorValidationError(
                f"relation {step.relation} does not unambiguously connect "
                f"entity type {entity_type}"
            )
        if step.related_type is not None and step.related_type not in candidates:
            raise OperatorValidationError(
                f"relation {step.relation} from {entity_type} reaches "
                f"{', '.join(candidates) or 'no entity type'}, "
                f"not {step.related_type}"
            )
        entity_type = related_type


def _plan_field_kind(
    plan: GroundingPlan,
    catalog: RuntimeSchemaCatalog,
    source: str,
    field: str | None,
) -> ValueKind:
    if field is None:
        return "unknown"
    if plan.grounding_domain == "runtime":
        return {
            "id": "string",
            "display_name": "string",
        }.get(field, "unknown")
    if source == "edge":
        if not plan.traversal:
            return "unknown"
        return catalog.relation_field_type(plan.traversal[-1].relation, field)
    entity_type = plan.subject.expected_type if plan.subject is not None else None
    for step in plan.traversal:
        if step.related_type is not None:
            entity_type = step.related_type
        elif entity_type is not None:
            inferred = _related_type_for_step(entity_type, step.relation, catalog)
            if inferred is not None:
                entity_type = inferred
    if entity_type is None:
        return "unknown"
    return catalog.entity_field_type(entity_type, field)


def _normalize_trusted_plan_roots(
    payload: Mapping[str, Any],
    messages: Sequence[Mapping[str, Any]],
    catalog: RuntimeSchemaCatalog | None,
) -> dict[str, Any]:
    """Keep conversational history from changing trusted graph roots."""
    completed = dict(payload)
    if completed.get("requires_grounding") is not True:
        return completed
    latest = next(
        (
            message.get("content")
            for message in reversed(messages)
            if message.get("role") == "user"
            and isinstance(message.get("content"), str)
        ),
        "",
    )
    subject = completed.get("subject")
    if not isinstance(subject, Mapping):
        return completed
    normalized_subject = dict(subject)
    reference_type = normalized_subject.get("reference_type")
    if reference_type is None:
        reference_type = {
            "authenticated_user": "speaker",
            "configured_home": "configured_home",
            "named_entity": "named_entity",
        }.get(normalized_subject.get("anchor"), normalized_subject.get("anchor"))
    contextual_type = _contextual_reference_type(
        reference_type,
        normalized_subject.get("reference"),
    )
    if contextual_type == "speaker":
        normalized_subject.update(
            {
                "reference_type": "speaker",
                "reference": None,
                "expected_type": "person",
            }
        )
        normalized_subject.pop("anchor", None)
        completed["grounding_domain"] = "household"
        completed["subject"] = normalized_subject
    elif contextual_type == "assistant":
        normalized_subject.update(
            {
                "reference_type": "assistant",
                "reference": None,
                "expected_type": None,
            }
        )
        normalized_subject.pop("anchor", None)
        completed["grounding_domain"] = "runtime"
        completed["subject"] = normalized_subject
    elif _references_configured_home(str(latest)):
        normalized_subject.update(
            {
                "reference_type": "configured_home",
                "reference": None,
                "expected_type": "address",
            }
        )
        normalized_subject.pop("anchor", None)
        completed["subject"] = normalized_subject
    transform = completed.get("transform")
    if (
        isinstance(transform, Mapping)
        and transform.get("operator") in {"first", "last"}
        and not completed.get("traversal")
        and normalized_subject.get("reference_type")
        in {"speaker", "assistant", "configured_home", "entity_id"}
    ):
        completed["transform"] = None
    root_type = normalized_subject.get("expected_type")
    if catalog is not None and isinstance(root_type, str):
        completed["traversal"] = _normalize_traversal_directions(
            completed.get("traversal"),
            root_type=root_type,
            catalog=catalog,
        )
    return completed


def _references_configured_home(text: str) -> bool:
    normalized = text.casefold()
    return bool(
        re.search(r"\b(?:(?:my|our|the)\s+)?(?:home|household)\b", normalized)
        or re.search(r"(?:我家|我们家|咱们家|家里|家中|家里的|家中的)", normalized)
    )


def _contextual_reference_type(reference_type: Any, reference: Any) -> str | None:
    if reference_type in {"speaker", "assistant"}:
        return str(reference_type)
    if reference_type != "named_entity" or not isinstance(reference, str):
        return None
    normalized = reference.casefold().strip(" \t\r\n?!？。！")
    if normalized in {"i", "me", "my", "mine", "我", "我的"}:
        return "speaker"
    if normalized in {"you", "your", "yours", "你", "您", "你的", "您的"}:
        return "assistant"
    return None


def _normalize_schema_backed_plan(
    payload: Mapping[str, Any],
    catalog: RuntimeSchemaCatalog,
) -> dict[str, Any]:
    """Correct model-declared field sources using the terminal runtime schema."""
    completed = dict(payload)
    transform = completed.get("transform")
    if isinstance(transform, Mapping) and transform.get("operator") == "select":
        field = transform.get("field")
        source = transform.get("source", "entity")
        projection_key = "edge_fields" if source == "edge" else "fields"
        projection = completed.get(projection_key)
        projected = list(projection) if isinstance(projection, list) else []
        if isinstance(field, str) and field not in projected:
            projected.append(field)
        completed[projection_key] = projected
        completed["transform"] = None
    subject = completed.get("subject")
    entity_type = (
        subject.get("expected_type") if isinstance(subject, Mapping) else None
    )
    traversal = completed.get("traversal")
    terminal_relation: str | None = None
    for step in traversal if isinstance(traversal, list) else []:
        if not isinstance(step, Mapping):
            continue
        relation = step.get("relation")
        if not isinstance(relation, str):
            continue
        terminal_relation = relation
        related_type = step.get("related_type")
        if isinstance(related_type, str):
            entity_type = related_type
        elif isinstance(entity_type, str):
            inferred = _related_type_for_step(entity_type, relation, catalog)
            if inferred is not None:
                entity_type = inferred

    entity_fields = (
        set(catalog.entities[entity_type].properties)
        if isinstance(entity_type, str) and entity_type in catalog.entities
        else set()
    )
    if completed.get("grounding_domain") == "runtime":
        entity_fields.update({"id", "display_name"})
    relation_schema = _relation_schema_for(terminal_relation, catalog)
    edge_fields = set(relation_schema.properties) if relation_schema else set()

    def source_for(field: Any, proposed: Any) -> str:
        source = proposed if proposed in {"entity", "edge"} else "entity"
        if not isinstance(field, str):
            return source
        belongs_to_entity = field in entity_fields
        belongs_to_edge = field in edge_fields
        if belongs_to_edge and not belongs_to_entity:
            return "edge"
        if belongs_to_entity and not belongs_to_edge:
            return "entity"
        return source

    projected: dict[str, list[str]] = {"entity": [], "edge": []}
    for source, key in (("entity", "fields"), ("edge", "edge_fields")):
        values = completed.get(key)
        for field in values if isinstance(values, list) else []:
            if not isinstance(field, str):
                continue
            destination = source_for(field, source)
            if field not in projected[destination]:
                projected[destination].append(field)
    completed["fields"] = projected["entity"]
    completed["edge_fields"] = projected["edge"]

    for key in ("filters", "sort"):
        operations = completed.get(key)
        if not isinstance(operations, list):
            continue
        normalized_operations: list[Any] = []
        for operation in operations:
            if not isinstance(operation, Mapping):
                normalized_operations.append(operation)
                continue
            normalized = dict(operation)
            normalized["source"] = source_for(
                normalized.get("field"),
                normalized.get("source"),
            )
            normalized_operations.append(normalized)
        completed[key] = normalized_operations

    transform = completed.get("transform")
    if isinstance(transform, Mapping):
        normalized_transform = dict(transform)
        transform_fields = [
            normalized_transform.get(key)
            for key in ("field", "other_field", "order_by")
            if isinstance(normalized_transform.get(key), str)
        ]
        inferred_sources = {
            source_for(field, normalized_transform.get("source"))
            for field in transform_fields
        }
        if len(inferred_sources) == 1:
            normalized_transform["source"] = inferred_sources.pop()
        completed["transform"] = normalized_transform

    requirements = completed.get("required_evidence")
    if isinstance(requirements, list):
        normalized_requirements: list[Any] = []
        for requirement in requirements:
            if not isinstance(requirement, Mapping):
                normalized_requirements.append(requirement)
                continue
            normalized = dict(requirement)
            if normalized.get("field") is not None:
                normalized["source"] = source_for(
                    normalized.get("field"),
                    normalized.get("source"),
                )
            normalized_requirements.append(normalized)
        completed["required_evidence"] = normalized_requirements
    return completed


def _relation_schema_for(
    relation: str | None,
    catalog: RuntimeSchemaCatalog,
) -> Any:
    if relation is None:
        return None
    schema = catalog.relations.get(relation)
    if schema is not None:
        return schema
    return next(
        (
            candidate
            for candidate in catalog.relations.values()
            if candidate.inverse_name == relation
        ),
        None,
    )


def _normalize_traversal_directions(
    raw_traversal: Any,
    *,
    root_type: str,
    catalog: RuntimeSchemaCatalog,
) -> Any:
    if not isinstance(raw_traversal, list):
        return raw_traversal
    current_type = root_type
    normalized: list[Any] = []
    for raw_step in raw_traversal:
        if not isinstance(raw_step, Mapping):
            normalized.append(raw_step)
            continue
        step = dict(raw_step)
        relation = step.get("relation")
        schema = catalog.relations.get(relation) if isinstance(relation, str) else None
        inverse = False
        if schema is None and isinstance(relation, str):
            schema = next(
                (
                    candidate
                    for candidate in catalog.relations.values()
                    if candidate.inverse_name == relation
                ),
                None,
            )
            inverse = schema is not None
        if schema is None:
            normalized.append(step)
            continue
        from_types = schema.to_types if inverse else schema.from_types
        to_types = schema.from_types if inverse else schema.to_types
        related_type = step.get("related_type")
        if schema.symmetric:
            step["direction"] = None
            endpoint_types = set(from_types) | set(to_types)
            if len(endpoint_types) == 1:
                related_type = next(iter(endpoint_types))
                step["related_type"] = related_type
        elif current_type in from_types and current_type not in to_types:
            step["direction"] = "out"
            if len(to_types) == 1:
                related_type = to_types[0]
                step["related_type"] = related_type
        elif current_type in to_types and current_type not in from_types:
            step["direction"] = "in"
            if len(from_types) == 1:
                related_type = from_types[0]
                step["related_type"] = related_type
        if isinstance(related_type, str):
            current_type = related_type
        normalized.append(step)
    return normalized


def _related_type_for_step(
    current_type: str,
    relation: str,
    catalog: RuntimeSchemaCatalog,
) -> str | None:
    candidates = _related_types_for_step(current_type, relation, catalog)
    return candidates[0] if len(candidates) == 1 else None


def _related_types_for_step(
    current_type: str,
    relation: str,
    catalog: RuntimeSchemaCatalog,
) -> tuple[str, ...]:
    schema = catalog.relations.get(relation)
    inverse = False
    if schema is None:
        schema = next(
            (
                candidate
                for candidate in catalog.relations.values()
                if candidate.inverse_name == relation
            ),
            None,
        )
        inverse = schema is not None
    if schema is None:
        return ()
    from_types = schema.to_types if inverse else schema.from_types
    to_types = schema.from_types if inverse else schema.to_types
    candidates: tuple[str, ...] = ()
    if current_type in from_types and current_type not in to_types:
        candidates = to_types
    elif current_type in to_types and current_type not in from_types:
        candidates = from_types
    elif current_type in from_types and current_type in to_types:
        candidates = tuple(sorted(set(from_types) | set(to_types)))
    return tuple(sorted(set(candidates)))


def _complete_evidence_requirements(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile structural query dependencies into evidence requirements.

    The planner chooses semantic operations. This compiler only makes their
    already-declared field and relation dependencies explicit, so an omitted
    bookkeeping entry cannot disable the evidence gate.
    """
    completed = dict(payload)
    if completed.get("requires_grounding") is not True:
        return completed
    raw_requirements = completed.get("required_evidence", [])
    if not isinstance(raw_requirements, list):
        return completed
    requirements = [
        dict(item) if isinstance(item, Mapping) else item
        for item in raw_requirements
    ]
    covered_fields = {
        (item.get("source", "entity"), item.get("field"))
        for item in requirements
        if isinstance(item, Mapping) and item.get("field") is not None
    }
    covered_relations = {
        item.get("relation")
        for item in requirements
        if isinstance(item, Mapping) and item.get("relation") is not None
    }

    def require_field(source: str, field: Any) -> None:
        if not isinstance(field, str) or not field:
            return
        marker = (source, field)
        if marker not in covered_fields:
            requirements.append({"source": source, "field": field})
            covered_fields.add(marker)

    def require_relation(relation: Any) -> None:
        if not isinstance(relation, str) or not relation:
            return
        if relation not in covered_relations:
            requirements.append({"source": "edge", "relation": relation})
            covered_relations.add(relation)

    entity_fields = completed.get("fields", [])
    if isinstance(entity_fields, list):
        for field in entity_fields:
            require_field("entity", field)
    edge_fields = completed.get("edge_fields", [])
    if isinstance(edge_fields, list):
        for field in edge_fields:
            require_field("edge", field)
    traversal = completed.get("traversal", [])
    for step in traversal if isinstance(traversal, list) else []:
        if not isinstance(step, Mapping):
            continue
        require_relation(step.get("relation"))
        field_equals = step.get("field_equals", {})
        if isinstance(field_equals, Mapping):
            for field in field_equals:
                require_field("entity", field)
    for operation_key in ("filters", "sort"):
        operations = completed.get(operation_key, [])
        for operation in operations if isinstance(operations, list) else []:
            if isinstance(operation, Mapping):
                field = operation.get("field")
                if operation_key == "filters" and operation.get("operator") == "exists":
                    field = "id"
                require_field(
                    str(operation.get("source", "entity")),
                    field,
                )
    transform = completed.get("transform")
    if isinstance(transform, Mapping):
        source = str(transform.get("source", "entity"))
        for key in ("field", "other_field", "order_by"):
            require_field(source, transform.get(key))
    completed["required_evidence"] = requirements
    return completed


class GroundingExecutor:
    """Execute a validated plan through bounded read-only graph tools."""

    def __init__(
        self,
        dispatcher: Any,
        catalog: RuntimeSchemaCatalog,
        *,
        home_entity_id: str | None,
        max_tool_calls: int = 8,
        max_records: int = 25,
        timeout_seconds: float = 5.0,
    ) -> None:
        if not 1 <= max_records < 100:
            raise ValueError("max_records must be between 1 and 99")
        self.dispatcher = dispatcher
        self.catalog = catalog
        self.home_entity_id = home_entity_id
        self.max_tool_calls = max_tool_calls
        self.max_records = max_records
        self.timeout_seconds = timeout_seconds

    async def execute(
        self,
        plan: GroundingPlan,
        *,
        context: AgentRequestContext | None = None,
        caller_entity_id: str | None = None,
        household_now: datetime | None = None,
    ) -> GroundingEvidence:
        if context is None:
            if household_now is None:
                raise ValueError("household_now or request context is required")
            context = AgentRequestContext(
                caller_entity_id=caller_entity_id,
                assistant_id="assistant",
                assistant_display_name="assistant",
                household_id=self.home_entity_id,
                current_time=household_now,
            )
        household_now = context.current_time
        try:
            validate_plan_operators(plan, self.catalog)
        except OperatorValidationError:
            return GroundingEvidence(
                "evidence_insufficient",
                requires_household_evidence=plan.grounding_domain == "household",
            )
        execution = _PlanExecution(
            self.dispatcher,
            caller_entity_id=context.caller_entity_id,
            maximum=self.max_tool_calls,
            timeout_seconds=self.timeout_seconds,
        )
        resolved_subject: str | None = None
        requires_household_evidence = plan.grounding_domain == "household"
        try:
            subjects, requires_household_evidence = await self._resolve_subject(
                plan.subject,
                execution,
                context,
            )
            resolved_subject = next(
                (
                    str(subject["id"])
                    for subject in subjects
                    if isinstance(subject.get("id"), str)
                ),
                None,
            )
            if not subjects:
                return GroundingEvidence(
                    "entity_not_found",
                    tool_calls=execution.tool_calls,
                    resolved_subject=resolved_subject,
                    requires_household_evidence=requires_household_evidence,
                )
            traversed_edges: list[Mapping[str, Any]] = []
            terminal_edges: list[Mapping[str, Any]] = []
            for step in plan.traversal:
                subjects, edges = await self._traverse(subjects, step, execution)
                traversed_edges.extend(edges)
                terminal_edges = list(edges)
                if not subjects:
                    return GroundingEvidence(
                        "entity_not_found",
                        tool_calls=execution.tool_calls,
                        resolved_subject=resolved_subject,
                        requires_household_evidence=requires_household_evidence,
                    )

            needs_entity_records = _needs_entity_records(plan)
            loaded_records = []
            if needs_entity_records:
                loaded_records = (
                    await self._load_records(subjects, execution)
                    if requires_household_evidence
                    else [dict(subject) for subject in subjects]
                )
            if needs_entity_records and not loaded_records:
                return GroundingEvidence(
                    "entity_not_found",
                    tool_calls=execution.tool_calls,
                    resolved_subject=resolved_subject,
                    requires_household_evidence=requires_household_evidence,
                )
            _validate_filter_inputs(loaded_records, terminal_edges, plan.filters)
            records = _apply_filters(
                loaded_records,
                [item for item in plan.filters if item.source == "entity"],
            )
            terminal_edges = _apply_filters(
                [dict(edge) for edge in terminal_edges],
                [item for item in plan.filters if item.source == "edge"],
            )
            records = _apply_sort(
                records,
                [item for item in plan.sort if item.source == "entity"],
            )
            terminal_edges = _apply_sort(
                terminal_edges,
                [item for item in plan.sort if item.source == "edge"],
            )
            if needs_entity_records and not records:
                return GroundingEvidence(
                    "evidence_insufficient",
                    tool_calls=execution.tool_calls,
                    resolved_subject=resolved_subject,
                    requires_household_evidence=requires_household_evidence,
                )
            status, missing = _validate_evidence(
                records,
                terminal_edges,
                plan.required_evidence,
                household_now,
                relationship_edges=traversed_edges,
            )
            if status != "sufficient":
                return GroundingEvidence(
                    status,
                    missing_fields=missing,
                    tool_calls=execution.tool_calls,
                    resolved_subject=resolved_subject,
                    requires_household_evidence=requires_household_evidence,
                )
            value = _apply_transform(
                records,
                terminal_edges,
                plan.transform,
                household_now,
            )
            evidence_records = records
            evidence_edges = terminal_edges
            if plan.transform is not None and plan.transform.operator in {
                "latest",
                "earliest",
            }:
                if plan.transform.source == "entity":
                    evidence_records = records[:1]
                else:
                    evidence_edges = terminal_edges[:1]
            scoped = _scope_records(evidence_records, plan)
            scoped_edges = _scope_edges(evidence_edges, plan)
            return GroundingEvidence(
                "sufficient",
                tuple(scoped),
                tuple(scoped_edges),
                value,
                tool_calls=execution.tool_calls,
                resolved_subject=resolved_subject,
                requires_household_evidence=requires_household_evidence,
            )
        except _PlanFailure as error:
            return GroundingEvidence(
                error.status,
                tool_calls=execution.tool_calls,
                resolved_subject=resolved_subject,
                requires_household_evidence=requires_household_evidence,
            )

    async def _resolve_subject(
        self,
        subject: GroundingSubject | None,
        execution: "_PlanExecution",
        context: AgentRequestContext,
    ) -> tuple[list[Mapping[str, Any]], bool]:
        if subject is None:
            return [], True
        if subject.reference_type == "assistant":
            return [
                {
                    "id": context.assistant_id,
                    "display_name": context.assistant_display_name,
                }
            ], False
        expected_type = subject.expected_type
        if subject.reference_type == "speaker":
            if context.caller_entity_id is None:
                raise _PlanFailure("caller_context_missing")
        if expected_type is None or not self.catalog.has_entity_type(expected_type):
            raise _PlanFailure("evidence_insufficient")
        if subject.reference_type == "speaker":
            if not _entity_matches_type(context.caller_entity_id, expected_type):
                raise _PlanFailure("evidence_insufficient")
            return [{"id": context.caller_entity_id}], True
        if subject.reference_type == "configured_home":
            if not _entity_matches_type(
                context.household_id,
                expected_type,
            ):
                return [], True
            return [{"id": context.household_id}], True
        if subject.reference_type == "entity_id":
            return [{"id": subject.reference}], True
        if subject.reference_type != "named_entity" or subject.reference is None:
            raise _PlanFailure("evidence_insufficient")
        result = await execution.call(
            "search_entities",
            {
                "text": subject.reference,
                "entity_type": expected_type,
                "limit": _probe_limit(self.max_records),
            },
        )
        records = _result_records(result)
        if len(records) > self.max_records:
            raise _PlanFailure("evidence_insufficient")
        normalized = subject.reference.casefold()
        exact = [
            record
            for record in records
            if normalized in {alias.casefold() for alias in record_aliases(record)}
        ]
        if len(exact) != 1:
            if len(exact) > 1:
                raise _PlanFailure("evidence_insufficient")
            return [], True
        return exact, True

    async def _traverse(
        self,
        subjects: Sequence[Mapping[str, Any]],
        step: TraversalStep,
        execution: "_PlanExecution",
    ) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
        if not self.catalog.has_relation(step.relation):
            raise _PlanFailure("evidence_insufficient")
        related: list[Mapping[str, Any]] = []
        edges: list[Mapping[str, Any]] = []
        seen: set[str] = set()
        for subject in subjects[: self.max_records]:
            if len(edges) >= self.max_records:
                raise _PlanFailure("evidence_insufficient")
            entity_id = subject.get("id")
            if not isinstance(entity_id, str):
                continue
            arguments: dict[str, Any] = {
                "entity_id": entity_id,
                "relation": step.relation,
                "include_ended": step.include_ended,
                "limit": _probe_limit(self.max_records),
            }
            if step.direction is not None:
                arguments["direction"] = step.direction
            result = await execution.call("get_relationships", arguments)
            result_edges = _result_records(result)
            remaining = self.max_records - len(edges)
            if len(result_edges) > remaining:
                raise _PlanFailure("evidence_insufficient")
            for edge in result_edges:
                candidate = edge.get("related_entity")
                if not isinstance(candidate, Mapping):
                    continue
                candidate_id = candidate.get("id")
                if not isinstance(candidate_id, str):
                    continue
                if step.related_type is not None and not candidate_id.startswith(
                    f"{step.related_type}:"
                ):
                    continue
                if not all(
                    candidate.get(key) == value
                    for key, value in step.field_equals.items()
                ):
                    continue
                edges.append(edge)
                if candidate_id not in seen:
                    seen.add(candidate_id)
                    related.append(candidate)
        return related, edges

    async def _load_records(
        self,
        subjects: Sequence[Mapping[str, Any]],
        execution: "_PlanExecution",
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for subject in subjects[: self.max_records]:
            entity_id = subject.get("id")
            if not isinstance(entity_id, str):
                continue
            result = await execution.call(
                "get_entity",
                {"entity_id": entity_id},
            )
            records.extend(dict(record) for record in _result_records(result))
            if len(records) >= self.max_records:
                break
        return records[: self.max_records]


class OpenWorldGroundingService:
    """Plan, execute, gate, and render open-world household facts."""

    def __init__(self, planner: GroundingPlanner, executor: GroundingExecutor) -> None:
        self.planner = planner
        self.executor = executor

    async def try_answer(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        context: AgentRequestContext | None = None,
        caller_entity_id: str | None = None,
        household_now: datetime | None = None,
        language: str | None = None,
        request_id: str = "-",
    ) -> GroundedAnswer | None:
        if context is None:
            if household_now is None or language is None:
                raise ValueError(
                    "request context or legacy grounding context is required"
                )
            context = AgentRequestContext(
                caller_entity_id=caller_entity_id,
                assistant_id="assistant",
                assistant_display_name="assistant",
                household_id=self.executor.home_entity_id,
                current_time=household_now,
                locale=language,
            )
        household_now = context.current_time
        language = context.locale or language or "en"
        try:
            plan = await self.planner.plan(
                messages,
                household_now=household_now,
            )
        except Exception as error:
            validation = (
                ";".join(
                    f"{'.'.join(str(item) for item in detail.get('loc', ()))}:"
                    f"{detail.get('type', 'unknown')}"
                    for detail in error.errors(include_input=False)
                )
                if isinstance(error, ValidationError)
                else str(error)
                if isinstance(error, OperatorValidationError)
                else "none"
            )
            logger.warning(
                "grounding_planner_failure request_id=%s error_type=%s "
                "validation=%s",
                safe_log_token(request_id),
                safe_log_token(type(error).__name__),
                safe_log_token(validation),
            )
            _log_grounding(
                request_id,
                required="unknown",
                status="planner_error",
            )
            if not _looks_like_factual_request(messages):
                return None
            return GroundedAnswer(
                (
                    "老管家目前无法判断这个请求需要哪些家庭事实。"
                    if language == "zh"
                    else (
                        "I could not determine which household evidence this "
                        "request requires."
                    )
                ),
                0,
                "tool_error",
            )
        if not plan.requires_grounding:
            if _has_private_household_reference(messages, self.planner.catalog):
                _log_grounding(
                    request_id,
                    required="false",
                    status="false_negative_blocked",
                )
                return GroundedAnswer(
                    (
                        "老管家无法确认这个请求是否需要家庭事实，因此不会猜测。"
                        if language == "zh"
                        else (
                            "I could not confirm whether this request requires "
                            "private household evidence, so I will not guess."
                        )
                    ),
                    0,
                    "tool_error",
                )
            _log_grounding(request_id, required="false", status="not_required")
            return None
        evidence = await self.executor.execute(
            plan,
            context=context,
        )
        if not evidence.sufficient:
            _log_grounding_plan(request_id, plan, evidence, context)
            return GroundedAnswer(
                _grounding_status_answer(evidence, plan, language),
                evidence.tool_calls,
                "timeout" if evidence.status == "timeout" else (
                    "tool_error" if evidence.status == "tool_error" else "answer"
                ),
            )
        text = _deterministic_evidence_answer(plan, evidence, language)
        _log_grounding_plan(request_id, plan, evidence, context)
        return GroundedAnswer(text, evidence.tool_calls)


class _PlanFailure(RuntimeError):
    def __init__(self, status: GroundingStatus) -> None:
        super().__init__(status)
        self.status = status


class _PlanExecution:
    def __init__(
        self,
        dispatcher: Any,
        *,
        caller_entity_id: str | None,
        maximum: int,
        timeout_seconds: float,
    ) -> None:
        self.dispatcher = dispatcher
        self.caller_entity_id = caller_entity_id
        self.maximum = maximum
        self.timeout_seconds = timeout_seconds
        self.tool_calls = 0

    async def call(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.tool_calls >= self.maximum:
            raise _PlanFailure("evidence_insufficient")
        self.tool_calls += 1
        try:
            result = await asyncio.wait_for(
                self.dispatcher.dispatch(
                    tool_name,
                    arguments,
                    caller_entity_id=self.caller_entity_id,
                ),
                timeout=self.timeout_seconds,
            )
        except TimeoutError as error:
            raise _PlanFailure("timeout") from error
        except Exception as error:
            raise _PlanFailure("tool_error") from error
        if result.get("ok") is not True:
            error = result.get("error")
            status: GroundingStatus = (
                "timeout"
                if isinstance(error, Mapping) and error.get("code") == "tool_timeout"
                else "tool_error"
            )
            raise _PlanFailure(status)
        return result


def _result_records(result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    records = result.get("result")
    if not isinstance(records, list):
        return []
    return [record for record in records if isinstance(record, Mapping)]


def _probe_limit(max_records: int) -> int:
    return min(max_records + 1, 100)


def _needs_entity_records(plan: GroundingPlan) -> bool:
    if plan.fields:
        return True
    if any(item.source == "entity" for item in (*plan.filters, *plan.sort)):
        return True
    if plan.transform is not None and plan.transform.source == "entity":
        return True
    return any(
        requirement.source == "entity"
        and (requirement.field is not None or requirement.freshness is not None)
        for requirement in plan.required_evidence
    )


def _entity_matches_type(entity_id: str | None, expected_type: str) -> bool:
    if entity_id is None:
        return False
    entity_type, separator, record_key = entity_id.partition(":")
    return separator == ":" and bool(record_key) and entity_type == expected_type


def _apply_filters(
    records: Sequence[dict[str, Any]],
    filters: Sequence[QueryFilter],
) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if all(
            _compare(record.get(item.field), item.operator, item.value)
            for item in filters
        )
    ]


def _validate_filter_inputs(
    records: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    filters: Sequence[QueryFilter],
) -> None:
    for item in filters:
        source_records = records if item.source == "entity" else edges
        if item.operator == "exists":
            continue
        if any(record.get(item.field) is None for record in source_records):
            raise _PlanFailure("evidence_insufficient")


def _compare(left: Any, operator: str, right: Any) -> bool:
    return evaluate_predicate(operator, left, right)


def _apply_sort(
    records: Sequence[dict[str, Any]],
    sorts: Sequence[QuerySort],
) -> list[dict[str, Any]]:
    ordered = list(records)
    for item in reversed(sorts):
        present = [
            record for record in ordered if record.get(item.field) is not None
        ]
        missing = [
            record for record in ordered if record.get(item.field) is None
        ]
        present.sort(
            key=lambda record: _sortable(record.get(item.field)),
            reverse=item.direction == "desc",
        )
        ordered = present + missing
    return ordered


def _sortable(value: Any) -> tuple[int, Any]:
    if _is_number(value):
        return 0, float(value)
    if isinstance(value, str):
        return 1, value
    if value is None:
        return 3, ""
    return 2, json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _validate_evidence(
    records: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    requirements: Sequence[RequiredEvidence],
    household_now: datetime,
    *,
    relationship_edges: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[GroundingStatus, tuple[str, ...]]:
    missing: list[str] = []
    relation_records = edges if relationship_edges is None else relationship_edges
    for requirement in requirements:
        if requirement.relation is not None:
            count = sum(
                edge.get("relation") == requirement.relation
                or edge.get("semantic_relation") == requirement.relation
                for edge in relation_records
            )
            if count < requirement.minimum_records:
                return "evidence_insufficient", ()
        if requirement.field is None:
            continue
        source_records = records if requirement.source == "entity" else edges
        matching = [
            record
            for record in source_records
            if requirement.field in record
        ]
        if not matching:
            missing.append(requirement.field)
            continue
        if len(matching) < requirement.minimum_records:
            return "evidence_insufficient", ()
        if requirement.freshness is not None:
            timestamps = [
                parsed
                for record in matching
                if (
                    parsed := _parse_datetime(
                        record.get(requirement.freshness.timestamp_field)
                    )
                )
                is not None
            ]
            if len(timestamps) < requirement.minimum_records:
                return "evidence_insufficient", ()
            latest = max(timestamps)
            now = household_now
            if now.tzinfo is None:
                now = now.replace(tzinfo=timezone.utc)
            age_seconds = (now.astimezone(timezone.utc) - latest).total_seconds()
            if age_seconds < 0:
                return "evidence_insufficient", ()
            if age_seconds > requirement.freshness.max_age_seconds:
                return "evidence_stale", ()
    if missing:
        return "field_not_available", tuple(sorted(set(missing)))
    return "sufficient", ()


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _apply_transform(
    records: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    transform: TransformSpec | None,
    household_now: datetime,
) -> Any:
    if transform is None:
        return None
    source_records = records if transform.source == "entity" else edges
    try:
        return execute_operator(
            transform.operator,
            OperatorInput(
                records=source_records,
                field=transform.field,
                other_field=transform.other_field,
                order_by=transform.order_by,
                mode=transform.mode,
                reference=transform.reference,
                from_unit=transform.from_unit,
                to_unit=transform.to_unit,
                now=household_now,
            ),
        )
    except (OperatorExecutionError, OperatorValidationError) as error:
        raise _PlanFailure("evidence_insufficient") from error


def _is_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _scope_records(
    records: Sequence[Mapping[str, Any]],
    plan: GroundingPlan,
) -> list[dict[str, Any]]:
    fields = set(plan.fields)
    fields.update(
        requirement.field
        for requirement in plan.required_evidence
        if requirement.source == "entity" and requirement.field is not None
    )
    fields.update(
        item.field for item in plan.sort if item.source == "entity"
    )
    fields.update(
        item.field for item in plan.filters if item.source == "entity"
    )
    fields.update(
        requirement.freshness.timestamp_field
        for requirement in plan.required_evidence
        if requirement.source == "entity" and requirement.freshness is not None
    )
    if plan.transform is not None and plan.transform.source == "entity":
        fields.update(
            field
            for field in (
                plan.transform.field,
                plan.transform.other_field,
                plan.transform.order_by,
            )
            if field is not None
        )
    fields.update({"id", "name"})
    return [
        {key: value for key, value in record.items() if str(key) in fields}
        for record in records
    ]


def _scope_edges(
    edges: Sequence[Mapping[str, Any]],
    plan: GroundingPlan,
) -> list[dict[str, Any]]:
    fields = set(plan.edge_fields)
    fields.update(
        requirement.field
        for requirement in plan.required_evidence
        if requirement.source == "edge" and requirement.field is not None
    )
    fields.update(
        item.field for item in plan.sort if item.source == "edge"
    )
    fields.update(
        item.field for item in plan.filters if item.source == "edge"
    )
    fields.update(
        requirement.freshness.timestamp_field
        for requirement in plan.required_evidence
        if requirement.source == "edge" and requirement.freshness is not None
    )
    if plan.transform is not None and plan.transform.source == "edge":
        fields.update(
            field
            for field in (
                plan.transform.field,
                plan.transform.other_field,
                plan.transform.order_by,
            )
            if field is not None
        )
    fields.update({"id", "relation", "semantic_relation"})
    return [
        {key: value for key, value in edge.items() if str(key) in fields}
        for edge in edges
    ]


def _grounding_status_answer(
    evidence: GroundingEvidence,
    plan: GroundingPlan,
    language: str,
) -> str:
    reference = (
        plan.subject.reference
        if plan.subject is not None and plan.subject.reference is not None
        else plan.subject.reference_type
        if plan.subject is not None
        else ""
    )
    if language == "zh":
        if evidence.status == "caller_context_missing":
            return "我无法确认当前登录者的身份。"
        if evidence.status == "entity_not_found":
            return (
                "家庭资料中没有找到与"
                f"“{reference}”对应的实体记录。"
            )
        if evidence.status == "field_not_available":
            fields = "、".join(evidence.missing_fields)
            return (
                "家庭资料中有相关实体记录，但没有记录所需字段："
                f"{fields}。"
            )
        if evidence.status == "evidence_stale":
            return (
                "家庭资料中有相关记录，但数据已过期，"
                "不能作为当前事实回答。"
            )
        if evidence.status == "evidence_insufficient":
            return (
                "家庭资料中有部分相关信息，"
                "但不足以核实这个问题。"
            )
        return "老管家目前无法从家庭资料中核实这项信息。"
    if evidence.status == "caller_context_missing":
        return "I cannot verify the identity of the current signed-in user."
    if evidence.status == "entity_not_found":
        return f'The home graph has no entity matching "{reference}".'
    if evidence.status == "field_not_available":
        fields = ", ".join(evidence.missing_fields)
        return f"The entity is recorded, but the required fields are absent: {fields}."
    if evidence.status == "evidence_stale":
        return "Relevant data exists, but it is too old to support a current answer."
    if evidence.status == "evidence_insufficient":
        return "Some relevant data exists, but it is insufficient to verify the answer."
    return "I could not verify that information from the home graph."


def _deterministic_evidence_answer(
    plan: GroundingPlan,
    evidence: GroundingEvidence,
    language: str,
) -> str:
    value = evidence.value
    if value is None and evidence.records:
        record_fields = plan.fields or tuple(
            requirement.field
            for requirement in plan.required_evidence
            if requirement.source == "entity" and requirement.field is not None
        )
        rendered_records = [
            _requested_values(
                record,
                record_fields,
                language=language,
                include_name=len(evidence.records) > 1,
            )
            for record in evidence.records
        ]
        value = (
            rendered_records[0]
            if len(rendered_records) == 1
            else rendered_records
        )
    if value is None and evidence.edges:
        edge_fields = plan.edge_fields or tuple(
            requirement.field
            for requirement in plan.required_evidence
            if requirement.source == "edge" and requirement.field is not None
        )
        if not edge_fields:
            edge_fields = ("semantic_relation", "relation")
        rendered_edges = [
            _requested_values(edge, edge_fields, language=language)
            for edge in evidence.edges
        ]
        value = rendered_edges[0] if len(rendered_edges) == 1 else rendered_edges
    reference_type = (
        plan.subject.reference_type if plan.subject is not None else None
    )
    if isinstance(value, Mapping):
        if reference_type == "assistant" and "display_name" in value:
            name = str(value["display_name"])
            return f"我是{name}。" if language == "zh" else f"I am {name}."
        if reference_type == "speaker" and "name" in value:
            name = str(value["name"])
            return f"您是{name}。" if language == "zh" else f"You are {name}."
    rendered = json.dumps(value, ensure_ascii=False, default=str)
    source = (
        "运行时上下文" if plan.grounding_domain == "runtime" else "家庭资料"
    )
    return (
        f"根据{source}，查询结果是：{rendered}。"
        if language == "zh"
        else f"According to {_evidence_source(plan)}, the result is: {rendered}."
    )


def _evidence_source(plan: GroundingPlan) -> str:
    return "runtime context" if plan.grounding_domain == "runtime" else "the home graph"


_PRIVATE_HOUSEHOLD_REFERENCE = re.compile(
    r"\b(?:i|my|mine|we|our|ours)\b|"
    r"我(?:的|家|们家|岳|爸|妈|父|母|妻|夫|儿|女|老公|老婆)|"
    r"咱们家",
    flags=re.IGNORECASE,
)
_FACTUAL_QUESTION_FORM = re.compile(
    r"\b(?:what|where|when|who)\b(?!\s+(?:should|can|could|would)\b)|"
    r"\bhow\s+(?:old|many|much|long)\b|"
    r"\b(?:tell|show|give)\s+me\b|"
    r"多少|多大|几岁|哪里|哪儿|在哪|是谁|什么时候|何时|有没有|是什么",
    flags=re.IGNORECASE,
)
_EXTERNAL_TOOL_REQUEST = re.compile(
    r"\b(?:calendar|schedule|appointment|event|meeting|availability)\b|"
    r"\b(?:what\s+do\s+(?:i|we)\s+have|what(?:'s|\s+is)\s+on\s+my)"
    r".{0,40}\b(?:today|tomorrow)\b|"
    r"\b(?:am\s+i|are\s+we)\b.{0,40}\b(?:available|free|busy)\b|"
    r"日程|行程|安排|约会|会议|(?:今天|明天).{0,12}(?:有什么|有空|忙不忙)|"
    r"(?:计算|算一下).*[0-9]",
    flags=re.IGNORECASE,
)


def _looks_like_factual_request(
    messages: Sequence[Mapping[str, Any]],
) -> bool:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        return isinstance(content, str) and bool(
            _FACTUAL_QUESTION_FORM.search(content)
        )
    return False


def _has_private_household_reference(
    messages: Sequence[Mapping[str, Any]],
    catalog: RuntimeSchemaCatalog,
) -> bool:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        return bool(
            isinstance(content, str)
            and not _EXTERNAL_TOOL_REQUEST.search(content)
            and (
                _PRIVATE_HOUSEHOLD_REFERENCE.search(content)
                or _mentions_catalog_entity(content, catalog)
            )
            and _FACTUAL_QUESTION_FORM.search(content)
        )
    return False


def _mentions_catalog_entity(
    text: str,
    catalog: RuntimeSchemaCatalog,
) -> bool:
    normalized = text.casefold()
    return any(alias.casefold() in normalized for alias in catalog.entity_aliases)


def _requested_values(
    record: Mapping[str, Any],
    requested_fields: Sequence[str],
    *,
    language: str,
    include_name: bool = False,
) -> dict[str, Any]:
    fields = list(requested_fields)
    if include_name and "name" in record and "name" not in fields:
        fields.insert(0, "name")
    values = {field: record[field] for field in fields if field in record}
    if "name" in values:
        display_name = resolve_display_name(record, language)
        if display_name is not None:
            values["name"] = display_name
    return values


def _log_grounding_plan(
    request_id: str,
    plan: GroundingPlan,
    evidence: GroundingEvidence,
    context: AgentRequestContext,
) -> None:
    subject_type = (
        plan.subject.expected_type
        if plan.subject is not None and plan.subject.expected_type is not None
        else "none"
    )
    reference_type = (
        plan.subject.reference_type if plan.subject is not None else "none"
    )
    operator = plan.transform.operator if plan.transform is not None else "none"
    fields = sorted(
        item.field for item in plan.required_evidence if item.field is not None
    )
    relations = sorted(
        item.relation
        for item in plan.required_evidence
        if item.relation is not None
    )
    logger.info(
        "grounding_plan request_id=%s grounding_required=true "
        "caller_context_present=%s subject_reference_type=%s subject_type=%s "
        "resolved_subject=%s requires_household_evidence=%s fields=%s "
        "relations=%s operator=%s evidence_status=%s",
        safe_log_token(request_id),
        str(context.caller_entity_id is not None).lower(),
        safe_log_token(reference_type),
        safe_log_token(subject_type),
        safe_log_token(evidence.resolved_subject or "none"),
        str(evidence.requires_household_evidence).lower(),
        safe_log_token(",".join(fields) or "none"),
        safe_log_token(",".join(relations) or "none"),
        safe_log_token(operator),
        safe_log_token(evidence.status),
    )


def _log_grounding(request_id: str, *, required: str, status: str) -> None:
    logger.info(
        "grounding_plan request_id=%s grounding_required=%s evidence_status=%s",
        safe_log_token(request_id),
        safe_log_token(required),
        safe_log_token(status),
    )
