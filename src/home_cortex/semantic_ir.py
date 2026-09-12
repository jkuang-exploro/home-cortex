"""Semantic household-fact IR: requests, results, and planner/executor types.

Authoritative structured types for interpretation and deterministic execution.
Physical storage names do not appear here.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .mutation_ir import (
    NamedWriteRequest,
    NamedCreateItem,
    NamedMoveItem,
    NamedDeleteItem,
    NamedUpdateAttributes,
)
from .operator_registry import FactOperation

FactStatus = Literal[
    "found",
    "discourse_context_missing",
    "caller_context_missing",
    "entity_not_found",
    "relationship_not_found",
    "property_unavailable",
    "relation_property_unavailable",
    "filter_input_missing",
    "filter_unsupported",
    "operator_unsupported",
    "semantic_plan_unsupported",
    "ambiguous",
    "computation_input_missing",
    "computation_impossible",
    "collection_incomplete",
]
ReferenceKind = Literal[
    "self",
    "assistant",
    "current_household",
    "named_entity",
    "discourse",
    "unresolved",
    "entity_id",
]
PlannerValidationCode = Literal[
    "VALID",
    "NOT_A_FACT",
    "MALFORMED_OUTPUT",
    "UNSUPPORTED_OPERATION",
    "UNKNOWN_PROPERTY",
    "UNKNOWN_RELATION",
    "MODEL_ORIGINATED_ENTITY_ID",
    "INVALID_PLAN",
]

_CONTEXT_ENTITY_TYPES = {
    "self": "person",
    "assistant": "person",
    "current_household": "address",
}


@dataclass(frozen=True)
class DiscourseContext:
    conversation_id: str
    caller_entity_id: str | None
    household_id: str | None
    assistant_id: str
    # Oldest to newest; empty entries preserve topic-change/failed-turn boundaries.
    turns: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class AgentRequestContext:
    """Trusted identities and clock used to resolve semantic references."""

    caller_entity_id: str | None
    assistant_id: str
    assistant_display_name: str
    household_id: str | None
    current_time: datetime
    locale: str | None = None
    conversation_id: str | None = None
    discourse: DiscourseContext | None = None


class _SemanticModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class SemanticFilter(_SemanticModel):
    property: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
        description="Advertised semantic property; never a status predicate.",
    )
    predicate: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
        description="Advertised collection predicate such as adult or minor.",
    )
    operator: Literal[
        "eq",
        "ne",
        "gt",
        "gte",
        "lt",
        "lte",
        "in",
        "exists",
        "date_range",
    ] = "eq"
    value: str | int | float | bool | tuple[str | int | float, ...] | None = None
    source: Literal["entity", "relation"] = "entity"
    value_from: Literal["anchor"] | None = None
    value_property: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
    )
    transform: Literal["date_difference"] | None = None
    mode: Literal["years", "months", "days"] | None = None

    @model_validator(mode="after")
    def validate_dynamic_value(self) -> "SemanticFilter":
        if (self.property is None) == (self.predicate is None):
            raise ValueError("filter requires exactly one of property or predicate")
        if self.predicate is not None and (
            self.operator != "eq"
            or self.value is not None
            or self.source != "entity"
            or self.value_from is not None
            or self.value_property is not None
            or self.transform is not None
            or self.mode is not None
        ):
            raise ValueError("semantic predicates do not accept field-filter options")
        if self.transform is None:
            if self.mode is not None:
                raise ValueError("mode requires transform")
        else:
            if self.mode not in {"years", "months", "days"}:
                raise ValueError("date_difference filters require mode years|months|days")
            if self.operator not in {"gt", "gte", "lt", "lte", "eq"}:
                raise ValueError("derived date filters require a scalar comparison")
            if type(self.value) is not int or self.value < 0 or self.value > 120:
                raise ValueError("derived date filters require an integer threshold")
            if self.value_from is not None or self.source != "entity":
                raise ValueError("derived date filters compare an entity date to household now")
        if self.value_from is not None and self.value is not None:
            raise ValueError("filter cannot define both value and value_from")
        if self.value_property is not None and self.value_from is None:
            raise ValueError("value_property requires value_from")
        if self.value_from is not None and self.source != "entity":
            raise ValueError("dynamic filter values require entity source")
        if self.value_from is not None and self.operator in {
            "in",
            "exists",
            "date_range",
        }:
            raise ValueError("dynamic filter value is incompatible with operator")
        return self


class SemanticConceptUse(_SemanticModel):
    """Explicit ontology path alias in model output; lowered before execution."""

    concept: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    filters: tuple[SemanticFilter, ...] = ()


class SemanticRelationStep(_SemanticModel):
    relation: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    filters: tuple[SemanticFilter, ...] = Field(
        default=(),
        description="Traversal disambiguation only; collection filters are request.filters.",
    )


class SemanticReference(_SemanticModel):
    kind: ReferenceKind
    value: str | None = Field(default=None, max_length=256)
    entity_type: str | None = Field(default=None, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    path: tuple[SemanticRelationStep, ...] = Field(default=(), max_length=8)
    turn_offset: int | None = Field(default=None, strict=True, ge=1, le=8)
    cardinality: Literal["single", "collection"] = "single"

    @model_validator(mode="after")
    def validate_value(self) -> "SemanticReference":
        if self.kind == "unresolved" and (self.value is not None or self.path):
            raise ValueError("unresolved reference cannot carry a value or path")
        if self.kind == "discourse":
            if self.value is not None or self.turn_offset is None or self.entity_type is None:
                raise ValueError("discourse requires turn_offset, entity_type and no literal value")
        elif self.turn_offset is not None or self.cardinality != "single":
            raise ValueError("discourse options require a discourse reference")
        if self.kind in _CONTEXT_ENTITY_TYPES and self.value is not None:
            raise ValueError("contextual references do not accept a literal value")
        if self.kind in {"named_entity", "entity_id"} and not self.value:
            raise ValueError(f"{self.kind} requires value")
        if self.kind == "entity_id" and self.value is not None and not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*:[A-Za-z0-9_-]+",
            self.value,
        ):
            raise ValueError("entity_id requires a canonical record ID")
        return self


class SemanticFactRequest(_SemanticModel):
    operation: FactOperation
    subject: SemanticReference
    property: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
        description="Semantic property projected or consumed by the outer operation.",
    )
    property_source: Literal["entity", "relationship"] = Field(
        default="entity",
        description="Whether property belongs to the resolved entity or final relationship edge.",
    )
    filters: tuple[SemanticFilter, ...] = Field(
        default=(),
        description="Collection filters evaluated before count/aggregation/selection.",
    )
    projection: Literal["scalar", "each"] = "scalar"
    exclude: tuple[SemanticReference, ...] = Field(default=(), max_length=8)
    amount: int | None = Field(default=None, strict=True, ge=-120000, le=120000)
    other: SemanticReference | None = None
    mode: Literal["years", "months", "days", "seconds"] | None = None
    from_unit: str | None = Field(default=None, max_length=16)
    to_unit: str | None = Field(default=None, max_length=16)

class SemanticPlan(_SemanticModel):
    """The only structured output accepted from the Tier-1 interpreter."""

    requires_fact: bool
    request: SemanticFactRequest | None = None
    mutation: NamedCreateItem | NamedMoveItem | NamedUpdateAttributes | NamedDeleteItem | None = None

    @model_validator(mode="after")
    def validate_request_presence(self) -> "SemanticPlan":
        if self.requires_fact != (self.request is not None):
            raise ValueError("requires_fact must match request presence")
        if self.mutation is not None and self.request is not None:
            raise ValueError("A plan cannot both query facts and mutate state")
        return self


@dataclass(frozen=True)
class FactRelationshipEvidence:
    relation: str
    source_id: str | None = None
    target_id: str | None = None
    start: Any = None
    end: Any = None


@dataclass(frozen=True)
class FactEvidence:
    entity_ids: tuple[str, ...] = ()
    relationship: str | None = None
    semantic_property: str | None = None
    relationships: tuple[FactRelationshipEvidence, ...] = ()


@dataclass(frozen=True)
class FactRow:
    entity: Mapping[str, Any]
    status: FactStatus
    value: Any = None
    unit: str | None = None
    evidence: FactEvidence = FactEvidence()
    missing_requirements: tuple[str, ...] = ()


@dataclass(frozen=True)
class FactContentGroup:
    space: Mapping[str, Any]
    entities: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class FactResult:
    status: FactStatus
    value: Any = None
    evidence: FactEvidence = FactEvidence()
    missing_requirements: tuple[str, ...] = ()
    candidates: tuple[Mapping[str, Any], ...] = ()
    unit: str | None = None
    shape: Literal["scalar", "entity", "entities", "rows"] = "scalar"
    rows: tuple[FactRow, ...] = ()
    focus_entity_ids: tuple[str, ...] = ()
    content_groups: tuple[FactContentGroup, ...] = ()


@dataclass(frozen=True)
class FactTimings:
    tier: int
    routing_ms: float = 0
    entity_resolution_ms: float = 0
    fact_query_ms: float = 0
    computation_ms: float = 0
    render_ms: float = 0
    llm_ms: float = 0
    total_ms: float = 0
    llm_call_count: int = 0
    db_query_count: int = 0


@dataclass(frozen=True)
class PlannerDiagnostics:
    input_summary: Mapping[str, Any]
    output_raw: Mapping[str, Any] | None = None
    normalized_plan: Mapping[str, Any] | None = None
    validation_result: PlannerValidationCode = "INVALID_PLAN"
    failure_detail: str | None = None
    attempt_count: int = 0
    latency_ms: float = 0
    prompt_build_ms: float = 0
    request_ms: float = 0
    validation_ms: float = 0
    prompt_eval_count: int = 0
    prompt_eval_duration_ms: float = 0
    eval_count: int = 0
    eval_duration_ms: float = 0
    load_duration_ms: float = 0
    transport: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class SemanticPlannerOutcome:
    plan: SemanticPlan
    latency_ms: float
    diagnostics: PlannerDiagnostics


class SemanticPlannerFailure(ValueError):
    def __init__(self, diagnostics: PlannerDiagnostics) -> None:
        super().__init__(diagnostics.failure_detail or diagnostics.validation_result)
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class FactAnswer:
    request: SemanticFactRequest
    result: FactResult
    text: str
    timings: FactTimings
    planner_diagnostics: PlannerDiagnostics | None = None


@dataclass(frozen=True)
class SemanticMutationIntent:
    mutation: NamedWriteRequest
    timings: FactTimings
    planner_diagnostics: PlannerDiagnostics


class _FactFailure(RuntimeError):
    def __init__(
        self,
        status: FactStatus,
        *,
        evidence: FactEvidence = FactEvidence(),
        missing: tuple[str, ...] = (),
        candidates: tuple[Mapping[str, Any], ...] = (),
    ) -> None:
        super().__init__(status)
        self.status = status
        self.evidence = evidence
        self.missing = missing
        self.candidates = candidates


def _unique_entities(entities: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for entity in entities:
        entity_id = entity.get("id")
        if isinstance(entity_id, str):
            unique.setdefault(entity_id, dict(entity))
    return list(unique.values())


def _related_entity_id(edge: Mapping[str, Any]) -> str | None:
    related = edge.get("related_entity")
    if not isinstance(related, Mapping):
        return None
    return _string_or_none(related.get("id"))


def _entity_type(entity: Mapping[str, Any]) -> str:
    entity_id = str(entity.get("id", ""))
    return entity_id.partition(":")[0]


def _string_or_none(value: Any) -> str | None:
    return str(value) if value is not None else None


def _last_relation(reference: SemanticReference) -> str | None:
    return reference.path[-1].relation if reference.path else None
