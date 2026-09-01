"""Open-world, schema-aware household evidence planning and execution."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .display import resolve_display_name
from .schema_catalog import RuntimeSchemaCatalog, record_aliases
from .text import safe_log_token

GroundingStatus = Literal[
    "sufficient",
    "entity_not_found",
    "field_not_available",
    "evidence_insufficient",
    "evidence_stale",
    "tool_error",
    "timeout",
]
GroundingOperator = Literal[
    "count",
    "sum",
    "average",
    "min",
    "max",
    "difference",
    "ratio",
    "latest",
    "earliest",
    "date_difference",
    "duration",
    "annual_occurrence",
    "unit_conversion",
]
GroundedStopReason = Literal["answer", "tool_error", "timeout"]

logger = logging.getLogger("uvicorn.error.home_cortex.grounding")


@dataclass(frozen=True)
class GroundedAnswer:
    text: str
    tool_calls: int
    stop_reason: GroundedStopReason = "answer"


class _PlanModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
    )


class GroundingSubject(_PlanModel):
    anchor: Literal["authenticated_user", "configured_home", "named_entity"]
    reference: str = Field(min_length=1, max_length=256)
    expected_type: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")


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
    operator: Literal["eq", "ne", "lt", "lte", "gt", "gte"]
    value: Any


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
    mode: Literal["completed_years", "days", "seconds"] | None = None
    reference: Literal["household_today", "household_now"] | None = None
    from_unit: str | None = None
    to_unit: str | None = None

    @model_validator(mode="after")
    def require_operator_inputs(self) -> "TransformSpec":
        if (
            self.operator in {"sum", "average", "min", "max"}
            and self.field is None
        ):
            raise ValueError(f"{self.operator} requires field")
        if self.operator in {"difference", "ratio"} and (
            self.field is None or self.other_field is None
        ):
            raise ValueError(f"{self.operator} requires field and other_field")
        if self.operator in {"latest", "earliest"} and (
            self.field is None or self.order_by is None
        ):
            raise ValueError(f"{self.operator} requires field and order_by")
        if self.operator in {"date_difference", "duration"}:
            if self.field is None or self.reference is None or self.mode is None:
                raise ValueError(
                    f"{self.operator} requires field, reference, and mode"
                )
            if self.operator == "duration" and self.mode == "completed_years":
                raise ValueError("duration supports only days or seconds")
        if self.operator == "annual_occurrence":
            if self.field is None or self.reference != "household_today":
                raise ValueError(
                    "annual_occurrence requires field and household_today"
                )
            if self.mode not in {None, "days"}:
                raise ValueError(
                    "annual_occurrence supports a projected date or days"
                )
        if self.operator == "unit_conversion" and (
            self.field is None or self.from_unit is None or self.to_unit is None
        ):
            raise ValueError(
                "unit_conversion requires field, from_unit, and to_unit"
            )
        return self


class GroundingPlan(_PlanModel):
    requires_grounding: bool
    grounding_domain: Literal["household", "external_tool", "none"]
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
            if self.grounding_domain != "household" or self.subject is None:
                raise ValueError("grounded plans require a household subject")
            if not self.required_evidence:
                raise ValueError("grounded plans require explicit evidence")
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
            return _validate_planner_payload(payload)
        except (ValidationError, ValueError, TypeError) as error:
            validation_errors = (
                error.errors(include_input=False)
                if isinstance(error, ValidationError)
                else [{"type": type(error).__name__}]
            )
            repair_instruction = {
                "role": "system",
                "content": (
                    "The previous grounding plan failed structural validation. "
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
            return _validate_planner_payload(repaired)


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


def _validate_planner_payload(payload: Mapping[str, Any]) -> GroundingPlan:
    completed = _complete_evidence_requirements(payload)
    return GroundingPlan.model_validate_json(json.dumps(completed))


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
    subject = completed.get("subject")
    if isinstance(subject, Mapping):
        normalized_subject = dict(subject)
        reference = normalized_subject.get("reference")
        normalized_reference = (
            reference.casefold().strip(" \t\r\n?!？。！")
            if isinstance(reference, str)
            else ""
        )
        if normalized_reference in {
            "i",
            "me",
            "myself",
            "authenticated user",
            "current user",
            "我",
            "本人",
            "当前用户",
        }:
            normalized_subject["anchor"] = "authenticated_user"
            normalized_subject["expected_type"] = "person"
        elif normalized_reference in {
            "home",
            "my home",
            "our home",
            "the home",
            "家",
            "家里",
            "家中",
            "我家",
            "我们家",
        }:
            normalized_subject["anchor"] = "configured_home"
            normalized_subject["expected_type"] = "address"
        completed["subject"] = normalized_subject
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
                require_field(
                    str(operation.get("source", "entity")),
                    operation.get("field"),
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
        caller_entity_id: str | None,
        household_now: datetime,
    ) -> GroundingEvidence:
        execution = _PlanExecution(
            self.dispatcher,
            caller_entity_id=caller_entity_id,
            maximum=self.max_tool_calls,
            timeout_seconds=self.timeout_seconds,
        )
        try:
            subjects = await self._resolve_subject(
                plan.subject,
                execution,
                caller_entity_id,
            )
            if not subjects:
                return GroundingEvidence(
                    "entity_not_found",
                    tool_calls=execution.tool_calls,
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
                    )

            needs_entity_records = _needs_entity_records(plan)
            loaded_records = (
                await self._load_records(subjects, execution)
                if needs_entity_records
                else []
            )
            if needs_entity_records and not loaded_records:
                return GroundingEvidence(
                    "entity_not_found",
                    tool_calls=execution.tool_calls,
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
                )
            _validate_transform_inputs(
                records,
                terminal_edges,
                plan.transform,
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
            )
        except _PlanFailure as error:
            return GroundingEvidence(
                error.status,
                tool_calls=execution.tool_calls,
            )

    async def _resolve_subject(
        self,
        subject: GroundingSubject | None,
        execution: "_PlanExecution",
        caller_entity_id: str | None,
    ) -> list[Mapping[str, Any]]:
        if subject is None:
            return []
        if not self.catalog.has_entity_type(subject.expected_type):
            raise _PlanFailure("evidence_insufficient")
        if subject.anchor == "authenticated_user":
            if not _entity_matches_type(caller_entity_id, subject.expected_type):
                return []
            return [{"id": caller_entity_id}]
        if subject.anchor == "configured_home":
            if not _entity_matches_type(
                self.home_entity_id,
                subject.expected_type,
            ):
                return []
            return [{"id": self.home_entity_id}]
        result = await execution.call(
            "search_entities",
            {
                "text": subject.reference,
                "entity_type": subject.expected_type,
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
            return []
        return exact

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
        caller_entity_id: str | None,
        household_now: datetime,
        language: str,
        request_id: str = "-",
    ) -> GroundedAnswer | None:
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
            caller_entity_id=caller_entity_id,
            household_now=household_now,
        )
        if not evidence.sufficient:
            _log_grounding_plan(request_id, plan, evidence.status)
            return GroundedAnswer(
                _grounding_status_answer(evidence, plan, language),
                evidence.tool_calls,
                "timeout" if evidence.status == "timeout" else (
                    "tool_error" if evidence.status == "tool_error" else "answer"
                ),
            )
        text = _deterministic_evidence_answer(plan, evidence, language)
        _log_grounding_plan(request_id, plan, evidence.status)
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
        if any(record.get(item.field) is None for record in source_records):
            raise _PlanFailure("evidence_insufficient")


def _compare(left: Any, operator: str, right: Any) -> bool:
    try:
        if operator == "eq":
            return left == right
        if operator == "ne":
            return left != right
        if operator == "lt":
            return left < right
        if operator == "lte":
            return left <= right
        if operator == "gt":
            return left > right
        if operator == "gte":
            return left >= right
        return False
    except TypeError:
        return False


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
            if requirement.field in record and record.get(requirement.field) is not None
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


def _validate_transform_inputs(
    records: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    transform: TransformSpec | None,
) -> None:
    if transform is None or transform.operator == "count":
        return
    source_records = records if transform.source == "entity" else edges
    if transform.operator in {"sum", "average", "min", "max"}:
        field = transform.field
        if field is None or not source_records or any(
            not _is_number(record.get(field)) for record in source_records
        ):
            raise _PlanFailure("evidence_insufficient")
        return
    if transform.operator in {"latest", "earliest"}:
        if (
            not source_records
            or transform.field is None
            or source_records[0].get(transform.field) is None
            or transform.order_by is None
            or source_records[0].get(transform.order_by) is None
        ):
            raise _PlanFailure("evidence_insufficient")
        return
    if len(source_records) != 1:
        raise _PlanFailure("evidence_insufficient")


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
    if transform.operator == "count":
        return len(source_records)
    field = transform.field
    values = (
        [record.get(field) for record in source_records]
        if field is not None
        else []
    )
    values = [value for value in values if value is not None]
    if transform.operator in {"latest", "earliest"}:
        if not source_records or field is None:
            raise _PlanFailure("evidence_insufficient")
        return source_records[0].get(field)
    if transform.operator in {"sum", "average", "min", "max"}:
        numeric = _numeric_values(values)
        if not numeric:
            raise _PlanFailure("evidence_insufficient")
        if transform.operator == "sum":
            return sum(numeric)
        if transform.operator == "average":
            return sum(numeric) / len(numeric)
        return min(numeric) if transform.operator == "min" else max(numeric)
    if transform.operator in {"difference", "ratio"}:
        if field is None or transform.other_field is None or not source_records:
            raise _PlanFailure("evidence_insufficient")
        left = source_records[0].get(field)
        right = source_records[0].get(transform.other_field)
        if not _is_number(left) or not _is_number(right):
            raise _PlanFailure("evidence_insufficient")
        if transform.operator == "ratio":
            if right == 0:
                raise _PlanFailure("evidence_insufficient")
            return left / right
        return left - right
    if transform.operator in {"date_difference", "duration"}:
        if not values:
            raise _PlanFailure("evidence_insufficient")
        return _date_transform(values[0], transform, household_now)
    if transform.operator == "annual_occurrence":
        if not values:
            raise _PlanFailure("evidence_insufficient")
        return _annual_occurrence(values[0], transform, household_now)
    if transform.operator == "unit_conversion":
        if not values or not _is_number(values[0]):
            raise _PlanFailure("evidence_insufficient")
        return _convert_unit(values[0], transform.from_unit, transform.to_unit)
    raise _PlanFailure("evidence_insufficient")


def _date_transform(value: Any, transform: TransformSpec, now: datetime) -> Any:
    if not isinstance(value, str):
        raise _PlanFailure("evidence_insufficient")
    try:
        stored_date = date.fromisoformat(value)
    except ValueError:
        parsed = _parse_datetime(value)
        if parsed is None:
            raise _PlanFailure("evidence_insufficient")
        delta = now.astimezone(timezone.utc) - parsed
        return delta.days if transform.mode == "days" else delta.total_seconds()
    if transform.mode == "completed_years":
        today = now.date()
        if stored_date > today:
            raise _PlanFailure("evidence_insufficient")
        passed = (today.month, today.day) >= (stored_date.month, stored_date.day)
        return today.year - stored_date.year - (not passed)
    delta = now.date() - stored_date
    return delta.days if transform.mode == "days" else delta.total_seconds()


def _annual_occurrence(
    value: Any,
    transform: TransformSpec,
    now: datetime,
) -> str | int:
    if not isinstance(value, str):
        raise _PlanFailure("evidence_insufficient")
    try:
        stored = date.fromisoformat(value)
    except ValueError:
        parsed = _parse_datetime(value)
        if parsed is None:
            raise _PlanFailure("evidence_insufficient")
        stored = parsed.date()
    today = now.date()
    year = today.year
    while year <= today.year + 8:
        try:
            occurrence = date(year, stored.month, stored.day)
        except ValueError:
            year += 1
            continue
        if occurrence >= today:
            return (
                (occurrence - today).days
                if transform.mode == "days"
                else occurrence.isoformat()
            )
        year += 1
    raise _PlanFailure("evidence_insufficient")


def _numeric_values(values: Sequence[Any]) -> list[float | int]:
    return [value for value in values if _is_number(value)]


def _is_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _convert_unit(value: float | int, source: str | None, target: str | None) -> float:
    conversions = {
        ("c", "f"): lambda number: number * 9 / 5 + 32,
        ("f", "c"): lambda number: (number - 32) * 5 / 9,
        ("kg", "lb"): lambda number: number * 2.2046226218,
        ("lb", "kg"): lambda number: number / 2.2046226218,
        ("cm", "in"): lambda number: number / 2.54,
        ("in", "cm"): lambda number: number * 2.54,
    }
    if source == target and source is not None:
        return float(value)
    conversion = conversions.get(((source or "").casefold(), (target or "").casefold()))
    if conversion is None:
        raise _PlanFailure("evidence_insufficient")
    return conversion(value)


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
    reference = plan.subject.reference if plan.subject is not None else ""
    if language == "zh":
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
    rendered = json.dumps(value, ensure_ascii=False, default=str)
    return (
        f"根据家庭资料，查询结果是：{rendered}。"
        if language == "zh"
        else f"According to the home graph, the result is: {rendered}."
    )


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
    status: str,
) -> None:
    subject_type = plan.subject.expected_type if plan.subject is not None else "none"
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
        "grounding_plan request_id=%s grounding_required=true subject_type=%s "
        "fields=%s relations=%s operator=%s evidence_status=%s",
        safe_log_token(request_id),
        safe_log_token(subject_type),
        safe_log_token(",".join(fields) or "none"),
        safe_log_token(",".join(relations) or "none"),
        safe_log_token(operator),
        safe_log_token(status),
    )


def _log_grounding(request_id: str, *, required: str, status: str) -> None:
    logger.info(
        "grounding_plan request_id=%s grounding_required=%s evidence_status=%s",
        safe_log_token(request_id),
        safe_log_token(required),
        safe_log_token(status),
    )
