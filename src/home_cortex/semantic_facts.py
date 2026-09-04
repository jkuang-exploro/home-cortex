"""Semantic household fact IR, deterministic execution, and Tier-0 parsing."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from time import perf_counter
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .display import resolve_display_name
from .grounding import AgentRequestContext, GroundedAnswer
from .schema_catalog import RuntimeSchemaCatalog
from .text import safe_log_token

logger = logging.getLogger("uvicorn.error.home_cortex.semantic_facts")

FactStatus = Literal[
    "found",
    "caller_context_missing",
    "entity_not_found",
    "relationship_not_found",
    "property_unavailable",
    "ambiguous",
    "computation_impossible",
]
FactOperation = Literal[
    "resolve_entity",
    "get_property",
    "list",
    "count",
    "exists",
    "argmin",
    "argmax",
    "completed_years",
    "compare",
]
ReferenceKind = Literal[
    "self",
    "assistant",
    "current_household",
    "named_entity",
    "entity_id",
]


class _SemanticModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class SemanticFilter(_SemanticModel):
    property: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    operator: Literal["eq"] = "eq"
    value: str | int | float | bool
    source: Literal["entity", "relation"] = "entity"


class SemanticRelationStep(_SemanticModel):
    relation: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    filters: tuple[SemanticFilter, ...] = ()


class SemanticReference(_SemanticModel):
    kind: ReferenceKind
    value: str | None = Field(default=None, max_length=256)
    entity_type: str | None = Field(default=None, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    path: tuple[SemanticRelationStep, ...] = ()

    @model_validator(mode="after")
    def validate_value(self) -> "SemanticReference":
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
    )
    other: SemanticReference | None = None
    comparison: Literal["older", "younger"] | None = None

    @model_validator(mode="after")
    def validate_operation(self) -> "SemanticFactRequest":
        if self.operation in {"get_property", "argmin", "argmax"} and not self.property:
            raise ValueError(f"{self.operation} requires property")
        if self.operation == "completed_years" and self.property is None:
            object.__setattr__(self, "property", "birth_date")
        if self.operation == "compare" and (
            self.other is None or self.comparison is None
        ):
            raise ValueError("compare requires other and comparison")
        return self


class SemanticPlan(_SemanticModel):
    """The only structured output accepted from the Tier-1 interpreter."""

    requires_fact: bool
    request: SemanticFactRequest | None = None

    @model_validator(mode="after")
    def validate_request_presence(self) -> "SemanticPlan":
        if self.requires_fact != (self.request is not None):
            raise ValueError("requires_fact must match request presence")
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
class FactResult:
    status: FactStatus
    value: Any = None
    evidence: FactEvidence = FactEvidence()
    missing_requirements: tuple[str, ...] = ()
    candidates: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class FactTimings:
    tier: int
    routing_ms: float = 0
    semantic_parse_ms: float = 0
    entity_resolution_ms: float = 0
    fact_query_ms: float = 0
    computation_ms: float = 0
    render_ms: float = 0
    llm_ms: float = 0
    total_ms: float = 0
    llm_call_count: int = 0
    db_query_count: int = 0


@dataclass(frozen=True)
class FactAnswer:
    request: SemanticFactRequest
    result: FactResult
    text: str
    timings: FactTimings


@dataclass(frozen=True)
class SemanticAttempt:
    """Tri-state route result: unclaimed, handled answer, or ordinary request."""

    claimed: bool
    answer: FactAnswer | None = None


class SemanticSchemaRegistry:
    """Map stable semantic concepts to deployment-specific schema names."""

    PROPERTY_CANDIDATES: Mapping[str, tuple[str, ...]] = {
        "display_name": ("display_name", "name", "full_name"),
        "birth_date": ("birth_date", "birthday", "dob", "date_of_birth"),
        "gender": ("gender", "sex"),
        "household_role": ("household_role", "member_role", "role"),
    }
    RELATION_CANDIDATES: Mapping[str, tuple[str, ...]] = {
        "spouse": ("spouse_of", "spouse"),
        "child": ("parent_of",),
        "parent": ("parent_of", "child_of"),
        "member": ("lives_in", "member_of"),
    }

    def __init__(self, catalog: RuntimeSchemaCatalog) -> None:
        self.catalog = catalog
        self._property_cache: dict[tuple[str, str], str | None] = {}
        self._capability_cache: dict[str, Any] | None = None

    def physical_property(self, entity_type: str, semantic: str) -> str | None:
        marker = (entity_type, semantic)
        if marker in self._property_cache:
            return self._property_cache[marker]
        schema = self.catalog.entities.get(entity_type)
        available = set(schema.properties) if schema else set()
        candidates = self.PROPERTY_CANDIDATES.get(semantic, (semantic,))
        physical = next((field for field in candidates if field in available), None)
        self._property_cache[marker] = physical
        return physical

    def physical_relation(self, semantic: str) -> tuple[str, str | None] | None:
        for candidate in self.RELATION_CANDIDATES.get(semantic, (semantic,)):
            if self.catalog.has_relation(candidate):
                direction = None
                if semantic == "child":
                    direction = "out"
                elif semantic == "parent":
                    direction = "in"
                elif semantic == "member":
                    direction = "in"
                return candidate, direction
        return None

    def relation_property(self, relation: str, semantic: str) -> str | None:
        schema = self.catalog.relations.get(relation)
        available = set(schema.properties) if schema else set()
        candidates = self.PROPERTY_CANDIDATES.get(semantic, (semantic,))
        return next((field for field in candidates if field in available), None)

    def capability_payload(self) -> dict[str, Any]:
        if self._capability_cache is not None:
            return self._capability_cache
        semantic_properties = {
            entity_type: sorted(self.semantic_properties(entity_type))
            for entity_type in self.catalog.entities
        }
        relations = {
            semantic
            for semantic in self.RELATION_CANDIDATES
            if self.physical_relation(semantic) is not None
        }
        aliased_relations = {
            physical
            for candidates in self.RELATION_CANDIDATES.values()
            for physical in candidates
        }
        relations.update(set(self.catalog.relations) - aliased_relations)
        self._capability_cache = {
            "entity_types": sorted(self.catalog.entities),
            "semantic_properties": semantic_properties,
            "semantic_relations": sorted(relations),
            "semantic_relation_properties": {
                relation: sorted(self.semantic_relation_properties(relation))
                for relation in relations
            },
            "operations": list(get_args(FactOperation)),
        }
        return self._capability_cache

    def semantic_properties(self, entity_type: str) -> frozenset[str]:
        schema = self.catalog.entities.get(entity_type)
        if schema is None:
            return frozenset()
        properties = {
            semantic
            for semantic in self.PROPERTY_CANDIDATES
            if self.physical_property(entity_type, semantic) is not None
        }
        aliased_physical = {
            physical
            for candidates in self.PROPERTY_CANDIDATES.values()
            for physical in candidates
        }
        properties.update(
            physical
            for physical in schema.properties
            if physical != "id" and physical not in aliased_physical
        )
        return frozenset(properties)

    def semantic_relation_properties(self, semantic: str) -> frozenset[str]:
        resolved = self.physical_relation(semantic)
        if resolved is None:
            return frozenset()
        relation = resolved[0]
        schema = self.catalog.relations.get(relation)
        if schema is None:
            return frozenset()
        properties = {
            semantic_property
            for semantic_property in self.PROPERTY_CANDIDATES
            if self.relation_property(relation, semantic_property) is not None
        }
        aliased_physical = {
            physical
            for candidates in self.PROPERTY_CANDIDATES.values()
            for physical in candidates
        }
        properties.update(
            physical
            for physical in schema.properties
            if physical not in aliased_physical
        )
        return frozenset(properties)

    def semantic_property_kinds(self, semantic: str) -> frozenset[str]:
        return frozenset(
            self.catalog.entity_field_type(entity_type, physical)
            for entity_type in self.catalog.entities
            if (physical := self.physical_property(entity_type, semantic)) is not None
        )

    def validates(self, request: SemanticFactRequest) -> bool:
        """Reject model plans outside the advertised semantic protocol."""
        references = (request.subject,) + ((request.other,) if request.other else ())
        for reference in references:
            if reference.kind == "assistant" and reference.path:
                return False
            if reference.entity_type is not None and not self.catalog.has_entity_type(
                reference.entity_type
            ):
                return False
            if any(self.physical_relation(step.relation) is None for step in reference.path):
                return False
            for step in reference.path:
                for item in step.filters:
                    if item.source == "entity" and not any(
                        item.property in self.semantic_properties(entity_type)
                        for entity_type in self.catalog.entities
                    ):
                        return False
                    if item.source == "relation":
                        resolved = self.physical_relation(step.relation)
                        if resolved is None or self.relation_property(
                            resolved[0], item.property
                        ) is None:
                            return False
        if request.property is not None and not any(
            request.property in self.semantic_properties(entity_type)
            for entity_type in self.catalog.entities
        ):
            return False
        property_kinds = (
            self.semantic_property_kinds(request.property)
            if request.property is not None
            else frozenset()
        )
        if request.operation == "completed_years" and not property_kinds.intersection(
            {"date", "datetime"}
        ):
            return False
        if request.operation in {"argmin", "argmax"} and not property_kinds.intersection(
            {"integer", "number", "date", "datetime"}
        ):
            return False
        if request.operation == "compare" and "birth_date" not in set().union(
            *(self.semantic_properties(item) for item in self.catalog.entities)
        ):
            return False
        return True


class SemanticFactPlanner:
    """One-call semantic interpreter that never sees storage field names."""

    def __init__(self, ollama: Any, schema: SemanticSchemaRegistry) -> None:
        self.ollama = ollama
        self.schema = schema

    async def plan(
        self,
        messages: Sequence[Mapping[str, Any]],
        context: AgentRequestContext,
    ) -> tuple[SemanticPlan, float]:
        started = perf_counter()
        payload = await self.ollama.plan_semantic_fact(
            messages,
            self.schema.capability_payload(),
            SemanticPlan.model_json_schema(),
            household_now=context.current_time.isoformat(),
        )
        return SemanticPlan.model_validate(payload), (perf_counter() - started) * 1000


class TierZeroSemanticParser:
    """Parse a bounded vocabulary into composable IR, never whole-query handlers."""

    _RELATIONS: tuple[tuple[tuple[str, ...], str, str | None], ...] = (
        (("老婆", "妻子", "wife"), "spouse", "female"),
        (("老公", "丈夫", "husband"), "spouse", "male"),
        (("儿子", "son"), "child", "male"),
        (("女儿", "daughter"), "child", "female"),
        (("孩子", "小孩", "children", "child"), "child", None),
        (("爸爸", "父亲", "father"), "parent", "male"),
        (("妈妈", "母亲", "mother"), "parent", "female"),
    )

    def parse(self, text: str) -> SemanticFactRequest | None:
        normalized = _normalize_request(text)
        if not normalized:
            return None
        if _is_household_oldest(normalized):
            return SemanticFactRequest(
                operation="argmin",
                subject=_household_members(),
                property="birth_date",
            )
        comparison = self._parse_comparison(normalized)
        if comparison is not None:
            return comparison
        household = _mentions_household(normalized)
        if household and _asks_household_children(normalized):
            return SemanticFactRequest(
                operation="count",
                subject=SemanticReference(
                    kind="current_household",
                    entity_type="address",
                    path=(
                        SemanticRelationStep(
                            relation="member",
                            filters=(
                                SemanticFilter(
                                    property="household_role",
                                    value="minor_dependent",
                                    source="relation",
                                ),
                            ),
                        ),
                    ),
                ),
            )
        if household and _asks_count(normalized):
            return SemanticFactRequest(operation="count", subject=_household_members())
        if household and _asks_list(normalized):
            return SemanticFactRequest(operation="list", subject=_household_members())

        property_name, stem = _extract_property(normalized)
        if property_name == "age":
            reference = self._parse_reference(stem)
            return (
                SemanticFactRequest(
                    operation="completed_years",
                    subject=reference,
                    property="birth_date",
                )
                if reference is not None
                else None
            )
        if property_name is not None:
            reference = self._parse_reference(stem)
            return (
                SemanticFactRequest(
                    operation="get_property",
                    subject=reference,
                    property=property_name,
                )
                if reference is not None
                else None
            )
        if _asks_count(normalized):
            stem = _strip_count_syntax(normalized)
            reference = self._parse_reference(stem)
            return (
                SemanticFactRequest(operation="count", subject=reference)
                if reference is not None and reference.path
                else None
            )
        if _asks_identity(normalized):
            reference = self._parse_reference(_strip_identity_syntax(normalized))
            return (
                SemanticFactRequest(operation="resolve_entity", subject=reference)
                if reference is not None
                else None
            )
        return None

    def _parse_reference(self, text: str) -> SemanticReference | None:
        normalized = text.strip("的 ")
        base: ReferenceKind
        remainder: str
        if normalized.startswith(("我", "my ")) or normalized in {"i", "me"}:
            base = "self"
            remainder = (
                normalized[1:]
                if normalized.startswith("我")
                else normalized.removeprefix("my ")
                if normalized.startswith("my ")
                else ""
            )
        elif normalized.startswith(("你", "您", "your ")) or normalized == "you":
            base = "assistant"
            remainder = (
                normalized[1:]
                if normalized[:1] in {"你", "您"}
                else normalized.removeprefix("your ")
                if normalized.startswith("your ")
                else ""
            )
        else:
            return None
        remainder = remainder.strip("的 ")
        steps: list[SemanticRelationStep] = []
        while remainder:
            matched = False
            for aliases, relation, gender in self._RELATIONS:
                alias = next((item for item in aliases if remainder.startswith(item)), None)
                if alias is None:
                    continue
                filters = (
                    (SemanticFilter(property="gender", value=gender),)
                    if gender is not None
                    else ()
                )
                steps.append(SemanticRelationStep(relation=relation, filters=filters))
                remainder = remainder[len(alias) :].strip("的 ")
                matched = True
                break
            if not matched:
                return None
        return SemanticReference(kind=base, entity_type="person", path=tuple(steps))

    def _parse_comparison(self, text: str) -> SemanticFactRequest | None:
        suffix = next(
            (
                item
                for item in ("谁年龄大", "谁年纪大", "谁大", "who is older")
                if text.endswith(item)
            ),
            None,
        )
        if suffix is None:
            return None
        operands = text[: -len(suffix)].strip("，, ?？")
        parts = re.split(r"和|与|跟| and ", operands, maxsplit=1)
        if len(parts) != 2:
            return None
        left = self._parse_reference(parts[0])
        right = self._parse_reference(parts[1])
        if left is None or right is None:
            return None
        return SemanticFactRequest(
            operation="compare",
            subject=left,
            other=right,
            comparison="older",
        )


class HouseholdFactEngine:
    def __init__(
        self,
        dispatcher: Any,
        schema: SemanticSchemaRegistry,
        *,
        max_records: int = 25,
    ) -> None:
        self.dispatcher = dispatcher
        self.schema = schema
        self.max_records = max_records

    async def execute(
        self,
        request: SemanticFactRequest,
        context: AgentRequestContext,
    ) -> tuple[FactResult, int, float, float]:
        execution = _FactExecution(self.dispatcher, context.caller_entity_id)
        resolution_started = perf_counter()
        allow_empty_collection = request.operation in {"count", "list", "exists"}
        try:
            entities = await self._resolve_reference(
                request.subject,
                context,
                execution,
                allow_empty_collection=allow_empty_collection,
            )
            other_entities = (
                await self._resolve_reference(
                    request.other,
                    context,
                    execution,
                    allow_empty_collection=False,
                )
                if request.other is not None
                else []
            )
        except _FactFailure as error:
            return (
                FactResult(
                    error.status,
                    evidence=error.evidence,
                    missing_requirements=error.missing,
                ),
                execution.query_count,
                (perf_counter() - resolution_started) * 1000,
                0,
            )
        entity_resolution_ms = (perf_counter() - resolution_started) * 1000
        computation_started = perf_counter()
        try:
            result = await self._operate(
                request,
                entities,
                other_entities,
                context,
                execution,
            )
        except _FactFailure as error:
            result = FactResult(
                error.status,
                evidence=error.evidence,
                missing_requirements=error.missing,
            )
        computation_ms = (perf_counter() - computation_started) * 1000
        return result, execution.query_count, entity_resolution_ms, computation_ms

    async def _resolve_reference(
        self,
        reference: SemanticReference | None,
        context: AgentRequestContext,
        execution: "_FactExecution",
        *,
        allow_empty_collection: bool,
    ) -> list[dict[str, Any]]:
        if reference is None:
            return []
        if reference.kind == "assistant":
            entities = [
                {"id": context.assistant_id, "display_name": context.assistant_display_name}
            ]
        elif reference.kind == "self":
            if context.caller_entity_id is None:
                raise _FactFailure("caller_context_missing")
            entities = [{"id": context.caller_entity_id}]
        elif reference.kind == "current_household":
            if context.household_id is None:
                raise _FactFailure("entity_not_found")
            entities = [{"id": context.household_id}]
        elif reference.kind == "entity_id":
            entities = [{"id": reference.value}]
        else:
            records = await execution.records(
                "search_entities",
                {
                    "text": reference.value,
                    "entity_type": reference.entity_type,
                    "limit": self.max_records,
                },
            )
            if not records:
                raise _FactFailure("entity_not_found")
            entities = records

        last_relation: str | None = None
        for step in reference.path:
            resolved = self.schema.physical_relation(step.relation)
            if resolved is None:
                raise _FactFailure(
                    "relationship_not_found",
                    evidence=FactEvidence(relationship=step.relation),
                )
            relation, direction = resolved
            last_relation = step.relation
            related: list[dict[str, Any]] = []
            for entity in entities:
                entity_id = entity.get("id")
                if not isinstance(entity_id, str):
                    continue
                arguments: dict[str, Any] = {
                    "entity_id": entity_id,
                    "relation": relation,
                    "include_ended": False,
                    "limit": self.max_records,
                }
                if direction is not None:
                    arguments["direction"] = direction
                edges = await execution.records("get_relationships", arguments)
                for edge in edges:
                    if not self._edge_matches(edge, relation, step.filters):
                        continue
                    execution.remember_relationship(step.relation, edge)
                    candidate = edge.get("related_entity")
                    if isinstance(candidate, Mapping):
                        related.append(dict(candidate))
            entities = _unique_entities(related)
            entities = await self._filter_entities(entities, step.filters, execution)
            if not entities:
                if allow_empty_collection:
                    return []
                raise _FactFailure(
                    "relationship_not_found",
                    evidence=FactEvidence(relationship=last_relation),
                )
        return entities

    def _edge_matches(
        self,
        edge: Mapping[str, Any],
        relation: str,
        filters: Sequence[SemanticFilter],
    ) -> bool:
        for item in filters:
            if item.source != "relation":
                continue
            physical = self.schema.relation_property(relation, item.property)
            if physical is None:
                raise _FactFailure(
                    "property_unavailable",
                    evidence=FactEvidence(semantic_property=item.property),
                    missing=(item.property,),
                )
            if edge.get(physical) != item.value:
                return False
        return True

    async def _filter_entities(
        self,
        entities: list[dict[str, Any]],
        filters: Sequence[SemanticFilter],
        execution: "_FactExecution",
    ) -> list[dict[str, Any]]:
        entity_filters = [item for item in filters if item.source == "entity"]
        if not entity_filters:
            return entities
        matched: list[dict[str, Any]] = []
        for entity in entities:
            entity_type = _entity_type(entity)
            mapped = [
                (item, self.schema.physical_property(entity_type, item.property))
                for item in entity_filters
            ]
            unavailable = next(
                (item.property for item, physical in mapped if physical is None),
                None,
            )
            if unavailable is not None:
                raise _FactFailure(
                    "property_unavailable",
                    evidence=FactEvidence(semantic_property=unavailable),
                    missing=(unavailable,),
                )
            needs_load = any(
                physical not in entity for _, physical in mapped if physical is not None
            )
            record = await self._load(entity, execution) if needs_load else dict(entity)
            if all(record.get(physical) == item.value for item, physical in mapped):
                matched.append(record)
        return matched

    async def _operate(
        self,
        request: SemanticFactRequest,
        entities: list[dict[str, Any]],
        other_entities: list[dict[str, Any]],
        context: AgentRequestContext,
        execution: "_FactExecution",
    ) -> FactResult:
        evidence = FactEvidence(
            tuple(
                str(item.get("id"))
                for item in (*entities, *other_entities)
                if item.get("id")
            ),
            request.subject.path[-1].relation if request.subject.path else None,
            request.property,
            tuple(execution.relationship_evidence),
        )
        if request.operation == "count":
            return FactResult("found", len(entities), evidence)
        if request.operation == "exists":
            return FactResult("found", bool(entities), evidence)
        if request.operation == "list":
            visible = [await self._load_if_unnamed(item, execution) for item in entities]
            return FactResult("found", visible, evidence)
        if request.operation == "resolve_entity":
            singular = await self._singular(
                entities,
                request,
                execution,
                load_full=False,
            )
            if isinstance(singular, FactResult):
                return singular
            return FactResult("found", singular, evidence)
        if request.operation in {"get_property", "completed_years"}:
            singular = await self._singular(entities, request, execution)
            if isinstance(singular, FactResult):
                return singular
            property_result = self._property(singular, request.property or "birth_date")
            if isinstance(property_result, FactResult):
                return property_result
            if request.operation == "completed_years":
                years = _completed_years(property_result, context.current_time.date())
                if years is None:
                    return FactResult(
                        "computation_impossible",
                        evidence=evidence,
                        missing_requirements=(request.property or "birth_date",),
                    )
                return FactResult("found", years, evidence)
            return FactResult("found", property_result, evidence)
        if request.operation in {"argmin", "argmax"}:
            candidates: list[tuple[Any, dict[str, Any]]] = []
            for entity in entities:
                record = await self._load(entity, execution)
                value = self._property(record, request.property or "")
                if isinstance(value, FactResult):
                    return FactResult(
                        "computation_impossible",
                        evidence=evidence,
                        missing_requirements=(request.property or "",),
                    )
                ordered = _ordered_value(value)
                if ordered is None:
                    return FactResult("computation_impossible", evidence=evidence)
                candidates.append((ordered, record))
            if not candidates:
                return FactResult("relationship_not_found", evidence=evidence)
            selected = (min if request.operation == "argmin" else max)(
                candidates,
                key=lambda item: item[0],
            )[1]
            return FactResult("found", selected, evidence)
        if request.operation == "compare":
            left = await self._singular(entities, request, execution)
            right = await self._singular(other_entities, request, execution)
            if isinstance(left, FactResult):
                return left
            if isinstance(right, FactResult):
                return right
            left_date = _parse_date(self._property(left, "birth_date"))
            right_date = _parse_date(self._property(right, "birth_date"))
            if left_date is None or right_date is None:
                return FactResult(
                    "computation_impossible",
                    evidence=evidence,
                    missing_requirements=("birth_date",),
                )
            older = left if left_date < right_date else right
            younger = right if older is left else left
            selected = older if request.comparison == "older" else younger
            other = younger if selected is older else older
            return FactResult(
                "found",
                {"selected": selected, "other": other, "equal": left_date == right_date},
                evidence,
            )
        return FactResult("computation_impossible", evidence=evidence)

    async def _singular(
        self,
        entities: list[dict[str, Any]],
        request: SemanticFactRequest,
        execution: "_FactExecution",
        *,
        load_full: bool = True,
    ) -> dict[str, Any] | FactResult:
        if not entities:
            status: FactStatus = (
                "relationship_not_found" if request.subject.path else "entity_not_found"
            )
            return FactResult(status)
        if len(entities) > 1:
            loaded = tuple(
                [
                    await self._load_if_unnamed(item, execution)
                    for item in entities
                ]
            )
            return FactResult("ambiguous", candidates=loaded)
        record = (
            await self._load(entities[0], execution)
            if load_full
            else await self._load_if_unnamed(entities[0], execution)
        )
        return record

    async def _load_if_unnamed(
        self,
        entity: Mapping[str, Any],
        execution: "_FactExecution",
    ) -> dict[str, Any]:
        if any(entity.get(field) for field in ("display_name", "name", "full_name")):
            return dict(entity)
        return await self._load(entity, execution)

    async def _load(
        self,
        entity: Mapping[str, Any],
        execution: "_FactExecution",
    ) -> dict[str, Any]:
        entity_id = entity.get("id")
        if not isinstance(entity_id, str):
            raise _FactFailure("entity_not_found")
        if not ":" in entity_id:
            return dict(entity)
        records = await execution.records("get_entity", {"entity_id": entity_id})
        if not records:
            raise _FactFailure("entity_not_found")
        return dict(records[0])

    def _property(self, entity: Mapping[str, Any], semantic: str) -> Any | FactResult:
        entity_type = _entity_type(entity)
        physical = self.schema.physical_property(entity_type, semantic)
        if physical is None or physical not in entity or entity.get(physical) is None:
            return FactResult(
                "property_unavailable",
                evidence=FactEvidence(
                    (str(entity.get("id")),) if entity.get("id") else (),
                    semantic_property=semantic,
                ),
                missing_requirements=(semantic,),
            )
        return entity[physical]


class FactRenderer:
    def render(
        self,
        request: SemanticFactRequest,
        result: FactResult,
        context: AgentRequestContext,
    ) -> str:
        language = context.locale or "en"
        if language.startswith("zh"):
            return self._zh(request, result, context)
        return self._en(request, result, context)

    def _zh(
        self,
        request: SemanticFactRequest,
        result: FactResult,
        context: AgentRequestContext,
    ) -> str:
        if result.status == "caller_context_missing":
            return "我无法确认当前登录者的身份。"
        if result.status == "entity_not_found":
            return "家庭资料中没有找到对应的人或实体。"
        if result.status == "relationship_not_found":
            relation = _relation_label(
                request.subject,
                result.evidence.relationship,
            )
            return f"家庭资料中没有找到{relation}关系记录。"
        if result.status == "property_unavailable":
            label = _property_label(result.missing_requirements)
            return (
                "家庭资料中有对应记录，但目前没有记录"
                f"{_subject_possessive(request.subject)}{label}。"
            )
        if result.status == "ambiguous":
            names = "、".join(_name(item, "zh") for item in result.candidates)
            return f"找到多个符合条件的家庭成员：{names}。请说明您指哪一位。"
        if result.status == "computation_impossible":
            label = _property_label(result.missing_requirements)
            suffix = f"，缺少{label}" if label else ""
            return f"家庭资料不足以完成这项计算{suffix}。"
        if request.operation == "count":
            count = int(result.value)
            noun = _count_noun(request.subject)
            if request.subject.kind == "current_household":
                return f"家里目前有{_zh_number(count)}{noun}。"
            return f"您目前有{_zh_number(count)}{noun}。"
        if request.operation == "list":
            values = result.value if isinstance(result.value, list) else []
            names = "、".join(_name(item, "zh") for item in values)
            return f"家里目前的成员有：{names}。"
        if request.operation == "exists":
            return "有相关记录。" if result.value else "没有相关记录。"
        if request.operation == "get_property":
            if request.property == "birth_date":
                return f"{_subject_possessive(request.subject)}生日是{result.value}。"
            return f"查询到的值是{result.value}。"
        if request.operation == "completed_years":
            return f"{_subject_nominative(request.subject)}今年{result.value}岁。"
        if request.operation in {"argmin", "argmax"}:
            if request.property == "birth_date":
                qualifier = "最年长" if request.operation == "argmin" else "最年轻"
                return f"家里{qualifier}的是{_name(result.value, 'zh')}。"
            return f"符合极值条件的是{_name(result.value, 'zh')}。"
        if request.operation == "compare" and isinstance(result.value, Mapping):
            selected = _name(result.value.get("selected"), "zh")
            other = _name(result.value.get("other"), "zh")
            if result.value.get("equal"):
                return f"{selected}和{other}年龄相同。"
            adjective = "大" if request.comparison == "older" else "小"
            return f"{selected}年龄比{other}{adjective}。"
        if request.subject.kind == "assistant":
            return f"我是{context.assistant_display_name}。"
        name = _name(result.value, "zh")
        if request.subject.kind == "self" and not request.subject.path:
            return f"您是{name}。"
        return f"{_subject_nominative(request.subject)}是{name}。"

    def _en(
        self,
        request: SemanticFactRequest,
        result: FactResult,
        context: AgentRequestContext,
    ) -> str:
        if result.status != "found":
            return {
                "caller_context_missing": "I cannot verify the current signed-in user.",
                "entity_not_found": "I could not find the corresponding household entity.",
                "relationship_not_found": "No matching household relationship is recorded.",
                "property_unavailable": "The entity is recorded, but that semantic property is unavailable.",
                "ambiguous": "More than one household entity matches; please clarify which one.",
                "computation_impossible": "The available evidence is insufficient for that computation.",
            }[result.status]
        if request.operation == "count":
            return f"The current count is {result.value}."
        if request.operation == "list":
            return "Current household members: " + ", ".join(
                _name(item, "en") for item in result.value
            ) + "."
        if request.operation == "get_property":
            return f"The requested value is {result.value}."
        if request.operation == "completed_years":
            return f"The completed age is {result.value}."
        if request.operation in {"argmin", "argmax"}:
            return f"The matching household member is {_name(result.value, 'en')}."
        if request.operation == "compare" and isinstance(result.value, Mapping):
            selected = _name(result.value.get("selected"), "en")
            other = _name(result.value.get("other"), "en")
            if result.value.get("equal"):
                return f"{selected} and {other} are the same age."
            adjective = "older" if request.comparison == "older" else "younger"
            return f"{selected} is {adjective} than {other}."
        if request.subject.kind == "assistant":
            return f"I am {context.assistant_display_name}, the Home Cortex household assistant."
        return f"The resolved person is {_name(result.value, 'en')}."


class SemanticFactService:
    def __init__(
        self,
        engine: HouseholdFactEngine,
        parser: TierZeroSemanticParser | None = None,
        renderer: FactRenderer | None = None,
        planner: SemanticFactPlanner | None = None,
    ) -> None:
        self.engine = engine
        self.parser = parser or TierZeroSemanticParser()
        self.renderer = renderer or FactRenderer()
        self.planner = planner

    async def attempt(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        context: AgentRequestContext,
        request_id: str = "-",
    ) -> SemanticAttempt:
        started = perf_counter()
        latest = _latest_user_text(messages)
        routing_started = perf_counter()
        parse_started = perf_counter()
        request = self.parser.parse(latest)
        semantic_parse_ms = (perf_counter() - parse_started) * 1000
        tier = 0
        llm_ms = 0.0
        llm_call_count = 0
        if request is None:
            if self.planner is None or not _likely_semantic_fact(latest):
                return SemanticAttempt(False)
            tier = 1
            llm_call_count = 1
            llm_started = perf_counter()
            try:
                plan, llm_ms = await self.planner.plan(messages, context)
            except Exception as error:
                llm_ms = (perf_counter() - llm_started) * 1000
                logger.warning(
                    "semantic_plan_invalid request_id=%s error=%s",
                    safe_log_token(request_id),
                    safe_log_token(type(error).__name__),
                )
                return SemanticAttempt(
                    True,
                    self._failure_answer(
                        context,
                        started,
                        request_id=request_id,
                        llm_ms=llm_ms,
                        llm_call_count=llm_call_count,
                    ),
                )
            if not plan.requires_fact:
                # The semantic classifier has claimed routing, but ordinary text
                # generation is still needed. Do not invoke the legacy planner.
                return SemanticAttempt(True)
            request = plan.request
            if request is None or not self.engine.schema.validates(request):
                return SemanticAttempt(
                    True,
                    self._failure_answer(
                        context,
                        started,
                        request=request,
                        request_id=request_id,
                        llm_ms=llm_ms,
                        llm_call_count=llm_call_count,
                    ),
                )
        routing_ms = (perf_counter() - routing_started) * 1000
        assert request is not None
        query_started = perf_counter()
        result, query_count, resolution_ms, computation_ms = await self.engine.execute(
            request,
            context,
        )
        fact_query_ms = (perf_counter() - query_started) * 1000
        render_started = perf_counter()
        rendered = self.renderer.render(request, result, context)
        render_ms = (perf_counter() - render_started) * 1000
        total_ms = (perf_counter() - started) * 1000
        timings = FactTimings(
            tier=tier,
            routing_ms=routing_ms,
            semantic_parse_ms=semantic_parse_ms,
            entity_resolution_ms=resolution_ms,
            fact_query_ms=fact_query_ms,
            computation_ms=computation_ms,
            render_ms=render_ms,
            llm_ms=llm_ms,
            total_ms=total_ms,
            llm_call_count=llm_call_count,
            db_query_count=query_count,
        )
        _log_fact_query(request_id, request, result, timings)
        return SemanticAttempt(True, FactAnswer(request, result, rendered, timings))

    def _failure_answer(
        self,
        context: AgentRequestContext,
        started: float,
        *,
        request: SemanticFactRequest | None = None,
        request_id: str,
        llm_ms: float,
        llm_call_count: int,
    ) -> FactAnswer:
        safe_request = request or SemanticFactRequest(
            operation="resolve_entity",
            subject=SemanticReference(kind="self", entity_type="person"),
        )
        result = FactResult("computation_impossible")
        render_started = perf_counter()
        rendered = self.renderer.render(safe_request, result, context)
        render_ms = (perf_counter() - render_started) * 1000
        timings = FactTimings(
            tier=1,
            render_ms=render_ms,
            llm_ms=llm_ms,
            llm_call_count=llm_call_count,
            total_ms=(perf_counter() - started) * 1000,
        )
        _log_fact_query(request_id, safe_request, result, timings)
        return FactAnswer(safe_request, result, rendered, timings)

    async def try_answer(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        context: AgentRequestContext,
        request_id: str = "-",
    ) -> FactAnswer | None:
        return (await self.attempt(
            messages,
            context=context,
            request_id=request_id,
        )).answer

    async def grounded_answer(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        context: AgentRequestContext,
        request_id: str = "-",
    ) -> GroundedAnswer | None:
        answer = await self.try_answer(messages, context=context, request_id=request_id)
        if answer is None:
            return None
        return GroundedAnswer(answer.text, answer.timings.db_query_count)


class _FactExecution:
    def __init__(self, dispatcher: Any, caller_entity_id: str | None) -> None:
        self.dispatcher = dispatcher
        self.caller_entity_id = caller_entity_id
        self.query_count = 0
        self.entity_cache: dict[str, dict[str, Any]] = {}
        self.relationship_evidence: list[FactRelationshipEvidence] = []

    def remember_relationship(
        self,
        semantic_relation: str,
        edge: Mapping[str, Any],
    ) -> None:
        evidence = FactRelationshipEvidence(
            relation=semantic_relation,
            source_id=_string_or_none(edge.get("in") or edge.get("from")),
            target_id=_string_or_none(edge.get("out") or edge.get("to")),
            start=edge.get("start"),
            end=edge.get("end"),
        )
        if evidence not in self.relationship_evidence:
            self.relationship_evidence.append(evidence)

    async def records(self, tool: str, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        entity_id = arguments.get("entity_id")
        if tool == "get_entity" and isinstance(entity_id, str):
            cached = self.entity_cache.get(entity_id)
            if cached is not None:
                return [dict(cached)]
        self.query_count += 1
        response = await self.dispatcher.dispatch(
            tool,
            arguments,
            caller_entity_id=self.caller_entity_id,
        )
        if response.get("ok") is not True:
            raise _FactFailure("computation_impossible")
        value = response.get("result")
        if not isinstance(value, list):
            raise _FactFailure("computation_impossible")
        records = [dict(item) for item in value if isinstance(item, Mapping)]
        if tool == "get_entity" and isinstance(entity_id, str) and records:
            self.entity_cache[entity_id] = dict(records[0])
        return records


class _FactFailure(RuntimeError):
    def __init__(
        self,
        status: FactStatus,
        *,
        evidence: FactEvidence = FactEvidence(),
        missing: tuple[str, ...] = (),
    ) -> None:
        super().__init__(status)
        self.status = status
        self.evidence = evidence
        self.missing = missing


def _latest_user_text(messages: Sequence[Mapping[str, Any]]) -> str:
    return next(
        (
            str(message.get("content"))
            for message in reversed(messages)
            if message.get("role") == "user"
            and isinstance(message.get("content"), str)
        ),
        "",
    )


def _likely_semantic_fact(text: str) -> bool:
    """Conservative Tier-1 routing; Tier-0 remains the canonical fast path."""
    normalized = _normalize_request(text)
    if not normalized:
        return False
    semantic_terms = re.search(
        r"家里|我家|家庭|老婆|妻子|丈夫|老公|儿子|女儿|孩子|"
        r"爸爸|父亲|妈妈|母亲|生日|出生|几岁|年纪|年龄|"
        r"household|wife|husband|spouse|son|daughter|child|parent|"
        r"birthday|birth date|how old",
        normalized,
    )
    question_shape = re.search(
        r"谁|什么|多少|几个|几位|哪|何时|什么时候|多大|几岁|"
        r"who|what|when|how many|how old|which|oldest|youngest",
        normalized,
    )
    return semantic_terms is not None and question_shape is not None


def _log_fact_query(
    request_id: str,
    request: SemanticFactRequest,
    result: FactResult,
    timings: FactTimings,
) -> None:
    logger.info(
        "fact_query request_id=%s tier=%d operation=%s semantic_plan=%s "
        "db_queries=%d llm_calls=%d routing_ms=%.2f semantic_parse_ms=%.2f "
        "entity_resolution_ms=%.2f fact_query_ms=%.2f computation_ms=%.2f "
        "render_ms=%.2f llm_ms=%.2f total_ms=%.2f status=%s",
        safe_log_token(request_id),
        timings.tier,
        safe_log_token(request.operation),
        safe_log_token(
            json.dumps(_plan_for_log(request), separators=(",", ":"))
        ),
        timings.db_query_count,
        timings.llm_call_count,
        timings.routing_ms,
        timings.semantic_parse_ms,
        timings.entity_resolution_ms,
        timings.fact_query_ms,
        timings.computation_ms,
        timings.render_ms,
        timings.llm_ms,
        timings.total_ms,
        safe_log_token(result.status),
    )


def _plan_for_log(request: SemanticFactRequest) -> dict[str, Any]:
    payload = request.model_dump(mode="json")
    for key in ("subject", "other"):
        reference = payload.get(key)
        if isinstance(reference, dict) and reference.get("value") is not None:
            reference["value"] = "[redacted]"
    return payload


def _normalize_request(text: str) -> str:
    normalized = text.casefold().strip()
    normalized = re.sub(r"^(?:请问|能否|能不能|能告诉我一下|告诉我一下|麻烦)", "", normalized)
    normalized = normalized.strip(" \t\r\n，,。.!！?？")
    return re.sub(r"\s+", " ", normalized)


def _extract_property(text: str) -> tuple[str | None, str]:
    birth = re.search(r"(?:的)?(?:生日|出生日期|出生时间)(?:是|在)?(?:哪天|什么时候|何时)?$", text)
    if birth:
        return "birth_date", text[: birth.start()].strip("的 ")
    age = re.search(r"(?:现在|今年)?(?:几岁(?:了)?|多大(?:了)?)$", text)
    if age:
        return "age", text[: age.start()].strip("的 ")
    english_birth = re.match(
        r"(?:what|when) (?:is|was) (.+?)(?:'s| ) (?:birthday|birth date)$",
        text,
    )
    if english_birth:
        return "birth_date", english_birth.group(1).strip()
    english_age = re.match(r"how old (?:is|are) (.+)$", text)
    if english_age:
        return "age", english_age.group(1).strip()
    return None, text


def _strip_identity_syntax(text: str) -> str:
    english = re.fullmatch(r"who (?:am|are) (i|you)", text)
    if english:
        return english.group(1)
    english_name = re.fullmatch(r"what is (my|your) name", text)
    if english_name:
        return "i" if english_name.group(1) == "my" else "you"
    return re.sub(r"(?:是)?谁$|(?:叫)?什么(?:名字)?$", "", text).strip("的 ")


def _strip_count_syntax(text: str) -> str:
    return re.sub(r"有(?:多少(?:个|位)?|几个|几位)", "", text).strip("的 ")


def _asks_identity(text: str) -> bool:
    return bool(
        re.search(r"(?:是)?谁$|(?:叫)?什么(?:名字)?$", text)
        or re.fullmatch(r"who (?:am|are) (?:i|you)", text)
        or re.fullmatch(r"what is (?:my|your) name", text)
    )


def _asks_count(text: str) -> bool:
    return bool(re.search(r"(?:有)?(?:多少|几)(?:个|位)?(?:人|成员|孩子|小孩)", text))


def _asks_list(text: str) -> bool:
    return bool(re.search(r"都有谁|成员名单|有哪些人|who lives|who is in", text))


def _mentions_household(text: str) -> bool:
    return bool(re.search(r"家里|我家|我们家|家庭|household|home", text))


def _asks_household_children(text: str) -> bool:
    return _asks_count(text) and bool(re.search(r"孩子|小孩|children", text))


def _is_household_oldest(text: str) -> bool:
    return (
        _mentions_household(text)
        and bool(re.search(r"谁最年长|谁年龄最大|谁年纪最大|oldest", text))
    ) or text in {"谁最年长", "谁年龄最大"}


def _household_members() -> SemanticReference:
    return SemanticReference(
        kind="current_household",
        entity_type="address",
        path=(SemanticRelationStep(relation="member"),),
    )


def _unique_entities(entities: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for entity in entities:
        entity_id = entity.get("id")
        if isinstance(entity_id, str):
            unique.setdefault(entity_id, dict(entity))
    return list(unique.values())


def _entity_type(entity: Mapping[str, Any]) -> str:
    entity_id = str(entity.get("id", ""))
    return entity_id.partition(":")[0]


def _string_or_none(value: Any) -> str | None:
    return str(value) if value is not None else None


def _parse_date(value: Any) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _ordered_value(value: Any) -> int | float | date | datetime | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    parsed_date = _parse_date(value)
    if parsed_date is not None:
        return parsed_date
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _completed_years(value: Any, today: date) -> int | None:
    born = _parse_date(value)
    if born is None or born > today:
        return None
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def _name(entity: Any, language: str) -> str:
    return resolve_display_name(entity, language) if isinstance(entity, Mapping) else str(entity)


def _last_relation(reference: SemanticReference) -> str | None:
    return reference.path[-1].relation if reference.path else None


def _counts_children(reference: SemanticReference) -> bool:
    if _last_relation(reference) == "child":
        return True
    return any(
        item.source == "relation"
        and item.property == "household_role"
        and item.value == "minor_dependent"
        for step in reference.path
        for item in step.filters
    )


def _count_noun(reference: SemanticReference) -> str:
    if _counts_children(reference):
        return "个孩子"
    if _last_relation(reference) == "member":
        return "个人"
    return "条记录"


def _relation_label(
    reference: SemanticReference,
    semantic_relation: str | None = None,
) -> str:
    return {
        "spouse": "配偶",
        "child": "亲子",
        "parent": "父母",
        "member": "家庭成员",
    }.get(semantic_relation or _last_relation(reference), "对应的")


def _subject_possessive(reference: SemanticReference) -> str:
    return f"{_subject_nominative(reference)}的"


def _subject_nominative(reference: SemanticReference) -> str:
    if not reference.path:
        return "您" if reference.kind == "self" else "对应实体"
    label = {
        "self": "您",
        "current_household": "家里",
    }.get(reference.kind, "对应实体")
    for index, step in enumerate(reference.path):
        connector = "" if reference.kind == "self" and index == 0 else "的"
        label = f"{label}{connector}{_relation_noun(step)}"
    return label


def _relation_noun(step: SemanticRelationStep) -> str:
    gender = next(
        (item.value for item in step.filters if item.property == "gender"),
        None,
    )
    return {
        ("spouse", "female"): "妻子",
        ("spouse", "male"): "丈夫",
        ("spouse", None): "配偶",
        ("child", "male"): "儿子",
        ("child", "female"): "女儿",
        ("child", None): "孩子",
        ("parent", "male"): "父亲",
        ("parent", "female"): "母亲",
        ("parent", None): "父母",
        ("member", None): "家庭成员",
    }.get((step.relation, gender), "关联实体")


def _property_label(properties: Sequence[str]) -> str:
    if not properties:
        return ""
    return {
        "birth_date": "出生日期",
        "display_name": "姓名",
        "gender": "性别",
        "household_role": "家庭角色",
    }.get(properties[0], "所需信息")


def _zh_number(value: int) -> str:
    return {
        0: "零",
        1: "一",
        2: "两",
        3: "三",
        4: "四",
        5: "五",
        6: "六",
        7: "七",
        8: "八",
        9: "九",
        10: "十",
    }.get(value, str(value))
