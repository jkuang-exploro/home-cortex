"""Semantic household fact IR, deterministic execution, and Tier-0 parsing."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .display import resolve_display_name
from .grounding import AgentRequestContext, GroundedAnswer
from .operator_registry import (
    OPERATORS,
    PREDICATE_OPERATORS,
    OperatorExecutionError,
    OperatorInput,
    OperatorValidationError,
    evaluate_predicate,
    execute_operator,
    infer_field_kind,
    operator_prompt_payload,
)
from .schema_catalog import RuntimeSchemaCatalog
from .semantic_ontology import SemanticOntology
from .text import safe_log_token

logger = logging.getLogger("uvicorn.error.home_cortex.semantic_facts")

FactStatus = Literal[
    "found",
    "caller_context_missing",
    "entity_not_found",
    "relationship_not_found",
    "property_unavailable",
    "ambiguous",
    "computation_input_missing",
    "computation_impossible",
]
FactOperation = Literal[
    "resolve_reference",
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
    "date_difference",
    "completed_years",
    "duration",
    "annual_occurrence",
    "unit_conversion",
]
ReferenceKind = Literal[
    "self",
    "assistant",
    "current_household",
    "named_entity",
    "entity_id",
]
ResolutionStatus = Literal[
    "resolved",
    "not_found",
    "ambiguous",
    "invalid_reference",
    "missing_context",
    "relationship_not_found",
    "property_unavailable",
]


class _SemanticModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class SemanticFilter(_SemanticModel):
    property: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
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
    mode: Literal["days", "seconds"] | None = None
    from_unit: str | None = Field(default=None, max_length=16)
    to_unit: str | None = Field(default=None, max_length=16)

    @model_validator(mode="after")
    def validate_operation(self) -> "SemanticFactRequest":
        if self.operation not in OPERATORS:
            raise ValueError("operation is not in the generic operator registry")
        if self.operation == "completed_years" and self.property is None:
            object.__setattr__(self, "property", "birth_date")
        return self


if not set(get_args(FactOperation)).issubset(OPERATORS):
    raise RuntimeError("FactOperation must be backed by the generic operator registry")


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


@dataclass(frozen=True)
class ResolutionResult:
    status: ResolutionStatus
    entities: tuple[Mapping[str, Any], ...] = ()
    entity_ids: tuple[str, ...] = ()
    confidence: float | None = None
    evidence: FactEvidence = FactEvidence()
    candidates: tuple[Mapping[str, Any], ...] = ()
    missing_requirements: tuple[str, ...] = ()


class SemanticSchemaRegistry:
    """Map stable semantic concepts to deployment-specific schema names."""

    def __init__(
        self,
        catalog: RuntimeSchemaCatalog,
        ontology: SemanticOntology | None = None,
    ) -> None:
        self.catalog = catalog
        self.ontology = ontology or SemanticOntology.load_default()
        self._aliased_physical_properties = frozenset(
            physical
            for definition in self.ontology.properties.values()
            for physical in definition.fields
        )
        self._property_cache: dict[tuple[str, str], str | None] = {}
        self._capability_cache: dict[str, Any] | None = None

    def physical_property(self, entity_type: str, semantic: str) -> str | None:
        marker = (entity_type, semantic)
        if marker in self._property_cache:
            return self._property_cache[marker]
        schema = self.catalog.entities.get(entity_type)
        available = set(schema.properties) if schema else set()
        candidates = self._property_candidates(semantic)
        physical = next((field for field in candidates if field in available), None)
        self._property_cache[marker] = physical
        return physical

    def physical_relation(self, semantic: str) -> tuple[str, str | None] | None:
        public_name = self.ontology.base_relations.get(semantic)
        if public_name is None:
            return None
        direct = self.catalog.relations.get(public_name)
        if direct is not None:
            return direct.name, None if direct.symmetric else "out"
        inverse = next(
            (
                schema
                for schema in self.catalog.relations.values()
                if schema.inverse_name == public_name
            ),
            None,
        )
        if inverse is not None:
            return inverse.name, "in"
        return None

    def relation_property(self, relation: str, semantic: str) -> str | None:
        schema = self.catalog.relations.get(relation)
        available = set(schema.properties) if schema else set()
        candidates = self._property_candidates(semantic)
        return next((field for field in candidates if field in available), None)

    def _property_candidates(self, semantic: str) -> tuple[str, ...]:
        if semantic in self.ontology.properties:
            return self.ontology.property_fields(semantic)
        if semantic in self._aliased_physical_properties:
            return ()
        return (semantic,)

    def capability_payload(self) -> dict[str, Any]:
        if self._capability_cache is not None:
            return self._capability_cache
        fact_operations = set(get_args(FactOperation))
        exposed_operators = fact_operations | set(PREDICATE_OPERATORS) | {"traverse"}
        contracts = operator_prompt_payload()
        semantic_properties = {
            entity_type: sorted(self.semantic_properties(entity_type))
            for entity_type in self.catalog.entities
        }
        relations = {
            semantic
            for semantic in self.ontology.base_relations
            if self.physical_relation(semantic) is not None
        }
        self._capability_cache = {
            "entity_types": sorted(self.catalog.entities),
            "semantic_properties": semantic_properties,
            "semantic_relations": sorted(relations),
            "semantic_relation_properties": {
                relation: sorted(self.semantic_relation_properties(relation))
                for relation in relations
            },
            "operations": sorted(fact_operations),
            "operator_contracts": {
                name: contracts[name] for name in sorted(exposed_operators)
            },
            "reference_ontology": self.ontology.prompt_payload(),
        }
        return self._capability_cache

    def semantic_properties(self, entity_type: str) -> frozenset[str]:
        schema = self.catalog.entities.get(entity_type)
        if schema is None:
            return frozenset()
        properties = {
            semantic
            for semantic in self.ontology.properties
            if self.physical_property(entity_type, semantic) is not None
        }
        properties.update(
            physical
            for physical in schema.properties
            if physical != "id"
            and physical not in self._aliased_physical_properties
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
            for semantic_property in self.ontology.properties
            if self.relation_property(relation, semantic_property) is not None
        }
        properties.update(
            physical
            for physical in schema.properties
            if physical not in self._aliased_physical_properties
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
        final_types: dict[int, frozenset[str]] = {}
        for reference in references:
            contextual_type = {
                "self": "person",
                "assistant": "person",
                "current_household": "address",
            }.get(reference.kind)
            if (
                contextual_type is not None
                and reference.entity_type is not None
                and reference.entity_type != contextual_type
            ):
                return False
            if (
                reference.kind == "entity_id"
                and reference.value is not None
                and reference.entity_type is not None
                and reference.entity_type != reference.value.partition(":")[0]
            ):
                return False
            if reference.kind == "assistant" and reference.path:
                return False
            resolved_types = self._reference_entity_types(reference)
            if resolved_types is None:
                return False
            final_types[id(reference)] = resolved_types
            step_types = self._base_entity_types(reference)
            for step in reference.path:
                next_types = self._traversal_target_types(step.relation, step_types)
                if next_types is None:
                    return False
                for item in step.filters:
                    if item.source == "entity":
                        kind = self._semantic_kind(next_types, item.property)
                        if kind is None or not self._valid_predicate(item, kind):
                            return False
                    if item.source == "relation":
                        resolved = self.physical_relation(step.relation)
                        if resolved is None:
                            return False
                        physical = self.relation_property(resolved[0], item.property)
                        if physical is None:
                            return False
                        kind = self.catalog.relation_field_type(resolved[0], physical)
                        if not self._valid_predicate(item, kind):
                            return False
                step_types = next_types

        operation = OPERATORS[request.operation]
        collection_input = bool(request.subject.path or request.other is not None)
        if operation.input_shape == "collection" and not collection_input:
            return False
        if request.operation == "resolve_reference":
            return request.property is None and request.other is None
        if request.operation == "select":
            if request.other is not None:
                return False
            if request.property is None:
                return bool(request.subject.path)
            return self._semantic_kind(
                final_types[id(request.subject)],
                request.property,
            ) is not None

        field_kind = (
            self._semantic_kind(final_types[id(request.subject)], request.property)
            if request.property is not None
            else "unknown"
        )
        if request.property is not None and field_kind is None:
            return False
        if (
            request.property is not None
            and field_kind == "unknown"
            and "any" not in operation.field_kinds
        ):
            return False
        if request.other is not None:
            other_kind = self._semantic_kind(
                final_types[id(request.other)],
                request.property,
            )
            if field_kind != other_kind:
                return False
        parameters = {
            "reference": "household_now",
            "mode": request.mode,
            "from_unit": request.from_unit,
            "to_unit": request.to_unit,
        }
        try:
            operation.validate(
                field=request.property,
                field_kind=field_kind or "unknown",
                order_by=(
                    request.property
                    if request.operation in {"latest", "earliest"}
                    else None
                ),
                order_by_kind=field_kind or "unknown",
                parameters=parameters,
            )
        except OperatorValidationError:
            return False
        return True

    def _base_entity_types(self, reference: SemanticReference) -> frozenset[str]:
        if reference.kind in {"self", "assistant"}:
            return frozenset({"person"})
        if reference.kind == "current_household":
            return frozenset({"address"})
        if reference.kind == "entity_id" and reference.value is not None:
            return frozenset({reference.value.partition(":")[0]})
        if reference.entity_type is not None:
            return frozenset({reference.entity_type})
        return frozenset(self.catalog.entities)

    def _reference_entity_types(
        self,
        reference: SemanticReference,
    ) -> frozenset[str] | None:
        types = self._base_entity_types(reference)
        if not types or any(not self.catalog.has_entity_type(item) for item in types):
            return None
        for step in reference.path:
            types = self._traversal_target_types(step.relation, types) or frozenset()
            if not types:
                return None
        return types

    def _traversal_target_types(
        self,
        semantic_relation: str,
        source_types: frozenset[str],
    ) -> frozenset[str] | None:
        resolved = self.physical_relation(semantic_relation)
        if resolved is None:
            return None
        relation, direction = resolved
        schema = self.catalog.relations.get(relation)
        if schema is None:
            return None
        from_types = frozenset(schema.from_types)
        to_types = frozenset(schema.to_types)
        if direction == "out":
            return to_types if source_types.intersection(from_types) else None
        if direction == "in":
            return from_types if source_types.intersection(to_types) else None
        targets: set[str] = set()
        if source_types.intersection(from_types):
            targets.update(to_types)
        if source_types.intersection(to_types):
            targets.update(from_types)
        return frozenset(targets) or None

    def _semantic_kind(
        self,
        entity_types: frozenset[str],
        semantic_property: str | None,
    ) -> str | None:
        if semantic_property is None:
            return None
        kinds = {
            self.catalog.entity_field_type(entity_type, physical)
            for entity_type in entity_types
            if (physical := self.physical_property(entity_type, semantic_property))
            is not None
        }
        if len(kinds) != 1:
            return None
        return next(iter(kinds))

    @staticmethod
    def _valid_predicate(item: SemanticFilter, field_kind: str) -> bool:
        definition = OPERATORS[item.operator]
        if field_kind == "unknown" and "any" not in definition.field_kinds:
            return False
        if item.operator == "exists" and not (
            item.value is None or isinstance(item.value, bool)
        ):
            return False
        if item.operator == "in" and not isinstance(item.value, tuple):
            return False
        if item.operator == "date_range" and (
            not isinstance(item.value, tuple)
            or len(item.value) != 2
            or any(not isinstance(value, str) for value in item.value)
        ):
            return False
        try:
            definition.validate(
                field=item.property,
                field_kind=field_kind,
            )
        except OperatorValidationError:
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
        plan = SemanticPlan.model_validate(payload)
        if plan.request is not None and any(
            reference.kind == "entity_id"
            for reference in (plan.request.subject, plan.request.other)
            if reference is not None
        ):
            raise ValueError("The semantic planner cannot originate entity IDs")
        return plan, (perf_counter() - started) * 1000


class TierZeroSemanticParser:
    """Parse a bounded vocabulary into composable IR, never whole-query handlers."""

    def __init__(self, ontology: SemanticOntology | None = None) -> None:
        self.ontology = ontology or SemanticOntology.load_default()

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
            return SemanticFactRequest(operation="select", subject=_household_members())

        if _asks_residence_address(normalized):
            return SemanticFactRequest(
                operation="select",
                subject=_self_residence(),
                property="full_address",
            )

        birthday_intent = _extract_birthday_intent(normalized, self.ontology)
        if birthday_intent is not None:
            operation, stem, mode = birthday_intent
            reference = self._parse_reference(stem)
            return (
                SemanticFactRequest(
                    operation=operation,
                    subject=reference,
                    property="birth_date",
                    mode=mode,
                )
                if reference is not None
                else None
            )

        property_name, stem = _extract_property(normalized, self.ontology)
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
                    operation="select",
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
                SemanticFactRequest(operation="resolve_reference", subject=reference)
                if reference is not None
                else None
            )
        return None

    def _parse_reference(self, text: str) -> SemanticReference | None:
        normalized = text.strip("的 ")
        base: ReferenceKind
        base_value: str | None = None
        remainder: str
        if normalized.startswith(("我", "my ")) or normalized in {"i", "me", "my"}:
            base = "self"
            remainder = (
                normalized[1:]
                if normalized.startswith("我")
                else normalized.removeprefix("my ")
                if normalized.startswith("my ")
                else ""
            )
        elif normalized.startswith(("你", "您", "your ")) or normalized in {
            "you",
            "your",
        }:
            base = "assistant"
            remainder = (
                normalized[1:]
                if normalized[:1] in {"你", "您"}
                else normalized.removeprefix("your ")
                if normalized.startswith("your ")
                else ""
            )
        else:
            if not normalized or len(normalized) > 256:
                return None
            name, separator, possible_path = normalized.partition("的")
            if not separator or not name or not self._starts_relation(possible_path):
                return SemanticReference(
                    kind="named_entity",
                    value=normalized,
                    entity_type="person",
                )
            base = "named_entity"
            base_value = name
            remainder = possible_path
        remainder = remainder.strip("的 ")
        steps: list[SemanticRelationStep] = []
        while remainder:
            matched = self.ontology.match_reference_prefix(remainder)
            if matched is None:
                return None
            concept, consumed = matched
            steps.extend(
                SemanticRelationStep(
                    relation=step.relation,
                    filters=tuple(
                        SemanticFilter(
                            property=item.property,
                            operator=item.operator,
                            value=item.value,
                            source=item.source,
                        )
                        for item in step.filters
                    ),
                )
                for step in concept.path
            )
            remainder = remainder[consumed:].strip("的 ")
        return SemanticReference(
            kind=base,
            value=base_value,
            entity_type="person",
            path=tuple(steps),
        )

    def _starts_relation(self, text: str) -> bool:
        return self.ontology.match_reference_prefix(text) is not None

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
            operation="argmin",
            subject=left,
            other=right,
            property="birth_date",
        )


class EntityResolver:
    """Authoritative resolver from semantic references to canonical entities."""

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

    async def resolve(
        self,
        reference: SemanticReference,
        context: AgentRequestContext,
        execution: "_FactExecution",
        *,
        allow_empty_collection: bool = False,
        expect_many: bool = False,
    ) -> ResolutionResult:
        try:
            entities = await self._resolve(
                reference,
                context,
                execution,
                allow_empty_collection=allow_empty_collection,
            )
            if len(entities) > 1 and not expect_many:
                candidates = tuple(
                    [await self._load_if_unnamed(item, execution) for item in entities]
                )
                return ResolutionResult("ambiguous", candidates=candidates)
            entity_ids = tuple(
                str(item["id"])
                for item in entities
                if isinstance(item.get("id"), str)
            )
            return ResolutionResult(
                "resolved",
                tuple(entities),
                entity_ids,
                1.0,
                FactEvidence(
                    entity_ids=entity_ids,
                    relationship=(
                        reference.path[-1].relation if reference.path else None
                    ),
                    relationships=tuple(execution.relationship_evidence),
                ),
            )
        except _FactFailure as error:
            status: ResolutionStatus = {
                "caller_context_missing": "missing_context",
                "entity_not_found": "not_found",
                "relationship_not_found": "relationship_not_found",
                "property_unavailable": "property_unavailable",
                "ambiguous": "ambiguous",
                "computation_input_missing": "invalid_reference",
                "computation_impossible": "invalid_reference",
            }[error.status]
            return ResolutionResult(
                status,
                evidence=error.evidence,
                candidates=error.candidates,
                missing_requirements=error.missing,
            )

    async def _resolve(
        self,
        reference: SemanticReference,
        context: AgentRequestContext,
        execution: "_FactExecution",
        *,
        allow_empty_collection: bool,
    ) -> list[dict[str, Any]]:
        if reference.kind == "assistant":
            entities = [
                {
                    "id": context.assistant_id,
                    "display_name": context.assistant_display_name,
                }
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
        elif reference.kind == "named_entity":
            records = await execution.records(
                "resolve_entity_alias",
                {
                    "text": reference.value,
                    "entity_type": reference.entity_type,
                    "limit": self.max_records,
                },
            )
            if not records:
                raise _FactFailure("entity_not_found")
            if len(records) > 1:
                raise _FactFailure(
                    "ambiguous",
                    evidence=FactEvidence(
                        entity_ids=tuple(
                            str(item["id"])
                            for item in records
                            if isinstance(item.get("id"), str)
                        )
                    ),
                    candidates=tuple(records),
                )
            entities = records
        else:
            raise _FactFailure("computation_impossible")

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
                    "include_residents": False,
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
            if not evaluate_predicate(item.operator, edge.get(physical), item.value):
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
            record = (
                await self._load(entity, execution) if needs_load else dict(entity)
            )
            if all(
                evaluate_predicate(item.operator, record.get(physical), item.value)
                for item, physical in mapped
            ):
                matched.append(record)
        return matched

    async def _load_if_unnamed(
        self,
        entity: Mapping[str, Any],
        execution: "_FactExecution",
    ) -> dict[str, Any]:
        if any(entity.get(field) for field in ("display_name", "name", "full_name")):
            return dict(entity)
        return await self._load(entity, execution)

    @staticmethod
    async def _load(
        entity: Mapping[str, Any],
        execution: "_FactExecution",
    ) -> dict[str, Any]:
        entity_id = entity.get("id")
        if not isinstance(entity_id, str):
            raise _FactFailure("entity_not_found")
        records = await execution.records("get_entity", {"entity_id": entity_id})
        if not records:
            raise _FactFailure("entity_not_found")
        return dict(records[0])


class HouseholdFactEngine:
    def __init__(
        self,
        dispatcher: Any,
        schema: SemanticSchemaRegistry,
        *,
        max_records: int = 25,
        resolver: EntityResolver | None = None,
    ) -> None:
        self.dispatcher = dispatcher
        self.schema = schema
        self.max_records = max_records
        self.resolver = resolver or EntityResolver(
            dispatcher,
            schema,
            max_records=max_records,
        )

    async def execute(
        self,
        request: SemanticFactRequest,
        context: AgentRequestContext,
    ) -> tuple[FactResult, int, float, float]:
        if not self.schema.validates(request):
            return FactResult("computation_impossible"), 0, 0, 0
        execution = _FactExecution(self.dispatcher, context.caller_entity_id)
        resolution_started = perf_counter()
        allow_empty_collection = request.operation in {"count", "select"}
        operation = OPERATORS[request.operation]
        expect_many = request.other is None and (
            operation.input_shape == "collection"
            or (request.operation == "select" and request.property is None)
        )
        resolution = await self.resolver.resolve(
            request.subject,
            context,
            execution,
            allow_empty_collection=allow_empty_collection,
            expect_many=expect_many,
        )
        other_resolution = (
            await self.resolver.resolve(
                request.other,
                context,
                execution,
                allow_empty_collection=False,
                expect_many=False,
            )
            if request.other is not None
            else ResolutionResult("resolved")
        )
        failed = next(
            (
                item
                for item in (resolution, other_resolution)
                if item.status != "resolved"
            ),
            None,
        )
        if failed is not None:
            status: FactStatus = {
                "not_found": "entity_not_found",
                "ambiguous": "ambiguous",
                "invalid_reference": "computation_impossible",
                "missing_context": "caller_context_missing",
                "relationship_not_found": "relationship_not_found",
                "property_unavailable": "property_unavailable",
            }[failed.status]
            return (
                FactResult(
                    status,
                    evidence=failed.evidence,
                    missing_requirements=failed.missing_requirements,
                    candidates=failed.candidates,
                ),
                execution.query_count,
                (perf_counter() - resolution_started) * 1000,
                0,
            )
        entities = [dict(item) for item in resolution.entities]
        other_entities = [dict(item) for item in other_resolution.entities]
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
            value = execute_operator("count", OperatorInput(records=entities))
            return FactResult("found", value, evidence)
        if request.operation == "select" and request.property is None:
            visible = [await self._load_if_unnamed(item, execution) for item in entities]
            return FactResult("found", visible, evidence)
        if request.operation == "resolve_reference":
            singular = await self._singular(
                entities,
                request,
                execution,
                load_full=False,
            )
            if isinstance(singular, FactResult):
                return singular
            return FactResult("found", singular, evidence)
        if request.operation == "select":
            singular = await self._singular(entities, request, execution)
            if isinstance(singular, FactResult):
                return singular
            property_result = self._property(singular, request.property or "")
            if isinstance(property_result, FactResult):
                return property_result
            return FactResult("found", property_result, evidence)
        definition = OPERATORS[request.operation]
        records = list(entities)
        if request.other is not None:
            left = await self._singular(entities, request, execution)
            right = await self._singular(other_entities, request, execution)
            if isinstance(left, FactResult):
                return left
            if isinstance(right, FactResult):
                return right
            records = [left, right]
        elif definition.input_shape == "scalar":
            singular = await self._singular(entities, request, execution)
            if isinstance(singular, FactResult):
                return singular
            records = [singular]
        else:
            records = [await self._load(item, execution) for item in records]
        normalized: list[dict[str, Any]] = []
        for record in records:
            value = (
                self._property(record, request.property)
                if request.property is not None
                else record
            )
            if isinstance(value, FactResult):
                return FactResult(
                    (
                        "computation_input_missing"
                        if value.status == "property_unavailable"
                        else value.status
                    ),
                    evidence=evidence,
                    missing_requirements=value.missing_requirements,
                    candidates=value.candidates,
                )
            normalized.append({"value": value, "entity": record})
        if not normalized:
            return FactResult("computation_impossible", evidence=evidence)
        if request.operation in {"latest", "earliest"}:
            normalized.sort(
                key=lambda item: item["value"],
                reverse=request.operation == "latest",
            )
        field = "value" if request.property is not None else None
        parameters = {
            "reference": "household_now",
            "mode": request.mode,
            "from_unit": request.from_unit,
            "to_unit": request.to_unit,
        }
        try:
            definition.validate(
                field=field,
                field_kind=(
                    infer_field_kind([item["value"] for item in normalized])
                    if field is not None
                    else "unknown"
                ),
                order_by=(field if request.operation in {"latest", "earliest"} else None),
                order_by_kind=(
                    infer_field_kind([item["value"] for item in normalized])
                    if field is not None
                    else "unknown"
                ),
                parameters=parameters,
            )
            value = execute_operator(
                request.operation,
                OperatorInput(
                    records=normalized,
                    field=field,
                    order_by=field if request.operation in {"latest", "earliest"} else None,
                    mode=request.mode,
                    reference="household_now",
                    from_unit=request.from_unit,
                    to_unit=request.to_unit,
                    now=context.current_time,
                ),
            )
        except (OperatorValidationError, OperatorExecutionError, TypeError, ValueError):
            return FactResult(
                "computation_impossible",
                evidence=evidence,
                missing_requirements=((request.property,) if request.property else ()),
            )
        if isinstance(value, Mapping) and isinstance(value.get("entity"), Mapping):
            selected = dict(value["entity"])
            if request.other is not None:
                selected_id = selected.get("id")
                other = next(
                    (
                        record
                        for record in records
                        if record.get("id") != selected_id
                    ),
                    records[0],
                )
                selected_value = self._property(selected, request.property or "")
                other_value = self._property(other, request.property or "")
                return FactResult(
                    "found",
                    {
                        "selected": selected,
                        "other": other,
                        "equal": selected_value == other_value,
                    },
                    evidence,
                )
            value = selected
        return FactResult("found", value, evidence)

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
        if result.status == "computation_input_missing":
            label = _property_label(result.missing_requirements) or "所需资料"
            return f"目前没有足够的{label}来完成这项计算。"
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
        if request.operation == "select" and request.property is None:
            values = result.value if isinstance(result.value, list) else []
            if not values:
                return "家庭资料中目前没有记录当前家庭成员。"
            names = "、".join(_name(item, "zh") for item in values)
            return f"家里目前的成员有：{names}。"
        if request.operation == "select":
            if request.property == "birth_date":
                return f"{_subject_possessive(request.subject)}出生日期是{result.value}。"
            if request.property == "full_address":
                return f"您的具体住址是{_format_address(result.value)}。"
            return f"查询到的值是{result.value}。"
        if request.operation == "completed_years":
            return f"{_subject_nominative(request.subject)}今年{result.value}岁。"
        if request.operation == "annual_occurrence":
            if request.mode == "days":
                days = int(result.value)
                if days == 0:
                    return f"{_subject_possessive(request.subject)}生日就是今天。"
                return f"{_subject_possessive(request.subject)}生日还有{days}天。"
            return f"{_subject_possessive(request.subject)}下次生日是{result.value}。"
        if request.operation in {"argmin", "argmax"}:
            if request.other is not None and isinstance(result.value, Mapping):
                selected = _name(result.value.get("selected"), "zh")
                other = _name(result.value.get("other"), "zh")
                if result.value.get("equal"):
                    return f"{selected}和{other}年龄相同。"
                adjective = "大" if request.operation == "argmin" else "小"
                return f"{selected}年龄比{other}{adjective}。"
            if request.property == "birth_date":
                qualifier = "最年长" if request.operation == "argmin" else "最年轻"
                return f"家里{qualifier}的是{_name(result.value, 'zh')}。"
            return f"符合极值条件的是{_name(result.value, 'zh')}。"
        if request.operation in {"sum", "average", "min", "max"}:
            return f"计算结果是{result.value}。"
        if request.operation in {
            "date_difference",
            "duration",
            "unit_conversion",
        }:
            return f"换算结果是{result.value}。"
        if request.operation in {"first", "last", "latest", "earliest"}:
            return f"符合条件的是{_name(result.value, 'zh')}。"
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
                "computation_input_missing": (
                    "A required semantic property is unavailable for this computation."
                ),
                "computation_impossible": "The available evidence is insufficient for that computation.",
            }[result.status]
        if request.operation == "count":
            return f"The current count is {result.value}."
        if request.operation == "select" and request.property is None:
            if not result.value:
                return "The household data has no current member records."
            return "Current household members: " + ", ".join(
                _name(item, "en") for item in result.value
            ) + "."
        if request.operation == "select":
            if request.property == "full_address":
                return f"Your street address is {_format_address(result.value)}."
            return f"The requested value is {result.value}."
        if request.operation == "completed_years":
            return f"The completed age is {result.value}."
        if request.operation == "annual_occurrence":
            if request.mode == "days":
                days = int(result.value)
                return (
                    "The birthday is today."
                    if days == 0
                    else f"The birthday is in {days} days."
                )
            return f"The next birthday is {result.value}."
        if request.operation in {"argmin", "argmax"}:
            if request.other is not None and isinstance(result.value, Mapping):
                selected = _name(result.value.get("selected"), "en")
                other = _name(result.value.get("other"), "en")
                if result.value.get("equal"):
                    return f"{selected} and {other} are the same age."
                adjective = "older" if request.operation == "argmin" else "younger"
                return f"{selected} is {adjective} than {other}."
            return f"The matching household member is {_name(result.value, 'en')}."
        if request.operation in {"sum", "average", "min", "max"}:
            return f"The computed result is {result.value}."
        if request.operation in {
            "date_difference",
            "duration",
            "unit_conversion",
        }:
            return f"The converted result is {result.value}."
        if request.operation in {"first", "last", "latest", "earliest"}:
            return f"The matching result is {_name(result.value, 'en')}."
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
        tier_zero_enabled: bool = True,
    ) -> None:
        self.engine = engine
        self.parser = parser or TierZeroSemanticParser(engine.schema.ontology)
        self.renderer = renderer or FactRenderer()
        self.planner = planner
        self.tier_zero_enabled = tier_zero_enabled

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
        request = self.parser.parse(latest) if self.tier_zero_enabled else None
        semantic_parse_ms = (perf_counter() - parse_started) * 1000
        tier = 0
        llm_ms = 0.0
        llm_call_count = 0
        if request is None:
            if self.planner is None:
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
            operation="resolve_reference",
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
        dispatch = (
            self.dispatcher.dispatch_internal
            if tool == "resolve_entity_alias"
            and hasattr(self.dispatcher, "dispatch_internal")
            else self.dispatcher.dispatch
        )
        response = await dispatch(
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
        candidates: tuple[Mapping[str, Any], ...] = (),
    ) -> None:
        super().__init__(status)
        self.status = status
        self.evidence = evidence
        self.missing = missing
        self.candidates = candidates


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
        if not isinstance(reference, dict):
            continue
        if reference.get("value") is not None:
            reference["value"] = "[redacted]"
        for step in reference.get("path", []):
            if not isinstance(step, dict):
                continue
            for item in step.get("filters", []):
                if isinstance(item, dict) and "value" in item:
                    item["value"] = "[redacted]"
    return payload


def _normalize_request(text: str) -> str:
    normalized = text.casefold().strip()
    normalized = re.sub(r"^(?:请问|能否|能不能|能告诉我一下|告诉我一下|麻烦)", "", normalized)
    normalized = normalized.strip(" \t\r\n，,。.!！?？")
    return re.sub(r"\s+", " ", normalized)


def _extract_property(
    text: str,
    ontology: SemanticOntology,
) -> tuple[str | None, str]:
    age = re.search(r"(?:现在|今年)?(?:几岁(?:了)?|多大(?:了)?)$", text)
    if age:
        return "age", text[: age.start()].strip("的 ")
    english_age = re.match(r"how old (?:is|are) (.+)$", text)
    if english_age:
        return "age", english_age.group(1).strip()
    for alias, semantic_property in ontology.fast_property_aliases():
        escaped = re.escape(alias.casefold())
        english = re.fullmatch(
            rf"(?:what|when) (?:is|was) (.+?)(?:'s)? {escaped}",
            text,
        )
        if english:
            return semantic_property, english.group(1).strip()
        english_postfix = re.fullmatch(rf"when was (.+?) {escaped}", text)
        if english_postfix:
            return semantic_property, english_postfix.group(1).strip()
        suffix = re.search(
            rf"(?:的)?{escaped}(?:是|在)?(?:哪天|什么时候|何时|是什么)?$",
            text,
        )
        if suffix:
            return semantic_property, text[: suffix.start()].strip("的 ")
    return None, text


def _extract_birthday_intent(
    text: str,
    ontology: SemanticOntology,
) -> tuple[Literal["annual_occurrence"], str, Literal["days"] | None] | None:
    birth_aliases = [
        alias
        for alias, semantic_property in ontology.fast_property_aliases()
        if semantic_property == "birth_date"
    ]
    alias_pattern = "|".join(re.escape(alias.casefold()) for alias in birth_aliases)
    countdown = re.fullmatch(
        rf"(?:距离|离)?(.+?)(?:的)?(?:{alias_pattern})"
        r"(?:还)?(?:有|剩)(?:多少|几)天(?:了)?",
        text,
    )
    if countdown:
        return "annual_occurrence", countdown.group(1).strip("的 "), "days"
    english_countdown = re.fullmatch(
        rf"how many days (?:are left )?until (.+?)(?:'s| )"
        rf"(?:{alias_pattern})",
        text,
    )
    if english_countdown:
        return "annual_occurrence", english_countdown.group(1).strip(), "days"
    upcoming = re.fullmatch(
        rf"(.+?)(?:的)?(?:下次(?:{alias_pattern})(?:是|在)?哪天|"
        rf"哪天过(?:{alias_pattern}))",
        text,
    )
    if upcoming:
        return "annual_occurrence", upcoming.group(1).strip("的 "), None
    english_upcoming = re.fullmatch(
        rf"when is (.+?)(?:'s| ) next (?:{alias_pattern})",
        text,
    )
    if english_upcoming:
        return "annual_occurrence", english_upcoming.group(1).strip(), None
    return None


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


def _asks_residence_address(text: str) -> bool:
    return bool(
        re.search(
            r"住哪里|住哪儿|具体住址|街道地址|家庭住址|家庭地址|"
            r"住址(?:是)?什么|where do i live|my (?:home |street )?address",
            text,
        )
    )


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


def _self_residence() -> SemanticReference:
    return SemanticReference(
        kind="self",
        entity_type="person",
        path=(SemanticRelationStep(relation="residence"),),
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
        "residence": "居住地",
    }.get(semantic_relation or _last_relation(reference), "对应的")


def _subject_possessive(reference: SemanticReference) -> str:
    return f"{_subject_nominative(reference)}的"


def _subject_nominative(reference: SemanticReference) -> str:
    if not reference.path:
        if reference.kind == "self":
            return "您"
        if reference.kind == "named_entity" and reference.value:
            return reference.value
        return "对应实体"
    label = {
        "self": "您",
        "current_household": "家里",
        "named_entity": reference.value or "对应实体",
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
        ("residence", None): "住所",
    }.get((step.relation, gender), "关联实体")


def _property_label(properties: Sequence[str]) -> str:
    if not properties:
        return ""
    return {
        "birth_date": "出生日期",
        "display_name": "姓名",
        "given_name": "名字",
        "family_name": "姓氏",
        "form_of_address": "称呼",
        "gender": "性别",
        "household_role": "家庭角色",
        "full_address": "具体住址",
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


def _format_address(value: Any) -> str:
    if not isinstance(value, Mapping):
        return str(value)
    street = value.get("street")
    city = value.get("city")
    state = value.get("state")
    postal = value.get("zip") or value.get("postal_code")
    locality = ", ".join(str(item) for item in (city, state) if item)
    if postal:
        locality = f"{locality} {postal}".strip()
    return ", ".join(str(item) for item in (street, locality) if item)
