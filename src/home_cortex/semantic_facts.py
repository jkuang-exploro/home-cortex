"""Semantic household fact IR, deterministic execution, and Tier-0 parsing."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .display import resolve_display_name
from .grounding import AgentRequestContext
from .operator_registry import (
    OPERATORS,
    OperatorExecutionError,
    OperatorInput,
    OperatorValidationError,
    evaluate_predicate,
    execute_operator,
    infer_field_kind,
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
    "relation_property_unavailable",
    "filter_input_missing",
    "filter_unsupported",
    "operator_unsupported",
    "semantic_plan_unsupported",
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
        ):
            raise ValueError("semantic predicates do not accept field-filter options")
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
class PlannerDiagnostics:
    input_summary: Mapping[str, Any]
    output_raw: Mapping[str, Any] | None = None
    normalized_plan: Mapping[str, Any] | None = None
    validation_result: PlannerValidationCode = "INVALID_PLAN"
    failure_detail: str | None = None
    attempt_count: int = 0
    latency_ms: float = 0


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
class ResolutionResult:
    status: ResolutionStatus
    entities: tuple[Mapping[str, Any], ...] = ()
    entity_ids: tuple[str, ...] = ()
    confidence: float | None = None
    evidence: FactEvidence = FactEvidence()
    candidates: tuple[Mapping[str, Any], ...] = ()
    missing_requirements: tuple[str, ...] = ()
    relationship_records: tuple[Mapping[str, Any], ...] = ()


class SemanticSchemaRegistry:
    """Map stable semantic concepts to deployment-specific schema names."""

    _RESOLVER_METADATA_PROPERTIES = frozenset({"aliases", "appellations"})

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
        if semantic in self._RESOLVER_METADATA_PROPERTIES:
            return ()
        if semantic in self.ontology.properties:
            return self.ontology.property_fields(semantic)
        if semantic in self._aliased_physical_properties:
            return ()
        return (semantic,)

    def capability_payload(self) -> dict[str, Any]:
        if self._capability_cache is not None:
            return self._capability_cache
        fact_operations = set(get_args(FactOperation))
        plan_operations = fact_operations | {"filter", "traverse"}
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
            "references": [
                "self",
                "assistant",
                "current_household",
                "named_entity",
            ],
            "entity_types": sorted(self.catalog.entities),
            "semantic_properties": semantic_properties,
            "semantic_relations": sorted(relations),
            "semantic_relation_properties": {
                relation: sorted(self.semantic_relation_properties(relation))
                for relation in relations
            },
            "operations": sorted(plan_operations),
            "operation_requirements": {
                "argmin": "collection + ordered property; returns entity",
                "argmax": "collection + ordered property; returns entity",
                "completed_years": "date property; reference=household_now",
                "duration": "date property + mode days|seconds",
                "annual_occurrence": "date property; mode=days only for countdown",
                "date_difference": "date property + mode days|seconds",
                "unit_conversion": "numeric property + from_unit + to_unit",
            },
            "reference_ontology": self.ontology.planner_payload(),
            "collection_predicates": sorted(self.ontology.collection_predicates),
            "property_sources": ["entity", "relationship"],
            "operation_semantics": {
                "list": "select with property=null over a collection reference",
                "get_property": "select with property set and source=entity",
                "get_relation_property": (
                    "select with property set and source=relationship"
                ),
                "compare": "argmin or argmax with subject and other",
                "days_until": "annual_occurrence with mode=days",
            },
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
            and physical not in self._RESOLVER_METADATA_PROPERTIES
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
            if physical not in self._RESOLVER_METADATA_PROPERTIES
            and physical not in self._aliased_physical_properties
        )
        return frozenset(properties)

    def semantic_property_kinds(self, semantic: str) -> frozenset[str]:
        return frozenset(
            self.catalog.entity_field_type(entity_type, physical)
            for entity_type in self.catalog.entities
            if (physical := self.physical_property(entity_type, semantic)) is not None
        )

    def validation_code(self, request: SemanticFactRequest) -> PlannerValidationCode:
        """Return a stable, non-sensitive reason for semantic-plan rejection."""
        references = (request.subject,) + ((request.other,) if request.other else ())
        for reference in references:
            for step in reference.path:
                if self.physical_relation(step.relation) is None:
                    return "UNKNOWN_RELATION"
        if request.property is not None:
            if request.property_source == "relationship":
                if self._final_relation_kind(request.subject, request.property) is None:
                    return "UNKNOWN_PROPERTY"
            else:
                final_types = self._reference_entity_types(request.subject)
                if (
                    final_types is not None
                    and self._semantic_kind(final_types, request.property) is None
                ):
                    return "UNKNOWN_PROPERTY"
        for reference in references:
            step_types = self._base_entity_types(reference)
            for step in reference.path:
                next_types = self._traversal_target_types(step.relation, step_types)
                if next_types is None:
                    break
                for item in step.filters:
                    if item.property is None:
                        continue
                    if item.source == "entity":
                        if self._semantic_kind(next_types, item.property) is None:
                            return "UNKNOWN_PROPERTY"
                    else:
                        resolved = self.physical_relation(step.relation)
                        if resolved is not None and self.relation_property(
                            resolved[0], item.property
                        ) is None:
                            return "UNKNOWN_PROPERTY"
                step_types = next_types
        for item in request.filters:
            if item.property is None:
                continue
            final_types = self._reference_entity_types(request.subject)
            if item.source == "entity" and (
                final_types is None
                or self._semantic_kind(final_types, item.property) is None
            ):
                return "UNKNOWN_PROPERTY"
            if item.source == "relation" and self._final_relation_kind(
                request.subject, item.property
            ) is None:
                return "UNKNOWN_PROPERTY"
        return "VALID" if self.validates(request) else "INVALID_PLAN"

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
            anchor_types = self._base_entity_types(reference)
            step_types = anchor_types
            for step in reference.path:
                next_types = self._traversal_target_types(step.relation, step_types)
                if next_types is None:
                    return False
                for item in step.filters:
                    if item.source == "entity":
                        kind = self._semantic_kind(next_types, item.property)
                        if kind is None or not self._valid_predicate(item, kind):
                            return False
                        if item.value_from == "anchor":
                            anchor_kind = self._semantic_kind(
                                anchor_types,
                                item.value_property or item.property,
                            )
                            if anchor_kind is None or anchor_kind != kind:
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

        if request.other is not None and request.filters:
            return False
        if request.filters and not request.subject.path:
            return False
        for item in request.filters:
            if item.predicate is not None:
                definition = self.ontology.collection_predicates.get(item.predicate)
                if definition is None or not final_types[id(request.subject)].issubset(
                    definition.entity_types
                ):
                    return False
                continue
            assert item.property is not None
            if item.source == "entity":
                kind = self._semantic_kind(
                    final_types[id(request.subject)], item.property
                )
            else:
                kind = self._final_relation_kind(request.subject, item.property)
            if kind is None or not self._valid_predicate(item, kind):
                return False

        if request.property_source == "relationship":
            if (
                request.other is not None
                or not request.subject.path
                or request.property is None
            ):
                return False

        operation = OPERATORS[request.operation]
        collection_input = bool(request.subject.path or request.other is not None)
        if operation.input_shape == "collection" and not collection_input:
            return False
        if request.operation == "resolve_reference":
            return (
                request.property is None
                and request.other is None
                and not request.filters
                and request.property_source == "entity"
            )
        if request.operation == "select":
            if request.other is not None:
                return False
            if request.property is None:
                return bool(request.subject.path)
            return self._request_property_kind(
                request,
                final_types[id(request.subject)],
            ) is not None

        field_kind = (
            self._request_property_kind(
                request,
                final_types[id(request.subject)],
            )
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

    def _request_property_kind(
        self,
        request: SemanticFactRequest,
        entity_types: frozenset[str],
    ) -> str | None:
        if request.property_source == "relationship":
            return self._final_relation_kind(request.subject, request.property)
        return self._semantic_kind(entity_types, request.property)

    def _final_relation_kind(
        self,
        reference: SemanticReference,
        semantic_property: str | None,
    ) -> str | None:
        if not reference.path or semantic_property is None:
            return None
        resolved = self.physical_relation(reference.path[-1].relation)
        if resolved is None:
            return None
        relation = resolved[0]
        physical = self.relation_property(relation, semantic_property)
        return (
            self.catalog.relation_field_type(relation, physical)
            if physical is not None
            else None
        )

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
    """Strict semantic interpreter that never sees storage field names."""

    def __init__(self, ollama: Any, schema: SemanticSchemaRegistry) -> None:
        self.ollama = ollama
        self.schema = schema

    async def plan(
        self,
        messages: Sequence[Mapping[str, Any]],
        context: AgentRequestContext,
    ) -> SemanticPlannerOutcome:
        started = perf_counter()
        output_schema = _compact_json_schema(SemanticPlan.model_json_schema())
        reference_kind = output_schema["$defs"]["SemanticReference"]["properties"][
            "kind"
        ]
        reference_kind["enum"] = [
            item for item in reference_kind["enum"] if item != "entity_id"
        ]
        capabilities = self.schema.capability_payload()
        input_summary = planner_input_summary(capabilities)
        payload: Mapping[str, Any] | None = None
        plan: SemanticPlan | None = None
        structural_error: Exception | None = None
        attempts = 0
        for attempts in (1, 2):
            planner_messages = list(messages)
            if attempts == 2:
                planner_messages.append(
                    {
                        "role": "system",
                        "content": (
                            "Your previous response failed strict structural "
                            "validation. Return exactly one JSON object conforming "
                            "to the supplied output schema. Do not change the "
                            "meaning of the original user request."
                        ),
                    }
                )
            try:
                payload = await self.ollama.plan_semantic_fact(
                    planner_messages,
                    capabilities,
                    output_schema,
                    household_now=context.current_time.isoformat(),
                )
                plan = SemanticPlan.model_validate(payload)
                structural_error = None
                break
            except (ValueError, TypeError) as error:
                structural_error = error
        latency_ms = (perf_counter() - started) * 1000
        if plan is None:
            code = _structural_validation_code(structural_error)
            raise SemanticPlannerFailure(
                PlannerDiagnostics(
                    input_summary=input_summary,
                    output_raw=payload,
                    validation_result=code,
                    failure_detail=(
                        type(structural_error).__name__
                        if structural_error is not None
                        else "planner returned no plan"
                    ),
                    attempt_count=attempts,
                    latency_ms=latency_ms,
                )
            )
        if plan.request is not None:
            plan = plan.model_copy(
                update={"request": self._normalize_collection_predicates(plan.request)}
            )
        if plan.request is not None and any(
            reference.kind == "entity_id"
            for reference in (plan.request.subject, plan.request.other)
            if reference is not None
        ):
            raise SemanticPlannerFailure(
                PlannerDiagnostics(
                    input_summary=input_summary,
                    output_raw=payload,
                    normalized_plan=plan.model_dump(mode="json"),
                    validation_result="MODEL_ORIGINATED_ENTITY_ID",
                    failure_detail="The semantic planner cannot originate entity IDs",
                    attempt_count=attempts,
                    latency_ms=latency_ms,
                )
            )
        validation: PlannerValidationCode = "NOT_A_FACT"
        if plan.request is not None:
            validation = self.schema.validation_code(plan.request)
            if validation != "VALID":
                raise SemanticPlannerFailure(
                    PlannerDiagnostics(
                        input_summary=input_summary,
                        output_raw=payload,
                        normalized_plan=plan.model_dump(mode="json"),
                        validation_result=validation,
                        failure_detail="semantic request violates advertised capabilities",
                        attempt_count=attempts,
                        latency_ms=latency_ms,
                    )
                )
        diagnostics = PlannerDiagnostics(
            input_summary=input_summary,
            output_raw=payload,
            normalized_plan=plan.model_dump(mode="json"),
            validation_result=validation,
            attempt_count=attempts,
            latency_ms=latency_ms,
        )
        return SemanticPlannerOutcome(plan, latency_ms, diagnostics)

    def _normalize_collection_predicates(
        self,
        request: SemanticFactRequest,
    ) -> SemanticFactRequest:
        """Canonicalize a model's semantic predicate without interpreting language."""
        predicates = self.schema.ontology.collection_predicates

        def normalized(item: SemanticFilter) -> SemanticFilter:
            predicate = (
                self.schema.ontology.resolve_collection_predicate(item.predicate)
                if item.predicate is not None
                else None
            )
            if predicate is not None:
                return SemanticFilter(predicate=predicate)
            property_predicate = (
                self.schema.ontology.resolve_collection_predicate(item.property)
                if item.property is not None
                else None
            )
            if (
                item.predicate is None
                and property_predicate in predicates
                and item.operator == "eq"
                and item.value is None
                and item.source == "entity"
                and item.value_from is None
            ):
                return SemanticFilter(predicate=property_predicate)
            return item

        collection_filters = tuple(normalized(item) for item in request.filters)
        subject = self._normalize_context_reference(request.subject)
        if subject.path:
            final_step = subject.path[-1]
            moved = tuple(
                normalized(item)
                for item in final_step.filters
                if normalized(item).predicate is not None
            )
            retained = tuple(
                item
                for item in final_step.filters
                if normalized(item).predicate is None
            )
            if moved:
                final_step = final_step.model_copy(update={"filters": retained})
                subject = subject.model_copy(
                    update={"path": (*subject.path[:-1], final_step)}
                )
                collection_filters = (*collection_filters, *moved)
        default_scope_relations = {
            definition.default_scope_relation
            for item in collection_filters
            if item.predicate is not None
            and (
                definition := self.schema.ontology.collection_predicates.get(
                    item.predicate
                )
            )
            is not None
            and definition.default_scope_relation is not None
        }
        if (
            subject.kind == "current_household"
            and len(default_scope_relations) == 1
            and (
                not subject.path
                or (
                    len(subject.path) == 1
                    and not subject.path[0].filters
                    and subject.path[0].relation not in default_scope_relations
                )
            )
        ):
            subject = subject.model_copy(
                update={
                    "path": (
                        SemanticRelationStep(
                            relation=next(iter(default_scope_relations))
                        ),
                    )
                }
            )
        return request.model_copy(
            update={
                "subject": subject,
                "filters": collection_filters,
                "other": (
                    self._normalize_context_reference(request.other)
                    if request.other is not None
                    else None
                ),
            }
        )

    @staticmethod
    def _normalize_context_reference(
        reference: SemanticReference,
    ) -> SemanticReference:
        entity_type = {
            "self": "person",
            "assistant": "person",
            "current_household": "address",
        }.get(reference.kind)
        return (
            reference.model_copy(update={"entity_type": entity_type})
            if entity_type is not None and reference.entity_type != entity_type
            else reference
        )


class TierZeroSemanticParser:
    """Recognize a deliberately tiny set of canonical, high-frequency requests.

    Tier 0 is only a latency optimization. It intentionally does not attempt
    open-ended language understanding; every plan it emits must also be covered
    by the semantic planner evaluation suite.
    """

    def parse(self, text: str) -> SemanticFactRequest | None:
        normalized = _normalize_request(text)
        if normalized in {"我是谁", "who am i"}:
            return SemanticFactRequest(
                operation="resolve_reference",
                subject=SemanticReference(kind="self", entity_type="person"),
            )
        if normalized in {"你是谁", "who are you"}:
            return SemanticFactRequest(
                operation="resolve_reference",
                subject=SemanticReference(kind="assistant", entity_type="person"),
            )
        if normalized == "家里有几个人":
            return SemanticFactRequest(
                operation="count",
                subject=_household_members(),
            )
        if normalized == "家里都有谁":
            return SemanticFactRequest(
                operation="select",
                subject=_household_members(),
            )
        return None

    def canonical_utterances(self) -> tuple[str, ...]:
        """Expose the optimization surface for Tier-0/Tier-1 parity checks."""
        return (
            "我是谁",
            "Who am I?",
            "你是谁",
            "Who are you?",
            "家里有几个人",
            "家里都有谁",
        )


class EntityResolver:
    """Authoritative resolver from semantic references to canonical entities."""

    def __init__(
        self,
        schema: SemanticSchemaRegistry,
        *,
        max_records: int = 25,
    ) -> None:
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
            entities, relationship_records = await self._resolve(
                reference,
                context,
                execution,
                allow_empty_collection=allow_empty_collection,
            )
            if len(entities) > 1 and not expect_many:
                candidates = tuple(
                    [await execution.load_if_unnamed(item) for item in entities]
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
                relationship_records=tuple(relationship_records),
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
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if reference.kind == "assistant":
            entities = [
                {
                    "id": context.assistant_id,
                    "display_name": context.assistant_display_name,
                }
            ]
        elif reference.kind == "self":
            if context.speaker.speaker_id is None:
                raise _FactFailure("caller_context_missing")
            entities = [{"id": context.speaker.speaker_id}]
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
                    "speaker_id": context.speaker.speaker_id,
                    "household_id": context.speaker.household_id,
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

        anchors = list(entities)
        last_relation: str | None = None
        last_relationship_records: list[dict[str, Any]] = []
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
            step_relationship_records: list[dict[str, Any]] = []
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
                        step_relationship_records.append(dict(edge))
            entities = _unique_entities(related)
            entities = await self._filter_entities(
                entities,
                step.filters,
                execution,
                anchors,
            )
            if not entities:
                if allow_empty_collection:
                    return [], []
                raise _FactFailure(
                    "relationship_not_found",
                    evidence=FactEvidence(relationship=last_relation),
                )
            entity_ids = {
                str(entity["id"])
                for entity in entities
                if isinstance(entity.get("id"), str)
            }
            last_relationship_records = [
                edge
                for edge in step_relationship_records
                if isinstance(edge.get("related_entity"), Mapping)
                and edge["related_entity"].get("id") in entity_ids
            ]
        return entities, last_relationship_records

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
        anchors: Sequence[Mapping[str, Any]],
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
                await execution.load(entity) if needs_load else dict(entity)
            )
            predicates: list[bool] = []
            for item, physical in mapped:
                expected = item.value
                if item.value_from == "anchor":
                    expected = await self._anchor_property(
                        anchors,
                        item.value_property or item.property,
                        execution,
                    )
                predicates.append(
                    evaluate_predicate(item.operator, record.get(physical), expected)
                )
            if all(predicates):
                matched.append(record)
        return matched

    async def _anchor_property(
        self,
        anchors: Sequence[Mapping[str, Any]],
        semantic_property: str,
        execution: "_FactExecution",
    ) -> Any:
        if len(anchors) != 1:
            raise _FactFailure("ambiguous", candidates=tuple(anchors))
        anchor = await execution.load(anchors[0])
        physical = self.schema.physical_property(
            _entity_type(anchor),
            semantic_property,
        )
        if physical is None or anchor.get(physical) is None:
            raise _FactFailure(
                "property_unavailable",
                evidence=FactEvidence(semantic_property=semantic_property),
                missing=(semantic_property,),
            )
        return anchor[physical]

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
        self.resolver = resolver or EntityResolver(
            schema,
            max_records=max_records,
        )

    async def execute(
        self,
        request: SemanticFactRequest,
        context: AgentRequestContext,
    ) -> tuple[FactResult, int, float, float]:
        if not self.schema.validates(request):
            return FactResult("semantic_plan_unsupported"), 0, 0, 0
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
        relationship_records = [dict(item) for item in resolution.relationship_records]
        entity_resolution_ms = (perf_counter() - resolution_started) * 1000
        computation_started = perf_counter()
        try:
            if request.filters:
                entities, relationship_records = await self._filter_collection(
                    request,
                    entities,
                    relationship_records,
                    context,
                    execution,
                )
            result = await self._operate(
                request,
                entities,
                other_entities,
                relationship_records,
                context,
                execution,
            )
        except _FactFailure as error:
            result = FactResult(
                error.status,
                evidence=error.evidence,
                missing_requirements=error.missing,
                candidates=error.candidates,
            )
        computation_ms = (perf_counter() - computation_started) * 1000
        return result, execution.query_count, entity_resolution_ms, computation_ms

    async def _operate(
        self,
        request: SemanticFactRequest,
        entities: list[dict[str, Any]],
        other_entities: list[dict[str, Any]],
        relationship_records: list[dict[str, Any]],
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
            visible = [await execution.load_if_unnamed(item) for item in entities]
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
            if request.property_source == "relationship":
                relationship = self._singular_relationship(
                    relationship_records,
                    evidence,
                )
                if isinstance(relationship, FactResult):
                    return relationship
                property_result = self._relationship_property(
                    relationship,
                    request,
                    evidence,
                )
                if isinstance(property_result, FactResult):
                    return property_result
                return FactResult("found", property_result, evidence)
            singular = await self._singular(entities, request, execution)
            if isinstance(singular, FactResult):
                return singular
            property_result = self._property(singular, request.property or "")
            if isinstance(property_result, FactResult):
                return property_result
            return FactResult("found", property_result, evidence)
        definition = OPERATORS[request.operation]
        records = (
            list(relationship_records)
            if request.property_source == "relationship"
            else list(entities)
        )
        if request.other is not None:
            left = await self._singular(entities, request, execution)
            right = await self._singular(other_entities, request, execution)
            if isinstance(left, FactResult):
                return left
            if isinstance(right, FactResult):
                return right
            records = [left, right]
        elif definition.input_shape == "scalar":
            singular = (
                self._singular_relationship(relationship_records, evidence)
                if request.property_source == "relationship"
                else await self._singular(entities, request, execution)
            )
            if isinstance(singular, FactResult):
                return singular
            records = [dict(singular)]
        else:
            records = (
                records
                if request.property_source == "relationship"
                else [await execution.load(item) for item in records]
            )
        normalized: list[dict[str, Any]] = []
        for record in records:
            value = (
                self._relationship_property(record, request, evidence)
                if request.property_source == "relationship"
                else self._property(record, request.property)
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

    async def _filter_collection(
        self,
        request: SemanticFactRequest,
        entities: list[dict[str, Any]],
        relationship_records: list[dict[str, Any]],
        context: AgentRequestContext,
        execution: "_FactExecution",
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        matched: list[dict[str, Any]] = []
        for entity in entities:
            entity_id = entity.get("id")
            edges = [
                edge
                for edge in relationship_records
                if _related_entity_id(edge) == entity_id
            ]
            include = True
            for item in request.filters:
                if item.predicate is not None:
                    include = await self._semantic_predicate_matches(
                        request,
                        item.predicate,
                        entity,
                        edges,
                        context,
                        execution,
                    )
                elif item.source == "relation":
                    include = self._relation_filter_matches(request, item, edges)
                else:
                    include = await self._entity_filter_matches(
                        item,
                        entity,
                        execution,
                    )
                if not include:
                    break
            if include:
                matched.append(entity)
        matched_ids = {
            entity.get("id")
            for entity in matched
            if isinstance(entity.get("id"), str)
        }
        return matched, [
            edge
            for edge in relationship_records
            if _related_entity_id(edge) in matched_ids
        ]

    async def _entity_filter_matches(
        self,
        item: SemanticFilter,
        entity: Mapping[str, Any],
        execution: "_FactExecution",
    ) -> bool:
        assert item.property is not None
        physical = self.schema.physical_property(_entity_type(entity), item.property)
        if physical is None:
            raise _FactFailure("filter_unsupported", missing=(item.property,))
        record = (
            dict(entity)
            if physical in entity
            else await execution.load(entity)
        )
        if physical not in record or record.get(physical) is None:
            raise _FactFailure("filter_input_missing", missing=(item.property,))
        return evaluate_predicate(item.operator, record.get(physical), item.value)

    def _relation_filter_matches(
        self,
        request: SemanticFactRequest,
        item: SemanticFilter,
        edges: Sequence[Mapping[str, Any]],
    ) -> bool:
        assert request.subject.path and item.property is not None
        resolved = self.schema.physical_relation(request.subject.path[-1].relation)
        if resolved is None:
            raise _FactFailure("filter_unsupported", missing=(item.property,))
        physical = self.schema.relation_property(resolved[0], item.property)
        if physical is None:
            raise _FactFailure("filter_unsupported", missing=(item.property,))
        values = [edge.get(physical) for edge in edges if edge.get(physical) is not None]
        if not values:
            raise _FactFailure("filter_input_missing", missing=(item.property,))
        return any(evaluate_predicate(item.operator, value, item.value) for value in values)

    async def _semantic_predicate_matches(
        self,
        request: SemanticFactRequest,
        predicate: str,
        entity: Mapping[str, Any],
        edges: Sequence[Mapping[str, Any]],
        context: AgentRequestContext,
        execution: "_FactExecution",
    ) -> bool:
        definition = self.schema.ontology.collection_predicates.get(predicate)
        if definition is None:
            raise _FactFailure("filter_unsupported", missing=(predicate,))
        relation_name = (
            self.schema.physical_relation(request.subject.path[-1].relation)
            or (None, None)
        )[0]
        role_property = (
            self.schema.relation_property(
                relation_name,
                definition.relation_property,
            )
            if relation_name is not None
            else None
        )
        recognized_roles = {
            edge.get(role_property)
            for edge in edges
            if role_property is not None
            and edge.get(role_property) in definition.recognized_values
        }
        if recognized_roles:
            decisions = {
                role in definition.matching_values for role in recognized_roles
            }
            if len(decisions) != 1:
                raise _FactFailure("filter_input_missing", missing=(predicate,))
            return decisions.pop()

        fallback = definition.fallback
        physical = self.schema.physical_property(
            _entity_type(entity), fallback.property
        )
        if physical is None:
            raise _FactFailure("filter_unsupported", missing=(predicate,))
        record = (
            dict(entity)
            if physical in entity
            else await execution.load(entity)
        )
        raw_value = record.get(physical)
        if raw_value is None:
            raise _FactFailure(
                "filter_input_missing",
                missing=(predicate, fallback.property),
            )
        normalized = {"value": raw_value}
        try:
            transform = OPERATORS[fallback.transform]
            transform.validate(
                field="value",
                field_kind=infer_field_kind([raw_value]),
                parameters={"reference": "household_now"},
            )
            derived = execute_operator(
                fallback.transform,
                OperatorInput(
                    records=[normalized],
                    field="value",
                    reference="household_now",
                    now=context.current_time,
                ),
            )
        except (OperatorValidationError, OperatorExecutionError, TypeError, ValueError):
            raise _FactFailure(
                "filter_input_missing",
                missing=(predicate, fallback.property),
            ) from None
        threshold = self.schema.ontology.policy_values[
            fallback.value_from_policy
        ]
        return evaluate_predicate(fallback.operator, derived, threshold)

    @staticmethod
    def _singular_relationship(
        records: Sequence[Mapping[str, Any]],
        evidence: FactEvidence,
    ) -> Mapping[str, Any] | FactResult:
        if not records:
            return FactResult("relationship_not_found", evidence=evidence)
        if len(records) > 1:
            return FactResult("ambiguous", evidence=evidence)
        return records[0]

    def _relationship_property(
        self,
        relationship: Mapping[str, Any],
        request: SemanticFactRequest,
        evidence: FactEvidence,
    ) -> Any | FactResult:
        assert request.subject.path and request.property is not None
        semantic_relation = request.subject.path[-1].relation
        resolved = self.schema.physical_relation(semantic_relation)
        physical = (
            self.schema.relation_property(resolved[0], request.property)
            if resolved is not None
            else None
        )
        if (
            physical is None
            or physical not in relationship
            or relationship.get(physical) is None
        ):
            return FactResult(
                "relation_property_unavailable",
                evidence=evidence,
                missing_requirements=(request.property,),
            )
        return relationship[physical]

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
                    await execution.load_if_unnamed(item)
                    for item in entities
                ]
            )
            return FactResult("ambiguous", candidates=loaded)
        record = (
            await execution.load(entities[0])
            if load_full
            else await execution.load_if_unnamed(entities[0])
        )
        return record

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
        if result.status == "relation_property_unavailable":
            if result.evidence.relationship == "spouse" and (
                "start_date" in result.missing_requirements
            ):
                return "家庭资料中有配偶关系记录，但目前没有记录结婚日期。"
            label = _property_label(result.missing_requirements)
            return f"家庭资料中有对应关系记录，但目前没有记录{label}。"
        if result.status == "filter_input_missing":
            if "adult" in result.missing_requirements:
                return "目前缺少足够的年龄或家庭角色资料，无法确定成年人数量。"
            if "minor" in result.missing_requirements:
                return "目前缺少足够的年龄或家庭角色资料，无法确定未成年人数。"
            label = _property_label(result.missing_requirements) or "筛选所需资料"
            return f"目前缺少足够的{label}，无法可靠完成筛选。"
        if result.status == "filter_unsupported":
            return "当前语义查询协议不支持这个筛选条件。"
        if result.status == "operator_unsupported":
            return "当前语义查询协议不支持这项计算。"
        if result.status == "semantic_plan_unsupported":
            return "老管家无法将这个请求转换为受支持的家庭事实查询。"
        if result.status == "ambiguous":
            names = "、".join(_name(item, "zh") for item in result.candidates)
            return f"找到多个符合条件的家庭成员：{names}。请说明您指哪一位。"
        if result.status == "computation_input_missing":
            if (
                request.operation in {"argmin", "argmax"}
                and request.property == "birth_date"
            ):
                return "目前缺少部分家庭成员的出生日期，因此无法可靠判断年龄排序。"
            label = _property_label(result.missing_requirements) or "所需资料"
            return f"目前没有足够的{label}来完成这项计算。"
        if result.status == "computation_impossible":
            label = _property_label(result.missing_requirements)
            suffix = f"，缺少{label}" if label else ""
            return f"家庭资料不足以完成这项计算{suffix}。"
        if request.operation == "count":
            count = int(result.value)
            noun = _count_noun(request)
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
            if (
                request.property_source == "relationship"
                and request.subject.path
                and request.subject.path[-1].relation == "spouse"
                and request.property == "start_date"
            ):
                return f"您和配偶的结婚日期是{result.value}。"
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
                "relation_property_unavailable": (
                    "The relationship is recorded, but that semantic property is unavailable."
                ),
                "filter_input_missing": (
                    "Required evidence is unavailable for deterministic filtering."
                ),
                "filter_unsupported": "That semantic filter is not supported.",
                "operator_unsupported": "That semantic operator is not supported.",
                "semantic_plan_unsupported": (
                    "The request could not be expressed in the supported household query protocol."
                ),
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
        planner: SemanticFactPlanner,
        parser: TierZeroSemanticParser | None = None,
        renderer: FactRenderer | None = None,
        tier_zero_enabled: bool = True,
    ) -> None:
        self.engine = engine
        self.parser = parser or TierZeroSemanticParser()
        self.renderer = renderer or FactRenderer()
        self.planner = planner
        self.tier_zero_enabled = tier_zero_enabled

    async def try_answer(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        context: AgentRequestContext,
        request_id: str = "-",
    ) -> FactAnswer | None:
        started = perf_counter()
        latest = _latest_user_text(messages)
        routing_started = perf_counter()
        parse_started = perf_counter()
        request = self.parser.parse(latest) if self.tier_zero_enabled else None
        semantic_parse_ms = (perf_counter() - parse_started) * 1000
        tier = 0
        llm_ms = 0.0
        llm_call_count = 0
        planner_diagnostics: PlannerDiagnostics | None = None
        if request is None:
            tier = 1
            llm_started = perf_counter()
            try:
                outcome = await self.planner.plan(messages, context)
                plan = outcome.plan
                llm_ms = outcome.latency_ms
                planner_diagnostics = outcome.diagnostics
                llm_call_count = outcome.diagnostics.attempt_count
            except SemanticPlannerFailure as error:
                planner_diagnostics = error.diagnostics
                llm_ms = error.diagnostics.latency_ms
                llm_call_count = error.diagnostics.attempt_count
                logger.warning(
                    "semantic_plan_invalid request_id=%s validation=%s attempts=%d",
                    safe_log_token(request_id),
                    safe_log_token(error.diagnostics.validation_result),
                    error.diagnostics.attempt_count,
                )
                return self._failure_answer(
                    context,
                    started,
                    request_id=request_id,
                    llm_ms=llm_ms,
                    llm_call_count=llm_call_count,
                    planner_diagnostics=planner_diagnostics,
                )
            except Exception as error:
                llm_ms = (perf_counter() - llm_started) * 1000
                llm_call_count = max(llm_call_count, 1)
                logger.warning(
                    "semantic_plan_invalid request_id=%s error=%s",
                    safe_log_token(request_id),
                    safe_log_token(type(error).__name__),
                )
                return self._failure_answer(
                    context,
                    started,
                    request_id=request_id,
                    llm_ms=llm_ms,
                    llm_call_count=llm_call_count,
                    planner_diagnostics=planner_diagnostics,
                )
            if not plan.requires_fact:
                return None
            request = plan.request
            if request is None or not self.engine.schema.validates(request):
                return self._failure_answer(
                    context,
                    started,
                    request=request,
                    request_id=request_id,
                    llm_ms=llm_ms,
                    llm_call_count=llm_call_count,
                    planner_diagnostics=planner_diagnostics,
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
        _log_fact_query(
            request_id,
            request,
            result,
            timings,
            planner_diagnostics=planner_diagnostics,
        )
        return FactAnswer(request, result, rendered, timings, planner_diagnostics)

    def _failure_answer(
        self,
        context: AgentRequestContext,
        started: float,
        *,
        request: SemanticFactRequest | None = None,
        request_id: str,
        llm_ms: float,
        llm_call_count: int,
        planner_diagnostics: PlannerDiagnostics | None = None,
        status: FactStatus = "semantic_plan_unsupported",
    ) -> FactAnswer:
        safe_request = request or SemanticFactRequest(
            operation="resolve_reference",
            subject=SemanticReference(kind="self", entity_type="person"),
        )
        result = FactResult(status)
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
        _log_fact_query(
            request_id,
            safe_request,
            result,
            timings,
            planner_diagnostics=planner_diagnostics,
        )
        return FactAnswer(
            safe_request,
            result,
            rendered,
            timings,
            planner_diagnostics,
        )

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

    async def load(self, entity: Mapping[str, Any]) -> dict[str, Any]:
        entity_id = entity.get("id")
        if not isinstance(entity_id, str):
            raise _FactFailure("entity_not_found")
        if ":" not in entity_id:
            return dict(entity)
        records = await self.records("get_entity", {"entity_id": entity_id})
        if not records:
            raise _FactFailure("entity_not_found")
        return records[0]

    async def load_if_unnamed(self, entity: Mapping[str, Any]) -> dict[str, Any]:
        if any(entity.get(field) for field in ("display_name", "name", "full_name")):
            return dict(entity)
        return await self.load(entity)


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


def planner_input_summary(capabilities: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize planner affordances without copying household fact values."""
    relation_properties = capabilities.get("semantic_relation_properties", {})
    semantic_properties = capabilities.get("semantic_properties", {})
    return {
        "references": list(capabilities.get("references", ())),
        "entity_types": list(capabilities.get("entity_types", ())),
        "relations": list(capabilities.get("semantic_relations", ())),
        "entity_properties": {
            str(entity_type): list(properties)
            for entity_type, properties in semantic_properties.items()
        }
        if isinstance(semantic_properties, Mapping)
        else {},
        "relationship_properties": {
            str(relation): list(properties)
            for relation, properties in relation_properties.items()
        }
        if isinstance(relation_properties, Mapping)
        else {},
        "operations": list(capabilities.get("operations", ())),
        "collection_predicates": list(
            capabilities.get("collection_predicates", ())
        ),
    }


def _compact_json_schema(value: Any) -> Any:
    """Remove model-irrelevant prose while preserving JSON Schema constraints."""
    if isinstance(value, Mapping):
        return {
            key: _compact_json_schema(item)
            for key, item in value.items()
            if key not in {"title", "description", "default"}
        }
    if isinstance(value, list):
        return [_compact_json_schema(item) for item in value]
    return value


def _structural_validation_code(
    error: Exception | None,
) -> PlannerValidationCode:
    if isinstance(error, ValidationError):
        if any(item.get("loc", ())[-1:] == ("operation",) for item in error.errors()):
            return "UNSUPPORTED_OPERATION"
    return "MALFORMED_OUTPUT"


def _log_fact_query(
    request_id: str,
    request: SemanticFactRequest,
    result: FactResult,
    timings: FactTimings,
    *,
    planner_diagnostics: PlannerDiagnostics | None = None,
) -> None:
    logger.info(
        "fact_query request_id=%s tier=%d operation=%s semantic_plan=%s "
        "db_queries=%d llm_calls=%d routing_ms=%.2f semantic_parse_ms=%.2f "
        "entity_resolution_ms=%.2f fact_query_ms=%.2f computation_ms=%.2f "
        "render_ms=%.2f llm_ms=%.2f total_ms=%.2f status=%s failure_stage=%s "
        "planner_validation=%s planner_attempts=%d",
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
        safe_log_token(_failure_stage(result.status) or "none"),
        safe_log_token(
            planner_diagnostics.validation_result
            if planner_diagnostics is not None
            else "not_run"
        ),
        planner_diagnostics.attempt_count if planner_diagnostics is not None else 0,
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
    for item in payload.get("filters", []):
        if isinstance(item, dict) and item.get("value") is not None:
            item["value"] = "[redacted]"
    return payload


def _failure_stage(status: FactStatus) -> str | None:
    return {
        "found": None,
        "caller_context_missing": "context",
        "entity_not_found": "entity_resolution",
        "relationship_not_found": "relationship_resolution",
        "property_unavailable": "entity_property",
        "relation_property_unavailable": "relationship_property",
        "ambiguous": "entity_resolution",
        "filter_input_missing": "filter",
        "filter_unsupported": "filter_validation",
        "operator_unsupported": "operator_validation",
        "computation_input_missing": "computation_input",
        "computation_impossible": "computation",
        "semantic_plan_unsupported": "semantic_plan_validation",
    }[status]


def _normalize_request(text: str) -> str:
    normalized = text.casefold().strip()
    normalized = re.sub(r"^(?:请问|能否|能不能|能告诉我一下|告诉我一下|麻烦)", "", normalized)
    normalized = normalized.strip(" \t\r\n，,。.!！?？")
    return re.sub(r"\s+", " ", normalized)


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


def _count_noun(request: SemanticFactRequest) -> str:
    predicates = {item.predicate for item in request.filters}
    if "adult" in predicates:
        return "位成年人"
    if "minor" in predicates:
        return "个未成年人"
    reference = request.subject
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
        "start_date": "开始日期",
        "end_date": "结束日期",
        "adult": "成年人判断资料",
        "minor": "未成年人判断资料",
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
