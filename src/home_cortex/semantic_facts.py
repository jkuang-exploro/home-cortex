"""Semantic household fact IR, interpretation, and deterministic execution."""

from __future__ import annotations

import json
import logging
import re
from copy import deepcopy

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from time import perf_counter
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .profiling import stage
from .display import resolve_display_name
from .edge_schema import EdgeSchemaRegistry, UnknownEdgeSchemaError
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
from .semantic_display import SemanticDisplay
from .semantic_contracts import ResolvedSemanticContract
from .text import latest_user_message, safe_log_token

logger = logging.getLogger("uvicorn.error.home_cortex.semantic_facts")

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
FactOperation = Literal[
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
    "discourse",
    "unresolved",
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
    "filter_input_missing",
    "filter_unsupported",
    "collection_incomplete",
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

    @model_validator(mode="after")
    def validate_operation(self) -> "SemanticFactRequest":
        if self.operation not in OPERATORS:
            raise ValueError("operation is not in the generic operator registry")
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
class ResolutionResult:
    status: ResolutionStatus
    entities: tuple[Mapping[str, Any], ...] = ()
    entity_ids: tuple[str, ...] = ()
    evidence: FactEvidence = FactEvidence()
    candidates: tuple[Mapping[str, Any], ...] = ()
    missing_requirements: tuple[str, ...] = ()
    relationship_records: tuple[Mapping[str, Any], ...] = ()
    content_groups: tuple[FactContentGroup, ...] = ()


class SemanticSchemaRegistry:
    """Map stable semantic concepts to deployment-specific schema names."""

    _RESOLVER_METADATA_PROPERTIES = frozenset({"aliases", "appellations", "collapse"})

    def __init__(
        self,
        catalog: RuntimeSchemaCatalog,
        ontology: SemanticOntology | None = None,
    ) -> None:
        self.catalog = catalog
        self.ontology = ontology or SemanticOntology.load_default()
        self.edge_registry = catalog.edge_registry or EdgeSchemaRegistry.load_default()
        self._aliased_physical_properties = frozenset(
            physical
            for definition in self.ontology.properties.values()
            for physical in definition.fields
        )
        self._property_cache: dict[tuple[str, str], str | None] = {}
        self._capability_cache: dict[str, Any] | None = None
        self._planner_capability_cache: dict[str, Any] | None = None
        self._planner_schema_cache: dict[str, Any] | None = None
        self.contracts = None
        if self.ontology.version == 2:
            self.contracts = ResolvedSemanticContract(self)
        self._available_predicates = frozenset(
            name for name, definition in self.ontology.collection_predicates.items()
            if self.contracts is None or all(
                (owner, definition.fallback.property) in self.contracts.entity_bindings
                for owner in definition.entity_types
            )
        )
        self._available_concepts = self._resolve_available_concepts()

    def _ontology_payload(self) -> dict[str, Any]:
        payload = self.ontology.planner_payload()
        payload['reference_concepts'] = {
            name: value for name, value in payload['reference_concepts'].items()
            if name in self._available_concepts
        }
        payload['collection_predicates'] = {
            name: value for name, value in payload['collection_predicates'].items()
            if name in self._available_predicates
        }
        if self.contracts is not None:
            available = self.contracts.payload()
            payload['properties'] = {name: value for name, value in payload['properties'].items() if name in available}
        return payload

    def _resolve_available_concepts(self) -> frozenset[str]:
        if self.contracts is None:
            return frozenset(self.ontology.reference_concepts)
        available = set()
        for name, concept in self.ontology.reference_concepts.items():
            for root in self.catalog.entities:
                types = frozenset({root})
                valid = True
                for step in concept.path:
                    targets = self._traversal_target_types(step.relation, types)
                    if not targets:
                        valid = False
                        break
                    for item in step.filters:
                        contract = self.contracts.properties[item.property]
                        if item.source == 'entity':
                            if not targets.issubset(contract.entities):
                                raise ValueError(f'Invalid entity owner in concept {name}')
                            valid &= self._semantic_kind(targets, item.property) is not None
                            if item.value_from:
                                anchor = self.contracts.properties[item.value_property or item.property]
                                valid &= (root, item.value_property or item.property) in self.contracts.entity_bindings
                                if contract.type != anchor.type:
                                    raise ValueError(f'Incompatible anchor in concept {name}')
                        else:
                            valid &= (step.relation, item.property) in self.contracts.relation_bindings
                    types = targets
                if valid:
                    available.add(name)
                    break
        return frozenset(available)

    def contract_error(self, request: SemanticFactRequest) -> str | None:
        """Internal, non-sensitive diagnostics; never rewrites the submitted IR."""
        names = {item.predicate for item in request.filters if item.predicate}
        for name in names:
            definition = self.ontology.collection_predicates.get(name)
            if definition and names.intersection(definition.disjoint_with):
                return 'CONTRADICTORY_PREDICATES'
        if self.contracts is None:
            return None
        references = (request.subject,) + ((request.other,) if request.other else ()) + request.exclude
        for reference in references:
            types = self._base_entity_types(reference)
            anchor_types = types
            for step in reference.path:
                types = self._traversal_target_types(step.relation, types)
                if not types:
                    return 'INVALID_PATH'
                for item in step.filters:
                    if error := self._contract_filter_error(item, types, step.relation, anchor_types):
                        return error
        types = self._reference_entity_types(request.subject)
        relation = request.subject.path[-1].relation if request.subject.path else None
        for item in request.filters:
            if item.predicate:
                continue  # Existing predicate applicability/shape checks remain authoritative.
            if error := self._contract_filter_error(item, types or frozenset(), relation, frozenset()):
                return error
        return None

    def _contract_filter_error(self, item, types, relation, anchor_types) -> str | None:
        if item.transform == "date_difference":
            contract = self.contracts.properties.get(item.property)
            if contract is None:
                return "UNKNOWN_PROPERTY"
            if not contract.type.kinds.intersection({"date", "datetime"}):
                return "INVALID_LITERAL_TYPE"
            if item.source != "entity" or (
                types and any((owner, item.property) not in self.contracts.entity_bindings for owner in types)
            ):
                return "PROPERTY_NOT_APPLICABLE"
            return None
        contract = self.contracts.properties.get(item.property)
        if contract is None:
            return 'UNKNOWN_PROPERTY'
        if item.source == 'entity':
            if not types or any((owner, item.property) not in self.contracts.entity_bindings for owner in types):
                return 'PROPERTY_NOT_APPLICABLE'
        elif (relation, item.property) not in self.contracts.relation_bindings:
            return 'PROPERTY_NOT_APPLICABLE'
        if item.value_from:
            name = item.value_property or item.property
            other = self.contracts.properties.get(name)
            if not anchor_types or other is None or other.type != contract.type or (
                (contract.values or other.values)
                and {value for value, _ in contract.values} != {value for value, _ in other.values}
            ) or any(
                (owner, name) not in self.contracts.entity_bindings for owner in anchor_types
            ):
                return 'INVALID_ANCHOR_TYPE'
        return contract.literal_error(item.operator, item.value, anchor=item.value_from is not None)

    def value_valid(self, semantic: str, value: Any) -> bool:
        return self.contracts is None or (
            semantic in self.contracts.properties and self.contracts.properties[semantic].accepts(value)
        )

    def validate_filter_value(self, semantic: str, value: Any) -> None:
        if self.contracts is None:
            return
        if value is None:
            raise _FactFailure('filter_input_missing', missing=(semantic,))
        if not self.value_valid(semantic, value):
            raise _FactFailure('filter_unsupported', missing=(semantic,))

    def physical_property(self, entity_type: str, semantic: str) -> str | None:
        if self.ontology.version == 2:
            definition = self.ontology.properties.get(semantic)
            if definition is None or entity_type not in definition.contract.entities:
                return None
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
        try:
            resolved = self.edge_registry.resolve(public_name)
        except UnknownEdgeSchemaError:
            return None
        if resolved.schema.id not in self.catalog.relations:
            return None
        direction = None if resolved.schema.symmetric else "in" if resolved.inverse else "out"
        return resolved.schema.id, direction

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
        if self.ontology.version == 2:
            return ()
        return (semantic,)

    def capability_payload(self) -> dict[str, Any]:
        if self._capability_cache is not None:
            return deepcopy(self._capability_cache)
        relations = {
            semantic
            for semantic in self.ontology.base_relations
            if self.physical_relation(semantic) is not None
        }
        self._capability_cache = {
            "references": [
                kind for kind in get_args(ReferenceKind) if kind != "entity_id"
            ],
            "entity_types": sorted(self.catalog.entities),
            "semantic_properties": {
                entity_type: sorted(self.semantic_properties(entity_type))
                for entity_type in self.catalog.entities
            },
            "semantic_relations": sorted(relations),
            "semantic_relation_properties": {
                relation: sorted(self.semantic_relation_properties(relation))
                for relation in relations
            },
            "operations": sorted(get_args(FactOperation)),
            "operation_requirements": {
                "select": "property=null returns all matching entities, including zero or many; property set returns a single stored property value. Filter properties are not output projections.",
                "resolve_reference": "returns exactly one entity; not a list of matching entities",
                "same_entity": "property=null; two resolved references subject and other; returns whether they are the same entity; not identity introduction and not a stored property",
                "argmin": "ordered property required; collection subject OR two references subject and other; returns entity",
                "argmax": "ordered property required; collection subject OR two references subject and other; returns entity",
                "completed_years": "legacy structured-call alias; interpreter uses date_difference with mode=years",
                "duration": "date property + mode days|seconds",
                "annual_occurrence": "date property; mode=days only for countdown",
                "date_add": "date property + strict integer amount + mode years|months|days; invalid target day becomes the following month's first day, return specified date even in past",
                "date_difference": "one entity OR relationship date to household_now; mode explicitly chooses years|months|days|seconds. Calendar years/months count full anniversaries, signed toward zero; never divide days by a fixed ratio. Age, tenure and elapsed relationship time use this same operation.",
                "unit_conversion": "numeric property + from_unit + to_unit",
            },
            "reference_ontology": self._ontology_payload(),
            "collection_predicates": sorted(self._available_predicates),
            "property_sources": ["entity", "relationship"],
        }
        if self.contracts is not None:
            self._capability_cache['property_contracts'] = self.contracts.payload()
        return deepcopy(self._capability_cache)

    @stage("schema.capabilities")
    def planner_capability_payload(self) -> dict[str, Any]:
        """Serialize the executable grammar, without storage or redundant aliases."""
        if self._planner_capability_cache is None:
            full = self.capability_payload()
            ontology = full["reference_ontology"]
            self._planner_capability_cache = {
                "references": full["references"],
                "reference_kinds": {
                    "self": "authenticated speaker; first-person I/me/我 only",
                    "assistant": "this household assistant; second-person you/你/您 addressing the agent",
                    "current_household": "configured home address",
                    "named_entity": "verbatim spoken name, never a pronoun",
                    "discourse": "prior user-turn focus; not a substitute for you/I",
                    "unresolved": "pronoun or description that cannot be grounded",
                },
                "composition": {
                    "projection": "each maps one scalar operation or select(property) over a collection; preserves per-entity rows including missing data",
                    "exclude": "up to eight resolved references subtracted by identity; other is only a comparison operand",
                    "discourse": "turn_offset 1..8 selects that prior user turn's trusted resolved focus; cardinality single or collection; entity_type required; never supply IDs",
                },
                "entity_types": full["entity_types"],
                "operations": [name for name in full["operations"] if name not in {"duration", "completed_years"}],
                "operation_requirements": {name: value for name, value in full["operation_requirements"].items() if name not in {"duration", "completed_years"}},
                "filter_requirements": {
                    "composition": "request.filters restricts the resolved collection before select/count/aggregation; all conditions are AND. The outer property selects the output, not the field used by a filter.",
                    "date_range": "date/datetime property with value=[inclusive_start, exclusive_end]; use ISO dates. A calendar year Y is [Y-01-01, (Y+1)-01-01).",
                    "derived_age": "满N岁/N岁以上 is {property:birth_date, transform:date_difference, mode:years, operator:gte, value:N}. N岁以下 uses lt/lte. The executor computes completed units from Household now. Do not invent an ISO cutoff or a birth-year date_range.",
                },
                "property_ownership": {
                    "entity": full["semantic_properties"],
                    "relationship": full["semantic_relation_properties"],
                },
                "relations": full["semantic_relations"],
                "relation_signatures": {
                    relation: {
                        source: sorted(targets)
                        for source in self.catalog.entities
                        if (targets := self._traversal_target_types(relation, frozenset({source}))) is not None
                    }
                    for relation in full["semantic_relations"]
                },
                "property_aliases": ontology["properties"],
                "reference_concepts": ontology["reference_concepts"],
                "collection_predicates": ontology["collection_predicates"],
            }
            if any(
                definition.disjoint_with
                for definition in self.ontology.collection_predicates.values()
            ):
                self._planner_capability_cache['predicate_disjointness'] = {
                    name: list(definition.disjoint_with)
                    for name, definition in self.ontology.collection_predicates.items()
                    if definition.disjoint_with and name in self._available_predicates
                }
            if self.contracts is not None:
                self._planner_capability_cache['property_contracts'] = self.contracts.payload()
        return deepcopy(self._planner_capability_cache)

    @stage("schema.output")
    def planner_output_schema(self) -> dict[str, Any]:
        """Constrain model output to this deployment's semantic vocabulary.

        Ownership is an explicit choice; predicates and field comparisons have
        separate productions. Runtime validation still checks types and paths.
        """
        if self._planner_schema_cache is None:
            schema = _planner_output_schema()
            definitions = schema["$defs"]
            full = self.capability_payload()
            properties = sorted({
                prop
                for group in (full["semantic_properties"], full["semantic_relation_properties"])
                for values in group.values()
                for prop in values
            })
            definitions["SemanticRelationStep"]["properties"]["relation"] = {
                "type": "string", "enum": full["semantic_relations"]
            }
            definitions["SemanticReference"]["properties"]["path"]["items"] = {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "concept": {"type": "string", "enum": sorted(self._available_concepts)},
                    "filters": {"type": "array", "items": {"$ref": "#/$defs/SemanticFilter"}},
                },
                "required": ["concept"],
            }
            field_filter = definitions["SemanticFilter"]
            field_filter["properties"].pop("predicate")
            field_filter["properties"].pop("transform", None)
            field_filter["properties"].pop("mode", None)
            field_filter["properties"]["property"] = {"type": "string", "enum": properties}
            field_filter["properties"]["value_property"] = {
                "anyOf": [{"type": "string", "enum": properties}, {"type": "null"}]
            }
            field_filter["required"] = ["property"]
            definitions["SemanticFilter"] = {"anyOf": [
                field_filter,
                {
                    "type": "object", "additionalProperties": False,
                    "properties": {"predicate": {
                        "type": "string", "enum": full["collection_predicates"]
                    }},
                    "required": ["predicate"],
                },
            ]}
            # Anchor-relative comparisons belong to a traversal step. Collection
            # filters have no anchor operand in their executor contract.
            date_properties = [
                name for name in properties
                if name in {"birth_date", "start_date", "end_date"}
            ]
            derived_age = {
                "type": "object",
                "additionalProperties": False,
                "description": (
                    "Compare completed calendar units of a date property "
                    "against Household now. Age thresholds use this form."
                ),
                "properties": {
                    "property": {"type": "string", "enum": date_properties or properties},
                    "transform": {"type": "string", "const": "date_difference"},
                    "mode": {"type": "string", "enum": ["years", "months", "days"]},
                    "operator": {"type": "string", "enum": ["gt", "gte", "lt", "lte", "eq"]},
                    "value": {"type": "integer", "minimum": 0, "maximum": 120},
                    "source": {"type": "string", "enum": ["entity"]},
                },
                "required": ["property", "transform", "mode", "operator", "value"],
            }
            definitions["SemanticFilter"] = {"anyOf": [
                *definitions["SemanticFilter"]["anyOf"],
                derived_age,
            ]}
            definitions["SemanticCollectionFilter"] = {"anyOf": [
                {
                    **field_filter,
                    "properties": {
                        key: value for key, value in field_filter["properties"].items()
                        if key not in {"value_from", "value_property"}
                    },
                    "required": ["property", "operator", "value"],
                },
                definitions["SemanticFilter"]["anyOf"][1],
                derived_age,
            ]}
            request = definitions["SemanticFactRequest"]
            request["properties"]["operation"]["enum"] = self.planner_capability_payload()["operations"]
            request["properties"]["filters"]["items"] = {
                "$ref": "#/$defs/SemanticCollectionFilter"
            }
            request["properties"]["property"] = {
                "anyOf": [{"type": "null"}, {"type": "string", "enum": properties}]
            }
            request["required"].extend(["property", "property_source"])
            path_schema = definitions["SemanticReference"]["properties"]["path"]
            entity_types = sorted(self.catalog.entities)
            definitions["SemanticReference"] = {"anyOf": [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "description": (
                        "Contextual reference. kind=self is the authenticated "
                        "speaker (first person). kind=assistant is this helper "
                        "when the user addresses it (second person). "
                        "kind=current_household is the configured home."
                    ),
                    "properties": {
                        "kind": {
                            "type": "string",
                            "enum": ["self", "assistant", "current_household"],
                            "description": (
                                "self: first-person speaker. assistant: this "
                                "helper under second-person address. "
                                "current_household: configured home."
                            ),
                        },
                        "value": {"type": "null"},
                        "entity_type": {
                            "anyOf": [
                                {"type": "null"},
                                {
                                    "type": "string",
                                    "enum": ["person", "address"],
                                },
                            ]
                        },
                        "path": path_schema,
                    },
                    "required": ["kind"],
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "description": (
                        "A name copied verbatim from the user utterance; never "
                        "a pronoun. entity_type=item for objects, space for "
                        "named rooms/areas, person for people. Object "
                        "在哪里/where is uses entity_type=item and path "
                        "concept location."
                    ),
                    "properties": {
                        "kind": {"type": "string", "const": "named_entity"},
                        "value": {"type": "string", "maxLength": 256},
                        "entity_type": {
                            "anyOf": [
                                {"type": "null"},
                                {"type": "string", "enum": entity_types},
                            ]
                        },
                        "path": path_schema,
                    },
                    "required": ["kind", "value"],
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "description": "Focus of a prior user turn. Not a substitute for first- or second-person identity.",
                    "properties": {
                        "kind": {"type": "string", "const": "discourse"},
                        "entity_type": {"type": "string", "enum": entity_types},
                        "turn_offset": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 8,
                        },
                        "cardinality": {
                            "type": "string",
                            "enum": ["single", "collection"],
                        },
                        "path": path_schema,
                    },
                    "required": ["kind", "entity_type", "turn_offset"],
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "description": "A referring expression that cannot be grounded.",
                    "properties": {
                        "kind": {"type": "string", "const": "unresolved"},
                        "entity_type": {
                            "anyOf": [
                                {"type": "null"},
                                {"type": "string", "enum": entity_types},
                            ]
                        },
                    },
                    "required": ["kind"],
                },
            ]}
            if self.contracts is not None:
                definitions['SemanticFilter'] = self.contracts.filters_schema(traversal=True, predicates=())
                definitions['SemanticCollectionFilter'] = self.contracts.filters_schema(
                    traversal=False, predicates=tuple(sorted(self._available_predicates))
                )
            self._planner_schema_cache = _prefer_null_union(schema)
        return deepcopy(self._planner_schema_cache)

    def expand_planner_concepts(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Expand only explicitly selected ontology aliases, without interpreting text.

        Base relations and their filters pass through unchanged. A concept use
        cannot override or drop part of its declared meaning.
        """
        raw = payload.get("request")
        if not isinstance(raw, Mapping):
            return payload
        request = dict(raw)
        concepts = self._ontology_payload()["reference_concepts"]
        if isinstance(request.get("exclude"), (list, tuple)):
            request["exclude"] = [
                self.expand_planner_concepts({"request": {"subject": reference}})["request"]["subject"]
                for reference in request["exclude"]
            ]
        for key in ("subject", "other"):
            reference = request.get(key)
            if not isinstance(reference, Mapping):
                continue
            path = reference.get("path", [])
            if not isinstance(path, (list, tuple)):
                continue  # ordinary structural validation reports the error
            expanded = []
            for step in path:
                if isinstance(step, Mapping) and "concept" in step:
                    use = SemanticConceptUse.model_validate(step)
                    if use.concept not in concepts:
                        raise ValueError("unknown reference concept")
                    steps = [dict(item) for item in concepts[use.concept]["path"]]
                    if use.filters:
                        steps[-1]["filters"] = [
                            *steps[-1].get("filters", []),
                            *(item.model_dump(mode="json", exclude_none=True) for item in use.filters),
                        ]
                    expanded.extend(steps)
                else:
                    expanded.append(step)
            completed = {**reference, "path": expanded}
            if completed.get("kind") == "named_entity" and expanded:
                relation = expanded[0].get("relation") if isinstance(expanded[0], Mapping) else None
                if isinstance(relation, str):
                    legal = {
                        entity_type
                        for entity_type in self.catalog.entities
                        if self._traversal_target_types(
                            relation, frozenset({entity_type})
                        )
                    }
                    if len(legal) == 1 and completed.get("entity_type") not in legal:
                        completed = {**completed, "entity_type": next(iter(legal))}
            request[key] = completed
        return {**payload, "request": request}

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
            and self.ontology.version == 1
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
            and (self.contracts is None or (semantic, semantic_property) in self.contracts.relation_bindings)
        }
        properties.update(
            physical
            for physical in schema.properties
            if physical not in self._RESOLVER_METADATA_PROPERTIES
            and physical not in self._aliased_physical_properties
            and self.ontology.version == 1
        )
        return frozenset(properties)

    def validation_code(self, request: SemanticFactRequest) -> PlannerValidationCode:
        """Return a stable, non-sensitive reason for semantic-plan rejection."""
        references = (request.subject,) + ((request.other,) if request.other else ()) + request.exclude
        if self.contract_error(request) is not None:
            return 'INVALID_PLAN'
        for reference in references:
            for step in reference.path:
                if self.physical_relation(step.relation) is None:
                    return "UNKNOWN_RELATION"
        if request.property is not None:
            if self.contracts is not None:
                contract = self.contracts.properties.get(request.property)
                if contract is not None and contract.values and request.operation in {
                    'argmin', 'argmax', 'min', 'max', 'latest', 'earliest'
                }:
                    return 'INVALID_PLAN'
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
        final_types: dict[int, frozenset[str]] = {}
        for reference in references:
            contextual_type = _CONTEXT_ENTITY_TYPES.get(reference.kind)
            if (
                contextual_type is not None
                and reference.entity_type is not None
                and reference.entity_type != contextual_type
            ):
                return "INVALID_PLAN"
            if (
                reference.kind == "entity_id"
                and reference.value is not None
                and reference.entity_type is not None
                and reference.entity_type != reference.value.partition(":")[0]
            ):
                return "INVALID_PLAN"
            if reference.kind == "assistant" and reference.path:
                return "INVALID_PLAN"
            resolved_types = self._reference_entity_types(reference)
            if resolved_types is None:
                return "INVALID_PLAN"
            final_types[id(reference)] = resolved_types
            anchor_types = self._base_entity_types(reference)
            step_types = anchor_types
            for step in reference.path:
                next_types = self._traversal_target_types(step.relation, step_types)
                if next_types is None:
                    return "INVALID_PLAN"
                for item in step.filters:
                    if item.source == "entity":
                        kind = self._semantic_kind(next_types, item.property)
                        if kind is None or not self._valid_predicate(item, kind):
                            return "INVALID_PLAN"
                        if item.value_from == "anchor":
                            anchor_kind = self._semantic_kind(
                                anchor_types,
                                item.value_property or item.property,
                            )
                            if anchor_kind is None or anchor_kind != kind:
                                return "INVALID_PLAN"
                    if item.source == "relation":
                        resolved = self.physical_relation(step.relation)
                        if resolved is None:
                            return "INVALID_PLAN"
                        physical = self.relation_property(resolved[0], item.property)
                        if physical is None:
                            return "INVALID_PLAN"
                        kind = self.catalog.relation_field_type(resolved[0], physical)
                        if not self._valid_predicate(item, kind):
                            return "INVALID_PLAN"
                step_types = next_types

        if request.other is not None and request.filters:
            return "INVALID_PLAN"
        if request.filters and not request.subject.path and request.subject.cardinality != "collection":
            return "INVALID_PLAN"
        for item in request.filters:
            if item.value_from is not None:
                return "INVALID_PLAN"
            if item.predicate is not None:
                definition = self.ontology.collection_predicates.get(item.predicate)
                if definition is None or item.predicate not in self._available_predicates or not final_types[id(request.subject)].issubset(
                    definition.entity_types
                ):
                    return "INVALID_PLAN"
                continue
            assert item.property is not None
            if item.source == "entity":
                kind = self._semantic_kind(
                    final_types[id(request.subject)], item.property
                )
            else:
                kind = self._final_relation_kind(request.subject, item.property)
            if kind is None or not self._valid_predicate(item, kind):
                return "INVALID_PLAN"

        if request.property_source == "relationship":
            if (
                request.other is not None
                or not request.subject.path
                or request.property is None
            ):
                return "INVALID_PLAN"

        operation = OPERATORS[request.operation]
        collection_input = bool(request.subject.path or request.other is not None
                                or request.subject.cardinality == "collection")
        if request.amount is not None and request.operation != "date_add":
            return "INVALID_PLAN"
        if request.projection == "each" and (
            not collection_input or request.other is not None or request.property is None
            or not (operation.input_shape == "scalar" or request.operation == "select")
        ):
            return "INVALID_PLAN"
        if request.exclude and (not collection_input or request.other is not None
                                or not (request.projection == "each" or operation.input_shape == "collection"
                                        or (request.operation == "select" and request.property is None))):
            return "INVALID_PLAN"
        if operation.input_shape == "collection" and not collection_input:
            return "INVALID_PLAN"
        if request.operation == "resolve_reference":
            valid = (
                request.property is None
                and request.other is None
                and not request.filters
                and request.property_source == "entity"
            )
            return "VALID" if valid else "INVALID_PLAN"
        if request.operation == "same_entity":
            valid = (
                request.property is None
                and request.other is not None
                and not request.filters
                and request.property_source == "entity"
            )
            return "VALID" if valid else "INVALID_PLAN"
        if request.operation == "select":
            if request.other is not None:
                return "INVALID_PLAN"
            if request.property is None:
                return "VALID" if collection_input else "INVALID_PLAN"
            valid = self._request_property_kind(
                request,
                final_types[id(request.subject)],
            ) is not None
            return "VALID" if valid else "INVALID_PLAN"

        field_kind = (
            self._request_property_kind(
                request,
                final_types[id(request.subject)],
            )
            if request.property is not None
            else "unknown"
        )
        if request.property is not None and field_kind is None:
            return "INVALID_PLAN"
        if (
            request.property is not None
            and field_kind == "unknown"
            and "any" not in operation.field_kinds
        ):
            return "INVALID_PLAN"
        if request.other is not None:
            other_kind = self._semantic_kind(
                final_types[id(request.other)],
                request.property,
            )
            if field_kind != other_kind:
                return "INVALID_PLAN"
        parameters = {
            "reference": "household_now",
            "mode": request.mode,
            "amount": request.amount,
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
            return "INVALID_PLAN"
        return "VALID"

    def validates(self, request: SemanticFactRequest) -> bool:
        """Return whether a request passes authoritative semantic validation."""
        return self.validation_code(request) == "VALID"

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
        if self.contracts is not None:
            if (reference.path[-1].relation, semantic_property) not in self.contracts.relation_bindings:
                return None
            return self.contracts.properties[semantic_property].type.execution_kind
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
        if contextual_type := _CONTEXT_ENTITY_TYPES.get(reference.kind):
            return frozenset({contextual_type})
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
            if semantic_relation == "contents":
                hosting = self.physical_relation("hosted_space")
                hosts = self.catalog.relations.get(hosting[0]) if hosting else None
                if hosts is not None:
                    to_types = to_types | frozenset(hosts.to_types)
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
        if self.contracts is not None:
            if not entity_types or any((owner, semantic_property) not in self.contracts.entity_bindings for owner in entity_types):
                return None
            return self.contracts.properties[semantic_property].type.execution_kind
        kinds = {
            self.catalog.entity_field_type(entity_type, physical)
            for entity_type in entity_types
            if (physical := self.physical_property(entity_type, semantic_property))
            is not None
        }
        if len(kinds) != 1:
            return None
        return next(iter(kinds))

    def _valid_predicate(self, item: SemanticFilter, field_kind: str) -> bool:
        if item.transform == "date_difference":
            return field_kind in {"date", "datetime"}
        if self.contracts is not None:
            contract = self.contracts.properties.get(item.property)
            return contract is not None and contract.literal_error(item.operator, item.value, anchor=item.value_from is not None) is None
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

    @stage("planner.total")
    async def plan(
        self,
        messages: Sequence[Mapping[str, Any]],
        context: AgentRequestContext,
    ) -> SemanticPlannerOutcome:
        started = perf_counter()
        build_started = perf_counter()
        output_schema = self.schema.planner_output_schema()
        capabilities = self.schema.planner_capability_payload()
        input_summary = planner_input_summary(self.schema.capability_payload())
        prompt_build_ms = (perf_counter() - build_started) * 1000
        payload: Mapping[str, Any] | None = None
        plan: SemanticPlan | None = None
        structural_error: Exception | None = None
        validation: PlannerValidationCode | None = None
        attempts = 0
        request_ms = 0.0
        validation_ms = 0.0
        runtime: Mapping[str, Any] = {}
        transport_attempts: list[dict[str, Any]] = []
        utterance = latest_user_message(messages)
        last_invalid_request: SemanticFactRequest | None = None
        for attempts in (1, 2):
            planner_messages: list[dict[str, Any]] = [
                {"role": "user", "content": str(message.get("content", ""))}
                for message in messages if message.get("role") == "user"
            ]
            if attempts == 2:
                previous = validation or _structural_validation_code(structural_error)
                hint = _identity_person_mismatch(utterance, last_invalid_request)
                location = _object_location_mismatch(utterance, last_invalid_request)
                grammar = _invalid_plan_retry_hint(self.schema, last_invalid_request)
                extra = " ".join(part for part in (hint, location, grammar) if part)
                planner_messages.append(
                    {
                        "role": "system",
                        "content": (
                            "Your previous response failed strict structural or "
                            f"semantic validation ({previous}). Recompile the "
                            "original meaning using only the advertised grammar; "
                            "check reference path, first vs second person, "
                            "property ownership, and operation requirements. "
                            + (f"{extra} " if extra else "")
                            + "Return exactly one structured value conforming to the "
                            "supplied output schema."
                        ),
                    }
                )
            request_started = perf_counter()
            try:
                payload = await self.ollama.plan_semantic_fact(
                    planner_messages,
                    capabilities,
                    output_schema,
                    household_now=_planner_clock(context.current_time),
                )
                runtime = getattr(self.ollama, "last_planner_runtime", {}) or {}
                validate_started = perf_counter()
                candidate = SemanticPlan.model_validate(
                    self.schema.expand_planner_concepts(payload)
                    if isinstance(payload, Mapping) else payload
                )
                validation = "NOT_A_FACT"
                if candidate.request is not None and any(
                    reference.kind == "entity_id"
                    for reference in (candidate.request.subject, candidate.request.other, *candidate.request.exclude)
                    if reference is not None
                ):
                    validation = "MODEL_ORIGINATED_ENTITY_ID"
                elif candidate.request is not None:
                    validation = self.schema.validation_code(candidate.request)
                    if validation == "VALID":
                        person_error = _identity_person_mismatch(
                            utterance, candidate.request
                        )
                        location_error = _object_location_mismatch(
                            utterance, candidate.request
                        )
                        if person_error:
                            validation = "INVALID_PLAN"
                        elif location_error:
                            completed = _complete_named_object_location(
                                candidate.request
                            )
                            if completed is None:
                                validation = "INVALID_PLAN"
                            else:
                                candidate = SemanticPlan(
                                    requires_fact=True, request=completed
                                )
                                validation = self.schema.validation_code(
                                    completed
                                )
                    elif _object_location_mismatch(utterance, candidate.request):
                        completed = _complete_named_object_location(
                            candidate.request
                        )
                        if completed is not None:
                            candidate = SemanticPlan(
                                requires_fact=True, request=completed
                            )
                            validation = self.schema.validation_code(completed)
                    if validation != "VALID":
                        last_invalid_request = candidate.request
                validation_ms += (perf_counter() - validate_started) * 1000
                if validation not in {"VALID", "NOT_A_FACT"}:
                    plan = None
                    structural_error = ValueError(validation)
                    continue
                plan = candidate
                structural_error = None
                break
            except (ValueError, TypeError) as error:
                runtime = getattr(self.ollama, "last_planner_runtime", {}) or runtime
                structural_error = error
            finally:
                attempt_runtime = getattr(self.ollama, "last_planner_runtime", {}) or {}
                if attempt_runtime.get("codec_version") is not None:
                    transport_attempts.append({
                        **_planner_runtime_fields(attempt_runtime),
                        "attempt": attempts,
                    })
                request_ms += (perf_counter() - request_started) * 1000
        latency_ms = (perf_counter() - started) * 1000
        timing = {
            "prompt_build_ms": prompt_build_ms,
            "request_ms": request_ms,
            "validation_ms": validation_ms,
            **_planner_runtime_fields(runtime),
        }
        if transport_attempts:
            timing["transport"]["attempts"] = transport_attempts
        if plan is None:
            code = validation or _structural_validation_code(structural_error)
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
                    **timing,
                )
            )
        assert validation is not None
        diagnostics = PlannerDiagnostics(
            input_summary=input_summary,
            output_raw=payload,
            normalized_plan=plan.model_dump(mode="json"),
            validation_result=validation,
            attempt_count=attempts,
            latency_ms=latency_ms,
            **timing,
        )
        return SemanticPlannerOutcome(plan, latency_ms, diagnostics)

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

    @stage("resolver.total")
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
            entities, relationship_records, content_groups = await self._resolve(
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
                FactEvidence(
                    entity_ids=entity_ids,
                    relationship=(
                        reference.path[-1].relation if reference.path else None
                    ),
                    relationships=tuple(execution.relationship_evidence),
                ),
                relationship_records=tuple(relationship_records),
                content_groups=tuple(content_groups),
            )
        except _FactFailure as error:
            status: ResolutionStatus = {
                "caller_context_missing": "missing_context",
                "entity_not_found": "not_found",
                "relationship_not_found": "relationship_not_found",
                "property_unavailable": "property_unavailable",
                "filter_input_missing": "filter_input_missing",
                "filter_unsupported": "filter_unsupported",
                "collection_incomplete": "collection_incomplete",
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
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[FactContentGroup]]:
        if reference.kind == "unresolved":
            raise _FactFailure("ambiguous", missing=("semantic_reference",))
        elif reference.kind == "discourse":
            discourse = context.discourse
            if (discourse is None or context.conversation_id is None
                or (discourse.conversation_id, discourse.caller_entity_id, discourse.household_id, discourse.assistant_id)
                != (context.conversation_id, context.caller_entity_id, context.household_id, context.assistant_id)
                or reference.turn_offset is None or reference.turn_offset > len(discourse.turns)
                or not discourse.turns[-reference.turn_offset]):
                raise _FactFailure("caller_context_missing", missing=("discourse_antecedent",))
            ids = discourse.turns[-reference.turn_offset]
            entities = [await execution.load({"id": entity_id}) for entity_id in ids]
            entities = [
                entity
                for entity in entities
                if _entity_type(entity) == reference.entity_type
            ]
            if not entities:
                raise _FactFailure(
                    "caller_context_missing", missing=("discourse_antecedent",)
                )
            if reference.cardinality == "single" and len(entities) != 1:
                raise _FactFailure("ambiguous", candidates=tuple(entities), missing=("discourse_antecedent",))
        elif reference.kind == "assistant":
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
                    "speaker_id": context.caller_entity_id,
                    "household_id": context.household_id,
                },
            )
            records = await self._scope_named_records(
                records,
                context.household_id,
                execution,
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
        last_content_groups: list[FactContentGroup] = []
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
            grouped: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
            collapsed = False
            for entity in entities:
                entity_id = entity.get("id")
                if not isinstance(entity_id, str):
                    continue
                sources = [entity]
                if step.relation == "contents":
                    stored = await execution.load(entity)
                    if stored.get("collapse") is True:
                        collapsed = True
                        sources.extend(await self._hosted_descendants(stored, execution))
                for source in sources:
                    edges = await self._relation_edges(
                        str(source["id"]), relation, direction, execution,
                    )
                    for edge in edges:
                        if not self._edge_matches(edge, relation, step.filters):
                            continue
                        execution.remember_relationship(step.relation, edge)
                        candidate = edge.get("related_entity")
                        if isinstance(candidate, Mapping):
                            related.append(dict(candidate))
                            step_relationship_records.append(dict(edge))
                            if step.relation == "contents":
                                group = grouped.setdefault(str(source["id"]), (dict(source), []))
                                group[1].append(dict(candidate))
            entities = _unique_entities(related)
            entities = await self._filter_entities(
                entities,
                step.filters,
                execution,
                anchors,
                context,
            )
            if not entities:
                if allow_empty_collection:
                    return [], [], []
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
            last_content_groups = [
                FactContentGroup(space, tuple(_unique_entities([
                    item for item in items if item.get("id") in entity_ids
                ])))
                for space, items in grouped.values()
            ] if collapsed else []
        return entities, last_relationship_records, last_content_groups

    async def _relation_edges(
        self, entity_id: str, relation: str, direction: str | None,
        execution: "_FactExecution",
    ) -> list[dict[str, Any]]:
        arguments: dict[str, Any] = {
            "entity_id": entity_id, "relation": relation,
            "include_ended": False, "limit": self.max_records + 1,
        }
        if direction is not None:
            arguments["direction"] = direction
        edges = await execution.records("get_relationships", arguments)
        if len(edges) > self.max_records:
            raise _FactFailure("collection_incomplete", missing=(relation,))
        return edges

    async def _hosted_descendants(
        self, parent: Mapping[str, Any], execution: "_FactExecution",
    ) -> list[dict[str, Any]]:
        """Expand a query view, retaining only authoritative graph edges."""
        hosting = self.schema.physical_relation("hosted_space")
        if hosting is None:
            return []
        descendants = []
        visited: set[str] = set()
        active: set[str] = set()
        stack = [(dict(parent), False)]
        while stack:
            entity, leaving = stack.pop()
            entity_id = str(entity["id"])
            if leaving:
                active.remove(entity_id)
                continue
            if entity_id in active:
                raise _FactFailure("computation_impossible", missing=("hosting_cycle",))
            if entity_id in visited:
                continue
            visited.add(entity_id)
            active.add(entity_id)
            stack.append((entity, True))
            edges = await self._relation_edges(entity_id, *hosting, execution)
            for edge in edges:
                child = edge.get("related_entity")
                if not isinstance(child, Mapping) or _entity_type(child) != "space":
                    continue
                execution.remember_relationship("hosted_space", edge)
                descendants.append(dict(child))
                stack.append((dict(child), False))
        return _unique_entities(descendants)

    async def _scope_named_records(
        self,
        records: Sequence[Mapping[str, Any]],
        household_id: str | None,
        execution: "_FactExecution",
    ) -> list[dict[str, Any]]:
        """Apply schema-declared containment scope to named records.

        Entity types without a declared scope-parent edge retain their existing
        resolution behavior. Types with such an edge must have a recorded path
        to the request's household. This keeps scoping generic across items,
        spaces, and future contained entity types.
        """
        if household_id is None:
            return [dict(record) for record in records]
        scoped: list[dict[str, Any]] = []
        for record in records:
            entity_id = record.get("id")
            entity_type = _entity_type(record)
            parents = self.schema.edge_registry.scope_parent_relations(entity_type)
            if not parents:
                scoped.append(dict(record))
                continue
            if isinstance(entity_id, str) and await self._belongs_to_household(
                entity_id,
                household_id,
                execution,
                visited=frozenset(),
            ):
                scoped.append(dict(record))
        return scoped

    async def _belongs_to_household(
        self,
        entity_id: str,
        household_id: str,
        execution: "_FactExecution",
        *,
        visited: frozenset[str],
    ) -> bool:
        if entity_id == household_id:
            return True
        if entity_id in visited or len(visited) >= 12:
            return False
        parents = self.schema.edge_registry.scope_parent_relations(
            entity_id.partition(":")[0]
        )
        if not parents:
            return False
        next_visited = visited | {entity_id}
        for relation in parents:
            edges = await execution.records(
                "get_relationships",
                {
                    "entity_id": entity_id,
                    "relation": relation,
                    "direction": "out",
                    "include_ended": False,
                    "limit": self.max_records + 1,
                },
            )
            if len(edges) > self.max_records:
                raise _FactFailure(
                    "collection_incomplete",
                    missing=(relation,),
                )
            for edge in edges:
                parent = edge.get("related_entity")
                parent_id = parent.get("id") if isinstance(parent, Mapping) else None
                if isinstance(parent_id, str) and await self._belongs_to_household(
                    parent_id,
                    household_id,
                    execution,
                    visited=next_visited,
                ):
                    return True
        return False

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
            self.schema.validate_filter_value(item.property, edge.get(physical))
            if not evaluate_predicate(item.operator, edge.get(physical), item.value):
                return False
        return True

    async def _filter_entities(
        self,
        entities: list[dict[str, Any]],
        filters: Sequence[SemanticFilter],
        execution: "_FactExecution",
        anchors: Sequence[Mapping[str, Any]],
        context: AgentRequestContext,
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
                self.schema.validate_filter_value(item.property, record.get(physical))
                if item.transform:
                    predicates.append(
                        self._date_transform_matches(item, record.get(physical), context)
                    )
                    continue
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
        self.schema.validate_filter_value(semantic_property, anchor[physical])
        return anchor[physical]

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
        if not 1 <= max_records < 100:
            raise ValueError("max_records must be between 1 and 99 for completeness checks")
        self.resolver = EntityResolver(
            schema,
            max_records=max_records,
        )

    @stage("executor.total")
    async def execute(
        self,
        request: SemanticFactRequest,
        context: AgentRequestContext,
    ) -> tuple[FactResult, int, float, float]:
        if not self.schema.validates(request):
            return FactResult("semantic_plan_unsupported"), 0, 0, 0
        execution = _FactExecution(self.dispatcher, context.caller_entity_id)
        resolution_started = perf_counter()
        allow_empty_collection = request.operation in {"count", "select"} or request.projection == "each"
        operation = OPERATORS[request.operation]
        expect_many = request.other is None and (
            request.projection == "each" or operation.input_shape == "collection"
            or (request.operation == "select" and request.property is None)
            or bool(request.filters)
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
        exclusions = [await self.resolver.resolve(
            reference, context, execution, expect_many=reference.cardinality == "collection"
        ) for reference in request.exclude]
        failed = next(
            (
                item
                for item in (resolution, other_resolution, *exclusions)
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
                "filter_input_missing": "filter_input_missing",
                "filter_unsupported": "filter_unsupported",
                "collection_incomplete": "collection_incomplete",
            }[failed.status]
            if "discourse_antecedent" in failed.missing_requirements and status == "caller_context_missing":
                status = "discourse_context_missing"
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
        excluded_ids = {entity_id for item in exclusions for entity_id in item.entity_ids}
        entities = [item for item in entities if item.get("id") not in excluded_ids]
        relationship_records = [edge for edge in relationship_records if _related_entity_id(edge) not in excluded_ids]
        computation_started = perf_counter()
        try:
            filter_failures: list[FactRow] = []
            if request.filters:
                entities, relationship_records, filter_failures = await self._filter_collection(
                    request,
                    entities,
                    relationship_records,
                    context,
                    execution,
                )
            if request.projection == "each":
                result = await self._project_each(
                    request, entities, other_entities, relationship_records,
                    context, execution, initial_rows=filter_failures,
                )
            else:
                result = await self._operate(
                    request, entities, other_entities, relationship_records,
                    context, execution,
                )
        except _FactFailure as error:
            result = FactResult(
                error.status,
                evidence=error.evidence,
                missing_requirements=error.missing,
                candidates=error.candidates,
            )
        if result.status == "found":
            if result.shape == "entities" and resolution.content_groups:
                visible = {item["id"]: item for item in result.value}
                groups = tuple(
                    FactContentGroup(group.space, tuple(
                        visible[item["id"]] for item in group.entities if item["id"] in visible
                    ))
                    for group in resolution.content_groups
                    if any(item["id"] in visible for item in group.entities)
                )
                result = replace(result, content_groups=groups)
            focus = tuple(str(item["id"]) for item in (*entities, *other_entities) if item.get("id"))
            if (request.other is None and request.property_source == "entity"
                and isinstance(result.value, Mapping) and result.value.get("id")):
                focus = (str(result.value["id"]),)
            final_types = {_entity_type(item) for item in entities}
            if (
                request.subject.kind == "named_entity"
                and request.subject.path
                and request.subject.entity_type not in final_types
            ):
                anchor = next(
                    (
                        entity_id
                        for edge in resolution.evidence.relationships
                        for entity_id in (edge.source_id, edge.target_id)
                        if entity_id is not None
                        and entity_id.partition(":")[0]
                        == request.subject.entity_type
                    ),
                    None,
                )
                if anchor is not None:
                    focus = (anchor, *focus)
            result = replace(result, focus_entity_ids=tuple(dict.fromkeys(focus)))
        computation_ms = (perf_counter() - computation_started) * 1000
        return result, execution.query_count, entity_resolution_ms, computation_ms

    async def _project_each(
        self,
        request: SemanticFactRequest,
        entities: list[dict[str, Any]],
        other_entities: list[dict[str, Any]],
        relationship_records: list[dict[str, Any]],
        context: AgentRequestContext,
        execution: "_FactExecution",
        *,
        initial_rows: Sequence[FactRow] = (),
    ) -> FactResult:
        rows: list[FactRow] = list(initial_rows)
        scalar = request.model_copy(update={"projection": "scalar", "exclude": ()})
        relation = _last_relation(request.subject)
        for entity in entities:
            edges = [
                edge for edge in relationship_records
                if _related_entity_id(edge) == entity.get("id")
            ]
            groups = (
                [[edge] for edge in edges]
                if request.property_source == "relationship" else [edges]
            )
            for group in groups or [[]]:
                # Share loaded records, but scope relationship evidence to this row.
                row_execution = _FactExecution(self.dispatcher, context.caller_entity_id)
                row_execution.entity_cache = execution.entity_cache
                for edge in group:
                    assert relation is not None
                    row_execution.remember_relationship(relation, edge)
                evidence = FactEvidence(
                    (str(entity["id"]),), relation, request.property,
                    tuple(row_execution.relationship_evidence),
                )
                try:
                    visible = await row_execution.load_if_unnamed(entity)
                    value = await self._operate(
                        scalar, [entity], [], group, context, row_execution,
                    )
                except _FactFailure as error:
                    visible = entity
                    value = FactResult(error.status, missing_requirements=error.missing)
                execution.query_count += row_execution.query_count
                rows.append(FactRow(
                    visible, value.status, value.value, _result_unit(request),
                    evidence, value.missing_requirements,
                ))
        return FactResult("found", shape="rows", rows=tuple(rows))

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
            return FactResult("found", visible, evidence, shape="entities")
        if request.operation == "resolve_reference":
            singular = await self._singular(
                entities,
                request,
                execution,
                load_full=False,
            )
            if isinstance(singular, FactResult):
                return singular
            return FactResult("found", singular, evidence, shape="entity")
        if request.operation == "same_entity":
            left = await self._singular(
                entities, request, execution, load_full=False
            )
            right = await self._singular(
                other_entities, request, execution, load_full=False
            )
            if isinstance(left, FactResult):
                return left
            if isinstance(right, FactResult):
                return right
            left_id = left.get("id")
            right_id = right.get("id")
            if not isinstance(left_id, str) or not isinstance(right_id, str):
                return FactResult("computation_impossible", evidence=evidence)
            return FactResult(
                "found", left_id == right_id, evidence, shape="scalar"
            )
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
            "amount": request.amount,
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
                    amount=request.amount,
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
        shape = (
            "entity" if request.property_source == "entity"
            and isinstance(value, Mapping) and value.get("id") else "scalar"
        )
        return FactResult("found", value, evidence, unit=_result_unit(request), shape=shape)

    async def _filter_collection(
        self,
        request: SemanticFactRequest,
        entities: list[dict[str, Any]],
        relationship_records: list[dict[str, Any]],
        context: AgentRequestContext,
        execution: "_FactExecution",
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[FactRow]]:
        # Relationship projections have edge rows: all edge conditions bind the
        # same final edge, rather than admitting every edge of a matching entity.
        edge_filters = tuple(item for item in request.filters if item.source == "relation")
        if request.projection == "each" and request.property_source == "relationship" and edge_filters:
            relationship_records = [
                edge for edge in relationship_records
                if all(self._relation_filter_matches(request, item, [edge]) for item in edge_filters)
            ]
            retained = {_related_entity_id(edge) for edge in relationship_records}
            entities = [entity for entity in entities if entity.get("id") in retained]
            request = request.model_copy(update={
                "filters": tuple(item for item in request.filters if item.source != "relation")
            })
        matched: list[dict[str, Any]] = []
        failures: list[FactRow] = []
        for entity in entities:
            entity_id = entity.get("id")
            edges = [
                edge
                for edge in relationship_records
                if _related_entity_id(edge) == entity_id
            ]
            include = True
            try:
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
                            context,
                        )
                    if not include:
                        break
            except _FactFailure as error:
                if request.projection != "each":
                    raise
                visible = await execution.load_if_unnamed(entity)
                failures.append(
                    FactRow(
                        visible,
                        error.status,
                        evidence=FactEvidence(
                            entity_ids=(str(entity_id),) if isinstance(entity_id, str) else (),
                            relationship=_last_relation(request.subject),
                            semantic_property=request.property,
                        ),
                        missing_requirements=error.missing,
                        unit=_result_unit(request),
                    )
                )
                continue
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
        ], failures

    async def _entity_filter_matches(
        self,
        item: SemanticFilter,
        entity: Mapping[str, Any],
        execution: "_FactExecution",
        context: AgentRequestContext,
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
        self.schema.validate_filter_value(item.property, record.get(physical))
        if item.transform:
            return self._date_transform_matches(item, record.get(physical), context)
        return evaluate_predicate(item.operator, record.get(physical), item.value)

    def _date_transform_matches(
        self,
        item: SemanticFilter,
        raw_value: Any,
        context: AgentRequestContext,
    ) -> bool:
        normalized = {"value": raw_value}
        try:
            if execute_operator(
                "date_difference",
                OperatorInput(
                    records=[normalized],
                    field="value",
                    mode="seconds",
                    now=context.current_time,
                ),
            ) < 0:
                raise OperatorExecutionError("derived date filter requires a past date")
            transform = OPERATORS[item.transform or "date_difference"]
            transform.validate(
                field="value",
                field_kind=infer_field_kind([raw_value]),
                parameters={"reference": "household_now", "mode": item.mode},
            )
            derived = execute_operator(
                item.transform or "date_difference",
                OperatorInput(
                    records=[normalized],
                    field="value",
                    reference="household_now",
                    now=context.current_time,
                    mode=item.mode,
                ),
            )
        except (OperatorValidationError, OperatorExecutionError, TypeError, ValueError):
            raise _FactFailure(
                "filter_input_missing",
                missing=(item.property,),
            ) from None
        return evaluate_predicate(item.operator, derived, item.value)

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
        for value in values:
            self.schema.validate_filter_value(item.property, value)
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
            (self.schema.physical_relation(request.subject.path[-1].relation)
             if request.subject.path else None) or (None, None)
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
        if self.schema.contracts is not None and role_property is not None:
            for edge in edges:
                if edge.get(role_property) is not None:
                    self.schema.validate_filter_value(definition.relation_property, edge[role_property])
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
        self.schema.validate_filter_value(fallback.property, raw_value)
        normalized = {"value": raw_value}
        try:
            if fallback.require_past and execute_operator(
                "date_difference",
                OperatorInput(records=[normalized],field="value",mode="seconds",now=context.current_time),
            ) < 0:
                raise OperatorExecutionError("predicate requires a past date")
            transform = OPERATORS[fallback.transform]
            transform.validate(
                field="value",
                field_kind=infer_field_kind([raw_value]),
                parameters={"reference": "household_now", "mode": fallback.mode},
            )
            derived = execute_operator(
                fallback.transform,
                OperatorInput(
                    records=[normalized],
                    field="value",
                    reference="household_now",
                    now=context.current_time,
                    mode=fallback.mode,
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
            or not self.schema.value_valid(request.property, relationship.get(physical))
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
        if (physical is None or physical not in entity or entity.get(physical) is None
            or not self.schema.value_valid(semantic, entity.get(physical))):
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
    def __init__(
        self,
        ontology: SemanticOntology | None = None,
        *,
        detailed: bool = False,
    ) -> None:
        self.ontology = ontology or SemanticOntology.load_default()
        self.detailed = detailed

    @stage("renderer")
    def render(
        self,
        request: SemanticFactRequest,
        result: FactResult,
        context: AgentRequestContext,
    ) -> str:
        rendered = self._render_result(request, result, context)
        # Describe the expanded, validated IR once, including empty/partial results.
        # Unsupported plans have no executed scope to describe.
        if self.detailed and result.status != "semantic_plan_unsupported" and (
            request.subject.path or request.filters or request.exclude or request.other
            or request.property is not None
            or request.operation == "count" or request.projection == "each"
            or (request.operation == "select" and request.property is None)
        ):
            description = SemanticDisplay(self.ontology, context.locale or "en").describe(request)
            return f"{description}\n{rendered}"
        if not self.detailed and result.status == "found" and not (
            request.operation == "count"
            or (request.operation == "select" and request.property is None)
        ):
            qualifier = self._condition_qualifier(request, context.locale or "en")
            if qualifier:
                separator = "\n" if result.shape == "rows" else ""
                if separator:
                    qualifier = qualifier.lstrip()
                rendered = f"{rendered}{separator}{qualifier}"
        return rendered

    def _condition_qualifier(
        self,
        request: SemanticFactRequest,
        language: str,
    ) -> str:
        """Faithfully expose conditions not already carried by relation nouns."""
        display = SemanticDisplay(self.ontology, language)
        remaining: list[SemanticFilter] = list(request.filters)
        noun_relations = {"spouse", "child", "parent"}
        for step in request.subject.path:
            for item in step.filters:
                absorbed_gender = (
                    step.relation in noun_relations
                    and item.source == "entity"
                    and item.property == "gender"
                    and item.operator == "eq"
                    and item.value_from is None
                    and item.value in {"male", "female"}
                )
                if not absorbed_gender:
                    remaining.append(item)
        clauses: list[str] = []
        if remaining:
            clauses.append(
                display.conditions(
                    tuple(remaining),
                    display.reference(request.subject.model_copy(update={"path": ()})),
                    collection=request.projection != "scalar",
                )
            )
        if request.exclude:
            excluded = " | ".join(display.reference(item) for item in request.exclude)
            clauses.append(("排除 " if display.zh else "exclude ") + excluded)
        if not clauses:
            return ""
        joined = ("；" if display.zh else "; ").join(clauses)
        return f"（条件：{joined}）" if display.zh else f" (Conditions: {joined})"

    def _render_result(
        self,
        request: SemanticFactRequest,
        result: FactResult,
        context: AgentRequestContext,
    ) -> str:
        language = context.locale or "en"
        if (result.status == "found" and result.shape == "entities"
            and request.operation == "select" and request.property is None
            and result.content_groups):
            display = SemanticDisplay(self.ontology, language)
            simple = (
                request.subject.kind == "named_entity"
                and len(request.subject.path) == 1
                and not request.subject.path[0].filters
                and not request.filters and not request.exclude and request.other is None
            )
            if simple:
                owner = request.subject.value
                heading = f"{owner}里有：" if display.zh else f"Contents of {owner}:"
            else:
                heading = display.describe(request)
            lines = [
                f"{_name(group.space, language)}" + ("：" if display.zh else ": ")
                + ("、" if display.zh else ", ").join(_name(item, language) for item in group.entities)
                for group in result.content_groups
            ]
            return heading + "\n" + ("；\n".join(lines) + "。" if display.zh else "\n".join(lines))
        if result.status == "ambiguous" and "discourse_antecedent" in result.missing_requirements:
            return "前文包含多个对象，请说明您指的是哪一个。" if language.startswith("zh") else "That earlier turn refers to multiple entities; please clarify which one you mean."
        if result.status == "discourse_context_missing":
            return "请说明您指的是谁；对应的前文对象尚未明确。" if language.startswith("zh") else "Please clarify who you mean; that earlier turn has no resolved referent."
        if result.shape == "rows":
            if not result.rows:
                return "没有找到符合条件的记录。" if language.startswith("zh") else "No matching records."
            return "\n".join(
                f"{_name(row.entity, language)}: " + self._render_result(
                    request.model_copy(update={"projection": "scalar"}),
                    FactResult(row.status, row.value, row.evidence, row.missing_requirements, unit=row.unit), context
                ) for row in result.rows
            )
        if result.status == "found" and request.operation == "date_add":
            return f"指定日期是{result.value}。" if language.startswith("zh") else f"The specified date is {result.value}."
        if not self.detailed and result.status == "found" and (
            request.operation == "count"
            or (request.operation == "select" and request.property is None)
        ):
            natural = self._collection_result(request, result, language)
            if natural is not None:
                return natural
        if language.startswith("zh"):
            return self._zh(request, result, context)
        return self._en(request, result, context)

    def _collection_result(
        self,
        request: SemanticFactRequest,
        result: FactResult,
        language: str,
    ) -> str | None:
        display = SemanticDisplay(self.ontology, language)
        described = display.collection_noun(request.subject, request.filters)
        if described is None:
            return self._generic_collection_result(request, result, display)
        noun, remaining = described
        if request.exclude or request.other:
            return self._generic_collection_result(request, result, display)
        zh = language.startswith("zh")
        if request.subject.kind == "current_household":
            owner = "家里" if zh else "the household"
        elif request.subject.kind == "self":
            owner = "您" if zh else "you"
        elif request.subject.kind == "named_entity" and request.subject.value:
            owner = str(request.subject.value)
        else:
            return self._generic_collection_result(request, result, display)
        condition = display.conditions(
            remaining,
            display.reference(request.subject.model_copy(update={"path": ()})),
            collection=True,
        ) if remaining else ""
        if request.operation == "count":
            count = int(result.value)
            if zh:
                counter = (
                    "个"
                    if noun == "人"
                    else "位" if noun.endswith(("人", "男性", "女性")) else "个"
                )
                if count == 0:
                    target = f"符合{condition}的{noun}" if condition else noun
                    return f"{owner}没有{target}。"
                suffix = f"，筛选条件还包括：{condition}" if condition else ""
                return f"{owner}有{count}{counter}{noun}{suffix}。"
            plural = noun if count == 1 else {
                "person": "people",
            }.get(noun, noun + ("es" if noun.endswith("s") else "s"))
            qualifier = f" matching {condition}" if condition else ""
            verb = "is" if count == 1 else "are"
            return f"There {verb} {count} {plural}{qualifier} in {owner}."
        values = result.value if isinstance(result.value, list) else []
        if not values:
            if zh:
                target = f"符合{condition}的{noun}" if condition else noun
                return f"{owner}没有{target}。"
            target = f"{noun} matching {condition}" if condition else noun
            return f"There are no {target}s in {owner}."
        names = ("、" if zh else ", ").join(_name(item, language) for item in values)
        if zh:
            suffix = f"（筛选条件还包括：{condition}）" if condition else ""
            return f"{owner}的{noun}有：{names}{suffix}。"
        suffix = f" matching {condition}" if condition else ""
        return f"The {noun}s in {owner}{suffix} are {names}."

    def _generic_collection_result(
        self,
        request: SemanticFactRequest,
        result: FactResult,
        display: SemanticDisplay,
    ) -> str:
        zh = display.zh
        scope = display.reference(request.subject)
        clauses: list[str] = []
        if request.filters:
            clauses.append(display.conditions(request.filters, scope, collection=True))
        if request.exclude:
            excluded = " | ".join(display.reference(item) for item in request.exclude)
            clauses.append(("排除 " if zh else "exclude ") + excluded)
        condition = ("；" if zh else "; ").join(clauses)
        if request.operation == "count":
            if zh:
                suffix = f"，条件为{condition}" if condition else ""
                return f"在{scope}中有{int(result.value)}条记录{suffix}。"
            suffix = f" matching {condition}" if condition else ""
            return f"There are {int(result.value)} records in {scope}{suffix}."
        values = result.value if isinstance(result.value, list) else []
        names = ("、" if zh else ", ").join(_name(item, "zh" if zh else "en") for item in values)
        if zh:
            suffix = f"，条件为{condition}" if condition else ""
            return f"在{scope}中找到的记录为：{names or '无'}{suffix}。"
        suffix = f" matching {condition}" if condition else ""
        return f"The records in {scope}{suffix} are {names or 'none'}."

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
            return "目前缺少筛选所需资料，无法可靠完成筛选。"
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
        if result.status == "collection_incomplete":
            return "查询结果超过当前完整性上限，无法给出可靠的总数或完整列表。"
        if request.operation == "count":
            count = int(result.value)
            return f"符合条件的记录数：{count}。"
        if request.operation == "select" and request.property is None:
            values = result.value if isinstance(result.value, list) else []
            if not values:
                if request.filters or any(step.filters for step in request.subject.path):
                    return "没有找到符合筛选条件的记录。"
                return "没有找到符合查询范围的记录。"
            names = "、".join(_name(item, "zh") for item in values)
            return f"符合条件的记录：{names}。"
        if request.operation == "select":
            if (
                request.property_source == "relationship"
                and request.subject.path
                and request.subject.path[-1].relation == "spouse"
                and request.property == "start_date"
            ):
                return f"该配偶关系的开始日期是{result.value}。"
            if request.property == "birth_date" and request.property_source == "entity":
                return f"{_subject_possessive(request.subject)}出生日期是{result.value}。"
            if request.property == "full_address":
                return f"具体住址是{_format_address(result.value)}。"
            return f"查询到的值是{result.value}。"
        if request.operation in {"date_difference", "duration", "completed_years"}:
            unit = result.unit or ("years" if request.operation == "completed_years" else request.mode)
            if unit == "years" and request.property == "birth_date" and request.property_source == "entity":
                return f"{_subject_nominative(request.subject)}今年{result.value}岁。"
            label = {"years": "年", "months": "个月", "days": "天", "seconds": "秒"}[unit]
            if result.value >= 0:
                prefix = "已满" if unit in {"years", "months"} else "已过"
                return f"从记录的日期到现在{prefix}{result.value}{label}。"
            return f"该日期与现在的间隔为{result.value}{label}。"
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
                if request.property == "birth_date" and request.property_source == "entity":
                    if result.value.get("equal"):
                        return f"{selected}和{other}年龄相同。"
                    adjective = "大" if request.operation == "argmin" else "小"
                    return f"{selected}年龄比{other}{adjective}。"
                if result.value.get("equal"):
                    return f"{selected}和{other}的比较值相同。"
                return f"{'最小值' if request.operation == 'argmin' else '最大值'}对应对象：{selected}。"
            if request.property == "birth_date" and request.property_source == "entity":
                qualifier = "最年长" if request.operation == "argmin" else "最年轻"
                return f"查询范围内{qualifier}的是{_name(result.value, 'zh')}。"
            return f"符合极值条件的是{_name(result.value, 'zh')}。"
        if request.operation in {"sum", "average", "min", "max"}:
            return f"计算结果是{result.value}。"
        if request.operation in {
            "unit_conversion",
        }:
            return f"换算结果是{result.value} {result.unit or request.to_unit}。"
        if request.operation in {"first", "last", "latest", "earliest"}:
            return f"符合条件的是{_name(result.value, 'zh')}。"
        if request.operation == "same_entity":
            return "是。" if result.value else "不是。"
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
                "collection_incomplete": "The result exceeds the completeness limit, so an exact total or complete list is unavailable.",
            }[result.status]
        if request.operation == "count":
            return f"The current count is {result.value}."
        if request.operation == "select" and request.property is None:
            if not result.value:
                if request.filters or any(step.filters for step in request.subject.path):
                    return "No records match the filters."
                return "No records match the query scope."
            return "Matching records: " + ", ".join(
                _name(item, "en") for item in result.value
            ) + "."
        if request.operation == "select":
            if request.property == "full_address":
                return f"The street address is {_format_address(result.value)}."
            return f"The requested value is {result.value}."
        if request.operation in {"date_difference", "duration", "completed_years"}:
            unit = result.unit or ("years" if request.operation == "completed_years" else request.mode)
            if unit == "years" and request.property == "birth_date" and request.property_source == "entity":
                return f"The age is {result.value} years."
            return f"The interval from the recorded date to now is {result.value} {unit}."
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
                if request.property == "birth_date" and request.property_source == "entity":
                    if result.value.get("equal"):
                        return f"{selected} and {other} are the same age."
                    adjective = "older" if request.operation == "argmin" else "younger"
                    return f"{selected} is {adjective} than {other}."
                if result.value.get("equal"):
                    return f"{selected} and {other} have equal comparison values."
                return f"The {'minimum' if request.operation == 'argmin' else 'maximum'} belongs to {selected}."
            return f"The matching entity is {_name(result.value, 'en')}."
        if request.operation in {"sum", "average", "min", "max"}:
            return f"The computed result is {result.value}."
        if request.operation in {
            "unit_conversion",
        }:
            return f"The converted result is {result.value} {result.unit or request.to_unit}."
        if request.operation in {"first", "last", "latest", "earliest"}:
            return f"The matching result is {_name(result.value, 'en')}."
        if request.operation == "same_entity":
            return "Yes." if result.value else "No."
        if request.subject.kind == "assistant":
            return f"I am {context.assistant_display_name}, the Home Cortex household assistant."
        return f"The resolved person is {_name(result.value, 'en')}."


class SemanticFactService:
    def __init__(
        self,
        engine: HouseholdFactEngine,
        planner: SemanticFactPlanner,
        renderer: FactRenderer | None = None,
    ) -> None:
        self.engine = engine
        self.renderer = renderer or FactRenderer(engine.schema.ontology)
        self.planner = planner

    async def try_answer(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        context: AgentRequestContext,
        request_id: str = "-",
    ) -> FactAnswer | None:
        started = perf_counter()
        routing_started = perf_counter()
        llm_ms = 0.0
        llm_call_count = 0
        planner_diagnostics: PlannerDiagnostics | None = None
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
        return await self.answer_request(
            request,
            context=context,
            request_id=request_id,
            started=started,
            routing_started=routing_started,
            llm_ms=llm_ms,
            llm_call_count=llm_call_count,
            planner_diagnostics=planner_diagnostics,
        )

    async def answer_request(
        self,
        request: SemanticFactRequest,
        *,
        context: AgentRequestContext,
        request_id: str = "-",
        started: float | None = None,
        routing_started: float | None = None,
        llm_ms: float = 0.0,
        llm_call_count: int = 0,
        planner_diagnostics: PlannerDiagnostics | None = None,
    ) -> FactAnswer:
        """Execute a previously validated semantic request without another LLM call."""
        started = perf_counter() if started is None else started
        routing_started = started if routing_started is None else routing_started
        if not self.engine.schema.validates(request):
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
            tier=1,
            routing_ms=routing_ms,
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

    @stage("graph.records")
    async def records(self, tool: str, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        entity_id = arguments.get("entity_id")
        if tool == "get_entity" and isinstance(entity_id, str):
            cached = self.entity_cache.get(entity_id)
            if cached is not None:
                return [dict(cached)]
        self.query_count += 1
        try:
            response = await self.dispatcher.dispatch_internal(
                tool,
                arguments,
                caller_entity_id=self.caller_entity_id,
            )
        except Exception:
            raise _FactFailure("computation_impossible") from None
        if not isinstance(response, Mapping) or response.get("ok") is not True:
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


_SECOND_PERSON_IDENTITY = re.compile(
    r"(?is)^(?:who are you\b|what(?:'s| is) your (?:name|role)\b)|"
    r"^(?:你|您)\s*(?:是\s*(?:谁|哪)|的名字|叫什么)"
)
_FIRST_PERSON_IDENTITY = re.compile(
    r"(?is)^(?:who am i\b|what(?:'s| is) my name\b)|"
    r"^我\s*(?:是\s*(?:谁|哪)|的名字|叫什么|的身份)"
)
_NAMED_OBJECT_LOCATION = re.compile(
    r"(?is)^(?:where(?:'s| is) (?:the )?(?!my\b).+|"
    r"(?!(?:我家|家里|咱家|我这个家)\s*).{1,24}在哪里)"
)


def _identity_person_hint(utterance: str) -> str | None:
    text = utterance.strip()
    if _SECOND_PERSON_IDENTITY.match(text):
        return (
            "The latest utterance addresses this helper in the second person. "
            "subject.kind must be assistant, path must be empty, property=null."
        )
    if _FIRST_PERSON_IDENTITY.match(text):
        return (
            "The latest utterance is first-person identity. "
            "subject.kind must be self, path must be empty, property=null."
        )
    return None


def _object_location_hint(utterance: str) -> str | None:
    if not _NAMED_OBJECT_LOCATION.match(utterance.strip()):
        return None
    return (
        "The latest utterance asks where a named object is. "
        "subject.kind=named_entity, entity_type=item, path concept "
        "location, resolve_reference, property=null. Not a person and "
        "not adult/minor."
    )


def _invalid_plan_retry_hint(
    schema: SemanticSchemaRegistry,
    request: SemanticFactRequest | None,
) -> str | None:
    """IR-based retry grammar; never matches on utterance text."""
    if request is None:
        return None
    notes: list[str] = []
    if schema.contract_error(request) == "CONTRADICTORY_PREDICATES":
        notes.append(
            "adult and minor cannot be combined. Gender is gender=male or "
            "gender=female, not those predicates."
        )
    hops = [step.relation for step in request.subject.path]
    if request.subject.kind == "self" and "member" in hops:
        notes.append(
            "self cannot traverse member; household people use "
            "current_household then member."
        )
    if "location" in hops and request.subject.kind == "current_household":
        notes.append(
            "item location uses named_entity entity_type=item then concept "
            "location; current_household cannot traverse location."
        )
    if (
        request.subject.kind == "named_entity"
        and "location" in hops
        and request.subject.entity_type != "item"
    ):
        notes.append(
            "location accepts items, not persons or spaces; named objects "
            "use entity_type=item."
        )
    named_predicates = {
        item.predicate for item in request.filters if item.predicate
    }
    if request.subject.kind == "named_entity" and named_predicates.intersection(
        {"adult", "minor"}
    ):
        notes.append(
            "named objects are not household members and do not take "
            "adult/minor. A named item's location is entity_type=item then "
            "concept location, resolve_reference, property=null."
        )
    if (
        request.subject.kind == "named_entity"
        and not hops
        and request.operation == "select"
        and request.property is None
    ):
        notes.append(
            "select with property=null needs a collection path. Item "
            "location uses named_entity entity_type=item then concept "
            "location; resolve_reference, property=null."
        )
    filter_properties = {
        item.property for item in request.filters if item.property
    }
    if "member" in hops and filter_properties.intersection({"space_type", "item_type"}):
        notes.append(
            "space_type and item_type are not household-member filters; "
            "rooms use current_household then concept room."
        )
    if request.subject.kind == "current_household" and not hops and (
        request.property in {"space_type", "item_type"}
        or filter_properties.intersection({"space_type", "item_type"})
    ):
        notes.append(
            "household rooms use current_household then concept room."
        )
    if request.operation == "same_entity" and (
        request.property is not None
        or request.other is None
        or request.filters
        or request.property_source != "entity"
    ):
        notes.append(
            "same_entity compares two resolved references; property=null; "
            "other is required; not a stored property and not resolve_reference."
        )
    return " ".join(notes) or None


def _object_location_mismatch(
    utterance: str, request: SemanticFactRequest | None
) -> str | None:
    """Reject object-location plans that name a person or omit location."""
    hint = _object_location_hint(utterance)
    if hint is None:
        return None
    if request is None:
        return hint
    hops = [step.relation for step in request.subject.path]
    if (
        request.subject.kind == "named_entity"
        and request.subject.entity_type in {None, "item"}
        and "location" in hops
        and request.property is None
        and not request.filters
    ):
        return None
    return hint


def _complete_named_object_location(
    request: SemanticFactRequest,
) -> SemanticFactRequest | None:
    """Fill item+location around a name the interpreter already extracted."""
    if request.subject.kind != "named_entity" or not request.subject.value:
        return None
    return SemanticFactRequest(
        operation="resolve_reference",
        subject=SemanticReference(
            kind="named_entity",
            value=request.subject.value,
            entity_type="item",
            path=(SemanticRelationStep(relation="location"),),
        ),
    )


def _identity_person_mismatch(
    utterance: str, request: SemanticFactRequest | None
) -> str | None:
    """Reject first/second-person identity plans that name the wrong referent."""
    if request is None:
        return _identity_person_hint(utterance)
    if (
        request.operation != "resolve_reference"
        or request.property is not None
        or request.subject.path
    ):
        return None
    hint = _identity_person_hint(utterance)
    if hint is None:
        return None
    if _SECOND_PERSON_IDENTITY.match(utterance.strip()):
        return hint if request.subject.kind != "assistant" else None
    if _FIRST_PERSON_IDENTITY.match(utterance.strip()):
        return hint if request.subject.kind != "self" else None
    return None


def _planner_clock(moment: datetime) -> str:
    """Hour-precision clock so the planner prefix stays cacheable within an hour."""
    return moment.replace(minute=0, second=0, microsecond=0).isoformat()


def _compact_json_schema(value: Any) -> Any:
    """Remove model-irrelevant prose while preserving JSON Schema constraints."""
    if isinstance(value, Mapping):
        return {
            key: _compact_json_schema(item)
            for key, item in value.items()
            if key not in {"title", "default", "description"}
        }
    if isinstance(value, list):
        return [_compact_json_schema(item) for item in value]
    return value


def _prefer_null_union(value: Any) -> Any:
    """Put explicit null first so constrained decoding does not invent values."""
    if isinstance(value, Mapping):
        mapped = {key: _prefer_null_union(item) for key, item in value.items()}
        options = mapped.get("anyOf")
        if isinstance(options, list):
            nulls = [item for item in options if item == {"type": "null"}]
            others = [item for item in options if item != {"type": "null"}]
            if nulls:
                mapped["anyOf"] = [*nulls, *others]
        return mapped
    if isinstance(value, list):
        return [_prefer_null_union(item) for item in value]
    return value


def _planner_output_schema() -> dict[str, Any]:
    """Fresh base schema; registry-specific constraints must not leak across homes."""
    schema = _compact_json_schema(SemanticPlan.model_json_schema())
    kind = schema["$defs"]["SemanticReference"]["properties"]["kind"]
    kind["enum"] = [item for item in kind["enum"] if item != "entity_id"]
    return schema


def _planner_runtime_fields(runtime: Mapping[str, Any]) -> dict[str, Any]:
    def number(name: str) -> float | int:
        value = runtime.get(name, 0) or 0
        return value

    return {
        "prompt_eval_count": int(number("prompt_eval_count")),
        "prompt_eval_duration_ms": float(number("prompt_eval_duration_ms")),
        "eval_count": int(number("eval_count")),
        "eval_duration_ms": float(number("eval_duration_ms")),
        "load_duration_ms": float(number("load_duration_ms")),
        "transport": {key: runtime[key] for key in (
            "codec_version", "schema_fingerprint", "compact_output_bytes",
            "expanded_output_bytes", "transport_parse_success", "transport_parse_ms",
            "compact_prompt_bytes", "compact_schema_bytes", "expanded_schema_bytes",
            "field_dictionary_fingerprint",
        ) if key in runtime},
    }


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
        "db_queries=%d llm_calls=%d routing_ms=%.2f "
        "entity_resolution_ms=%.2f fact_query_ms=%.2f computation_ms=%.2f "
        "render_ms=%.2f llm_ms=%.2f total_ms=%.2f status=%s failure_stage=%s "
        "planner_validation=%s planner_attempts=%d "
        "planner_prompt_build_ms=%.2f planner_request_ms=%.2f "
        "planner_validation_ms=%.2f prompt_eval_count=%d "
        "prompt_eval_ms=%.2f eval_count=%d eval_ms=%.2f load_ms=%.2f",
        safe_log_token(request_id),
        timings.tier,
        safe_log_token(request.operation),
        safe_log_token(
            json.dumps(_plan_for_log(request), separators=(",", ":"))
        ),
        timings.db_query_count,
        timings.llm_call_count,
        timings.routing_ms,
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
        planner_diagnostics.prompt_build_ms if planner_diagnostics is not None else 0,
        planner_diagnostics.request_ms if planner_diagnostics is not None else 0,
        planner_diagnostics.validation_ms if planner_diagnostics is not None else 0,
        planner_diagnostics.prompt_eval_count if planner_diagnostics is not None else 0,
        planner_diagnostics.prompt_eval_duration_ms
        if planner_diagnostics is not None
        else 0,
        planner_diagnostics.eval_count if planner_diagnostics is not None else 0,
        planner_diagnostics.eval_duration_ms if planner_diagnostics is not None else 0,
        planner_diagnostics.load_duration_ms if planner_diagnostics is not None else 0,
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


def _result_unit(request: SemanticFactRequest) -> str | None:
    if request.operation == "completed_years":
        return "years"
    if request.operation in {"date_difference", "duration", "annual_occurrence"}:
        return request.mode
    if request.operation == "unit_conversion":
        return request.to_unit
    return None


def _failure_stage(status: FactStatus) -> str | None:
    return {
        "found": None,
        "caller_context_missing": "context",
        "discourse_context_missing": "context",
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
        "collection_incomplete": "collection_completeness",
        "semantic_plan_unsupported": "semantic_plan_validation",
    }[status]


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
        "location": "位置",
        "contents": "包含",
        "host": "承载位置",
        "hosted_space": "空间",
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
        (item.value for item in step.filters if item.property == "gender"
         and item.source == "entity" and item.operator == "eq" and item.value_from is None
         and item.value in {"male", "female"}),
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
        ("location", None): "所在位置",
        ("contents", None): "所含物品",
        ("host", None): "承载物",
        ("hosted_space", None): "空间",
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
