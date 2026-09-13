"""Map declared semantic vocabulary onto this deployment's catalog."""
from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
from typing import Any, get_args

from .request_tracing import stage
from .edge_schema import EdgeSchemaRegistry, UnknownEdgeSchemaError
from .operator_registry import FACT_OPERATORS, OperatorValidationError
from .schema_catalog import RuntimeSchemaCatalog
from .semantic_ontology import SemanticOntology
from .semantic_contracts import ResolvedSemanticContract
from .semantic_ir import (
    PlannerValidationCode,
    ReferenceKind,
    SemanticConceptUse,
    SemanticFactRequest,
    SemanticFilter,
    SemanticPlan,
    SemanticReference,
    _CONTEXT_ENTITY_TYPES,
    _FactFailure,
)

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
        self._property_cache: dict[tuple[str, str], str | None] = {}
        self._capability_cache: dict[str, Any] | None = None
        self._planner_capability_cache: dict[str, Any] | None = None
        self._planner_schema_cache: dict[str, Any] | None = None
        self.contracts = ResolvedSemanticContract(self)
        self._available_predicates = frozenset(
            name for name, definition in self.ontology.collection_predicates.items()
            if all(
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
        available = self.contracts.payload()
        payload['properties'] = {name: value for name, value in payload['properties'].items() if name in available}
        return payload

    def _resolve_available_concepts(self) -> frozenset[str]:
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
        return semantic in self.contracts.properties and self.contracts.properties[semantic].accepts(value)

    def validate_filter_value(self, semantic: str, value: Any) -> None:
        if value is None:
            raise _FactFailure('filter_input_missing', missing=(semantic,))
        if not self.value_valid(semantic, value):
            raise _FactFailure('filter_unsupported', missing=(semantic,))

    def physical_property(self, entity_type: str, semantic: str) -> str | None:
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
        return ()

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
                for relation in sorted(relations)
            },
            "operations": sorted(FACT_OPERATORS),
            "operation_requirements": {
                "select": "property=null returns all matching entities, including zero or many; property set returns a single stored property value. Filter properties are not output projections.",
                "inspect": "property=null; returns all declared semantic attributes of one entity, marking missing values null; no storage IDs or metadata",
                "resolve_reference": "returns exactly one entity; not a list of matching entities",
                "same_entity": "property=null; two resolved references subject and other; returns whether they are the same entity; not identity introduction and not a stored property",
                "argmin": "ordered property required; collection subject OR two references subject and other; returns entity",
                "argmax": "ordered property required; collection subject OR two references subject and other; returns entity",
                "annual_occurrence": "date property; mode=days only for countdown",
                "date_add": "date property + strict integer amount + mode years|months|days; invalid target day becomes the following month's first day, return specified date even in past",
                "date_difference": "one entity OR relationship date to household_now; mode explicitly chooses years|months|days|seconds. Calendar years/months count full anniversaries, signed toward zero; never divide days by a fixed ratio. Age, tenure and elapsed relationship time use this same operation.",
                "unit_conversion": "numeric property + from_unit + to_unit",
            },
            "reference_ontology": self._ontology_payload(),
            "collection_predicates": sorted(self._available_predicates),
            "property_sources": ["entity", "relationship"],
        }
        self._capability_cache['property_contracts'] = self.contracts.payload()
        return deepcopy(self._capability_cache)

    @stage("schema.capabilities")
    def planner_capability_payload(self) -> dict[str, Any]:
        """Serialize the executable grammar, without storage or redundant aliases."""
        if self._planner_capability_cache is None:
            full = self.capability_payload()
            ontology = full["reference_ontology"]
            self._planner_capability_cache = {
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
                "operations": full["operations"],
                "operation_requirements": full["operation_requirements"],
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
                    "properties": {
                        "kind": {
                            "type": "string",
                            "enum": ["self", "assistant", "current_household"],
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
        return frozenset(
            semantic
            for semantic in self.ontology.properties
            if self.physical_property(entity_type, semantic) is not None
        )

    def semantic_relation_properties(self, semantic: str) -> frozenset[str]:
        resolved = self.physical_relation(semantic)
        if resolved is None:
            return frozenset()
        relation = resolved[0]
        schema = self.catalog.relations.get(relation)
        if schema is None:
            return frozenset()
        return frozenset(
            semantic_property
            for semantic_property in self.ontology.properties
            if self.relation_property(relation, semantic_property) is not None
            and (semantic, semantic_property) in self.contracts.relation_bindings
        )

    def validation_code(self, request: SemanticFactRequest) -> PlannerValidationCode:
        """Return a stable, non-sensitive reason for semantic-plan rejection."""
        operation = FACT_OPERATORS.get(request.operation)
        if operation is None:
            return "UNSUPPORTED_OPERATION"
        references = (request.subject,) + ((request.other,) if request.other else ()) + request.exclude
        if self.contract_error(request) is not None:
            return 'INVALID_PLAN'
        for reference in references:
            for step in reference.path:
                if self.physical_relation(step.relation) is None:
                    return "UNKNOWN_RELATION"
        if request.property is not None:
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
        if request.operation in {"resolve_reference", "inspect"}:
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
        if (reference.path[-1].relation, semantic_property) not in self.contracts.relation_bindings:
            return None
        return self.contracts.properties[semantic_property].type.execution_kind

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
        if not entity_types or any((owner, semantic_property) not in self.contracts.entity_bindings for owner in entity_types):
            return None
        return self.contracts.properties[semantic_property].type.execution_kind

    def _valid_predicate(self, item: SemanticFilter, field_kind: str) -> bool:
        if item.transform == "date_difference":
            return field_kind in {"date", "datetime"}
        contract = self.contracts.properties.get(item.property)
        return contract is not None and contract.literal_error(item.operator, item.value, anchor=item.value_from is not None) is None



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
