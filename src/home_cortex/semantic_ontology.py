"""Declarative semantic vocabulary and relation composition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml

from .operator_registry import OPERATORS, PREDICATE_OPERATORS


@dataclass(frozen=True)
class OntologyFilter:
    property: str
    operator: str = "eq"
    value: str | int | float | bool | None = None
    source: str = "entity"
    value_from: str | None = None
    value_property: str | None = None


@dataclass(frozen=True)
class OntologyRelationStep:
    relation: str
    filters: tuple[OntologyFilter, ...] = ()


@dataclass(frozen=True)
class OntologyReferenceConcept:
    name: str
    aliases: tuple[str, ...]
    path: tuple[OntologyRelationStep, ...]


@dataclass(frozen=True)
class OntologyProperty:
    name: str
    fields: tuple[str, ...]
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class OntologyPredicateFallback:
    property: str
    transform: str
    operator: str
    value_from_policy: str


@dataclass(frozen=True)
class OntologyCollectionPredicate:
    name: str
    aliases: tuple[str, ...]
    entity_types: tuple[str, ...]
    relation_property: str
    recognized_values: tuple[str, ...]
    matching_values: tuple[str, ...]
    fallback: OntologyPredicateFallback
    default_scope_relation: str | None = None


class SemanticOntology:
    """Validated domain ontology loaded from repository schema data."""

    def __init__(
        self,
        *,
        properties: Mapping[str, OntologyProperty],
        base_relations: Mapping[str, str],
        reference_concepts: Mapping[str, OntologyReferenceConcept],
        policy_values: Mapping[str, int | float],
        collection_predicates: Mapping[str, OntologyCollectionPredicate],
    ) -> None:
        self.properties = MappingProxyType(dict(properties))
        self.base_relations = MappingProxyType(dict(base_relations))
        self.reference_concepts = MappingProxyType(dict(reference_concepts))
        self.policy_values = MappingProxyType(dict(policy_values))
        self.collection_predicates = MappingProxyType(dict(collection_predicates))
        alias_pairs = [
            (alias.casefold(), concept)
            for concept in self.reference_concepts.values()
            for alias in concept.aliases
        ]
        duplicates = _duplicates(alias for alias, _ in alias_pairs)
        if duplicates:
            raise ValueError(
                "Semantic relation aliases must be unique: "
                + ", ".join(sorted(duplicates))
            )
        predicate_alias_pairs = [
            (alias.casefold(), definition.name)
            for definition in self.collection_predicates.values()
            for alias in (definition.name, *definition.aliases)
        ]
        predicate_duplicates = _duplicates(alias for alias, _ in predicate_alias_pairs)
        if predicate_duplicates:
            raise ValueError(
                "Semantic predicate aliases must be unique: "
                + ", ".join(sorted(predicate_duplicates))
            )
        self._collection_predicate_aliases = MappingProxyType(
            dict(predicate_alias_pairs)
        )

    @classmethod
    def from_file(cls, path: Path) -> "SemanticOntology":
        if not path.is_file():
            raise FileNotFoundError(f"Semantic ontology does not exist: {path}")
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"{path} must contain a YAML object")
        allowed = {
            "version",
            "properties",
            "base_relations",
            "reference_concepts",
            "policy_values",
            "collection_predicates",
        }
        extra = sorted(set(raw) - allowed)
        if extra:
            raise ValueError(f"Unknown semantic ontology fields: {', '.join(extra)}")
        if raw.get("version") != 1:
            raise ValueError("Semantic ontology version must be 1")

        properties = _parse_properties(raw.get("properties"), path)
        base_relations = _parse_base_relations(raw.get("base_relations"), path)
        concepts = _parse_reference_concepts(
            raw.get("reference_concepts"),
            base_relations,
            path,
        )
        policy_values = _parse_policy_values(raw.get("policy_values", {}), path)
        collection_predicates = _parse_collection_predicates(
            raw.get("collection_predicates", {}),
            policy_values,
            path,
        )
        invalid_scope_relations = {
            definition.default_scope_relation
            for definition in collection_predicates.values()
            if definition.default_scope_relation is not None
            and definition.default_scope_relation not in base_relations
        }
        if invalid_scope_relations:
            raise ValueError(
                "Collection predicates reference unknown scope relations: "
                + ", ".join(sorted(invalid_scope_relations))
            )
        return cls(
            properties=properties,
            base_relations=base_relations,
            reference_concepts=concepts,
            policy_values=policy_values,
            collection_predicates=collection_predicates,
        )

    @classmethod
    def load_default(cls) -> "SemanticOntology":
        candidates = (
            Path(__file__).resolve().parents[2] / "schemas" / "semantic" / "ontology.yaml",
            Path("/app/schemas/semantic/ontology.yaml"),
        )
        for candidate in candidates:
            if candidate.is_file():
                return cls.from_file(candidate)
        raise FileNotFoundError("Could not locate schemas/semantic/ontology.yaml")

    def property_fields(self, semantic: str) -> tuple[str, ...]:
        definition = self.properties.get(semantic)
        return definition.fields if definition is not None else (semantic,)

    def resolve_collection_predicate(self, value: str) -> str | None:
        return self._collection_predicate_aliases.get(value.casefold())

    def prompt_payload(self) -> dict[str, Any]:
        return {
            "properties": {
                name: {"aliases": list(definition.aliases)}
                for name, definition in self.properties.items()
            },
            "base_relations": sorted(self.base_relations),
            "reference_concepts": {
                name: {
                    "aliases": list(concept.aliases),
                    "path": [
                        {
                            "relation": step.relation,
                            "filters": [
                                {
                                    "property": item.property,
                                    "operator": item.operator,
                                    "value": item.value,
                                    "source": item.source,
                                    **(
                                        {"value_from": item.value_from}
                                        if item.value_from is not None
                                        else {}
                                    ),
                                    **(
                                        {"value_property": item.value_property}
                                        if item.value_property is not None
                                        else {}
                                    ),
                                }
                                for item in step.filters
                            ],
                        }
                        for step in concept.path
                    ],
                }
                for name, concept in self.reference_concepts.items()
            },
            "collection_predicates": {
                name: {
                    "aliases": list(definition.aliases),
                    "entity_types": list(definition.entity_types),
                    "default_scope_relation": definition.default_scope_relation,
                    "precedence": [
                        {
                            "source": "relation",
                            "property": definition.relation_property,
                            "recognized_values": list(definition.recognized_values),
                            "matching_values": list(definition.matching_values),
                        },
                        {
                            "source": "derived",
                            "property": definition.fallback.property,
                            "transform": definition.fallback.transform,
                            "operator": definition.fallback.operator,
                            "value": self.policy_values[
                                definition.fallback.value_from_policy
                            ],
                        },
                    ],
                }
                for name, definition in self.collection_predicates.items()
            },
        }

    def planner_payload(self) -> dict[str, Any]:
        """Compact declarative vocabulary for a small-context local planner."""
        return {
            "properties": {
                name: list(definition.aliases)
                for name, definition in self.properties.items()
            },
            "base_relations": sorted(self.base_relations),
            "reference_concepts": {
                name: {
                    "aliases": list(concept.aliases),
                    "path": [
                        {
                            "relation": step.relation,
                            **(
                                {
                                    "filters": [
                                        {
                                            "property": item.property,
                                            **(
                                                {"operator": item.operator}
                                                if item.operator != "eq"
                                                else {}
                                            ),
                                            **(
                                                {"value": item.value}
                                                if item.value is not None
                                                else {}
                                            ),
                                            **(
                                                {"source": item.source}
                                                if item.source != "entity"
                                                else {}
                                            ),
                                            **(
                                                {"value_from": item.value_from}
                                                if item.value_from is not None
                                                else {}
                                            ),
                                            **(
                                                {"value_property": item.value_property}
                                                if item.value_property is not None
                                                else {}
                                            ),
                                        }
                                        for item in step.filters
                                    ]
                                }
                                if step.filters
                                else {}
                            ),
                        }
                        for step in concept.path
                    ],
                }
                for name, concept in self.reference_concepts.items()
            },
            "collection_predicates": {
                name: {
                    "aliases": list(definition.aliases),
                    "scope_relation": definition.default_scope_relation,
                }
                for name, definition in self.collection_predicates.items()
            },
        }


def _parse_policy_values(raw: Any, path: Path) -> dict[str, int | float]:
    values = _mapping(raw, "policy_values", path)
    result: dict[str, int | float] = {}
    for name, value in values.items():
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or value < 0
        ):
            raise ValueError(f"policy_values.{name} must be a non-negative number")
        result[name] = value
    return result


def _parse_collection_predicates(
    raw: Any,
    policy_values: Mapping[str, int | float],
    path: Path,
) -> dict[str, OntologyCollectionPredicate]:
    values = _mapping(raw, "collection_predicates", path)
    result: dict[str, OntologyCollectionPredicate] = {}
    for name, definition in values.items():
        label = f"collection_predicates.{name}"
        item = _mapping(definition, label, path)
        allowed = {
            "aliases",
            "entity_types",
            "relation_property",
            "recognized_values",
            "matching_values",
            "fallback",
            "default_scope_relation",
        }
        if extra := sorted(set(item) - allowed):
            raise ValueError(f"Unknown {label} fields: {', '.join(extra)}")
        relation_property = item.get("relation_property")
        if not isinstance(relation_property, str) or not relation_property:
            raise ValueError(f"{label}.relation_property must be a string")
        default_scope_relation = item.get("default_scope_relation")
        if default_scope_relation is not None and (
            not isinstance(default_scope_relation, str)
            or not default_scope_relation
        ):
            raise ValueError(f"{label}.default_scope_relation must be a string")
        recognized = _strings(
            item.get("recognized_values"),
            f"{label}.recognized_values",
            path,
        )
        matching = _strings(
            item.get("matching_values"),
            f"{label}.matching_values",
            path,
        )
        if not set(matching).issubset(recognized):
            raise ValueError(f"{label}.matching_values must be recognized values")
        fallback_raw = _mapping(item.get("fallback"), f"{label}.fallback", path)
        if set(fallback_raw) != {
            "property",
            "transform",
            "operator",
            "value_from_policy",
        }:
            raise ValueError(f"{label}.fallback has invalid fields")
        property_name = fallback_raw.get("property")
        transform = fallback_raw.get("transform")
        operator = fallback_raw.get("operator")
        value_from_policy = fallback_raw.get("value_from_policy")
        if not all(
            isinstance(value, str) and value
            for value in (property_name, transform, operator, value_from_policy)
        ):
            raise ValueError(f"{label}.fallback values must be strings")
        if transform not in OPERATORS or OPERATORS[transform].implementation is None:
            raise ValueError(f"{label}.fallback transform is not executable")
        if operator not in PREDICATE_OPERATORS:
            raise ValueError(f"{label}.fallback operator is not allowlisted")
        if value_from_policy not in policy_values:
            raise ValueError(f"{label}.fallback references an unknown policy value")
        result[name] = OntologyCollectionPredicate(
            name=name,
            aliases=_strings(item.get("aliases"), f"{label}.aliases", path),
            entity_types=_strings(
                item.get("entity_types"), f"{label}.entity_types", path
            ),
            relation_property=relation_property,
            recognized_values=recognized,
            matching_values=matching,
            fallback=OntologyPredicateFallback(
                property=property_name,
                transform=transform,
                operator=operator,
                value_from_policy=value_from_policy,
            ),
            default_scope_relation=default_scope_relation,
        )
    return result


def _parse_properties(raw: Any, path: Path) -> dict[str, OntologyProperty]:
    values = _mapping(raw, "properties", path)
    result: dict[str, OntologyProperty] = {}
    for name, definition in values.items():
        item = _mapping(definition, f"properties.{name}", path)
        extra = sorted(set(item) - {"fields", "aliases"})
        if extra:
            raise ValueError(f"Unknown properties.{name} fields: {', '.join(extra)}")
        result[name] = OntologyProperty(
            name,
            _strings(item.get("fields"), f"properties.{name}.fields", path),
            _strings(item.get("aliases", []), f"properties.{name}.aliases", path),
        )
    return result


def _parse_base_relations(raw: Any, path: Path) -> dict[str, str]:
    values = _mapping(raw, "base_relations", path)
    result: dict[str, str] = {}
    for name, definition in values.items():
        item = _mapping(definition, f"base_relations.{name}", path)
        if set(item) != {"edge"} or not isinstance(item["edge"], str):
            raise ValueError(f"base_relations.{name} requires only a string edge")
        result[name] = item["edge"]
    return result


def _parse_reference_concepts(
    raw: Any,
    base_relations: Mapping[str, str],
    path: Path,
) -> dict[str, OntologyReferenceConcept]:
    values = _mapping(raw, "reference_concepts", path)
    result: dict[str, OntologyReferenceConcept] = {}
    for name, definition in values.items():
        item = _mapping(definition, f"reference_concepts.{name}", path)
        extra = sorted(set(item) - {"aliases", "path"})
        if extra:
            raise ValueError(
                f"Unknown reference_concepts.{name} fields: {', '.join(extra)}"
            )
        aliases = _strings(
            item.get("aliases"), f"reference_concepts.{name}.aliases", path
        )
        raw_steps = item.get("path")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise ValueError(f"reference_concepts.{name}.path must be a list")
        steps: list[OntologyRelationStep] = []
        for index, raw_step in enumerate(raw_steps):
            step = _mapping(
                raw_step, f"reference_concepts.{name}.path[{index}]", path
            )
            extra_step = sorted(set(step) - {"relation", "filters"})
            relation = step.get("relation")
            if extra_step or not isinstance(relation, str):
                raise ValueError(
                    f"Invalid reference_concepts.{name}.path[{index}]"
                )
            if relation not in base_relations:
                raise ValueError(
                    f"Unknown base relation {relation!r} in reference concept {name}"
                )
            filters = tuple(
                _parse_filter(raw_filter, name, index, filter_index, path)
                for filter_index, raw_filter in enumerate(step.get("filters", []))
            )
            steps.append(OntologyRelationStep(relation, filters))
        result[name] = OntologyReferenceConcept(name, aliases, tuple(steps))
    return result


def _parse_filter(
    raw: Any,
    concept: str,
    step_index: int,
    filter_index: int,
    path: Path,
) -> OntologyFilter:
    label = f"reference_concepts.{concept}.path[{step_index}].filters[{filter_index}]"
    item = _mapping(raw, label, path)
    extra = sorted(
        set(item)
        - {
            "property",
            "operator",
            "value",
            "source",
            "value_from",
            "value_property",
        }
    )
    property_name = item.get("property")
    operator = item.get("operator", "eq")
    source = item.get("source", "entity")
    value_from = item.get("value_from")
    value_property = item.get("value_property")
    if extra or not isinstance(property_name, str):
        raise ValueError(f"Invalid {label}")
    if operator not in PREDICATE_OPERATORS:
        raise ValueError(f"{label} uses an unregistered predicate")
    if source not in {"entity", "relation"}:
        raise ValueError(f"{label}.source must be entity or relation")
    if value_from not in {None, "anchor"}:
        raise ValueError(f"{label}.value_from must be anchor when present")
    if value_property is not None and not isinstance(value_property, str):
        raise ValueError(f"{label}.value_property must be a string")
    if value_property is not None and value_from is None:
        raise ValueError(f"{label}.value_property requires value_from")
    if value_from is not None and source != "entity":
        raise ValueError(f"{label}.value_from requires entity source")
    value = item.get("value")
    if value is not None and not isinstance(value, (str, int, float, bool)):
        raise ValueError(f"{label}.value must be a scalar")
    if value_from is not None and value is not None:
        raise ValueError(f"{label} cannot define both value and value_from")
    return OntologyFilter(
        property_name,
        operator,
        value,
        source,
        value_from,
        value_property,
    )


def _mapping(value: Any, field: str, path: Path) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{field} in {path} must be an object")
    return dict(value)


def _strings(value: Any, field: str, path: Path) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"{field} in {path} must be a string list")
    values = tuple(item.strip() for item in value)
    if len(values) != len(set(values)):
        raise ValueError(f"{field} in {path} contains duplicates")
    return values


def _duplicates(values: Any) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates
