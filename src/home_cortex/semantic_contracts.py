"""Versioned declarative property contracts; no language routing or value repair."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, fields, asdict
from datetime import date, datetime
from typing import Any, Mapping

from .operator_registry import OPERATORS, PREDICATE_OPERATORS


@dataclass(frozen=True)
class SemanticType:
    kind: str
    items: SemanticType | None = None
    alternatives: tuple[SemanticType, ...] = ()

    @classmethod
    def parse(cls, raw: Any, *, nested: bool = False) -> SemanticType:
        if not isinstance(raw, dict):
            raise ValueError('contract type must be an object')
        if set(raw) == {'any_of'} and not nested:
            choices = raw['any_of']
            if not isinstance(choices, list) or len(choices) < 2:
                raise ValueError('type.any_of requires at least two alternatives')
            alternatives = tuple(cls.parse(item, nested=True) for item in choices)
            if len(set(alternatives)) != len(alternatives):
                raise ValueError('duplicate type alternatives')
            return cls('union', alternatives=alternatives)
        kind = raw.get('kind')
        if kind == 'collection' and set(raw) == {'kind', 'items'}:
            items = cls.parse(raw['items'], nested=True)
            if items.kind not in {'string', 'boolean', 'integer', 'number', 'date', 'datetime'}:
                raise ValueError('collection items must be atomic')
            return cls(kind, items=items)
        if set(raw) != {'kind'} or kind not in {'string', 'boolean', 'integer', 'number', 'date', 'datetime', 'object'}:
            raise ValueError('unsupported contract type')
        return cls(kind)

    def accepts(self, value: Any) -> bool:
        if self.alternatives:
            return any(item.accepts(value) for item in self.alternatives)
        if self.kind == 'string':
            return isinstance(value, str)
        if self.kind == 'boolean':
            return type(value) is bool
        if self.kind == 'integer':
            return type(value) is int
        if self.kind == 'number':
            return type(value) in {int, float} and (type(value) is int or math.isfinite(value))
        if self.kind == 'collection':
            return isinstance(value, (list, tuple)) and all(self.items.accepts(item) for item in value)
        if self.kind == 'object':
            return isinstance(value, Mapping)
        if not isinstance(value, str):
            return False
        try:
            if self.kind == 'date':
                return bool(re.fullmatch(r'\d{4}-\d{2}-\d{2}', value)) and bool(date.fromisoformat(value))
            return bool(re.fullmatch(r'\d{4}-\d{2}-\d{2}T.+(?:Z|[+-]\d{2}:\d{2})', value)) and datetime.fromisoformat(value).utcoffset() is not None
        except ValueError:
            return False

    def payload(self) -> dict[str, Any]:
        if self.alternatives:
            return {'any_of': [item.payload() for item in self.alternatives]}
        return {'kind': self.kind, **({'items': self.items.payload()} if self.items else {})}

    def schema(self) -> dict[str, Any]:
        if self.alternatives:
            return {'anyOf': [item.schema() for item in self.alternatives]}
        if self.kind == 'collection':
            return {'type': 'array', 'items': self.items.schema()}
        if self.kind in {'date', 'datetime'}:
            return {'type': 'string', 'format': 'date' if self.kind == 'date' else 'date-time'}
        return {'type': self.kind}

    @property
    def kinds(self) -> frozenset[str]:
        return frozenset(item.kind for item in self.alternatives) if self.alternatives else frozenset({self.kind})

    @property
    def execution_kind(self) -> str:
        return next(iter(self.kinds)) if len(self.kinds) == 1 else 'unknown'


def strings(raw: Any, name: str) -> tuple[str, ...]:
    if not isinstance(raw, list) or any(not isinstance(item, str) or not item for item in raw) or len(set(raw)) != len(raw):
        raise ValueError(f'{name} must be a list of unique strings')
    return tuple(raw)


@dataclass(frozen=True)
class PropertyContract:
    type: SemanticType
    entities: tuple[str, ...]
    relationships: tuple[str, ...]
    operators: tuple[str, ...]
    # Canonical values and their linguistic aliases; labels remain in the ontology.
    values: tuple[tuple[str, tuple[str, ...]], ...] = ()

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> PropertyContract:
        kind = SemanticType.parse(raw.get('type'))
        applies = raw.get('applies_to')
        if not isinstance(applies, dict) or set(applies) != {'entity', 'relationship'}:
            raise ValueError('applies_to requires entity and relationship lists')
        entities = strings(applies['entity'], 'applies_to.entity')
        relationships = strings(applies['relationship'], 'applies_to.relationship')
        if not entities and not relationships:
            raise ValueError('property requires at least one applicable owner')
        operators = strings(raw.get('filter_operators'), 'filter_operators')
        values = raw.get('values', {})
        if not isinstance(values, dict) or ('values' in raw and not values) or (values and kind.kind != 'string'):
            raise ValueError('closed values require a nonempty string domain')
        parsed = []
        seen: dict[str, str] = {}
        for value, definition in values.items():
            if not isinstance(value, str) or not isinstance(definition, dict) or set(definition) - {'aliases', 'label'}:
                raise ValueError('invalid canonical value declaration')
            aliases = strings(definition.get('aliases', []), 'value aliases')
            for alias in (value, *aliases):
                folded = alias.casefold()
                if folded in seen and seen[folded] != value:
                    raise ValueError('colliding value aliases')
                seen[folded] = value
            parsed.append((value, aliases))
        for operator in operators:
            if operator not in PREDICATE_OPERATORS:
                raise ValueError('unknown filter operator')
            allowed = OPERATORS[operator].field_kinds
            if 'any' not in allowed and not kind.kinds.issubset(allowed):
                raise ValueError('operator incompatible with property type')
            if operator != 'exists' and kind.kinds.intersection({'collection', 'object'}):
                raise ValueError('structured values support projection and exists only')
            if values and operator not in {'eq', 'ne', 'in', 'exists'}:
                raise ValueError('closed domains are unordered')
        return cls(kind, entities, relationships, operators, tuple(parsed))

    def accepts(self, value: Any) -> bool:
        return self.type.accepts(value) and (not self.values or value in dict(self.values))

    def literal_error(self, operator: str, value: Any, *, anchor: bool = False) -> str | None:
        if operator not in self.operators:
            return 'OPERATOR_NOT_APPLICABLE'
        if anchor:
            return None if operator in {'eq', 'ne', 'gt', 'gte', 'lt', 'lte'} else 'INVALID_OPERAND'
        if operator == 'exists':
            return None if value is None or type(value) is bool else 'INVALID_LITERAL_TYPE'
        values = value if operator in {'in', 'date_range'} else (value,)
        if not isinstance(values, tuple) or not values:
            return 'INVALID_OPERAND'
        if operator == 'date_range' and len(values) != 2:
            return 'INVALID_OPERAND'
        if any(not self.type.accepts(item) for item in values):
            return 'INVALID_LITERAL_TYPE'
        if self.type.alternatives and not any(
            all(alternative.accepts(item) for item in values)
            for alternative in self.type.alternatives
        ):
            return 'INVALID_LITERAL_TYPE'
        if self.values and any(item not in dict(self.values) for item in values):
            return 'VALUE_OUT_OF_DOMAIN'
        if operator == 'date_range':
            try:
                parsed = [datetime.fromisoformat(item) if 'T' in item else date.fromisoformat(item) for item in values]
                if type(parsed[0]) is not type(parsed[1]) or parsed[0] >= parsed[1]:
                    return 'INVALID_RANGE'
            except (TypeError, ValueError):
                return 'INVALID_RANGE'
        return None

    def payload(self) -> dict[str, Any]:
        return {'type': self.type.payload(), 'applies_to': {'entity': list(self.entities), 'relationship': list(self.relationships)},
                'filter_operators': list(self.operators), **({'values': {key: {'aliases': list(aliases)} for key, aliases in self.values}} if self.values else {})}

    def filter_branches(self, name: str, *, traversal: bool, sources: tuple[str, ...]) -> list[dict[str, Any]]:
        literal = self.type.schema()
        if self.values:
            literal = {'type': 'string', 'enum': [key for key, _ in self.values]}
        branches = []
        for source in sources:
            groups = []
            scalar = [op for op in self.operators if op in {'eq', 'ne', 'gt', 'gte', 'lt', 'lte'}]
            if scalar:
                groups.append(scalar)
            groups.extend([op] for op in self.operators if op not in scalar)
            for operators in groups:
                operator = operators[0]
                operand = literal
                if operator == 'exists':
                    operand = {'anyOf': [{'type': 'boolean'}, {'type': 'null'}]}
                elif operator in {'in', 'date_range'}:
                    operand = {'type': 'array', 'items': literal, 'minItems': 2 if operator == 'date_range' else 1}
                    if operator == 'date_range':
                        operand['maxItems'] = 2
                props = {'property': {'enum': [name]}, 'source': {'enum': [source]}, 'operator': {'enum': operators}, 'value': operand}
                required = ['property', 'operator', 'value'] + (['source'] if source == 'relation' else [])
                branches.append({'type': 'object', 'additionalProperties': False, 'properties': props, 'required': required})
                if traversal and source == 'entity' and operator in {'eq', 'ne', 'gt', 'gte', 'lt', 'lte'}:
                    dynamic = {key: value for key, value in props.items() if key != 'value'}
                    dynamic.update({'value_from': {'enum': ['anchor']}, 'value_property': {'type': 'string'}})
                    branches.append({'type': 'object', 'additionalProperties': False, 'properties': dynamic,
                                     'required': ['property', 'operator', 'value_from']})
        return branches


@dataclass(frozen=True)
class ResolvedSemanticContract:
    """Deployment bindings resolved once; public views never contain storage names."""
    properties: Mapping[str, PropertyContract]
    entity_bindings: Mapping[tuple[str, str], str]
    relation_bindings: Mapping[tuple[str, str], str]
    fingerprint: str

    def __init__(self, registry: Any) -> None:
        from types import MappingProxyType
        import hashlib
        import json

        properties = {name: definition.contract for name, definition in registry.ontology.properties.items()}
        object.__setattr__(self, 'properties', MappingProxyType(properties))
        entity_bindings = {}
        relation_bindings = {}
        for name, contract in properties.items():
            for entity in contract.entities:
                physical = registry.physical_property(entity, name)
                if physical is not None:
                    observed = registry.catalog.entity_field_type(entity, physical)
                    self._check_kind(name, contract, observed)
                    entity_bindings[(entity, name)] = physical
            for relation in contract.relationships:
                resolved = registry.physical_relation(relation)
                physical = registry.relation_property(resolved[0], name) if resolved else None
                if physical is not None:
                    observed = registry.catalog.relation_field_type(resolved[0], physical)
                    self._check_kind(name, contract, observed)
                    relation_bindings[(relation, name)] = physical
        object.__setattr__(self, 'entity_bindings', MappingProxyType(entity_bindings))
        object.__setattr__(self, 'relation_bindings', MappingProxyType(relation_bindings))
        fingerprint = hashlib.sha256(json.dumps({
            'properties': {key: value.payload() for key, value in sorted(properties.items())},
            'entities': sorted((list(key), value) for key, value in entity_bindings.items()),
            'relations': sorted((list(key), value) for key, value in relation_bindings.items()),
            'ontology': registry.ontology.planner_payload(),
            'operators': {name: {field.name: (sorted(value) if isinstance(value, frozenset) else value) for field in fields(definition) if field.name != 'implementation' for value in [getattr(definition, field.name)]} for name, definition in sorted(OPERATORS.items())},
            'display_and_declarations': {name: asdict(value) for name, value in registry.ontology.properties.items()},
            'predicates': {name: asdict(value) for name, value in registry.ontology.collection_predicates.items()},
            'concepts': {name: asdict(value) for name, value in registry.ontology.reference_concepts.items()},
            'catalog': {
                'entities': {name: {'properties': value.properties, 'types': dict(value.property_types)} for name, value in registry.catalog.entities.items()},
                'relations': {name: {'from': value.from_types, 'to': value.to_types,
                    'properties': value.properties, 'types': dict(value.property_types),
                    'symmetric': value.symmetric, 'temporal': value.temporal}
                    for name, value in registry.catalog.relations.items()},
                'bindings': {name: registry.physical_relation(name) for name in registry.ontology.base_relations},
            },
        }, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        object.__setattr__(self, 'fingerprint', fingerprint)

    @staticmethod
    def _check_kind(name: str, contract: PropertyContract, observed: str) -> None:
        if observed == 'unknown':
            return
        allowed = contract.type.kinds
        if observed in allowed or (observed == 'integer' and 'number' in allowed):
            return
        # A string declaration also accepts ISO-looking strings catalogued as dates.
        if 'string' in allowed and observed in {'date', 'datetime'}:
            return
        raise ValueError(f'Incompatible catalog type for semantic property {name}: {observed}')

    def payload(self) -> dict[str, Any]:
        payload = {}
        for name, contract in sorted(self.properties.items()):
            entities = sorted(owner for owner, prop in self.entity_bindings if prop == name)
            relationships = sorted(owner for owner, prop in self.relation_bindings if prop == name)
            if entities or relationships:
                payload[name] = {**contract.payload(), 'applies_to': {'entity': entities, 'relationship': relationships}}
        return payload

    def filters_schema(self, *, traversal: bool, predicates: tuple[str, ...]) -> dict[str, Any]:
        branches = []
        for name, payload in self.payload().items():
            sources = tuple(source for source, owner in (('entity', 'entity'), ('relation', 'relationship')) if payload['applies_to'][owner])
            generated = self.properties[name].filter_branches(name, traversal=traversal, sources=sources)
            anchors = sorted(other for other, contract in self.properties.items()
                if contract.type == self.properties[name].type
                and {value for value, _ in contract.values} == {value for value, _ in self.properties[name].values}
                and any(prop == other for _, prop in self.entity_bindings))
            for branch in generated:
                if 'value_property' in branch['properties']:
                    branch['properties']['value_property'] = {'enum': anchors}
            branches.extend(generated)
        if not traversal and predicates:
            branches.append({'type': 'object', 'additionalProperties': False,
                             'properties': {'predicate': {'enum': list(predicates)}}, 'required': ['predicate']})
        return {'anyOf': branches} if branches else {'not': {}}
