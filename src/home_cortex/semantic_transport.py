"""Versioned, schema-derived serialization at the interpreter boundary only."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from copy import deepcopy
from functools import lru_cache
from typing import Any

from jsonschema import Draft202012Validator

CODEC_VERSION = 2


def canonical_json(value: Any) -> str:
    """Maps/sets are unordered; arrays and tuples retain semantic order."""
    def normalize(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {key: normalize(value) for key, value in sorted(item.items())}
        if isinstance(item, (set, frozenset)):
            return sorted((normalize(value) for value in item), key=canonical_json)
        if isinstance(item, (list, tuple)):
            return [normalize(value) for value in item]
        return item
    return json.dumps(normalize(value), ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _fields(schema: Any) -> set[str]:
    result: set[str] = set()
    if isinstance(schema, dict):
        result.update(schema.get('properties', {}))
        for value in schema.values():
            result.update(_fields(value))
    elif isinstance(schema, list):
        for value in schema:
            result.update(_fields(value))
    return result


def _symbol(index: int) -> str:
    # Bijective base 26: a..z, aa..az. No manually maintained domain vocabulary.
    result = ''
    while index >= 0:
        result = chr(97 + index % 26) + result
        index = index // 26 - 1
    return result


class SemanticTransport:
    def __init__(self, schema: Mapping[str, Any]):
        from .semantic_facts import SemanticPlan, SemanticConceptUse
        self.expanded_schema = deepcopy(dict(schema))
        fields = _fields(SemanticPlan.model_json_schema()) | _fields(SemanticConceptUse.model_json_schema())
        self.aliases = {name: _symbol(i) for i, name in enumerate(sorted(fields))}
        self.names = {value: key for key, value in self.aliases.items()}
        self.schema_fingerprint = fingerprint(schema)
        self.version = CODEC_VERSION
        self.expanded_validator = Draft202012Validator(self.expanded_schema)
        self.layouts: set[tuple[str, ...]] = set()
        body = self._schema(self.expanded_schema)
        definitions = body.pop('$defs', {})
        self.schema = {
            'type': 'array', 'prefixItems': [{'const': self.version, 'type': 'integer'}, body],
            'minItems': 2, 'maxItems': 2, '$defs': definitions,
        }
        self.validator = Draft202012Validator(self.schema)

    def _schema(self, node: Any) -> Any:
        if isinstance(node, list):
            return [self._schema(value) for value in node]
        if not isinstance(node, dict):
            return node
        if node.get('type') == 'object' and 'properties' in node:
            supported = {'type', 'properties', 'required', 'additionalProperties', '$defs', 'title', 'description', 'default'}
            if set(node) - supported or node.get('additionalProperties') is not False:
                raise ValueError('Unsupported semantic object schema constraints')
            required = sorted(node.get('required', []))
            optional = sorted(set(node['properties']) - set(required))
            self.layouts.add(tuple(required))
            prefix = [self._schema(node['properties'][name]) for name in required]
            if optional:
                prefix.append({'type': 'object', 'additionalProperties': False,
                               'properties': {self.aliases[name]: self._schema(node['properties'][name]) for name in optional}})
            result = {'type': 'array', 'prefixItems': prefix, 'minItems': len(required), 'maxItems': len(prefix)}
            if '$defs' in node:
                result['$defs'] = {key: self._schema(value) for key, value in node['$defs'].items()}
            return result
        return {key: self._schema(value) for key, value in node.items() if key not in {'title', 'description'}}

    def _resolve(self, node):
        while '$ref' in node:
            ref = node['$ref']
            if not ref.startswith('#/$defs/'):
                raise ValueError('Nonlocal semantic schema reference')
            node = self.expanded_schema['$defs'][ref.removeprefix('#/$defs/')]
        return node

    def _shape_schema(self, node):
        if isinstance(node, dict):
            return {key: self._shape_schema(child) for key, child in node.items() if key not in {'enum', 'const'}}
        if isinstance(node, list):
            return [self._shape_schema(child) for child in node]
        return node

    def _convert(self, value: Any, node: Any, *, decode: bool) -> Any:
        node = self._resolve(node)
        if 'anyOf' in node:
            for branch in node['anyOf']:
                if decode:
                    wire_schema = {**self._schema(branch), '$defs': self.schema['$defs']}
                    matches = Draft202012Validator(wire_schema).is_valid(value)
                else:
                    matches = Draft202012Validator({**self._shape_schema(branch), '$defs': self._shape_schema(self.expanded_schema.get('$defs', {}))}).is_valid(value)
                if matches:
                    return self._convert(value, branch, decode=decode)
            raise ValueError('Invalid semantic transport union')
        if node.get('type') == 'object' and 'properties' in node:
            required = sorted(node.get('required', []))
            properties = node['properties']
            if decode:
                result = {name: self._convert(child, properties[name], decode=True)
                          for name, child in zip(required, value[:len(required)], strict=True)}
                if len(value) > len(required):
                    for alias, child in value[-1].items():
                        name = self.names[alias]
                        result[name] = self._convert(child, properties[name], decode=True)
                return result
            if not isinstance(value, Mapping) or set(value) - set(properties):
                raise ValueError('Unknown semantic transport field')
            result = [self._convert(value[name], properties[name], decode=False) for name in required]
            tail = {self.aliases[name]: self._convert(child, properties[name], decode=False)
                    for name, child in value.items() if name not in required}
            if tail:
                result.append(tail)
            return result
        if node.get('type') == 'array':
            return [self._convert(child, node.get('items', {}), decode=decode) for child in value]
        return value

    def encode(self, value: Any, *, validate: bool = True) -> str:
        """Encode canonical IR or pre-expansion concept mapping; never repair."""
        if hasattr(value, 'model_dump'):
            value = value.model_dump(mode='json', exclude_defaults=True)
        if validate and not self.expanded_validator.is_valid(value):
            raise ValueError('Invalid expanded semantic schema')
        return canonical_json([self.version, self._convert(value, self.expanded_schema, decode=False)])

    def decode(self, text: str) -> dict[str, Any]:
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError('Duplicate semantic transport field')
                result[key] = value
            return result
        def invalid_constant(_):
            raise ValueError('Nonfinite semantic transport number')
        def finite_float(value):
            parsed = float(value)
            if not math.isfinite(parsed):
                raise ValueError('Nonfinite semantic transport number')
            return parsed
        wire = json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_constant, parse_float=finite_float)
        if not isinstance(wire, list) or len(wire) != 2 or type(wire[0]) is not int or wire[0] != self.version:
            raise ValueError('Unsupported semantic transport version')
        if not self.validator.is_valid(wire):
            raise ValueError('Invalid semantic transport schema')
        result = self._convert(wire[1], self.expanded_schema, decode=True)
        if not self.expanded_validator.is_valid(result):
            raise ValueError('Invalid expanded semantic schema')
        return result

    def decode_plan(self, text: str, registry=None):
        from .semantic_facts import SemanticPlan
        payload = self.decode(text)
        if registry is not None:
            payload = registry.expand_planner_concepts(payload)
        return SemanticPlan.model_validate(payload)

    def input_metrics(self, messages) -> dict[str, Any]:
        return {
            'codec_version': self.version,
            'schema_fingerprint': self.schema_fingerprint,
            'field_dictionary_fingerprint': fingerprint(self.aliases),
            'compact_prompt_bytes': len(canonical_json(messages).encode()),
            'compact_schema_bytes': len(canonical_json(self.schema).encode()),
            'expanded_schema_bytes': len(canonical_json(self.expanded_schema).encode()),
        }

    def instructions(self) -> str:
        return (
            '\nTransport v1: emit [1,plan]. Every semantic object becomes an array: '
            'required fields in alphabetical order, followed optionally by ONE object '
            'of compact-key optional fields. Never remove a required slot. '
            'Arrays that were lists remain lists, including path and filters. '
            'Null and literal values remain unchanged. Semantic instructions use expanded names. '
            'Required slot layouts: ' + ';'.join(','.join(layout) for layout in sorted(self.layouts))
            + '. Optional keys: ' + ','.join(f'{alias}={name}' for name, alias in self.aliases.items())
            + '. Capabilities $table=[columns,rows], each row [map_key,column_values...]; '
            '$literal is an escaped object as key/value pairs.\n'
        )


@lru_cache(maxsize=16)
def _cached_transport(schema_text: str) -> SemanticTransport:
    return SemanticTransport(json.loads(schema_text))


def transport_for(schema: Mapping[str, Any]) -> SemanticTransport:
    return _cached_transport(canonical_json(schema))


def pack_capabilities(value: Any) -> Any:
    """Lossless tables for repeated object shapes, only when bytes decrease."""
    if isinstance(value, (list, tuple)):
        return [pack_capabilities(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return pack_capabilities(json.loads(canonical_json(value)))
    if not isinstance(value, Mapping):
        return value
    packed = {key: pack_capabilities(child) for key, child in sorted(value.items())}
    if '$table' in packed or '$literal' in packed:
        return {'$literal': list(packed.items())}
    children = list(value.values())
    if len(children) > 1 and all(isinstance(child, Mapping) for child in children):
        columns = sorted(children[0])
        if all(set(child) == set(columns) for child in children):
            table = {'$table': [columns, [
                [key, *(pack_capabilities(child[column]) for column in columns)]
                for key, child in sorted(value.items())
            ]]}
            if len(canonical_json(table)) < len(canonical_json(packed)):
                return table
    return packed


def unpack_capabilities(value: Any) -> Any:
    if isinstance(value, list):
        return [unpack_capabilities(item) for item in value]
    if not isinstance(value, dict):
        return value
    if set(value) == {'$literal'}:
        return {key: unpack_capabilities(child) for key, child in value['$literal']}
    if set(value) == {'$table'}:
        columns, rows = value['$table']
        return {row[0]: {key: unpack_capabilities(child) for key, child in zip(columns, row[1:], strict=True)} for row in rows}
    return {key: unpack_capabilities(child) for key, child in value.items()}


def decode_response(codec: SemanticTransport, content: str, runtime: dict[str, Any]) -> dict[str, Any]:
    from time import perf_counter
    started = perf_counter()
    runtime.update(codec_version=codec.version, schema_fingerprint=codec.schema_fingerprint,
                   compact_output_bytes=len(content.encode()), transport_parse_success=False)
    try:
        result = codec.decode(content)
        runtime.update(transport_parse_success=True, expanded_output_bytes=len(canonical_json(result).encode()))
        return result
    finally:
        runtime['transport_parse_ms'] = (perf_counter() - started) * 1000
