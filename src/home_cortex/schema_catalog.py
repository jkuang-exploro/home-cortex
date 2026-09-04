"""Compact runtime schema discovery for open-world household grounding."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from collections.abc import Sequence
from typing import Any, Mapping, get_args

from .edge_schema import EdgeSchemaRegistry
from .operator_registry import ValueKind, infer_field_kind


@dataclass(frozen=True)
class EntityTypeSchema:
    name: str
    properties: tuple[str, ...]
    property_types: Mapping[str, ValueKind] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "property_types",
            _validated_property_types(self.properties, self.property_types),
        )


@dataclass(frozen=True)
class RelationTypeSchema:
    name: str
    from_types: tuple[str, ...]
    to_types: tuple[str, ...]
    properties: tuple[str, ...]
    symmetric: bool
    temporal: bool
    inverse_name: str | None
    property_types: Mapping[str, ValueKind] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "property_types",
            _validated_property_types(self.properties, self.property_types),
        )


@dataclass(frozen=True)
class RuntimeSchemaCatalog:
    """Queryable fields and relations discovered from deployment data/schema."""

    entities: Mapping[str, EntityTypeSchema]
    relations: Mapping[str, RelationTypeSchema]
    entity_aliases: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "entities", MappingProxyType(dict(self.entities)))
        object.__setattr__(self, "relations", MappingProxyType(dict(self.relations)))
        object.__setattr__(self, "entity_aliases", frozenset(self.entity_aliases))

    @classmethod
    def from_data_dir(
        cls,
        data_dir: Path,
        edge_registry: EdgeSchemaRegistry,
    ) -> "RuntimeSchemaCatalog":
        entities: dict[str, EntityTypeSchema] = {}
        aliases: set[str] = set()
        node_dir = data_dir / "nodes"
        if node_dir.is_dir():
            for path in sorted(node_dir.glob("*.json")):
                records = _json_records(path)
                fields = sorted(
                    {
                        str(key)
                        for record in records
                        for key in record
                    }
                )
                entities[path.stem] = EntityTypeSchema(
                    path.stem,
                    tuple(fields),
                    _infer_property_types(records, fields),
                )
                aliases.update(
                    alias
                    for record in records
                    for alias in record_aliases(record)
                )

        relations: dict[str, RelationTypeSchema] = {}
        edge_dir = data_dir / "edges"
        for name in edge_registry.relationship_names:
            schema = edge_registry.get(name)
            path = edge_dir / f"{name}.json"
            records = _json_records(path) if path.is_file() else []
            properties = {
                str(key)
                for record in records
                for key in record
                if str(key) not in {"from", "to", "id"}
            }
            if schema.temporal:
                properties.update({"start", "end"})
            relations[name] = RelationTypeSchema(
                name=name,
                from_types=schema.from_types,
                to_types=schema.to_types,
                properties=tuple(sorted(properties)),
                symmetric=schema.symmetric,
                temporal=schema.temporal,
                inverse_name=schema.inverse_name,
                property_types=_infer_property_types(
                    records,
                    tuple(sorted(properties)),
                ),
            )
        return cls(entities, relations, frozenset(aliases))

    def has_entity_type(self, entity_type: str) -> bool:
        return entity_type in self.entities

    def has_entity_field(self, entity_type: str, field: str) -> bool:
        schema = self.entities.get(entity_type)
        return schema is not None and field in schema.properties

    def has_relation(self, relation: str) -> bool:
        return relation in self.relations or any(
            schema.inverse_name == relation for schema in self.relations.values()
        )

    def entity_field_type(self, entity_type: str, field: str) -> ValueKind:
        schema = self.entities.get(entity_type)
        return schema.property_types.get(field, "unknown") if schema else "unknown"

    def relation_field_type(self, relation: str, field: str) -> ValueKind:
        schema = self.relations.get(relation)
        if schema is None:
            schema = next(
                (
                    candidate
                    for candidate in self.relations.values()
                    if candidate.inverse_name == relation
                ),
                None,
            )
        return schema.property_types.get(field, "unknown") if schema else "unknown"

    def prompt_payload(self) -> dict[str, Any]:
        return {
            "entities": {
                name: {
                    "properties": list(schema.properties),
                    "property_types": dict(schema.property_types),
                }
                for name, schema in self.entities.items()
            },
            "relations": {
                name: {
                    "from": list(schema.from_types),
                    "to": list(schema.to_types),
                    "properties": list(schema.properties),
                    "property_types": dict(schema.property_types),
                    "symmetric": schema.symmetric,
                    "temporal": schema.temporal,
                    **(
                        {"inverse_name": schema.inverse_name}
                        if schema.inverse_name is not None
                        else {}
                    ),
                }
                for name, schema in self.relations.items()
            },
        }


def _json_records(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a JSON list")
    return [record for record in raw if isinstance(record, dict)]


def _infer_property_types(
    records: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> Mapping[str, ValueKind]:
    return {
        field: kind
        for field in fields
        if (
            kind := infer_field_kind(
                [record.get(field) for record in records if field in record]
            )
        )
        != "unknown"
    }


def _validated_property_types(
    properties: Sequence[str],
    property_types: Mapping[str, ValueKind],
) -> Mapping[str, ValueKind]:
    unknown_fields = set(property_types) - set(properties)
    if unknown_fields:
        raise ValueError(
            "property types reference undeclared fields: "
            + ", ".join(sorted(unknown_fields))
        )
    valid_kinds = set(get_args(ValueKind))
    invalid_kinds = set(property_types.values()) - valid_kinds
    if invalid_kinds:
        raise ValueError(
            "invalid property types: " + ", ".join(sorted(invalid_kinds))
        )
    return MappingProxyType(dict(property_types))


def record_aliases(record: Mapping[str, Any]) -> tuple[str, ...]:
    """Return stored display aliases used for exact entity resolution."""
    names = record.get("name")
    aliases: list[str]
    if isinstance(names, str):
        aliases = [names]
    elif isinstance(names, Mapping):
        aliases = [
            value.strip()
            for value in names.values()
            if isinstance(value, str) and value.strip()
        ]
    elif isinstance(names, Sequence) and not isinstance(
        names, (str, bytes, bytearray)
    ):
        aliases = [
            value.strip()
            for value in names
            if isinstance(value, str) and value.strip()
        ]
    else:
        aliases = []
    stored_aliases = record.get("aliases")
    if isinstance(stored_aliases, Sequence) and not isinstance(
        stored_aliases, (str, bytes, bytearray)
    ):
        aliases.extend(
            item
            for item in stored_aliases
            if isinstance(item, str) and item.strip()
        )
        aliases.extend(
            value
            for item in stored_aliases
            if isinstance(item, Mapping)
            and isinstance((value := item.get("value")), str)
            and value.strip()
        )
    for field in (
        "display_name",
        "preferred_name",
        "nickname",
        "english_name",
        "chinese_name",
    ):
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            aliases.append(value)
    first_name = record.get("first_name")
    last_name = record.get("last_name")
    if isinstance(first_name, str):
        aliases.append(first_name)
    if isinstance(first_name, str) and isinstance(last_name, str):
        aliases.append(f"{first_name} {last_name}")
    return tuple(dict.fromkeys(alias.strip() for alias in aliases if alias.strip()))


def normalize_entity_alias(value: str) -> str:
    """Normalize lookup syntax without erasing meaningful name characters."""
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    normalized = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in normalized
    )
    return re.sub(r"\s+", " ", normalized).strip()
