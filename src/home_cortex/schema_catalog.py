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
class ScopedAppellation:
    """A household name whose meaning depends on trusted request scope."""

    value: str
    household_id: str | None = None
    speaker_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuntimeSchemaCatalog:
    """Queryable fields and relations discovered from deployment data/schema."""

    entities: Mapping[str, EntityTypeSchema]
    relations: Mapping[str, RelationTypeSchema]
    edge_registry: EdgeSchemaRegistry | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "entities", MappingProxyType(dict(self.entities)))
        object.__setattr__(self, "relations", MappingProxyType(dict(self.relations)))

    @classmethod
    def from_data_dir(
        cls,
        data_dir: Path,
        edge_registry: EdgeSchemaRegistry,
    ) -> "RuntimeSchemaCatalog":
        entities: dict[str, EntityTypeSchema] = {}
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
        return cls(entities, relations, edge_registry)

    def has_entity_type(self, entity_type: str) -> bool:
        return entity_type in self.entities

    def entity_field_type(self, entity_type: str, field: str) -> ValueKind:
        schema = self.entities.get(entity_type)
        return schema.property_types.get(field, "unknown") if schema else "unknown"

    def relation_field_type(self, relation: str, field: str) -> ValueKind:
        schema = self.relations.get(relation)
        if schema is None and self.edge_registry is not None:
            try:
                schema = self.relations.get(self.edge_registry.resolve(relation).schema.id)
            except LookupError:
                pass
        return schema.property_types.get(field, "unknown") if schema else "unknown"

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


def record_appellations(record: Mapping[str, Any]) -> tuple[ScopedAppellation, ...]:
    """Return valid, explicitly scoped appellations stored on an entity."""
    raw = record.get("appellations")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return ()
    result: list[ScopedAppellation] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        value = item.get("value")
        household_id = item.get("household_id")
        raw_speakers = item.get("speaker_ids", ())
        if not isinstance(value, str) or not value.strip():
            continue
        if household_id is not None and not isinstance(household_id, str):
            continue
        if not isinstance(raw_speakers, Sequence) or isinstance(
            raw_speakers,
            (str, bytes, bytearray),
        ):
            continue
        speaker_ids = tuple(
            speaker
            for speaker in raw_speakers
            if isinstance(speaker, str) and speaker
        )
        if household_id is None and not speaker_ids:
            continue
        result.append(
            ScopedAppellation(value.strip(), household_id, speaker_ids)
        )
    return tuple(result)


def matches_scoped_appellation(
    record: Mapping[str, Any],
    value: str,
    *,
    speaker_id: str | None,
    household_id: str | None,
) -> bool:
    """Match an appellation only when every stored scope constraint holds."""
    normalized = normalize_entity_alias(value)
    return any(
        normalize_entity_alias(item.value) == normalized
        and (
            item.household_id is None
            or (
                household_id is not None
                and item.household_id == household_id
            )
        )
        and (
            not item.speaker_ids
            or (speaker_id is not None and speaker_id in item.speaker_ids)
        )
        for item in record_appellations(record)
    )
