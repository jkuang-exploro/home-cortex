"""Schema-driven recurring household dates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping

import yaml

NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class MemorableDateSchema:
    id: str
    aliases: tuple[str, ...]
    label: Mapping[str, str]
    recurrence: Literal["annual"]
    source_kind: Literal["node", "edge"]
    source_type: str
    source_field: str


@dataclass(frozen=True)
class MemorableDateOccurrence:
    schema: MemorableDateSchema
    stored_date: date
    next_occurrence: date
    days_until: int


class MemorableDateRegistry:
    """Load date semantics and resolve recurring occurrences generically."""

    def __init__(self, schemas: Mapping[str, MemorableDateSchema]) -> None:
        if not schemas:
            raise ValueError("At least one memorable-date schema is required")
        self._schemas = MappingProxyType(dict(schemas))
        sources: dict[tuple[str, str, str], str] = {}
        aliases: dict[str, str] = {}
        for schema in self._schemas.values():
            key = (schema.source_kind, schema.source_type, schema.source_field)
            previous = sources.setdefault(key, schema.id)
            if previous != schema.id:
                raise ValueError(f"Memorable-date source {key!r} is defined twice")
            for alias in schema.aliases:
                normalized = alias.casefold()
                previous = aliases.setdefault(normalized, schema.id)
                if previous != schema.id:
                    raise ValueError(
                        f"Memorable-date alias {alias!r} is defined twice"
                    )
        self._sources = MappingProxyType(sources)

    @classmethod
    def from_file(cls, path: Path) -> "MemorableDateRegistry":
        if not path.is_file():
            raise FileNotFoundError(f"Memorable-date schema does not exist: {path}")
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or set(raw) != {"dates"}:
            raise ValueError(f"{path} must contain only a dates list")
        values = raw["dates"]
        if not isinstance(values, list) or not values:
            raise ValueError(f"dates in {path} must be a non-empty list")
        schemas: dict[str, MemorableDateSchema] = {}
        for value in values:
            schema = _parse_schema(value, path)
            if schema.id in schemas:
                raise ValueError(f"Duplicate memorable-date schema {schema.id!r}")
            schemas[schema.id] = schema
        return cls(schemas)

    @property
    def aliases(self) -> tuple[str, ...]:
        return tuple(
            alias
            for schema in self._schemas.values()
            for alias in schema.aliases
        )

    def match(self, text: str) -> MemorableDateSchema | None:
        normalized = text.casefold()
        matches = [
            (len(alias), schema)
            for schema in self._schemas.values()
            for alias in schema.aliases
            if alias.casefold() in normalized
        ]
        return max(matches, default=(0, None), key=lambda item: item[0])[1]

    def get(self, schema_id: str) -> MemorableDateSchema:
        try:
            return self._schemas[schema_id]
        except KeyError as error:
            raise LookupError(f"Unknown memorable date {schema_id!r}") from error

    def for_source(
        self,
        source_kind: Literal["node", "edge"],
        source_type: str,
        source_field: str,
    ) -> MemorableDateSchema:
        key = (source_kind, source_type, source_field)
        try:
            return self._schemas[self._sources[key]]
        except KeyError as error:
            raise LookupError(f"Unknown memorable-date source {key!r}") from error

    def occurrence(
        self,
        schema: MemorableDateSchema,
        value: Any,
        *,
        as_of: date,
    ) -> MemorableDateOccurrence | None:
        if not isinstance(value, str):
            return None
        try:
            stored = date.fromisoformat(value)
        except ValueError:
            return None
        if schema.recurrence != "annual":
            raise ValueError(f"Unsupported recurrence {schema.recurrence!r}")
        next_occurrence = _next_annual_occurrence(stored, as_of)
        return MemorableDateOccurrence(
            schema=schema,
            stored_date=stored,
            next_occurrence=next_occurrence,
            days_until=(next_occurrence - as_of).days,
        )


@lru_cache(maxsize=1)
def default_memorable_date_registry() -> MemorableDateRegistry:
    candidates = (
        Path(__file__).resolve().parents[2] / "schemas" / "memorable_dates.yaml",
        Path("/app/schemas/memorable_dates.yaml"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return MemorableDateRegistry.from_file(candidate)
    raise FileNotFoundError("Could not locate schemas/memorable_dates.yaml")


def _next_annual_occurrence(stored: date, as_of: date) -> date:
    first_year = max(stored.year, as_of.year)
    for year in range(first_year, first_year + 9):
        try:
            candidate = date(year, stored.month, stored.day)
        except ValueError:
            continue
        if candidate >= stored and candidate >= as_of:
            return candidate
    raise ValueError("Could not resolve the next annual occurrence")


def _parse_schema(raw: Any, path: Path) -> MemorableDateSchema:
    if not isinstance(raw, dict):
        raise ValueError(f"Each date in {path} must be an object")
    allowed = {"id", "aliases", "label", "recurrence", "source"}
    extra = sorted(set(raw) - allowed)
    if extra:
        raise ValueError(f"Unknown fields in {path}: {', '.join(extra)}")
    schema_id = _name(raw.get("id"), "id", path)
    aliases = _strings(raw.get("aliases"), "aliases", path)
    label = raw.get("label")
    if not isinstance(label, dict) or not label or not all(
        isinstance(language, str)
        and language.strip()
        and isinstance(text, str)
        and text.strip()
        for language, text in label.items()
    ):
        raise ValueError(f"label in {path} must be a localized text object")
    recurrence = raw.get("recurrence")
    if recurrence != "annual":
        raise ValueError(f"recurrence in {path} must be 'annual'")
    source = raw.get("source")
    if not isinstance(source, dict) or set(source) != {"kind", "type", "field"}:
        raise ValueError(f"source in {path} requires kind, type, and field")
    source_kind = source["kind"]
    if source_kind not in {"node", "edge"}:
        raise ValueError(f"source kind in {path} must be node or edge")
    return MemorableDateSchema(
        id=schema_id,
        aliases=aliases,
        label=MappingProxyType(dict(label)),
        recurrence=recurrence,
        source_kind=source_kind,
        source_type=_name(source["type"], "source.type", path),
        source_field=_name(source["field"], "source.field", path),
    )


def _name(value: Any, field: str, path: Path) -> str:
    if not isinstance(value, str) or NAME_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} in {path} must be a valid name")
    return value


def _strings(value: Any, field: str, path: Path) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"{field} in {path} must be a non-empty string list")
    values = tuple(item.strip() for item in value)
    if len({item.casefold() for item in values}) != len(values):
        raise ValueError(f"{field} in {path} contains duplicates")
    return values
