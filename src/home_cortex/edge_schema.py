import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml

NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class UnknownEdgeSchemaError(LookupError):
    """Raised when a relationship name is not registered."""


@dataclass(frozen=True)
class EdgeSchema:
    id: str
    symmetric: bool
    from_types: tuple[str, ...]
    to_types: tuple[str, ...]
    temporal: bool
    inverse_name: str | None = None
    unique_from: bool = False


@dataclass(frozen=True)
class ResolvedEdgeSchema:
    schema: EdgeSchema
    inverse: bool = False


class EdgeSchemaRegistry:
    """Authoritative relationship semantics loaded from YAML schemas."""

    def __init__(self, schemas: Mapping[str, EdgeSchema]) -> None:
        if not schemas:
            raise ValueError("At least one edge schema is required")
        self._schemas = MappingProxyType(dict(schemas))
        inverse_names: dict[str, str] = {}
        for schema in self._schemas.values():
            if schema.inverse_name is None:
                continue
            if schema.inverse_name in self._schemas:
                raise ValueError(
                    f"Inverse name {schema.inverse_name!r} conflicts with a schema ID"
                )
            previous = inverse_names.setdefault(schema.inverse_name, schema.id)
            if previous != schema.id:
                raise ValueError(
                    f"Inverse name {schema.inverse_name!r} is defined more than once"
                )
        self._inverse_names = MappingProxyType(inverse_names)

    @classmethod
    def from_directory(cls, directory: Path) -> "EdgeSchemaRegistry":
        if not directory.is_dir():
            raise FileNotFoundError(f"Edge schema directory does not exist: {directory}")
        schemas: dict[str, EdgeSchema] = {}
        for path in sorted(directory.glob("*.yaml")):
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            schema = _parse_schema(raw, path)
            if schema.id != path.stem:
                raise ValueError(
                    f"Edge schema ID {schema.id!r} must match filename {path.name!r}"
                )
            if schema.id in schemas:
                raise ValueError(f"Duplicate edge schema {schema.id!r}")
            schemas[schema.id] = schema
        return cls(schemas)

    @property
    def relationship_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._schemas))

    @property
    def public_names(self) -> tuple[str, ...]:
        return tuple(sorted((*self._schemas, *self._inverse_names)))

    def get(self, relationship: str) -> EdgeSchema:
        try:
            return self._schemas[relationship]
        except KeyError as error:
            raise UnknownEdgeSchemaError(
                f"Unknown relationship {relationship!r}; expected one of "
                f"{', '.join(self.public_names)}"
            ) from error

    def resolve(self, relationship: str) -> ResolvedEdgeSchema:
        if relationship in self._schemas:
            return ResolvedEdgeSchema(self._schemas[relationship])
        canonical = self._inverse_names.get(relationship)
        if canonical is not None:
            return ResolvedEdgeSchema(self._schemas[canonical], inverse=True)
        self.get(relationship)
        raise AssertionError("unreachable")

    def validate_endpoints(self, relationship: str, source_type: str, target_type: str) -> None:
        schema = self.get(relationship)
        forward = source_type in schema.from_types and target_type in schema.to_types
        reverse = (
            schema.symmetric
            and source_type in schema.to_types
            and target_type in schema.from_types
        )
        if not (forward or reverse):
            raise ValueError(
                f"Invalid {relationship} endpoints: {source_type} -> {target_type}; "
                f"expected {schema.from_types} -> {schema.to_types}"
            )


def _parse_schema(raw: Any, path: Path) -> EdgeSchema:
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML object")
    allowed = {
        "id",
        "symmetric",
        "from_types",
        "to_types",
        "temporal",
        "inverse_name",
        "unique_from",
    }
    extra = sorted(set(raw) - allowed)
    if extra:
        raise ValueError(f"Unknown fields in {path}: {', '.join(extra)}")
    schema_id = _name(raw.get("id"), "id", path)
    symmetric = _boolean(raw.get("symmetric"), "symmetric", path)
    from_types = _names(raw.get("from_types"), "from_types", path)
    to_types = _names(raw.get("to_types"), "to_types", path)
    temporal = _boolean(raw.get("temporal"), "temporal", path)
    unique_from = _boolean(raw.get("unique_from", False), "unique_from", path)
    inverse = raw.get("inverse_name")
    if inverse is not None:
        inverse = _name(inverse, "inverse_name", path)
        if symmetric:
            raise ValueError(f"Symmetric edge schema {schema_id!r} cannot define an inverse")
        if inverse == schema_id:
            raise ValueError(f"Edge schema {schema_id!r} cannot be its own inverse")
    if symmetric and unique_from:
        raise ValueError(
            f"Symmetric edge schema {schema_id!r} cannot define unique_from"
        )
    return EdgeSchema(
        schema_id,
        symmetric,
        from_types,
        to_types,
        temporal,
        inverse,
        unique_from,
    )


def _name(value: Any, field: str, path: Path) -> str:
    if not isinstance(value, str) or NAME_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} in {path} must be a valid relationship/type name")
    return value


def _names(value: Any, field: str, path: Path) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} in {path} must be a non-empty list")
    names = tuple(_name(item, field, path) for item in value)
    if len(names) != len(set(names)):
        raise ValueError(f"{field} in {path} contains duplicate names")
    return names


def _boolean(value: Any, field: str, path: Path) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} in {path} must be true or false")
    return value
