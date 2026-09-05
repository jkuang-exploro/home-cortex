from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

from surrealdb import RecordID

from .db import Database
from .edge_schema import (
    EdgeSchema,
    EdgeSchemaRegistry,
    ResolvedEdgeSchema,
    UnknownEdgeSchemaError,
)
from .record_ids import canonical_record_id, split_record_id
from .schema_catalog import (
    matches_scoped_appellation,
    normalize_entity_alias,
    record_aliases,
)

ENTITY_SUMMARY_FIELDS = frozenset(
    {
        "id",
        "name",
        "display_name",
        "address_as",
        "gender",
        "first_name",
        "last_name",
        "address_type",
        "item_type",
        "space_type",
        "type",
    }
)


class RetrievalService:
    """Run safe, read-only graph operations over registered tables."""

    def __init__(
        self,
        database: Database,
        limit: int = 100,
        data_dir: Path | None = None,
        edge_registry: EdgeSchemaRegistry | None = None,
    ) -> None:
        self.database = database
        self.limit = limit
        self.node_tables = self._table_names(data_dir, "nodes") or (
            "address",
            "item",
            "person",
            "space",
        )
        self.edge_registry = edge_registry or EdgeSchemaRegistry.load_default(data_dir)
        self.edge_tables = self.edge_registry.relationship_names

    async def get_entity(self, record_id: str) -> dict[str, Any] | None:
        """Return the record for a canonical ID, or None if it does not exist.

        This is a point-get, not named-entity resolution.
        """
        entity = _parse_record_id(record_id)
        if entity.table_name not in self.node_tables:
            raise ValueError(
                f"Unknown entity type {entity.table_name!r}; expected one of "
                f"{', '.join(self.node_tables)}"
            )
        result = await self.database.query(
            "SELECT * FROM $id;",
            {"id": entity},
        )
        for record in _query_records(result):
            if record.get("id") == record_id:
                return record
        return None

    async def resolve_entity_alias(
        self,
        text: str,
        entity_type: str | None = None,
        limit: int | None = None,
        *,
        speaker_id: str | None = None,
        household_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Resolve stored aliases, then scoped appellations, from runtime data."""
        normalized = normalize_entity_alias(text)
        if not normalized:
            raise ValueError("Entity alias cannot be empty")
        result_limit = self._validated_limit(limit)
        if entity_type is not None:
            if entity_type not in self.node_tables:
                raise ValueError(
                    f"Unknown entity type {entity_type!r}; expected one of "
                    f"{', '.join(self.node_tables)}"
                )
            tables = (entity_type,)
        else:
            tables = self.node_tables

        statement = """
            SELECT * FROM type::table($table)
            ORDER BY id;
        """
        alias_matches: list[dict[str, Any]] = []
        appellation_matches: list[dict[str, Any]] = []
        for table in tables:
            records = _query_records(
                await self.database.query(
                    statement,
                    {"table": table},
                )
            )
            alias_matches.extend(
                _entity_summary(record)
                for record in records
                if any(
                    normalize_entity_alias(alias) == normalized
                    for alias in record_aliases(record)
                )
            )
            appellation_matches.extend(
                _entity_summary(record)
                for record in records
                if matches_scoped_appellation(
                    record,
                    text,
                    speaker_id=speaker_id,
                    household_id=household_id,
                )
            )
        matches = alias_matches or appellation_matches
        matches.sort(key=lambda record: str(record.get("id", "")))
        return matches[:result_limit]

    async def get_relationships(
        self,
        entity_id: str,
        relation: str | None = None,
        direction: Literal["out", "in", "both"] | None = None,
        limit: int | None = None,
        *,
        include_ended: bool = False,
    ) -> list[dict[str, Any]]:
        entity = _parse_record_id(entity_id)
        result_limit = self._validated_limit(limit)

        if relation is not None:
            try:
                resolved_relations = (self.edge_registry.resolve(relation),)
            except UnknownEdgeSchemaError as error:
                raise ValueError(str(error)) from error
        else:
            if direction is not None:
                raise ValueError("Direction requires a specific relation")
            resolved_relations = tuple(
                ResolvedEdgeSchema(self.edge_registry.get(name))
                for name in self.edge_tables
            )

        statement_template = """
            SELECT *,
                in.* AS source_entity,
                out.* AS target_entity
            FROM type::table($relation)
            WHERE {predicate}
            ORDER BY id
            LIMIT $limit;
        """
        relationships: list[dict[str, Any]] = []
        for resolved in resolved_relations:
            relation_name = resolved.schema.id
            stored_direction = _stored_direction(
                resolved,
                entity.table_name,
                direction,
            )
            predicate = {
                "out": "in = $entity",
                "in": "out = $entity",
                "both": "in = $entity OR out = $entity",
            }[stored_direction]
            if not include_ended:
                predicate = f"({predicate}) AND (end = NONE OR end = NULL)"
            result = await self.database.query(
                statement_template.format(predicate=predicate),
                {
                    "relation": relation_name,
                    "entity": entity,
                    "limit": result_limit,
                },
            )
            for edge in _query_records(result):
                if not include_ended and not _is_current_relationship(edge):
                    continue
                edge["relation"] = relation_name
                edge_direction = _relationship_direction(edge, entity_id)
                edge["semantic_relation"] = _semantic_relation(
                    resolved.schema,
                    edge_direction,
                )
                source_entity = edge.pop("source_entity", None)
                target_entity = edge.pop("target_entity", None)
                edge["direction"] = edge_direction
                subject_entity = (
                    source_entity if edge_direction == "outgoing" else target_entity
                )
                if isinstance(subject_entity, dict):
                    edge["entity"] = _entity_summary(subject_entity)
                related_entity = (
                    source_entity if edge_direction == "incoming" else target_entity
                )
                if isinstance(related_entity, dict):
                    edge["related_entity"] = _entity_summary(related_entity)
                relationships.append(edge)

        relationships.sort(
            key=lambda edge: (str(edge.get("relation", "")), str(edge.get("id", "")))
        )
        return relationships[:result_limit]

    def _validated_limit(self, limit: int | None) -> int:
        if limit is None:
            return self.limit
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= self.limit
        ):
            raise ValueError(f"Limit must be an integer between 1 and {self.limit}")
        return limit

    @staticmethod
    def _table_names(data_dir: Path | None, category: str) -> tuple[str, ...]:
        if data_dir is None:
            return ()
        directory = data_dir / category
        if not directory.is_dir():
            return ()
        return tuple(sorted(path.stem for path in directory.glob("*.json")))


def to_json_value(value: Any) -> Any:
    """Convert SurrealDB SDK values into JSON-compatible Python values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_json_value(item) for item in value]

    table = getattr(value, "table_name", None)
    record_id = getattr(value, "id", None)
    if table is not None and record_id is not None:
        return canonical_record_id(value)
    return str(value)


def _query_records(value: Any) -> list[dict[str, Any]]:
    normalized = to_json_value(value)
    if normalized is None:
        return []
    if isinstance(normalized, dict):
        return [normalized]
    if isinstance(normalized, list) and all(
        isinstance(record, dict) for record in normalized
    ):
        return normalized
    raise RuntimeError("SurrealDB returned an unexpected query result")


def _parse_record_id(value: str) -> RecordID:
    if not isinstance(value, str):
        raise ValueError("Entity ID must use the table:record_id format")
    table, record_id = split_record_id(value)
    return RecordID(table, record_id)


def _relationship_direction(edge: dict[str, Any], entity_id: str) -> str:
    is_source = edge.get("in") == entity_id
    is_target = edge.get("out") == entity_id
    if is_source and is_target:
        return "both"
    if is_source:
        return "outgoing"
    return "incoming"


def _stored_direction(
    resolved: ResolvedEdgeSchema,
    entity_type: str,
    requested: Literal["out", "in", "both"] | None,
) -> Literal["out", "in", "both"]:
    schema = resolved.schema
    if schema.symmetric:
        return "both"
    source_only = (
        entity_type in schema.from_types and entity_type not in schema.to_types
    )
    target_only = (
        entity_type in schema.to_types and entity_type not in schema.from_types
    )
    # For differently typed endpoints, the entity's type determines the only
    # structurally valid stored direction. Do not let a model-supplied direction
    # turn a valid query into an impossible traversal.
    if source_only:
        return "out"
    if target_only:
        return "in"
    if requested is not None:
        if not resolved.inverse or requested == "both":
            return requested
        return "in" if requested == "out" else "out"
    if resolved.inverse:
        return "in"
    return "both"


def _semantic_relation(schema: EdgeSchema, direction: str) -> str:
    if schema.symmetric or direction in {"outgoing", "both"}:
        return schema.id
    return schema.inverse_name or schema.id


def _is_current_relationship(edge: dict[str, Any]) -> bool:
    return edge.get("end") is None


def _entity_summary(record: dict[str, Any]) -> dict[str, Any]:
    """Expose identity metadata without private profile fields."""
    return {
        field: value
        for field, value in record.items()
        if field in ENTITY_SUMMARY_FIELDS
    }
