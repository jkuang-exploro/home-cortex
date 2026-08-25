import base64
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from surrealdb import RecordID

from .db import Database
from .edge_schema import EdgeSchema, EdgeSchemaRegistry, UnknownEdgeSchemaError
from .record_ids import (
    RECORD_ID_RE,
    TABLE_NAME_RE,
    canonical_record_id,
    split_record_id,
)

TABLE_PATTERN = TABLE_NAME_RE
RECORD_PATTERN = RECORD_ID_RE


@dataclass(frozen=True)
class IngestionResult:
    node_files: int = 0
    edge_files: int = 0
    nodes_upserted: int = 0
    edges_upserted: int = 0


@dataclass(frozen=True)
class _PreparedNode:
    record_id: RecordID
    content: dict[str, Any]


@dataclass(frozen=True)
class _PreparedEdge:
    relation: str
    record_id: RecordID
    source: RecordID
    target: RecordID
    content: dict[str, Any]


def _records_from_file(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)

    records = value if isinstance(value, list) else [value]
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"{path} must contain a JSON object or array of objects")
    return records


def _validate_node_name(record: dict[str, Any], path: Path) -> None:
    """Accept legacy alias lists and explicit localized-name objects."""
    if "name" not in record:
        return
    names = record["name"]
    if isinstance(names, list):
        if not names or any(
            not isinstance(name, str) or not name.strip() for name in names
        ):
            raise ValueError(
                f"Node in {path} must use 'name' as a non-empty list of strings "
                "or localized object"
            )
        if len(names) != len(set(names)):
            raise ValueError(f"Node in {path} contains duplicate values in 'name'")
        return
    if not _is_localized_text(names):
        raise ValueError(
            f"Node in {path} must use 'name' as a non-empty list of strings "
            "or localized object"
        )


def _validate_address_as(
    record: dict[str, Any],
    path: Path,
    table: str,
) -> None:
    if "address_as" not in record:
        return
    if table != "person":
        raise ValueError(f"Only Person nodes may define 'address_as' in {path}")
    if not _is_localized_text(record["address_as"]):
        raise ValueError(
            f"Person in {path} must use 'address_as' as a non-empty localized "
            "object of strings"
        )


def _validate_person_relationship_status(
    record: dict[str, Any],
    path: Path,
    table: str,
) -> None:
    if table != "person":
        return
    if "is_guest" in record or record.get("person_type") == "guest":
        raise ValueError(
            f"Guest status in {path} must be stored on a household relationship, "
            "not on a Person node"
        )


def _is_localized_text(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and bool(value)
        and all(
            isinstance(language, str)
            and bool(language.strip())
            and isinstance(text, str)
            and bool(text.strip())
            for language, text in value.items()
        )
    )


def parse_record_id(value: str, *, source: Path) -> RecordID:
    try:
        table, record_id = split_record_id(value)
    except ValueError as error:
        raise ValueError(
            f"Invalid record ID {value!r} in {source}; expected table:record_id"
        ) from error
    return RecordID(table, record_id)


def _implicit_edge_component(record_id: RecordID) -> str:
    canonical = canonical_record_id(record_id)
    if ":" not in str(record_id.id):
        return canonical.replace(":", "_", 1)
    encoded = base64.urlsafe_b64encode(canonical.encode()).decode().rstrip("=")
    return f"b64_{encoded}"


def _edge_record_id(
    relation: str,
    record: dict[str, Any],
    source: RecordID,
    target: RecordID,
    path: Path,
) -> RecordID:
    raw_id = record.get("id")
    if raw_id is not None:
        if not isinstance(raw_id, str):
            raise ValueError(f"Edge in {path} has a non-string 'id'")
        edge_id = parse_record_id(raw_id, source=path)
        if edge_id.table_name != relation:
            raise ValueError(
                f"Edge ID {raw_id!r} in {path} must use the {relation!r} table"
            )
        return edge_id

    identifier = (
        f"{_implicit_edge_component(source)}__{_implicit_edge_component(target)}"
    )
    return RecordID(relation, identifier)


async def ingest_directory(
    database: Database,
    data_dir: Path,
    edge_registry: EdgeSchemaRegistry | None = None,
) -> IngestionResult:
    nodes_dir = data_dir / "nodes"
    edges_dir = data_dir / "edges"
    if not nodes_dir.is_dir() or not edges_dir.is_dir():
        raise FileNotFoundError(
            f"Expected {nodes_dir} and {edges_dir} to both be directories"
        )

    registry = edge_registry or _default_edge_registry(data_dir)
    node_files = sorted(nodes_dir.glob("*.json"))
    edge_files = sorted(edges_dir.glob("*.json"))
    source_relationships = {path.stem for path in edge_files}
    missing_relationships = sorted(
        set(registry.relationship_names) - source_relationships
    )
    if missing_relationships:
        raise ValueError(
            "Missing relationship data files for registered schemas: "
            f"{', '.join(f'{name}.json' for name in missing_relationships)}; "
            "use an empty JSON array when a relationship has no facts"
        )
    prepared_nodes: dict[str, list[_PreparedNode]] = {}
    prepared_edges: dict[str, list[_PreparedEdge]] = {}

    # Validate the entire source before mutating the database.
    for path in node_files:
        table_nodes: list[_PreparedNode] = []
        seen_node_ids: set[str] = set()
        for record in _records_from_file(path):
            _validate_node_name(record, path)
            raw_id = record.get("id")
            if not isinstance(raw_id, str):
                raise ValueError(f"Node in {path} is missing a string 'id'")
            record_id = parse_record_id(raw_id, source=path)
            if record_id.table_name != path.stem:
                raise ValueError(
                    f"Node ID {raw_id!r} in {path} must use the {path.stem!r} table"
                )
            canonical_id = canonical_record_id(record_id)
            if canonical_id in seen_node_ids:
                raise ValueError(f"Duplicate node ID {record_id} in {path}")
            seen_node_ids.add(canonical_id)
            _validate_address_as(record, path, record_id.table_name)
            _validate_person_relationship_status(
                record,
                path,
                record_id.table_name,
            )
            content = {key: value for key, value in record.items() if key != "id"}
            table_nodes.append(_PreparedNode(record_id, content))
        prepared_nodes[path.stem] = table_nodes

    known_node_ids = {
        canonical_record_id(node.record_id)
        for nodes in prepared_nodes.values()
        for node in nodes
    }

    for path in edge_files:
        relation = path.stem
        if TABLE_PATTERN.fullmatch(relation) is None:
            raise ValueError(f"Invalid relation table name derived from {path.name}")

        try:
            schema = registry.get(relation)
        except UnknownEdgeSchemaError as error:
            raise ValueError(str(error)) from error

        seen_pairs: set[tuple[str, str]] = set()
        seen_sources: set[str] = set()
        seen_ids: set[str] = set()
        table_edges: list[_PreparedEdge] = []
        for record in _records_from_file(path):
            raw_from = record.get("from")
            raw_to = record.get("to")
            if not isinstance(raw_from, str) or not isinstance(raw_to, str):
                raise ValueError(f"Edge in {path} requires string 'from' and 'to'")
            household_role = record.get("household_role")
            if household_role is not None and (
                not isinstance(household_role, str)
                or not household_role.strip()
            ):
                raise ValueError(
                    f"Edge in {path} has an invalid 'household_role'"
                )

            source = parse_record_id(raw_from, source=path)
            target = parse_record_id(raw_to, source=path)
            registry.validate_endpoints(
                relation,
                source.table_name,
                target.table_name,
            )
            for endpoint, role in ((source, "from"), (target, "to")):
                canonical_endpoint = canonical_record_id(endpoint)
                if canonical_endpoint not in known_node_ids:
                    raise ValueError(
                        f"Edge in {path} references unknown {role} node "
                        f"{canonical_endpoint!r}"
                    )
            _validate_temporal_fields(record, path, schema)

            pair = _edge_pair(schema, source, target)
            if pair in seen_pairs:
                qualifier = "symmetric " if schema.symmetric else ""
                raise ValueError(
                    f"Duplicate {qualifier}{relation} relationship in {path}: "
                    f"{source} -> {target}"
                )
            canonical_source = canonical_record_id(source)
            if schema.unique_from and canonical_source in seen_sources:
                raise ValueError(
                    f"Relationship {relation!r} in {path} allows only one "
                    f"target for source {canonical_source!r}"
                )
            seen_pairs.add(pair)
            seen_sources.add(canonical_source)

            edge = _edge_record_id(relation, record, source, target, path)
            canonical_edge_id = canonical_record_id(edge)
            if canonical_edge_id in seen_ids:
                raise ValueError(f"Duplicate edge ID {edge} in {path}")
            seen_ids.add(canonical_edge_id)
            content = {
                key: value
                for key, value in record.items()
                if key not in {"id", "from", "to", "in", "out"}
            }

            table_edges.append(
                _PreparedEdge(relation, edge, source, target, content)
            )
        prepared_edges[relation] = table_edges

    nodes_upserted = 0
    edges_upserted = 0
    # Nodes must exist before relationship records are related.
    for table, nodes in prepared_nodes.items():
        for node in nodes:
            await database.upsert(node.record_id, node.content)
            nodes_upserted += 1
        await _prune_table(database, table, [node.record_id for node in nodes])

    for relation, edges in prepared_edges.items():
        for edge in edges:
            await database.query(
                "RELATE $source->$edge->$target CONTENT $content;",
                {
                    "source": edge.source,
                    "edge": edge.record_id,
                    "target": edge.target,
                    "content": edge.content,
                },
            )
            edges_upserted += 1
        await _prune_table(
            database,
            relation,
            [edge.record_id for edge in edges],
        )

    return IngestionResult(
        node_files=len(node_files),
        edge_files=len(edge_files),
        nodes_upserted=nodes_upserted,
        edges_upserted=edges_upserted,
    )


def _default_edge_registry(data_dir: Path) -> EdgeSchemaRegistry:
    candidates = (
        data_dir.parent / "schemas" / "edge",
        Path(__file__).resolve().parents[2] / "schemas" / "edge",
        Path("/app/schemas/edge"),
    )
    for candidate in candidates:
        if candidate.is_dir():
            return EdgeSchemaRegistry.from_directory(candidate)
    raise FileNotFoundError("Could not locate schemas/edge")


def _validate_temporal_fields(
    record: dict[str, Any],
    path: Path,
    schema: EdgeSchema,
) -> None:
    present = {field for field in ("start", "end") if field in record}
    if present and not schema.temporal:
        raise ValueError(
            f"Non-temporal relationship {schema.id!r} in {path} cannot define "
            f"{', '.join(sorted(present))}"
        )
    for field in present:
        value = record[field]
        if field == "end" and value is None:
            continue
        if not isinstance(value, str):
            raise ValueError(f"{field!r} in {path} must be an ISO date string or null")
        try:
            date.fromisoformat(value)
        except ValueError as error:
            raise ValueError(f"{field!r} in {path} must be an ISO date") from error


def _edge_pair(
    schema: EdgeSchema,
    source: RecordID,
    target: RecordID,
) -> tuple[str, str]:
    pair = (canonical_record_id(source), canonical_record_id(target))
    return tuple(sorted(pair)) if schema.symmetric else pair


async def _prune_table(
    database: Database,
    table: str,
    record_ids: list[RecordID],
) -> None:
    """Make each JSON file the source of truth for its corresponding table."""
    await database.query(
        "DELETE FROM type::table($table) WHERE id NOT IN $record_ids;",
        {"table": table, "record_ids": record_ids},
    )
