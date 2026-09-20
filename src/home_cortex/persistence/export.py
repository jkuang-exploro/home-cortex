"""Deterministic SurrealDB → canonical JSON export.

This is the inverse of ``ingest_directory``. The LLM, planner, and conversational
context are not involved. Records are converted structurally:

- Node table ``person`` → ``nodes/person.json`` with ``id: "person:<key>"``.
- Registered relationship ``lives_in`` → ``edges/lives_in.json`` with ``from`` /
  ``to`` in place of SurrealDB ``in`` / ``out``.
- Implicit ingest-generated edge identities are omitted; explicit source IDs
  are preserved.
- Sharded ingest directories such as ``nodes/item/*.json`` cannot be recovered
  from the database. Export writes one ``nodes/<table>.json`` per table.

Empty node tables are omitted. Every registered relationship is written, using
``[]`` when the table has no facts, so the result is a valid ingest source.
Known retired tables leftover from schema renames (``resides_in``,
``contained_in``, ``location``) are omitted and reported; they are not part of
the canonical JSON schema and ingest would prune them. Unknown relationship
tables still fail the export.

SurrealDB drops JSON null properties at ingest time. Temporal ``end`` is restored
as ``null`` when the schema marks the relationship temporal and the stored row
has no end, matching the canonical "current relationship" convention. Other
nulls cannot be reconstructed and are documented as an ingest-time asymmetry.
"""

from __future__ import annotations

import errno
import json
import os
import shutil
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .db import Database
from .edge_schema import EdgeSchema, EdgeSchemaRegistry
from .ingestion import (
    _RETIRED_EDGE_TABLES,
    _RETIRED_NODE_TABLES,
    implicit_edge_record_id,
)
from .record_ids import TABLE_NAME_RE, as_record_id, canonical_record_id, split_record_id

_SURREAL_EDGE_IDENTITY_FIELDS = frozenset({"id", "in", "out"})
_NODE_LEADING_FIELDS = ("id",)
_EDGE_LEADING_FIELDS = ("id", "from", "to")


@dataclass(frozen=True)
class ExportResult:
    node_files: int = 0
    edge_files: int = 0
    nodes_exported: int = 0
    edges_exported: int = 0
    omitted_retired_tables: tuple[str, ...] = ()
    omitted_retired_records: int = 0
    target_dir: str = ""


def canonical_json_value(value: Any) -> Any:
    """Convert a SurrealDB SDK value into canonical JSON-compatible data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): canonical_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical_json_value(item) for item in value]
    table = getattr(value, "table_name", None)
    record_id = getattr(value, "id", None)
    if table is not None and record_id is not None:
        return canonical_record_id(value)
    raise ValueError(
        "Cannot losslessly represent a database value of type "
        f"{type(value).__name__} in canonical JSON"
    )


async def export_directory(
    database: Database,
    target_dir: Path,
    edge_registry: EdgeSchemaRegistry | None = None,
) -> ExportResult:
    """Export the connected database into canonical ``nodes/`` and ``edges/`` JSON.

    ``target_dir`` is required. The live deployment ``data/`` directory is never
    assumed. Existing ``nodes/`` and ``edges/`` subdirectories under the target
    are replaced only after the database has been read and the staged snapshot
    is complete. Other files in the target, such as ``Readme.md``, are left
    in place. The returned ``target_dir`` is the resolved absolute path that
    was written.
    """
    target = _explicit_target_dir(target_dir)
    if target.exists() and not target.is_dir():
        raise ValueError(f"{target} exists and is not a directory")
    for name in ("nodes", "edges"):
        existing = target / name
        if existing.exists() and not existing.is_dir():
            raise ValueError(f"{existing} exists and is not a directory")

    registry = edge_registry or EdgeSchemaRegistry.load_default()
    payload = await _read_canonical_payload(database, registry)

    staging = _staging_directory(target)
    try:
        nodes_dir = staging / "nodes"
        edges_dir = staging / "edges"
        nodes_dir.mkdir(parents=True)
        edges_dir.mkdir()
        for table, records in payload.nodes.items():
            _write_json(nodes_dir / f"{table}.json", records)
        for relation, records in payload.edges.items():
            _write_json(edges_dir / f"{relation}.json", records)
        _replace_export_tree(target, staging)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    return ExportResult(
        node_files=len(payload.nodes),
        edge_files=len(payload.edges),
        nodes_exported=sum(len(records) for records in payload.nodes.values()),
        edges_exported=sum(len(records) for records in payload.edges.values()),
        omitted_retired_tables=payload.omitted_retired_tables,
        omitted_retired_records=payload.omitted_retired_records,
        target_dir=str(target),
    )


@dataclass(frozen=True)
class _CanonicalPayload:
    nodes: dict[str, list[dict[str, Any]]]
    edges: dict[str, list[dict[str, Any]]]
    omitted_retired_tables: tuple[str, ...] = ()
    omitted_retired_records: int = 0


async def _read_canonical_payload(
    database: Database,
    registry: EdgeSchemaRegistry,
) -> _CanonicalPayload:
    db_tables = await _database_tables(database)
    edge_names = set(registry.relationship_names)
    inverse_names = set(registry.public_names) - edge_names
    retired = set(_RETIRED_EDGE_TABLES) | set(_RETIRED_NODE_TABLES)

    raw_tables = set(db_tables) | edge_names
    raw_records: dict[str, list[dict[str, Any]]] = {}
    for table in sorted(raw_tables):
        raw_records[table] = await _select_table(database, table)

    omitted_retired = tuple(
        sorted(table for table in retired if raw_records.get(table))
    )
    omitted_retired_records = sum(len(raw_records[table]) for table in omitted_retired)

    unregistered_relations: list[str] = []
    node_tables: dict[str, list[dict[str, Any]]] = {}
    for table, records in raw_records.items():
        if table in edge_names or table in retired:
            continue
        if table in inverse_names or _looks_like_relationship(records):
            if records:
                unregistered_relations.append(table)
            continue
        if not TABLE_NAME_RE.fullmatch(table):
            raise ValueError(f"Cannot export table {table!r}; it is not a canonical table name")
        if records:
            node_tables[table] = records
    if unregistered_relations:
        raise ValueError(
            "Cannot export unregistered relationship tables: "
            + ", ".join(sorted(unregistered_relations))
            + "; they are not part of the canonical JSON schema"
        )

    nodes = {
        table: _canonical_nodes(table, records)
        for table, records in sorted(node_tables.items())
    }
    edges = {
        relation: _canonical_edges(relation, raw_records.get(relation, []), registry.get(relation))
        for relation in registry.relationship_names
    }
    return _CanonicalPayload(
        nodes,
        edges,
        omitted_retired,
        omitted_retired_records,
    )


def _canonical_nodes(table: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    exported: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        converted = canonical_json_value(record)
        if not isinstance(converted, dict):
            raise ValueError(f"Node in table {table!r} is not an object")
        if "in" in converted or "out" in converted:
            raise ValueError(
                f"Table {table!r} contains relationship identity fields; "
                "refusing to serialize it as a node file"
            )
        record_id = converted.get("id")
        if not isinstance(record_id, str):
            raise ValueError(f"Node in table {table!r} is missing a canonical string id")
        record_table, _ = split_record_id(record_id)
        if record_table != table:
            raise ValueError(
                f"Node ID {record_id!r} in table {table!r} must use the {table!r} table"
            )
        if record_id in seen:
            raise ValueError(f"Duplicate node ID {record_id!r} in table {table!r}")
        seen.add(record_id)
        exported.append(_ordered_record(converted, _NODE_LEADING_FIELDS))
    exported.sort(key=lambda record: str(record["id"]))
    return exported


def _canonical_edges(
    relation: str,
    records: list[dict[str, Any]],
    schema: EdgeSchema,
) -> list[dict[str, Any]]:
    exported: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        converted = canonical_json_value(record)
        if not isinstance(converted, dict):
            raise ValueError(f"Edge in {relation!r} is not an object")
        raw_from = converted.get("in")
        raw_to = converted.get("out")
        if not isinstance(raw_from, str) or not isinstance(raw_to, str):
            raise ValueError(
                f"Edge in {relation!r} is missing SurrealDB in/out endpoints"
            )
        source = as_record_id(raw_from)
        target = as_record_id(raw_to)
        stored_id = converted.get("id")
        if not isinstance(stored_id, str):
            raise ValueError(f"Edge in {relation!r} is missing a canonical string id")
        if stored_id in seen:
            raise ValueError(f"Duplicate edge ID {stored_id!r} in {relation}")
        seen.add(stored_id)

        canonical: dict[str, Any] = {"from": raw_from, "to": raw_to}
        implicit_id = canonical_record_id(
            implicit_edge_record_id(relation, source, target)
        )
        if stored_id != implicit_id:
            canonical["id"] = stored_id
        for key, value in converted.items():
            if key in _SURREAL_EDGE_IDENTITY_FIELDS:
                continue
            canonical[key] = value
        if schema.temporal and "end" not in canonical:
            canonical["end"] = None
        exported.append(_ordered_record(canonical, _EDGE_LEADING_FIELDS))
    exported.sort(
        key=lambda record: (
            str(record.get("from", "")),
            str(record.get("to", "")),
            str(record.get("id", "")),
        )
    )
    return exported


def _looks_like_relationship(records: list[dict[str, Any]]) -> bool:
    return any("in" in record and "out" in record for record in records)


def _ordered_record(record: dict[str, Any], leading: tuple[str, ...]) -> dict[str, Any]:
    ordered: dict[str, Any] = {}
    for key in leading:
        if key in record:
            ordered[key] = _ordered_value(record[key])
    for key in sorted(record):
        if key in ordered:
            continue
        ordered[key] = _ordered_value(record[key])
    return ordered


def _ordered_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _ordered_value(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_ordered_value(item) for item in value]
    return value


async def _database_tables(database: Database) -> set[str]:
    info = await database.query("INFO FOR DB;")
    if isinstance(info, list) and len(info) == 1:
        info = info[0]
    if not isinstance(info, dict) or "tables" not in info:
        raise RuntimeError("SurrealDB INFO FOR DB returned an unexpected result")
    tables = info["tables"]
    if tables is None:
        return set()
    if isinstance(tables, dict):
        return {str(name) for name in tables}
    if isinstance(tables, list):
        names: set[str] = set()
        for item in tables:
            if isinstance(item, str):
                names.add(item)
            elif isinstance(item, dict) and "name" in item:
                names.add(str(item["name"]))
            else:
                raise RuntimeError("SurrealDB INFO FOR DB returned an unexpected table list")
        return names
    raise RuntimeError("SurrealDB INFO FOR DB returned an unexpected tables field")


async def _select_table(database: Database, table: str) -> list[dict[str, Any]]:
    result = await database.query(
        "SELECT * FROM type::table($table);",
        {"table": table},
    )
    if result is None:
        return []
    if isinstance(result, dict):
        return [result]
    if isinstance(result, list) and all(isinstance(record, dict) for record in result):
        return result
    raise RuntimeError(
        f"SurrealDB returned an unexpected query result for table {table!r}"
    )


def _write_json(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _explicit_target_dir(target_dir: Path) -> Path:
    if not str(target_dir).strip():
        raise ValueError("An explicit target directory is required")
    target = Path(target_dir).expanduser()
    if str(target) in {".", "./", "..", "../"} or target.name in {"", ".", ".."}:
        raise ValueError(
            "target_dir cannot be '.' or '..'; pass an explicit directory. "
            "The HTTP API writes inside the API process filesystem "
            "(Docker: /app/export, host: tmp/db-export)."
        )
    return target.resolve()


def _staging_directory(target: Path) -> Path:
    # Stage on the same filesystem as the target so rename works when the
    # destination is a Docker bind mount (EXDEV across /app vs /app/export).
    target.mkdir(parents=True, exist_ok=True)
    return target / f".export-tmp-{os.getpid()}-{uuid4().hex}"


def _move_directory(source: Path, destination: Path) -> None:
    try:
        source.rename(destination)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        shutil.copytree(source, destination)
        shutil.rmtree(source)


def _replace_export_tree(target: Path, staged: Path) -> None:
    unique = f"{os.getpid()}-{uuid4().hex}"
    backups: dict[str, Path] = {}
    try:
        for name in ("nodes", "edges"):
            destination = target / name
            staged_dir = staged / name
            if destination.exists():
                backup = target / f".{name}.export-backup-{unique}"
                _move_directory(destination, backup)
                backups[name] = backup
            _move_directory(staged_dir, destination)
    except BaseException:
        for name, backup in backups.items():
            destination = target / name
            if destination.exists():
                shutil.rmtree(destination)
            _move_directory(backup, destination)
        raise
    for backup in backups.values():
        shutil.rmtree(backup)
