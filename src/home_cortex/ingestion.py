import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from surrealdb import RecordID

from .db import Database

TABLE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
RECORD_PATTERN = re.compile(
    r"^(?P<table>[A-Za-z_][A-Za-z0-9_]*):(?P<id>[A-Za-z0-9_-]+)$"
)


@dataclass(frozen=True)
class IngestionResult:
    node_files: int = 0
    edge_files: int = 0
    nodes_upserted: int = 0
    edges_upserted: int = 0


def _records_from_file(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)

    records = value if isinstance(value, list) else [value]
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"{path} must contain a JSON object or array of objects")
    return records


def parse_record_id(value: str, *, source: Path) -> RecordID:
    match = RECORD_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(
            f"Invalid record ID {value!r} in {source}; expected table:record_id"
        )
    return RecordID(match.group("table"), match.group("id"))


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
        f"{source.table_name}_{source.id}__{target.table_name}_{target.id}"
    )
    return RecordID(relation, identifier)


async def ingest_directory(database: Database, data_dir: Path) -> IngestionResult:
    nodes_dir = data_dir / "nodes"
    edges_dir = data_dir / "edges"
    if not nodes_dir.is_dir() or not edges_dir.is_dir():
        raise FileNotFoundError(
            f"Expected {nodes_dir} and {edges_dir} to both be directories"
        )

    node_files = sorted(nodes_dir.glob("*.json"))
    edge_files = sorted(edges_dir.glob("*.json"))
    nodes_upserted = 0
    edges_upserted = 0

    # Load nodes first so every relation endpoint exists before edges are created.
    for path in node_files:
        for record in _records_from_file(path):
            raw_id = record.get("id")
            if not isinstance(raw_id, str):
                raise ValueError(f"Node in {path} is missing a string 'id'")
            record_id = parse_record_id(raw_id, source=path)
            content = {key: value for key, value in record.items() if key != "id"}
            await database.upsert(record_id, content)
            nodes_upserted += 1

    for path in edge_files:
        relation = path.stem
        if TABLE_PATTERN.fullmatch(relation) is None:
            raise ValueError(f"Invalid relation table name derived from {path.name}")

        implicit_pairs: set[tuple[str, str]] = set()
        for record in _records_from_file(path):
            raw_from = record.get("from")
            raw_to = record.get("to")
            if not isinstance(raw_from, str) or not isinstance(raw_to, str):
                raise ValueError(f"Edge in {path} requires string 'from' and 'to'")

            source = parse_record_id(raw_from, source=path)
            target = parse_record_id(raw_to, source=path)
            if "id" not in record:
                pair = (str(source), str(target))
                if pair in implicit_pairs:
                    raise ValueError(
                        f"Multiple {relation} edges in {path} connect {source} to "
                        "the same target; give each edge a unique 'id'"
                    )
                implicit_pairs.add(pair)

            edge = _edge_record_id(relation, record, source, target, path)
            content = {
                key: value
                for key, value in record.items()
                if key not in {"id", "from", "to", "in", "out"}
            }

            # A stable edge record ID makes repeated ingestion an upsert.
            statement = "RELATE $source->$edge->$target CONTENT $content;"
            await database.query(
                statement,
                {
                    "source": source,
                    "edge": edge,
                    "target": target,
                    "content": content,
                },
            )
            edges_upserted += 1

    return IngestionResult(
        node_files=len(node_files),
        edge_files=len(edge_files),
        nodes_upserted=nodes_upserted,
        edges_upserted=edges_upserted,
    )
