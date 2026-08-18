import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from surrealdb import RecordID

from .db import Database

RECORD_PATTERN = re.compile(
    r"^(?P<table>[A-Za-z_][A-Za-z0-9_]*):(?P<id>[A-Za-z0-9_-]+)$"
)


@dataclass(frozen=True)
class RetrievedContext:
    question: str
    nodes: dict[str, list[dict[str, Any]]]
    edges: dict[str, list[dict[str, Any]]]
    text: str


class RetrievalService:
    """Run safe, read-only searches over known node and relationship tables."""

    def __init__(
        self,
        database: Database,
        limit: int = 100,
        data_dir: Path | None = None,
    ) -> None:
        self.database = database
        self.limit = limit
        self.node_tables = self._table_names(data_dir, "nodes") or (
            "location",
            "person",
        )
        self.edge_tables = self._table_names(data_dir, "edges") or ("resides_in",)

    async def search_entities(
        self,
        text: str,
        entity_type: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        search_text = text.strip().lower()
        if not search_text:
            raise ValueError("Search text cannot be empty")

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
            WHERE string::contains(
                string::lowercase(type::string($this)),
                $text
            )
            ORDER BY id
            LIMIT $limit;
        """
        matches: list[dict[str, Any]] = []
        for table in tables:
            result = await self.database.query(
                statement,
                {"table": table, "text": search_text, "limit": result_limit},
            )
            matches.extend(_query_records(result))

        matches.sort(key=lambda record: str(record.get("id", "")))
        return matches[:result_limit]

    async def get_entity(self, record_id: str) -> dict[str, Any] | None:
        """Return the record for a canonical ID, or None if it does not exist.

        This is a point-get, not a search. Callers that already know a
        record ID must not go through search_entities.
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

    async def get_relationships(
        self,
        entity_id: str,
        relation: str | None = None,
        limit: int | None = None,
        *,
        include_residents: bool = True,
    ) -> list[dict[str, Any]]:
        entity = _parse_record_id(entity_id)
        result_limit = self._validated_limit(limit)

        if relation is not None:
            if relation not in self.edge_tables:
                raise ValueError(
                    f"Unknown relation {relation!r}; expected one of "
                    f"{', '.join(self.edge_tables)}"
                )
            relations = (relation,)
        else:
            relations = self.edge_tables

        statement = """
            SELECT *,
                in.* AS source_entity,
                out.* AS target_entity
            FROM type::table($relation)
            WHERE in = $entity OR out = $entity
            ORDER BY id
            LIMIT $limit;
        """
        relationships: list[dict[str, Any]] = []
        for relation_name in relations:
            result = await self.database.query(
                statement,
                {
                    "relation": relation_name,
                    "entity": entity,
                    "limit": result_limit,
                },
            )
            for edge in _query_records(result):
                edge["relation"] = relation_name
                direction = _relationship_direction(edge, entity_id)
                source_entity = edge.pop("source_entity", None)
                target_entity = edge.pop("target_entity", None)
                edge["direction"] = direction
                subject_entity = (
                    source_entity if direction == "outgoing" else target_entity
                )
                if isinstance(subject_entity, dict):
                    edge["entity"] = subject_entity
                related_entity = (
                    source_entity if direction == "incoming" else target_entity
                )
                if isinstance(related_entity, dict):
                    edge["related_entity"] = related_entity
                relationships.append(edge)

        relationships.sort(
            key=lambda edge: (str(edge.get("relation", "")), str(edge.get("id", "")))
        )
        relationships = relationships[:result_limit]
        if include_residents and entity.table_name == "person":
            await self._attach_residence_rosters(relationships, result_limit)
        return relationships

    async def _attach_residence_rosters(
        self,
        relationships: list[dict[str, Any]],
        result_limit: int,
    ) -> None:
        """A person's resides_in edge names a home, not the household roster."""
        if "resides_in" not in self.edge_tables:
            return
        residents_by_home: dict[str, list[dict[str, Any]]] = {}
        for edge in relationships:
            if edge.get("relation") != "resides_in":
                continue
            home_id = edge.get("out")
            if not isinstance(home_id, str) or not home_id.startswith("location:"):
                continue
            if home_id not in residents_by_home:
                household = await self.get_relationships(
                    home_id,
                    relation="resides_in",
                    limit=result_limit,
                    include_residents=False,
                )
                residents: list[dict[str, Any]] = []
                for resident_edge in household:
                    person = resident_edge.get("related_entity")
                    if isinstance(person, dict) and str(
                        person.get("id", "")
                    ).startswith("person:"):
                        _append_unique_record(residents, person)
                residents.sort(key=lambda record: str(record.get("id", "")))
                residents_by_home[home_id] = residents
            edge["residents"] = residents_by_home[home_id]

    async def retrieve(self, question: str) -> RetrievedContext:
        entities = await self.search_entities(question)
        nodes = {table: [] for table in self.node_tables}
        edges = {table: [] for table in self.edge_tables}

        for entity in entities:
            record_id = str(entity.get("id", ""))
            table = record_id.partition(":")[0]
            if table in nodes:
                _append_unique_record(nodes[table], entity)
            if record_id:
                for relationship in await self.get_relationships(record_id):
                    edge = dict(relationship)
                    edge.pop("entity", None)
                    related_entity = edge.pop("related_entity", None)
                    if isinstance(related_entity, dict):
                        related_id = str(related_entity.get("id", ""))
                        related_table = related_id.partition(":")[0]
                        if related_table in nodes:
                            _append_unique_record(
                                nodes[related_table],
                                related_entity,
                            )
                    for resident in edge.get("residents") or []:
                        if not isinstance(resident, dict):
                            continue
                        resident_id = str(resident.get("id", ""))
                        resident_table = resident_id.partition(":")[0]
                        if resident_table in nodes:
                            _append_unique_record(
                                nodes[resident_table],
                                resident,
                            )
                    relation = str(edge.get("relation", ""))
                    if relation in edges and edge not in edges[relation]:
                        edges[relation].append(edge)

        for records in nodes.values():
            records.sort(key=lambda record: str(record.get("id", "")))
        for records in edges.values():
            records.sort(key=lambda record: str(record.get("id", "")))

        graph = {"nodes": nodes, "edges": edges}
        text = json.dumps(graph, ensure_ascii=False, indent=2, sort_keys=True)
        return RetrievedContext(question=question, nodes=nodes, edges=edges, text=text)

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
        return f"{table}:{record_id}"
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
    match = RECORD_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError("Entity ID must use the table:record_id format")
    return RecordID(match.group("table"), match.group("id"))


def _relationship_direction(edge: dict[str, Any], entity_id: str) -> str:
    is_source = edge.get("in") == entity_id
    is_target = edge.get("out") == entity_id
    if is_source and is_target:
        return "both"
    if is_source:
        return "outgoing"
    return "incoming"


def _append_unique_record(
    records: list[dict[str, Any]],
    record: dict[str, Any],
) -> None:
    record_id = str(record.get("id", ""))
    is_new = all(
        str(existing.get("id", "")) != record_id for existing in records
    )
    if record_id and is_new:
        records.append(record)
