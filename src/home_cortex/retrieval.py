import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from .db import Database


@dataclass(frozen=True)
class RetrievedContext:
    question: str
    nodes: dict[str, list[dict[str, Any]]]
    edges: dict[str, list[dict[str, Any]]]
    text: str


class RetrievalService:
    """Retrieve the small graph as structured prompt context."""

    def __init__(
        self,
        database: Database,
        limit: int = 100,
        data_dir: Path | None = None,
    ) -> None:
        self.database = database
        self.limit = limit
        self.node_tables = self._table_names(data_dir, "nodes") or ("home", "person")
        self.edge_tables = self._table_names(data_dir, "edges") or ("resides_in",)

    async def retrieve(self, question: str) -> RetrievedContext:
        nodes = {
            table: to_json_value((await self.database.select(table))[: self.limit])
            for table in self.node_tables
        }
        edges = {
            table: to_json_value((await self.database.select(table))[: self.limit])
            for table in self.edge_tables
        }
        graph = {"nodes": nodes, "edges": edges}
        text = json.dumps(graph, ensure_ascii=False, indent=2, sort_keys=True)
        return RetrievedContext(question=question, nodes=nodes, edges=edges, text=text)

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

