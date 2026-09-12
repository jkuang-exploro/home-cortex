"""Shared synthetic graph adapter for benchmarks, probes, and deterministic tests."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from home_cortex.edge_schema import EdgeSchemaRegistry
from home_cortex.retrieval import ENTITY_SUMMARY_FIELDS
from home_cortex.schema_catalog import matching_named_entities, node_table_sources


class JsonGraphDispatcher:
    """Read-only debug adapter over the same node/edge source documents."""

    def __init__(self, data_dir: Path, registry: EdgeSchemaRegistry) -> None:
        self.registry = registry
        self.entities = {
            record["id"]: record
            for paths in node_table_sources(data_dir / "nodes").values()
            for path in paths
            for record in json.loads(path.read_text(encoding="utf-8"))
        }
        self.edges = {
            path.stem: json.loads(path.read_text(encoding="utf-8"))
            for path in (data_dir / "edges").glob("*.json")
        }
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def dispatch_internal(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        **_: Any,
    ) -> dict[str, Any]:
        self.calls.append((tool_name, arguments))
        if tool_name == "get_entity":
            entity = self.entities.get(arguments["entity_id"])
            records = [entity] if entity is not None else []
        elif tool_name == "resolve_entity_alias":
            expected = arguments.get("entity_type")
            candidates = [
                entity
                for entity in self.entities.values()
                if expected is None or entity["id"].startswith(f"{expected}:")
            ]
            records = [_summary(entity) for entity in matching_named_entities(
                candidates, arguments["text"], limit=arguments.get("limit", 25),
                speaker_id=arguments.get("speaker_id"),
                household_id=arguments.get("household_id"),
            )]
        elif tool_name == "get_relationships":
            records = self._relationships(arguments)
        else:
            records = []
        return {"ok": True, "tool": tool_name, "result": records}

    def _relationships(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        entity_id = arguments["entity_id"]
        resolved = self.registry.resolve(arguments["relation"])
        requested = arguments.get("direction")
        if resolved.inverse and requested in {"in", "out"}:
            requested = "out" if requested == "in" else "in"
        records: list[dict[str, Any]] = []
        for raw in self.edges.get(resolved.schema.id, []):
            if not arguments.get("include_ended") and raw.get("end") is not None:
                continue
            is_out = raw.get("from") == entity_id
            is_in = raw.get("to") == entity_id
            matches = (
                is_out or is_in
                if resolved.schema.symmetric or requested not in {"in", "out"}
                else is_out
                if requested == "out"
                else is_in
            )
            if not matches:
                continue
            related_id = raw["to"] if is_out else raw["from"]
            edge = dict(raw)
            edge["relation"] = resolved.schema.id
            edge["related_entity"] = _summary(self.entities[related_id])
            records.append(edge)
        return records[: arguments.get("limit", 25)]


def _summary(entity: dict[str, Any]) -> dict[str, Any]:
    return {
        field: value
        for field, value in entity.items()
        if field in ENTITY_SUMMARY_FIELDS
    }


