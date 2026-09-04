"""CLI benchmark for the deterministic semantic household fact pipeline."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence
from zoneinfo import ZoneInfo

from .agents import get_agent
from .config import get_settings
from .db import Database
from .edge_schema import EdgeSchemaRegistry
from .grounding import AgentRequestContext
from .retrieval import RetrievalService
from .schema_catalog import RuntimeSchemaCatalog
from .semantic_facts import (
    HouseholdFactEngine,
    SemanticFactService,
    SemanticSchemaRegistry,
)
from .tools import GRAPH_TOOL_NAMES, ToolDispatcher

QUESTIONS = (
    "我是谁",
    "你是谁",
    "家里都有谁",
    "家里都有哪些人",
    "家里有几个人",
    "我家住哪里",
    "请问我的具体住址是什么？",
    "我老婆是谁",
    "我生日是哪天",
    "我有几个孩子",
    "我儿子是谁",
    "我儿子几岁了",
    "谁最年长",
    "我和我老婆谁年龄大",
)


async def benchmark_runtime(caller_entity_id: str, repeat: int) -> dict[str, Any]:
    settings = get_settings()
    steward = get_agent("steward")
    database = Database(settings)
    await database.connect()
    try:
        edge_registry = EdgeSchemaRegistry.from_directory(settings.edge_schema_dir)
        catalog = RuntimeSchemaCatalog.from_data_dir(settings.data_dir, edge_registry)
        retrieval = RetrievalService(
            database,
            settings.retrieval_limit,
            settings.data_dir,
            edge_registry,
        )
        dispatcher = ToolDispatcher(retrieval, sorted(GRAPH_TOOL_NAMES))
        service = SemanticFactService(
            HouseholdFactEngine(
                dispatcher,
                SemanticSchemaRegistry(catalog),
                max_records=settings.retrieval_limit,
            )
        )
        localized = steward.settings.get("localized_identity", {})
        context = AgentRequestContext(
            caller_entity_id=caller_entity_id,
            assistant_id=steward.id,
            assistant_display_name=localized.get("zh", steward.display_name),
            household_id=steward.settings.get("home_entity_id"),
            current_time=datetime.now(ZoneInfo(settings.calendar_timezone)),
            locale="zh",
        )
        return await _run_suite(service, context, repeat, backend="surrealdb")
    finally:
        await database.close()


async def benchmark_json(
    caller_entity_id: str,
    repeat: int,
    data_dir: Path,
    schema_dir: Path,
) -> dict[str, Any]:
    steward = get_agent("steward")
    edge_registry = EdgeSchemaRegistry.from_directory(schema_dir)
    catalog = RuntimeSchemaCatalog.from_data_dir(data_dir, edge_registry)
    dispatcher = _JsonGraphDispatcher(data_dir, edge_registry)
    service = SemanticFactService(
        HouseholdFactEngine(dispatcher, SemanticSchemaRegistry(catalog))
    )
    localized = steward.settings.get("localized_identity", {})
    context = AgentRequestContext(
        caller_entity_id=caller_entity_id,
        assistant_id=steward.id,
        assistant_display_name=localized.get("zh", steward.display_name),
        household_id=steward.settings.get("home_entity_id"),
        current_time=datetime.now(ZoneInfo("America/Los_Angeles")),
        locale="zh",
    )
    return await _run_suite(service, context, repeat, backend="json")


async def _run_suite(
    service: SemanticFactService,
    context: AgentRequestContext,
    repeat: int,
    *,
    backend: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    for question in QUESTIONS:
        latest = None
        samples: list[float] = []
        for _ in range(repeat):
            latest = await service.try_answer(
                [{"role": "user", "content": question}],
                context=context,
                request_id="fact-benchmark",
            )
            if latest is None:
                raise RuntimeError(f"Tier-0 did not recognize {question!r}")
            samples.append(latest.timings.total_ms)
            latencies.append(latest.timings.total_ms)
        assert latest is not None
        rows.append(
            {
                "question": question,
                "answer": latest.text,
                "semantic_plan": latest.request.model_dump(mode="json"),
                "tier": latest.timings.tier,
                "llm_call_count": latest.timings.llm_call_count,
                "db_query_count": latest.timings.db_query_count,
                "total_latency_ms": round(statistics.median(samples), 3),
            }
        )
    return {
        "backend": backend,
        "queries": rows,
        "aggregate": {
            "samples": len(latencies),
            "p50_ms": round(_percentile(latencies, 0.50), 3),
            "p95_ms": round(_percentile(latencies, 0.95), 3),
            "llm_call_count": sum(row["llm_call_count"] for row in rows),
        },
    }


class _JsonGraphDispatcher:
    """Read-only debug adapter over the same node/edge source documents."""

    def __init__(self, data_dir: Path, registry: EdgeSchemaRegistry) -> None:
        self.registry = registry
        self.entities = {
            record["id"]: record
            for path in (data_dir / "nodes").glob("*.json")
            for record in json.loads(path.read_text(encoding="utf-8"))
        }
        self.edges = {
            path.stem: json.loads(path.read_text(encoding="utf-8"))
            for path in (data_dir / "edges").glob("*.json")
        }

    async def dispatch(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        **_: Any,
    ) -> dict[str, Any]:
        if tool_name == "get_entity":
            entity = self.entities.get(arguments["entity_id"])
            records = [entity] if entity is not None else []
        elif tool_name == "search_entities":
            query = arguments["text"].casefold()
            expected = arguments.get("entity_type")
            records = [
                _summary(entity)
                for entity in self.entities.values()
                if (expected is None or entity["id"].startswith(f"{expected}:"))
                and any(query == alias.casefold() for alias in _aliases(entity))
            ][: arguments.get("limit", 25)]
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


def _aliases(entity: dict[str, Any]) -> list[str]:
    name = entity.get("name")
    if isinstance(name, str):
        return [name]
    if isinstance(name, list):
        return [item for item in name if isinstance(item, str)]
    if isinstance(name, dict):
        return [item for item in name.values() if isinstance(item, str)]
    return []


def _summary(entity: dict[str, Any]) -> dict[str, Any]:
    return {
        field: value
        for field, value in entity.items()
        if field in {"id", "name", "display_name", "gender"}
    }


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caller", default="person:jian_kuang")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument(
        "--backend",
        choices=("json", "surrealdb"),
        default="json",
        help="Use source JSON for offline timing or the configured runtime database.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--schema-dir", type=Path, default=Path("schemas/edge"))
    arguments = parser.parse_args()
    if arguments.repeat < 1:
        parser.error("--repeat must be at least 1")
    coroutine = (
        benchmark_json(
            arguments.caller,
            arguments.repeat,
            arguments.data_dir,
            arguments.schema_dir,
        )
        if arguments.backend == "json"
        else benchmark_runtime(arguments.caller, arguments.repeat)
    )
    result = asyncio.run(coroutine)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
