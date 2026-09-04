"""CLI benchmark for the deterministic semantic household fact pipeline."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Sequence
from zoneinfo import ZoneInfo

from .agents import get_agent
from .config import get_settings
from .db import Database
from .edge_schema import EdgeSchemaRegistry
from .grounding import AgentRequestContext
from .ollama import OllamaService
from .retrieval import RetrievalService
from .schema_catalog import (
    RuntimeSchemaCatalog,
    matches_scoped_appellation,
    normalize_entity_alias,
    record_aliases,
)
from .semantic_facts import (
    HouseholdFactEngine,
    SemanticFactPlanner,
    SemanticFactService,
    SemanticSchemaRegistry,
    _failure_stage,
)
from .tools import GRAPH_TOOL_NAMES, ToolDispatcher

QUESTIONS = (
    "我是谁",
    "你是谁",
    "家里都有谁",
    "家里都有哪些人",
    "家里有多少人",
    "家里有几个人",
    "我家住哪里",
    "请问我的具体住址是什么？",
    "我老婆是谁",
    "我生日是哪天",
    "我有几个孩子",
    "我儿子是谁",
    "我儿子几岁",
    "家里谁最年长",
    "谁最年长",
    "谁最年幼",
    "谁年纪最小",
    "有几个成年人",
    "有几个孩子",
    "我和我老婆谁年龄大",
    "匡德伦的生日是哪天",
    "匡德伦哪天出生",
    "匡德伦是谁",
    "Dylan是谁",
    "德伦是谁",
    "Dylan Kuang是谁",
    "巴璞的儿子是谁",
    "巴璞哪天过生日",
    "我儿子哪天过生日",
    "我儿子的生日还有多少天",
    "我岳父是谁",
    "我岳母是谁",
    "我们什么时候结婚的",
)

TEMPORAL_OPERATIONS = frozenset(
    {"date_difference", "completed_years", "duration", "annual_occurrence"}
)

BenchmarkMode = Literal["tier0_enabled", "tier0_disabled"]


@dataclass(frozen=True)
class BenchmarkCase:
    speaker_id: str
    utterance: str


SPEAKER_CASES = (
    BenchmarkCase("person:jian_kuang", "我儿子是谁"),
    BenchmarkCase("person:pu_ba", "我儿子是谁"),
    BenchmarkCase("person:guiqiu_wang", "我孙子是谁"),
    BenchmarkCase("person:zhigang_ba", "我外孙是谁"),
    BenchmarkCase("person:evelyn_kuang", "我哥哥是谁"),
)


async def benchmark_runtime(
    caller_entity_id: str,
    repeat: int,
    mode: BenchmarkMode = "tier0_enabled",
    questions: Sequence[str] = QUESTIONS,
) -> dict[str, Any]:
    settings = get_settings()
    steward = get_agent("steward")
    database = Database(settings)
    llm: OllamaService | None = None
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
        schema = SemanticSchemaRegistry(catalog)
        llm = OllamaService(settings.ollama_url, settings.ollama_model)
        service = SemanticFactService(
            HouseholdFactEngine(dispatcher, schema, max_records=settings.retrieval_limit),
            planner=(SemanticFactPlanner(llm, schema) if llm is not None else None),
            tier_zero_enabled=mode == "tier0_enabled",
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
        return await _run_suite(
            service,
            context,
            repeat,
            backend="surrealdb",
            mode=mode,
            questions=questions,
            speaker_cases=(SPEAKER_CASES if tuple(questions) == QUESTIONS else ()),
        )
    finally:
        if llm is not None:
            await llm.close()
        await database.close()


async def benchmark_json(
    caller_entity_id: str,
    repeat: int,
    data_dir: Path,
    schema_dir: Path,
    mode: BenchmarkMode = "tier0_enabled",
    questions: Sequence[str] = QUESTIONS,
) -> dict[str, Any]:
    steward = get_agent("steward")
    edge_registry = EdgeSchemaRegistry.from_directory(schema_dir)
    catalog = RuntimeSchemaCatalog.from_data_dir(data_dir, edge_registry)
    dispatcher = _JsonGraphDispatcher(data_dir, edge_registry)
    schema = SemanticSchemaRegistry(catalog)
    llm: OllamaService | None = None
    settings = get_settings()
    llm = OllamaService(settings.ollama_url, settings.ollama_model)
    service = SemanticFactService(
        HouseholdFactEngine(dispatcher, schema),
        planner=(SemanticFactPlanner(llm, schema) if llm is not None else None),
        tier_zero_enabled=mode == "tier0_enabled",
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
    try:
        return await _run_suite(
            service,
            context,
            repeat,
            backend="json",
            mode=mode,
            questions=questions,
            speaker_cases=(SPEAKER_CASES if tuple(questions) == QUESTIONS else ()),
        )
    finally:
        if llm is not None:
            await llm.close()


async def _run_suite(
    service: SemanticFactService,
    context: AgentRequestContext,
    repeat: int,
    *,
    backend: str,
    mode: BenchmarkMode,
    questions: Sequence[str] = QUESTIONS,
    speaker_cases: Sequence[BenchmarkCase] = SPEAKER_CASES,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    cases = (
        *(BenchmarkCase(context.caller_entity_id or "", question) for question in questions),
        *speaker_cases,
    )
    for case in cases:
        case_context = replace(context, caller_entity_id=case.speaker_id or None)
        latest = None
        samples: list[float] = []
        for _ in range(repeat):
            latest = await service.try_answer(
                [{"role": "user", "content": case.utterance}],
                context=case_context,
                request_id="fact-benchmark",
            )
            if latest is None:
                raise RuntimeError(
                    f"Semantic pipeline did not answer {case.utterance!r} "
                    f"for {case.speaker_id!r} in {mode}"
                )
            samples.append(latest.timings.total_ms)
            latencies.append(latest.timings.total_ms)
        assert latest is not None
        references = (latest.request.subject,) + (
            (latest.request.other,) if latest.request.other is not None else ()
        )
        relations = [
            step.relation for reference in references for step in reference.path
        ]
        semantic_properties = {
            item
            for item in (
                latest.request.property,
                *(
                    query_filter.property
                    for reference in references
                    for step in reference.path
                    for query_filter in step.filters
                ),
                *(query_filter.property for query_filter in latest.request.filters),
            )
            if item is not None
        }
        filters = [
            {
                "stage": "collection",
                **query_filter.model_dump(mode="json"),
            }
            for query_filter in latest.request.filters
        ] + [
            {
                "stage": "traversal",
                "relation": step.relation,
                **query_filter.model_dump(mode="json"),
            }
            for reference in references
            for step in reference.path
            for query_filter in step.filters
        ]
        entity_properties = {
            item.property
            for item in latest.request.filters
            if item.property is not None and item.source == "entity"
        }
        relationship_properties = {
            item.property
            for item in latest.request.filters
            if item.property is not None and item.source == "relation"
        }
        for item in latest.request.filters:
            if item.predicate is None:
                continue
            definition = service.engine.schema.ontology.collection_predicates.get(
                item.predicate
            )
            if definition is None:
                continue
            entity_properties.add(definition.fallback.property)
            relationship_properties.add(definition.relation_property)
        if latest.request.property is not None:
            target = (
                relationship_properties
                if latest.request.property_source == "relationship"
                else entity_properties
            )
            target.add(latest.request.property)
        operators = [latest.request.operation]
        if any(reference.path for reference in references):
            operators.insert(0, "traverse")
        if latest.request.filters:
            operators.insert(-1, "filter")
        rows.append(
            {
                "speaker_id": case.speaker_id,
                "utterance": case.utterance,
                "question": case.utterance,
                "answer": latest.text,
                "semantic_plan": latest.request.model_dump(mode="json"),
                "scope": latest.request.subject.model_dump(mode="json"),
                "filters": filters,
                "operators": operators,
                "resolution_path": [
                    {
                        "anchor": reference.kind,
                        "anchor_text": reference.value,
                        "relations": [
                            step.model_dump(mode="json") for step in reference.path
                        ],
                    }
                    for reference in references
                ],
                "route": f"tier_{latest.timings.tier}",
                "tier": latest.timings.tier,
                "resolved_entities": list(latest.result.evidence.entity_ids),
                "relationship_refs": [
                    {
                        "relation": item.relation,
                        "source_id": item.source_id,
                        "target_id": item.target_id,
                        "start": item.start,
                        "end": item.end,
                    }
                    for item in latest.result.evidence.relationships
                ],
                "entity_properties": sorted(entity_properties),
                "relationship_properties": sorted(relationship_properties),
                "semantic_properties": sorted(semantic_properties),
                "relations": relations,
                "temporal_operations": (
                    [latest.request.operation]
                    if latest.request.operation in TEMPORAL_OPERATIONS
                    else []
                ),
                "result_status": latest.result.status,
                "failure_stage": _failure_stage(latest.result.status),
                "missing_requirements": list(latest.result.missing_requirements),
                "llm_call_count": latest.timings.llm_call_count,
                "db_query_count": latest.timings.db_query_count,
                "total_latency_ms": round(statistics.median(samples), 3),
                "stage_latency_ms": {
                    "routing": round(latest.timings.routing_ms, 3),
                    "semantic_parse": round(latest.timings.semantic_parse_ms, 3),
                    "entity_resolution": round(
                        latest.timings.entity_resolution_ms, 3
                    ),
                    "fact_query": round(latest.timings.fact_query_ms, 3),
                    "computation": round(latest.timings.computation_ms, 3),
                    "render": round(latest.timings.render_ms, 3),
                    "llm": round(latest.timings.llm_ms, 3),
                },
            }
        )
    row_by_question = {row["utterance"]: row for row in rows}
    comparisons = {
        "age_extrema": {
            question: row_by_question[question]["semantic_plan"]
            for question in ("谁最年长", "谁最年幼")
            if question in row_by_question
        },
        "household_counts": {
            question: row_by_question[question]["semantic_plan"]
            for question in ("家里有几个人", "有几个成年人")
            if question in row_by_question
        },
        "spouse_entity_vs_relationship": {
            question: row_by_question[question]["semantic_plan"]
            for question in ("我老婆是谁", "我们什么时候结婚的")
            if question in row_by_question
        },
    }
    return {
        "backend": backend,
        "mode": mode,
        "queries": rows,
        "diagnostic_comparisons": comparisons,
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
        elif tool_name in {"search_entities", "resolve_entity_alias"}:
            query = normalize_entity_alias(arguments["text"])
            expected = arguments.get("entity_type")
            candidates = [
                entity
                for entity in self.entities.values()
                if expected is None or entity["id"].startswith(f"{expected}:")
            ]
            aliases = [
                _summary(entity)
                for entity in candidates
                if any(
                    query == normalize_entity_alias(alias)
                    for alias in record_aliases(entity)
                )
            ]
            appellations = [
                _summary(entity)
                for entity in candidates
                if matches_scoped_appellation(
                    entity,
                    arguments["text"],
                    speaker_id=arguments.get("speaker_id"),
                    household_id=arguments.get("household_id"),
                )
            ]
            records = (aliases or appellations)[: arguments.get("limit", 25)]
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
        "--mode",
        choices=("tier0_enabled", "tier0_disabled"),
        default=os.environ.get("MODE", "tier0_enabled"),
        help=(
            "Enable the deterministic fast parser or force every query through "
            "the configured semantic planner. May also be set with MODE."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=("json", "surrealdb"),
        default="json",
        help="Use source JSON for offline timing or the configured runtime database.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--schema-dir", type=Path, default=Path("schemas/edge"))
    parser.add_argument(
        "--question",
        action="append",
        dest="questions",
        help="Benchmark only this utterance; may be repeated.",
    )
    arguments = parser.parse_args()
    if arguments.repeat < 1:
        parser.error("--repeat must be at least 1")
    coroutine = (
        benchmark_json(
            arguments.caller,
            arguments.repeat,
            arguments.data_dir,
            arguments.schema_dir,
            arguments.mode,
            arguments.questions or QUESTIONS,
        )
        if arguments.backend == "json"
        else benchmark_runtime(
            arguments.caller,
            arguments.repeat,
            arguments.mode,
            arguments.questions or QUESTIONS,
        )
    )
    result = asyncio.run(coroutine)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
