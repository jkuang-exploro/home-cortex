#!/usr/bin/env python3
"""Tier-1 planner latency probe. Uses JSON graph. Tier-0 disabled."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from home_cortex.agents import get_agent
from home_cortex.edge_schema import EdgeSchemaRegistry
from home_cortex.fact_benchmark import _JsonGraphDispatcher, _percentile
from home_cortex.ollama import OllamaService
from home_cortex.schema_catalog import RuntimeSchemaCatalog
from home_cortex.semantic_facts import (
    AgentRequestContext,
    HouseholdFactEngine,
    SemanticFactPlanner,
    SemanticFactService,
    SemanticSchemaRegistry,
)
from home_cortex.semantic_planner_benchmark import load_semantic_eval_cases

LATENCY_UTTERANCES = (
    "咱家最晚出生的是谁？",
    "家里已经成年的有多少位？",
    "我爱人的爸爸是谁？",
    "德伦再过多久过生日？",
    "我和我老婆结婚多久了？",
    "我老婆生日是哪天",
    "我儿子哪天出生",
    "家里最年长的是谁",
    "有几个孩子",
    "我岳父是谁",
    "我家住哪里",
    "我和我老婆谁年龄大",
    "匡德伦是谁",
    "我的女儿叫什么",
    "我们什么时候结婚的",
    "请列出家庭成员",
    "谁年纪最小",
    "家中未满十八岁的有几位",
    "我妻子的父亲叫什么",
    "从今天到我儿子生日相隔几天",
)


def _metrics(diag) -> dict:
    data = {
        "attempt_count": getattr(diag, "attempt_count", None),
        "validation": getattr(diag, "validation_result", None),
        "latency_ms": getattr(diag, "latency_ms", None),
        "prompt_build_ms": getattr(diag, "prompt_build_ms", None),
        "request_ms": getattr(diag, "request_ms", None),
        "validation_ms": getattr(diag, "validation_ms", None),
        "prompt_eval_count": getattr(diag, "prompt_eval_count", None),
        "prompt_eval_duration_ms": getattr(diag, "prompt_eval_duration_ms", None),
        "eval_count": getattr(diag, "eval_count", None),
        "eval_duration_ms": getattr(diag, "eval_duration_ms", None),
        "load_duration_ms": getattr(diag, "load_duration_ms", None),
    }
    return {key: value for key, value in data.items() if value not in (None, 0, 0.0)}


async def run(args: argparse.Namespace) -> dict:
    registry = EdgeSchemaRegistry.from_directory(args.schema_dir)
    catalog = RuntimeSchemaCatalog.from_data_dir(args.data_dir, registry)
    schema = SemanticSchemaRegistry(catalog)
    ollama = OllamaService(args.ollama_url, args.model)
    service = SemanticFactService(
        HouseholdFactEngine(_JsonGraphDispatcher(args.data_dir, registry), schema),
        planner=SemanticFactPlanner(ollama, schema),
        tier_zero_enabled=False,
    )
    steward = get_agent("steward")
    context = AgentRequestContext(
        caller_entity_id="person:jian_kuang",
        assistant_id=steward.id,
        assistant_display_name="老管家",
        household_id=steward.settings.get("home_entity_id"),
        current_time=datetime.fromisoformat("2026-09-03T12:00:00-07:00"),
        locale="zh",
    )
    eval_cases = {
        case.utterance: case
        for case in load_semantic_eval_cases(args.eval)
    }
    rows = []
    try:
        for index, utterance in enumerate(LATENCY_UTTERANCES):
            answer = await service.try_answer(
                [
                    {"role": "assistant", "content": "prior turn should not be planned"},
                    {"role": "user", "content": utterance},
                ],
                context=context,
            )
            assert answer is not None
            diag = answer.planner_diagnostics
            expected = eval_cases.get(utterance.rstrip("？?"))
            plan_ok = None
            if expected is not None:
                from home_cortex.semantic_planner_benchmark import (
                    normalize_semantic_request,
                )

                actual = normalize_semantic_request(answer.request)
                wanted = normalize_semantic_request(expected.expected)
                alternatives = [
                    normalize_semantic_request(item)
                    for item in expected.acceptable_alternatives
                ]
                plan_ok = actual == wanted or actual in alternatives
            row = {
                "utterance": utterance,
                "phase": "cold" if index == 0 else "warm",
                "tier": answer.timings.tier,
                "total_ms": round(answer.timings.total_ms, 1),
                "llm_ms": round(answer.timings.llm_ms, 1),
                "entity_resolution_ms": round(answer.timings.entity_resolution_ms, 1),
                "fact_query_ms": round(answer.timings.fact_query_ms, 1),
                "computation_ms": round(answer.timings.computation_ms, 1),
                "render_ms": round(answer.timings.render_ms, 1),
                "status": answer.result.status,
                "attempts": answer.timings.llm_call_count,
                "plan_match": plan_ok,
                "planner": _metrics(diag) if diag is not None else {},
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if index == 0 and args.repeat_first:
                second = await service.try_answer(
                    [{"role": "user", "content": utterance}],
                    context=context,
                )
                assert second is not None
                warm = {
                    "utterance": utterance,
                    "phase": "warm_repeat",
                    "tier": second.timings.tier,
                    "total_ms": round(second.timings.total_ms, 1),
                    "llm_ms": round(second.timings.llm_ms, 1),
                    "attempts": second.timings.llm_call_count,
                    "status": second.result.status,
                    "planner": _metrics(second.planner_diagnostics),
                }
                rows.append(warm)
                print(json.dumps(warm, ensure_ascii=False), flush=True)
    finally:
        await ollama.close()

    warm = [row for row in rows if row["phase"] != "cold"]
    planner = [row["llm_ms"] for row in warm]
    e2e = [row["total_ms"] for row in warm]
    matched = [row["plan_match"] for row in rows if row.get("plan_match") is not None]
    summary = {
        "n_warm": len(warm),
        "planner_p50": round(statistics.median(planner), 1) if planner else None,
        "planner_p95": round(_percentile(planner, 0.95), 1) if planner else None,
        "e2e_p50": round(statistics.median(e2e), 1) if e2e else None,
        "e2e_p95": round(_percentile(e2e, 0.95), 1) if e2e else None,
        "avg_attempts": round(
            sum(row["attempts"] for row in warm) / len(warm), 3
        )
        if warm
        else None,
        "eval_accuracy": (
            round(sum(int(item) for item in matched) / len(matched), 4)
            if matched
            else None
        ),
        "cold_total_ms": rows[0]["total_ms"] if rows else None,
        "cold_llm_ms": rows[0]["llm_ms"] if rows else None,
    }
    print(json.dumps({"summary": summary}, ensure_ascii=False), flush=True)
    return {"summary": summary, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--schema-dir", type=Path, default=Path("schemas/edge"))
    parser.add_argument(
        "--eval", type=Path, default=Path("benchmarks/semantic_planner_eval.yaml")
    )
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--repeat-first", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
