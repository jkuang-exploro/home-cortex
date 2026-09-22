"""Paired shadow evaluation of the existing and unified semantic planners."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import yaml

from home_cortex.providers.ollama import OllamaService
from home_cortex.mutation.ir import NAMED_WRITE_ADAPTER
from home_cortex.semantic.ir import SemanticPlannerFailure
from scripts import PROJECT_ROOT
from scripts.benchmarks.planner_prompt_experiment import (
    answer_signature,
    regression_cases,
)
from scripts.benchmarks.semantic_planner_benchmark import (
    build_json_fact_service,
    collect_provenance,
    normalize_semantic_request,
    serialize_fact_result,
    summarize_latencies,
)
from scripts.probes.unified_semantic_planner import (
    UnifiedShadowPlanner,
    unified_chat_messages,
    unified_output_schema,
)
from scripts.probes.two_stage_semantic_planner import LegacyTwoStagePlanner


class ObservedOllama(OllamaService):
    def reset_observations(self):
        self.observations = []

    async def plan_item_mutation(self, messages):
        decision, metrics = await super().plan_item_mutation(messages)
        self.observations.append({"stage": "mutation", **metrics})
        return decision, metrics

    async def plan_semantic_fact(self, *args, **kwargs):
        result = await super().plan_semantic_fact(*args, **kwargs)
        self.observations.append({"stage": "fact", **self.last_planner_runtime})
        return result


def _compact(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _kind(plan):
    if plan is None:
        return "invalid"
    if getattr(plan, "multi_intent", False):
        return "multi_intent"
    if plan.mutation is not None:
        return "mutation"
    if plan.request is not None:
        return "fact"
    return "none"


def _mutation_payload(plan):
    return (
        plan.mutation.model_dump(mode="json")
        if plan is not None and plan.mutation is not None else None
    )


async def _existing(client, service, context, utterance):
    client.reset_observations()
    started = perf_counter()
    outcome = None
    error = None
    try:
        outcome = await service.planner.plan(
            [{"role": "user", "content": utterance}], context
        )
    except SemanticPlannerFailure as exc:
        error = exc.diagnostics.validation_result
    return {
        "plan": outcome.plan if outcome else None,
        "latency_ms": (perf_counter() - started) * 1000,
        "calls": tuple(client.observations),
        "error": error,
    }


async def _unified(planner, context, utterance):
    result = await planner.plan(
        [{"role": "user", "content": utterance}],
        household_now=context.current_time,
    )
    return {
        "plan": result.plan,
        "latency_ms": result.latency_ms,
        "calls": result.calls,
        "error": result.error,
        "output_raw": result.output_raw,
    }


async def _read_row(route, sample, case, run, service, context, gold):
    plan = run["plan"]
    normalized = (
        normalize_semantic_request(plan.request)
        if plan is not None and plan.request is not None else None
    )
    accepted = {
        _compact(normalize_semantic_request(candidate))
        for candidate in (case.expected, *case.acceptable_alternatives)
    }
    actual_result = None
    if plan is not None and plan.request is not None:
        result, _, _, _ = await service.engine.execute(plan.request, context)
        actual_result = serialize_fact_result(result)
    return _common_row(route, sample, "read", case.utterance, run) | {
        "case_id": case.case_id,
        "decision_kind": _kind(plan),
        "plan_correct": normalized is not None and _compact(normalized) in accepted,
        "answer_correct": answer_signature(actual_result) == answer_signature(gold),
        "normalized_plan": normalized,
    }


def _common_row(route, sample, category, utterance, run):
    calls = run["calls"]
    return {
        "route": route,
        "sample": sample,
        "category": category,
        "utterance": utterance,
        "llm_calls": len(calls),
        "input_tokens": sum(call.get("prompt_eval_count", call.get("prompt_tokens", 0)) or 0 for call in calls),
        "output_tokens": sum(call.get("eval_count", call.get("output_tokens", 0)) or 0 for call in calls),
        "latency_ms": run["latency_ms"],
        "validation_error": run["error"],
        "length_stops": sum(call.get("done_reason") == "length" for call in calls),
        "stages": [call.get("stage", "unified") for call in calls],
        "output_raw": run.get("output_raw"),
    }


def _intent_row(route, sample, category, utterance, run, expected, expected_kind):
    plan = run["plan"]
    actual = _mutation_payload(plan)
    return _common_row(route, sample, category, utterance, run) | {
        "decision_kind": _kind(plan),
        "classification_correct": _kind(plan) == expected_kind if expected_kind else None,
        "payload_correct": actual == expected if expected is not None else None,
        "partial_plan": bool(
            category == "mixed"
            and plan is not None
            and (plan.request is not None or plan.mutation is not None)
        ),
        "mutation": actual,
    }


def _summary(rows):
    result = {}
    for route in ("existing", "unified"):
        selected = [row for row in rows if row["route"] == route]
        result[route] = {
            "requests": len(selected),
            "llm_calls": summarize_latencies([row["llm_calls"] for row in selected]),
            "input_tokens": summarize_latencies([row["input_tokens"] for row in selected]),
            "output_tokens": summarize_latencies([row["output_tokens"] for row in selected]),
            "latency_ms": summarize_latencies([row["latency_ms"] for row in selected]),
            "validation_failures": sum(row["validation_error"] is not None for row in selected),
            "length_stops": sum(row["length_stops"] for row in selected),
            "plan_correct": sum(row.get("plan_correct") is True for row in selected),
            "answer_correct": sum(row.get("answer_correct") is True for row in selected),
            "classification_correct": sum(row.get("classification_correct") is True for row in selected),
            "payload_correct": sum(row.get("payload_correct") is True for row in selected),
            "mutation_as_nonmutation": sum(
                row.get("classification_correct") is False
                and row.get("decision_kind") in {"fact", "none"}
                for row in selected
            ),
            "read_as_mutation": sum(
                row["category"] == "read" and row.get("decision_kind") == "mutation"
                for row in selected
            ),
            "multi_intent_correct": sum(
                row["category"] == "mixed"
                and row.get("decision_kind") == "multi_intent"
                for row in selected
            ),
            "mixed_partial_plans": sum(
                row.get("partial_plan") is True for row in selected
            ),
            "explicit_preview_as_commit": sum(
                row["category"] == "write"
                and (row.get("expected_mutation") or {}).get("mode") == "preview"
                and (row.get("mutation") or {}).get("mode") == "commit"
                for row in selected
            ),
        }
    return result


async def run(args):
    dataset = yaml.safe_load(args.routing_eval.read_text())
    client = ObservedOllama(args.ollama_url, args.model)
    client.reset_observations()
    schema_dir = getattr(args, "schema_dir", None) or (PROJECT_ROOT / "schemas/edge")
    service, context = build_json_fact_service(args.data_dir, schema_dir, client)
    service.planner = LegacyTwoStagePlanner(client, service.engine.schema)
    unified = UnifiedShadowPlanner(client, service.engine.schema)
    rows = []
    try:
        if args.group == "reads":
            cases = regression_cases(service, context)
            if args.utterance:
                missing = set(args.utterance) - {case.utterance for case in cases}
                if missing:
                    raise ValueError(f"Unknown read utterances: {sorted(missing)}")
                cases = [case for case in cases if case.utterance in args.utterance]
            gold = {}
            for case in cases:
                result, _, _, _ = await service.engine.execute(case.expected, context)
                gold[case.case_id] = serialize_fact_result(result)
            await _existing(client, service, context, cases[0].utterance)
            await _unified(unified, context, cases[0].utterance)
            for route in ("existing", "unified"):
                for sample in range(args.repeat):
                    for case in cases:
                        invocation = (
                            await _existing(client, service, context, case.utterance)
                            if route == "existing"
                            else await _unified(unified, context, case.utterance)
                        )
                        rows.append(await _read_row(
                            route, sample, case, invocation, service, context,
                            gold[case.case_id],
                        ))
        else:
            categories = ("write", "mixed", "ambiguous")
            if args.utterance:
                available = {
                    utterance
                    for category in categories
                    for utterance in dataset[category]
                }
                missing = set(args.utterance) - available
                if missing:
                    raise ValueError(f"Unknown intent utterances: {sorted(missing)}")
            first = dataset["write"][0]
            await _existing(client, service, context, first)
            await _unified(unified, context, first)
            for route in ("existing", "unified"):
                for sample in range(args.repeat):
                    for category in categories:
                        gold_map = dataset.get(f"{category}_gold", {})
                        for utterance in dataset[category]:
                            if args.utterance and utterance not in args.utterance:
                                continue
                            expected = (
                                NAMED_WRITE_ADAPTER.validate_python(gold_map[utterance]).model_dump(mode="json")
                                if utterance in gold_map else None
                            )
                            expected_kind = (
                                "mutation" if category == "write"
                                else "multi_intent" if category == "mixed"
                                else None
                            )
                            invocation = (
                                await _existing(client, service, context, utterance)
                                if route == "existing"
                                else await _unified(unified, context, utterance)
                            )
                            rows.append(_intent_row(
                                route, sample, category, utterance, invocation,
                                expected, expected_kind,
                            ))
                            rows[-1]["expected_mutation"] = expected
    finally:
        await client.close()

    prompt = unified_chat_messages(
        [{"role": "user", "content": "我是谁"}], service.engine.schema,
        household_now=context.current_time.isoformat(),
    )
    report = {
        "mode": "shadow_only_no_dispatch",
        "group": args.group,
        "repeat": args.repeat,
        "summary": _summary(rows),
        "rows": rows,
        "unified_contract": {
            "type": "SemanticPlan",
            "branches": ["fact", "mutation", "conversation", "multi_intent"],
            "schema_bytes": len(_compact(unified_output_schema(service.engine.schema)).encode()),
            "message_content_bytes": sum(len(message["content"].encode()) for message in prompt),
            "messages_sha256": hashlib.sha256(_compact(prompt).encode()).hexdigest(),
        },
        "provenance": collect_provenance(
            root=PROJECT_ROOT,
            eval_path=(
                PROJECT_ROOT / "benchmarks/planner_prompt_compression.yaml"
                if args.group == "reads" else args.routing_eval
            ),
            ollama_url=args.ollama_url,
            ollama_model=args.model,
            backend="unified-shadow-frozen-json",
            frozen_time=context.current_time,
            warmup=2,
            repeat=args.repeat,
            verified_cold=False,
            data_dir=args.data_dir,
            schema_dir=PROJECT_ROOT / "schemas",
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"summary": report["summary"], "unified_contract": report["unified_contract"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("reads", "intents"), required=True)
    parser.add_argument("--routing-eval", type=Path, default=PROJECT_ROOT / "benchmarks/mutation_routing.yaml")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--ollama-url", default="http://ollama:11434")
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--utterance", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeat < 1 or args.output.exists():
        parser.error("Require positive repeats and a new output path")
    asyncio.run(run(args))
