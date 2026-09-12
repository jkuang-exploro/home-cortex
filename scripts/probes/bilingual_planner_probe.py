#!/usr/bin/env python3
"""In-situ bilingual planner probe on an isolated package.

Uses the deployed Ollama model. Does not read live household data or secrets.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from scripts import PROJECT_ROOT
from time import perf_counter
from typing import Any

from home_cortex.ollama import (
    PLANNER_KEEP_ALIVE,
    PLANNER_NUM_CTX,
    OllamaService,
    _PLANNER_INSTRUCTIONS,
    _semantic_planner_examples,
    planner_chat_messages,
)
from home_cortex.semantic_ir import SemanticFactRequest, SemanticPlannerFailure
from scripts.benchmarks.semantic_planner_benchmark import (
    build_json_fact_service,
    collect_provenance,
    load_bilingual_dataset,
    normalize_semantic_request,
    summarize_latencies,
)

ROOT = PROJECT_ROOT
FIXTURE = ROOT / "benchmarks/fixtures/semantic-contract"
SCHEMA = ROOT / "schemas/edge"
DATASET = ROOT / "benchmarks/semantic_planner_bilingual.yaml"
MODEL = "qwen3.5:9b"


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def classify(text: str) -> str:
    has_cjk = any("\u4e00" <= char <= "\u9fff" for char in text)
    has_latin = any("A" <= char <= "Z" or "a" <= char <= "z" for char in text)
    if has_cjk and has_latin:
        return "mixed"
    if has_cjk:
        return "zh"
    if has_latin:
        return "en"
    return "other"


def strip_named_values(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {
            key: strip_named_values(item)
            for key, item in payload.items()
            if not (key == "value" and payload.get("kind") == "named_entity")
        }
    if isinstance(payload, list):
        return [strip_named_values(item) for item in payload]
    return payload


def canonical(schema, request: dict[str, Any] | None, *, ignore_named_value: bool = False):
    if request is None:
        return None
    expanded = schema.expand_planner_concepts({"requires_fact": True, "request": request})
    normalized = normalize_semantic_request(
        SemanticFactRequest.model_validate(expanded["request"])
    )
    return strip_named_values(normalized) if ignore_named_value else normalized


def path_concepts(request: dict[str, Any] | None) -> list[str]:
    if not request:
        return []
    subject = request.get("subject") or {}
    return [
        step.get("concept") or step.get("relation")
        for step in subject.get("path") or ()
        if isinstance(step, dict)
    ]


def iter_cases(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    plans = {pair["id"]: pair["expected"] for pair in dataset["pairs"]}
    cases: list[dict[str, Any]] = []
    for pair in dataset["pairs"]:
        expected = pair["expected"]
        ignore = bool(pair.get("ignore_named_value"))
        for language, utterance in (("zh", pair["zh"]), ("en", pair["en"])):
            cases.append({
                "id": f"{pair['id']}:{language}",
                "pair_id": pair["id"],
                "kind": "pair",
                "category": pair["category"],
                "language": language,
                "utterance": utterance,
                "expected": expected,
                "ignore_named_value": ignore,
                "requires_fact": True,
            })
    for item in dataset["mixed"]:
        expected = item.get("expected") or plans[item["equivalent_to"]]
        cases.append({
            "id": item["id"],
            "pair_id": item.get("equivalent_to") or item["id"],
            "kind": "mixed",
            "category": item["category"],
            "language": "mixed",
            "utterance": item["utterance"],
            "expected": expected,
            "ignore_named_value": bool(item.get("ignore_named_value")),
            "requires_fact": True,
        })
    for item in dataset["stress"]:
        expected = item.get("expected")
        if expected is None and item.get("plan") in plans:
            expected = plans[item["plan"]]
        cases.append({
            "id": item["id"],
            "pair_id": item.get("plan") or item["id"],
            "kind": "stress",
            "category": item["category"],
            "language": item.get("language") or classify(item["utterance"]),
            "utterance": item["utterance"],
            "expected": expected,
            "ignore_named_value": bool(item.get("ignore_named_value")),
            "requires_fact": item.get("requires_fact", True),
        })
    return cases


async def plan_case(service, context, schema, case: dict[str, Any], history: list[str]):
    messages = [{"role": "user", "content": utterance} for utterance in (*history, case["utterance"])]
    started = perf_counter()
    raw = None
    actual = None
    requires_fact = None
    validation = None
    runtime: dict[str, Any] = {}
    try:
        outcome = await service.planner.plan(messages, context)
        runtime = {
            "prompt_eval_count": outcome.diagnostics.prompt_eval_count,
            "prompt_eval_duration_ms": outcome.diagnostics.prompt_eval_duration_ms,
            "eval_count": outcome.diagnostics.eval_count,
            "eval_duration_ms": outcome.diagnostics.eval_duration_ms,
            "latency_ms": outcome.diagnostics.latency_ms,
        }
        raw = outcome.diagnostics.output_raw
        validation = outcome.diagnostics.validation_result
        requires_fact = outcome.plan.requires_fact
        if outcome.plan.request is not None:
            actual = normalize_semantic_request(outcome.plan.request)
    except SemanticPlannerFailure as error:
        runtime = {
            "prompt_eval_count": getattr(error.diagnostics, "prompt_eval_count", None),
            "prompt_eval_duration_ms": getattr(error.diagnostics, "prompt_eval_duration_ms", None),
            "eval_count": getattr(error.diagnostics, "eval_count", None),
            "eval_duration_ms": getattr(error.diagnostics, "eval_duration_ms", None),
            "latency_ms": getattr(error.diagnostics, "latency_ms", None),
        }
        raw = error.diagnostics.output_raw
        validation = error.diagnostics.validation_result
        if isinstance(raw, dict):
            requires_fact = raw.get("requires_fact")
            request = None
            if isinstance(error.diagnostics.normalized_plan, dict):
                request = error.diagnostics.normalized_plan.get("request")
            elif isinstance(raw.get("request"), dict):
                request = raw.get("request")
            if isinstance(request, dict):
                try:
                    actual = canonical(schema, request, ignore_named_value=False)
                except Exception:
                    actual = request
    latency_ms = (perf_counter() - started) * 1000
    ignore = bool(case.get("ignore_named_value"))
    expected = None
    if case.get("requires_fact") is False:
        match = requires_fact is False and actual is None
    else:
        expected = canonical(schema, case["expected"], ignore_named_value=ignore)
        compared = strip_named_values(actual) if ignore and actual is not None else actual
        match = compared == expected
    forbidden = []
    raw_request = raw.get("request") if isinstance(raw, dict) else None
    raw_subject = raw_request.get("subject") if isinstance(raw_request, dict) else {}
    if case.get("id", "").startswith("who_am_i") or case["utterance"] == "Who am I?":
        if (raw_subject or {}).get("kind") == "assistant":
            forbidden.append("who_am_i_became_assistant")
    if case["utterance"] == "Who are you?":
        if (raw_subject or {}).get("kind") == "self":
            forbidden.append("who_are_you_became_self")
    if case["utterance"] in {"Who is in my household?", "Who is in our household?", "我家里都有谁？"}:
        if (raw_subject or {}).get("kind") == "self" and "member" in path_concepts(raw_request):
            forbidden.append("household_used_self_member")
    if case["utterance"] == "Who is my son?" and "son" not in path_concepts(raw_request) and "child" in path_concepts(raw_request):
        forbidden.append("son_simplified_to_child")
    if "林青" in case["utterance"] and isinstance(raw_subject, dict):
        if raw_subject.get("kind") == "named_entity" and raw_subject.get("value") not in {None, "林青"}:
            forbidden.append("named_entity_translated")
    return {
        "id": case["id"],
        "pair_id": case["pair_id"],
        "kind": case["kind"],
        "category": case["category"],
        "language": case["language"],
        "utterance": case["utterance"],
        "match": bool(match),
        "requires_fact": requires_fact,
        "validation": validation,
        "actual": actual,
        "expected": expected,
        "raw": raw,
        "forbidden": forbidden,
        "wall_ms": latency_ms,
        **runtime,
    }


def rate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if "match" in row]
    correct = sum(1 for row in scored if row["match"])
    return {
        "correct": correct,
        "scored": len(scored),
        "rate": round(correct / len(scored), 4) if scored else None,
    }


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return round(ordered[index], 3)


async def measure_prompt_components(ollama_url: str, model: str, service) -> dict[str, Any]:
    from ollama import AsyncClient

    parts = {
        "instructions": _PLANNER_INSTRUCTIONS,
        "examples": "".join(message["content"] for message in _semantic_planner_examples()),
        "capabilities": compact(service.engine.schema.planner_capability_payload()),
    }
    client = AsyncClient(host=ollama_url)
    rows = []
    try:
        for name, text in parts.items():
            response = await client.generate(
                model=model,
                prompt=text,
                raw=True,
                stream=False,
                think=False,
                keep_alive=PLANNER_KEEP_ALIVE,
                options={"num_predict": 1, "num_ctx": PLANNER_NUM_CTX, "temperature": 0, "seed": 0},
            )
            rows.append({
                "component": name,
                "utf8_bytes": len(text.encode()),
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "raw_prompt_tokens": response.prompt_eval_count,
            })
    finally:
        await client.close()
    built = planner_chat_messages(
        [{"role": "user", "content": "Who am I?"}],
        service.engine.schema.planner_capability_payload(),
        household_now="2026-09-03T12:00:00-07:00",
    )
    return {
        "example_user_count": sum(1 for message in _semantic_planner_examples() if message["role"] == "user"),
        "components": rows,
        "chat_prefix_messages": sum(1 for message in built if message["role"] != "user" or message["content"] != "Who am I?"),
        "instruction_chars": len(_PLANNER_INSTRUCTIONS),
        "example_chars": sum(len(message["content"]) for message in _semantic_planner_examples()),
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    ollama = OllamaService(args.ollama_url, args.model)
    service, context = build_json_fact_service(FIXTURE, SCHEMA, ollama)
    schema = service.engine.schema
    dataset = load_bilingual_dataset(DATASET)
    cases = iter_cases(dataset)
    prompt = await measure_prompt_components(args.ollama_url, args.model, service)
    rows: list[dict[str, Any]] = []
    try:
        for _ in range(args.warmup):
            await plan_case(service, context, schema, cases[0], [])
        for sample in range(args.repeat):
            for case in cases:
                rows.append({**await plan_case(service, context, schema, case, []), "sample_index": sample})
            for sequence in dataset["discourse"]:
                history: list[str] = []
                for index, turn in enumerate(sequence["turns"]):
                    case = {
                        "id": f"{sequence['id']}:t{index}",
                        "pair_id": sequence["id"],
                        "kind": "discourse",
                        "category": sequence["category"],
                        "language": classify(turn["utterance"]),
                        "utterance": turn["utterance"],
                        "expected": turn["expected"],
                        "ignore_named_value": False,
                        "requires_fact": True,
                    }
                    row = await plan_case(service, context, schema, case, history)
                    row["sample_index"] = sample
                    rows.append(row)
                    history.append(turn["utterance"])
            for item in dataset["negatives"]:
                case = {
                    "id": item["id"],
                    "pair_id": item["id"],
                    "kind": "negative",
                    "category": "negative",
                    "language": classify(item["utterance"]),
                    "utterance": item["utterance"],
                    "expected": item.get("expected") or {
                        "operation": "resolve_reference",
                        "subject": {
                            "kind": item.get("expected_subject_kind", "self"),
                            "entity_type": "person",
                        },
                    },
                    "ignore_named_value": False,
                    "requires_fact": True,
                }
                row = await plan_case(service, context, schema, case, [])
                row["sample_index"] = sample
                rows.append(row)
    finally:
        await ollama.close()

    measured = [row for row in rows if row.get("sample_index") == 0] if args.repeat else rows
    by_language = defaultdict(list)
    by_kind = defaultdict(list)
    for row in measured:
        by_language[row["language"]].append(row)
        by_kind[row["kind"]].append(row)
    pair_ids = sorted({row["pair_id"] for row in measured if row["kind"] == "pair"})
    parity = []
    for pair_id in pair_ids:
        zh = next((row for row in measured if row["id"] == f"{pair_id}:zh"), None)
        en = next((row for row in measured if row["id"] == f"{pair_id}:en"), None)
        if zh is None or en is None:
            continue
        ignore = pair_id == "storage_box"
        zh_ir = strip_named_values(zh["actual"]) if ignore else zh["actual"]
        en_ir = strip_named_values(en["actual"]) if ignore else en["actual"]
        parity.append({
            "pair_id": pair_id,
            "both_correct": bool(zh["match"] and en["match"]),
            "equivalent_actual": zh_ir == en_ir and zh_ir is not None,
            "zh_match": zh["match"],
            "en_match": en["match"],
        })
    latencies = [float(row["wall_ms"]) for row in measured if row.get("wall_ms") is not None]
    prompt_tokens = [row["prompt_eval_count"] for row in measured if row.get("prompt_eval_count")]
    eval_counts = [row["eval_count"] for row in measured if row.get("eval_count")]
    summary = {
        "mode": "bilingual_planner_probe",
        "model": args.model,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "prompt": prompt,
        "scores": {
            "overall": rate(measured),
            "all_samples": rate(rows),
            "zh": rate(by_language["zh"]),
            "en": rate(by_language["en"]),
            "mixed": rate(by_language["mixed"]),
            "pairs": rate(by_kind["pair"]),
            "discourse": rate(by_kind["discourse"]),
            "negatives": rate(by_kind["negative"]),
        },
        "parity": {
            "both_correct": rate([
                {"match": item["both_correct"]} for item in parity
            ]),
            "equivalent_actual": rate([
                {"match": item["equivalent_actual"]} for item in parity
            ]),
            "pairs": parity,
        },
        "latency_ms": {
            **summarize_latencies(latencies),
            "p50": percentile(latencies, 0.5),
            "p95": percentile(latencies, 0.95),
        },
        "tokens": {
            "prompt_eval_count": {
                "n": len(prompt_tokens),
                "median": statistics.median(prompt_tokens) if prompt_tokens else None,
            },
            "eval_count": {
                "n": len(eval_counts),
                "median": statistics.median(eval_counts) if eval_counts else None,
            },
        },
        "forbidden": [
            {"id": row["id"], "utterance": row["utterance"], "flags": row["forbidden"]}
            for row in measured if row["forbidden"]
        ],
        "failures": [
            {
                "id": row["id"],
                "utterance": row["utterance"],
                "language": row["language"],
                "validation": row["validation"],
                "actual": row["actual"],
                "expected": row["expected"],
            }
            for row in measured if not row["match"]
        ],
        "provenance": collect_provenance(
            root=ROOT,
            eval_path=DATASET,
            ollama_url=args.ollama_url,
            ollama_model=args.model,
            backend="json",
            frozen_time=context.current_time,
            warmup=args.warmup,
            repeat=args.repeat,
            verified_cold=False,
            data_dir=FIXTURE,
            schema_dir=SCHEMA,
        ),
        "queries": rows,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ollama-url", default="http://ollama:11434")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        key: report[key]
        for key in ("mode", "model", "prompt", "scores", "parity", "latency_ms", "tokens", "forbidden")
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
