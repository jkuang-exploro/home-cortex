#!/usr/bin/env python3
"""Trace interpretation, execution, and rendering on synthetic data.

Does not change prompts, ontology, executor, or scoring. Isolated GPU runs
must keep the already-resident model and num_ctx=8192.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from home_cortex.ollama import (
    PLANNER_KEEP_ALIVE,
    PLANNER_NUM_CTX,
    PLANNER_NUM_PREDICT,
    PLANNER_SEED,
    OllamaService,
    planner_chat_messages,
)
from home_cortex.semantic_conversation import SemanticConversationService
from home_cortex.fact_renderer import FactRenderer
from home_cortex.semantic_ir import SemanticFactRequest, SemanticPlannerFailure
from home_cortex.semantic_planner_benchmark import (
    build_json_fact_service,
    normalize_semantic_request,
    primary_entity_ids,
    semantic_mismatch_reason,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "benchmarks/fixtures/semantic-contract"
SCHEMA = ROOT / "schemas/edge"
MODEL = "qwen3.5:9b"
REPEATS = 3
WARMUP = 1


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dump_request(request: SemanticFactRequest | None) -> dict[str, Any] | None:
    if request is None:
        return None
    return normalize_semantic_request(request)


def expected_request(schema, spec: dict[str, Any]) -> SemanticFactRequest:
    raw = {"requires_fact": True, "request": spec}
    return SemanticFactRequest.model_validate(schema.expand_planner_concepts(raw)["request"])


def result_view(result) -> dict[str, Any]:
    return {
        "status": result.status,
        "value": result.value if not isinstance(result.value, (list, dict)) else compact(result.value)[:500],
        "entity_ids": list(primary_entity_ids(result)),
        "shape": result.shape,
        "unit": result.unit,
        "missing": list(result.missing_requirements),
    }


def render_omits_plan_conditions(request: SemanticFactRequest, text: str) -> list[str]:
    omitted = []
    predicates = {item.predicate for item in request.filters if item.predicate}
    if "adult" in predicates and "成年" not in text and "adult" not in text.lower():
        omitted.append("adult")
    if "minor" in predicates and "未成年" not in text and "孩子" not in text and "minor" not in text.lower():
        omitted.append("minor")
    genders = set()
    for item in request.filters:
        if item.property == "gender":
            genders.add(item.value)
    for step in request.subject.path:
        for item in step.filters:
            if item.property == "gender":
                genders.add(item.value)
    if "male" in genders and "男" not in text and "丈夫" not in text and "父亲" not in text and "male" not in text.lower():
        omitted.append("gender=male")
    if "female" in genders and "女" not in text and "妻子" not in text and "母亲" not in text and "女儿" not in text and "female" not in text.lower():
        omitted.append("gender=female")
    return omitted


def attribute(
    *,
    expected: SemanticFactRequest | None,
    raw: Any,
    actual: SemanticFactRequest | None,
    validation: str | None,
    executed: dict[str, Any] | None,
    gold: dict[str, Any] | None,
    text: str | None,
    omitted: list[str],
) -> str:
    if expected is None:
        return "unknown"
    if not isinstance(raw, dict):
        return "interpretation"
    if raw.get("requires_fact") is not True or actual is None:
        return "interpretation"
    if validation not in {None, "VALID"}:
        expected_norm = dump_request(expected)
        actual_norm = dump_request(actual)
        if actual_norm != expected_norm:
            return "interpretation"
        return "validation"
    expected_norm = dump_request(expected)
    actual_norm = dump_request(actual)
    if actual_norm != expected_norm:
        return "interpretation"
    if executed is None or gold is None:
        return "unknown"
    if executed.get("status") != gold.get("status") or executed.get("entity_ids") != gold.get("entity_ids"):
        return "execution"
    if executed.get("value") != gold.get("value") or executed.get("unit") != gold.get("unit"):
        return "execution"
    if omitted:
        return "rendering"
    return "none"


async def gold_execute(service, request: SemanticFactRequest, context) -> dict[str, Any]:
    result, *_ = await service.engine.execute(request, context)
    text = FactRenderer().render(request, result, context)
    return {"result": result_view(result), "text": text}


async def trace_one(conversations, service, messages, context, expected: SemanticFactRequest | None) -> dict[str, Any]:
    gold = await gold_execute(service, expected, context) if expected is not None else None
    started = perf_counter()
    try:
        answer = await conversations.try_answer(messages, context=context)
        wall_ms = (perf_counter() - started) * 1000
        diagnostics = None if answer is None else answer.planner_diagnostics
        raw = None if diagnostics is None else diagnostics.output_raw
        actual = None if answer is None else answer.request
        validation = None if diagnostics is None else diagnostics.validation_result
        executed = None if answer is None else result_view(answer.result)
        text = None if answer is None else answer.text
        omitted = render_omits_plan_conditions(actual, text or "") if actual is not None else []
        layer = attribute(
            expected=expected, raw=raw, actual=actual, validation=validation,
            executed=executed, gold=None if gold is None else gold["result"],
            text=text, omitted=omitted,
        )
        reason = None
        if expected is not None and actual is not None:
            reason = semantic_mismatch_reason(dump_request(actual), dump_request(expected))
            if reason == "PLAN_MISMATCH" and dump_request(actual) == dump_request(expected):
                reason = None
        return {
            "wall_ms": wall_ms,
            "requires_fact": None if raw is None else raw.get("requires_fact"),
            "raw_ir": raw,
            "expanded_request": dump_request(actual),
            "expected_request": dump_request(expected),
            "validation": validation,
            "mismatch": None if layer == "none" else reason,
            "attempts": None if diagnostics is None else diagnostics.attempt_count,
            "executed": executed,
            "gold_executed": None if gold is None else gold["result"],
            "rendered": text,
            "gold_rendered": None if gold is None else gold["text"],
            "render_omitted_conditions": omitted,
            "layer": layer,
            "semantic_correct": layer == "none",
            "found_only": executed is not None and executed.get("status") == "found",
        }
    except SemanticPlannerFailure as error:
        wall_ms = (perf_counter() - started) * 1000
        raw = error.diagnostics.output_raw
        return {
            "wall_ms": wall_ms,
            "requires_fact": None if not isinstance(raw, dict) else raw.get("requires_fact"),
            "raw_ir": raw,
            "expanded_request": None,
            "expected_request": dump_request(expected),
            "validation": error.diagnostics.validation_result,
            "mismatch": "NO_SEMANTIC_PLAN",
            "attempts": error.diagnostics.attempt_count,
            "executed": None,
            "gold_executed": None if gold is None else gold["result"],
            "rendered": None,
            "gold_rendered": None if gold is None else gold["text"],
            "render_omitted_conditions": [],
            "layer": "interpretation",
            "semantic_correct": False,
            "found_only": False,
            "failure_detail": error.diagnostics.failure_detail,
        }


def specs(schema):
    self_ref = {"kind": "self", "entity_type": "person"}
    household = {
        "kind": "current_household",
        "entity_type": "address",
        "path": [{"concept": "member"}],
    }
    cases = {
        "who_am_i": {"operation": "resolve_reference", "subject": self_ref},
        "my_birthday_countdown": {
            "operation": "annual_occurrence", "subject": self_ref,
            "property": "birth_date", "property_source": "entity", "mode": "days",
        },
        "my_birth_date": {
            "operation": "select", "subject": self_ref,
            "property": "birth_date", "property_source": "entity",
        },
        "my_father": {
            "operation": "resolve_reference",
            "subject": {**self_ref, "path": [{"concept": "father"}]},
        },
        "my_father_in_law": {
            "operation": "resolve_reference",
            "subject": {**self_ref, "path": [{"concept": "father_in_law"}]},
        },
        "wife_birthday_countdown": {
            "operation": "annual_occurrence",
            "subject": {**self_ref, "path": [{"concept": "wife"}]},
            "property": "birth_date", "property_source": "entity", "mode": "days",
        },
        "daughter_birthday_countdown": {
            "operation": "annual_occurrence",
            "subject": {**self_ref, "path": [{"concept": "daughter"}]},
            "property": "birth_date", "property_source": "entity", "mode": "days",
        },
        "wife_birth_date": {
            "operation": "select",
            "subject": {**self_ref, "path": [{"concept": "wife"}]},
            "property": "birth_date", "property_source": "entity",
        },
        "household_list": {"operation": "select", "subject": household},
        "male_count": {
            "operation": "count", "subject": household,
            "filters": [{"property": "gender", "operator": "eq", "value": "male", "source": "entity"}],
        },
        "female_count": {
            "operation": "count", "subject": household,
            "filters": [{"property": "gender", "operator": "eq", "value": "female", "source": "entity"}],
        },
        "adult_count": {
            "operation": "count", "subject": household,
            "filters": [{"predicate": "adult"}],
        },
    }
    return {name: expected_request(schema, spec) for name, spec in cases.items()}


REPORTED = [
    ("who_am_i", "我是谁"),
    ("my_birthday_countdown", "我什么时候过生日"),
    ("my_birth_date", "我的生日是哪天"),
    ("my_father", "我父亲是谁"),
    ("my_father_in_law", "我岳父是谁"),
    ("wife_birthday_countdown", "我老婆什么时候过生日"),
    ("daughter_birthday_countdown", "我女儿什么时候过生日"),
    ("wife_birth_date", "我老婆生日是哪天"),
]

HOUSEHOLD = [
    ("household_list", "家里都有谁"),
    ("household_list", "我家里都有谁"),
]

GENDER = [
    ("male_count", "家里有几个男的"),
    ("female_count", "家里有几个女的"),
    ("male_count", "有几个男的"),
    ("adult_count", "家里有几个成年人"),
]


async def run(args) -> None:
    model = OllamaService(args.ollama_url, args.model)
    service, context = build_json_fact_service(FIXTURE, SCHEMA, model)
    context = replace(
        context,
        caller_entity_id="person:a",
        household_id="address:fictional",
        locale="zh",
    )
    expected = specs(service.engine.schema)
    conversations = SemanticConversationService(service)
    capabilities = service.engine.schema.planner_capability_payload()
    prompt = planner_chat_messages(
        [{"role": "user", "content": "我是谁"}],
        capabilities,
        household_now=context.current_time.isoformat(),
    )
    report: dict[str, Any] = {
        "objective": "Layer-attribute reported failures; no behavior change",
        "model": args.model,
        "request_settings": {
            "think": False, "temperature": 0, "seed": 0,
            "num_ctx": PLANNER_NUM_CTX, "num_predict": PLANNER_NUM_PREDICT,
            "keep_alive": PLANNER_KEEP_ALIVE,
        },
        "clock": context.current_time.isoformat(),
        "speaker": context.caller_entity_id,
        "household": context.household_id,
        "locale": context.locale,
        "prompt_sha256": sha256_bytes(compact(prompt).encode()),
        "capability_sha256": sha256_bytes(compact(capabilities).encode()),
        "output_schema_sha256": sha256_bytes(compact(service.engine.schema.planner_output_schema()).encode()),
        "repeats": REPEATS,
        "warmup": WARMUP,
        "rows": [],
    }
    try:
        for _ in range(WARMUP):
            await conversations.try_answer(
                [{"role": "user", "content": "我是谁"}],
                context=replace(context, conversation_id=uuid4().hex),
            )
        for repeat in range(REPEATS):
            for case_id, utterance in [*REPORTED, *HOUSEHOLD, *GENDER]:
                row = await trace_one(
                    conversations, service,
                    [{"role": "user", "content": utterance}],
                    replace(context, conversation_id=uuid4().hex),
                    expected[case_id],
                )
                row.update(group="standalone", case_id=case_id, utterance=utterance, repeat=repeat, history=[])
                report["rows"].append(row)
            history: list[dict[str, str]] = []
            conv = uuid4().hex
            for case_id, utterance in REPORTED:
                history.append({"role": "user", "content": utterance})
                row = await trace_one(
                    conversations, service, history,
                    replace(context, conversation_id=conv),
                    expected[case_id],
                )
                row.update(
                    group="reported_conversation", case_id=case_id, utterance=utterance,
                    repeat=repeat, history=[item["content"] for item in history[:-1]],
                )
                report["rows"].append(row)
            conv = uuid4().hex
            leak_steps = [
                ("adult_count", "家里有几个成年人"),
                ("male_count", "家里有几个男的"),
            ]
            history = []
            for case_id, utterance in leak_steps:
                history.append({"role": "user", "content": utterance})
                row = await trace_one(
                    conversations, service, history,
                    replace(context, conversation_id=conv),
                    expected[case_id],
                )
                row.update(
                    group="adult_then_male", case_id=case_id, utterance=utterance,
                    repeat=repeat, history=[item["content"] for item in history[:-1]],
                )
                report["rows"].append(row)
            conv = uuid4().hex
            leak_steps = [
                ("my_father_in_law", "我岳父是谁"),
                ("wife_birthday_countdown", "我老婆什么时候过生日"),
            ]
            history = []
            for case_id, utterance in leak_steps:
                history.append({"role": "user", "content": utterance})
                row = await trace_one(
                    conversations, service, history,
                    replace(context, conversation_id=conv),
                    expected[case_id],
                )
                row.update(
                    group="inlaw_then_wife", case_id=case_id, utterance=utterance,
                    repeat=repeat, history=[item["content"] for item in history[:-1]],
                )
                report["rows"].append(row)
    finally:
        await model.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n")
    print(compact({"wrote": str(args.output), "rows": len(report["rows"])}))


def summarize(path: Path, output: Path) -> None:
    data = json.loads(path.read_text())
    groups: dict[str, dict[str, Any]] = {}
    for row in data["rows"]:
        key = f"{row['group']}/{row['case_id']}/{row['utterance']}"
        bucket = groups.setdefault(key, {
            "group": row["group"], "case_id": row["case_id"], "utterance": row["utterance"],
            "n": 0, "semantic_correct": 0, "found_only": 0, "layers": {}, "traces": [],
        })
        bucket["n"] += 1
        bucket["semantic_correct"] += int(row["semantic_correct"])
        bucket["found_only"] += int(bool(row.get("found_only")))
        layer = row["layer"]
        bucket["layers"][layer] = bucket["layers"].get(layer, 0) + 1
        bucket["traces"].append({
            "repeat": row["repeat"],
            "layer": layer,
            "mismatch": row.get("mismatch"),
            "validation": row.get("validation"),
            "raw_subject": None if not isinstance(row.get("raw_ir"), dict) else (row["raw_ir"].get("request") or {}).get("subject"),
            "expanded": row.get("expanded_request"),
            "executed": row.get("executed"),
            "gold_executed": row.get("gold_executed"),
            "rendered": row.get("rendered"),
            "gold_rendered": row.get("gold_rendered"),
            "render_omitted_conditions": row.get("render_omitted_conditions"),
            "history": row.get("history"),
        })
    failures = [item for item in groups.values() if item["semantic_correct"] < item["n"]]
    output.write_text(json.dumps({
        "model": data["model"],
        "clock": data["clock"],
        "prompt_sha256": data["prompt_sha256"],
        "capability_sha256": data["capability_sha256"],
        "output_schema_sha256": data["output_schema_sha256"],
        "request_settings": data["request_settings"],
        "repeats": data["repeats"],
        "cases": list(groups.values()),
        "failures": failures,
        "scoring": "Exact expanded request vs expected; found/value match is not semantic correctness",
    }, ensure_ascii=False, indent=2) + "\n")
    print(compact({
        "cases": len(groups),
        "failing_case_keys": [f"{item['group']}/{item['utterance']}" for item in failures],
        "wrote": str(output),
    }))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--ollama-url", default="http://ollama:11434")
    run_parser.add_argument("--model", default=MODEL)
    run_parser.add_argument("--output", type=Path, required=True)
    score_parser = commands.add_parser("summarize")
    score_parser.add_argument("--results", type=Path, required=True)
    score_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        os.environ.setdefault("PYTHONHASHSEED", "0")
        asyncio.run(run(args))
    else:
        summarize(args.results, args.output)
