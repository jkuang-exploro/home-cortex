#!/usr/bin/env python3
"""Focused named-item location IR probe. Synthetic fixture only."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from home_cortex.ollama import OllamaService
from home_cortex.semantic_ir import SemanticFactRequest, SemanticPlannerFailure
from home_cortex.semantic_planner_benchmark import (
    build_json_fact_service,
    normalize_semantic_request,
)

ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ("fridge", "冰箱在哪里", "冰箱"),
    ("milk", "牛奶在哪里", "牛奶"),
    ("washer", "洗衣机在哪里", "洗衣机"),
    ("box", "收纳盒在哪里？", "收纳盒"),
    ("vase", "花瓶在哪里？", "花瓶"),
    ("kettle", "Where is the kettle?", "kettle"),
)


def matches(actual: dict | None, value: str) -> bool:
    if not actual:
        return False
    subject = actual.get("subject") or {}
    relations = [
        step.get("relation") for step in subject.get("path") or () if isinstance(step, dict)
    ]
    return (
        actual.get("operation") in {"select", "resolve_reference"}
        and subject.get("kind") == "named_entity"
        and subject.get("entity_type") == "item"
        and subject.get("value") == value
        and relations == ["location"]
    )


async def run(args: argparse.Namespace) -> dict:
    ollama = OllamaService(args.ollama_url, args.model)
    service, context = build_json_fact_service(
        ROOT / "tests/static_test_data",
        ROOT / "schemas/edge",
        ollama,
    )
    rows = []
    try:
        for _ in range(args.warmup):
            try:
                await service.planner.plan(
                    [{"role": "user", "content": "花瓶在哪里？"}], context
                )
            except SemanticPlannerFailure:
                pass
        for sample in range(args.repeat):
            for case_id, utterance, value in CASES:
                raw = None
                actual = None
                validation = None
                try:
                    outcome = await service.planner.plan(
                        [{"role": "user", "content": utterance}], context
                    )
                    raw = outcome.diagnostics.output_raw
                    validation = outcome.diagnostics.validation_result
                    if outcome.plan.request is not None:
                        actual = normalize_semantic_request(outcome.plan.request)
                except SemanticPlannerFailure as error:
                    raw = error.diagnostics.output_raw
                    validation = error.diagnostics.validation_result or error.diagnostics.failure_detail
                    payload = error.diagnostics.normalized_plan
                    request = payload.get("request") if isinstance(payload, dict) else None
                    if isinstance(request, dict):
                        try:
                            actual = normalize_semantic_request(
                                SemanticFactRequest.model_validate(request)
                            )
                        except Exception:
                            actual = request
                rows.append({
                    "id": case_id,
                    "sample_index": sample,
                    "utterance": utterance,
                    "match": matches(actual, value),
                    "validation": validation,
                    "actual": actual,
                    "raw": raw,
                    "raw_subject": (
                        (raw.get("request") or {}).get("subject")
                        if isinstance(raw, dict)
                        else None
                    ),
                })
    finally:
        await ollama.close()
    first = [row for row in rows if row["sample_index"] == 0]
    return {
        "first_pass": {
            "correct": sum(row["match"] for row in first),
            "scored": len(first),
        },
        "all_samples": {
            "correct": sum(row["match"] for row in rows),
            "scored": len(rows),
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ollama-url", default="http://ollama:11434")
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = asyncio.run(run(args))
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {key: report[key] for key in ("first_pass", "all_samples")}
    summary["failures"] = [
        {
            "id": row["id"],
            "utterance": row["utterance"],
            "sample_index": row["sample_index"],
            "validation": row["validation"],
            "raw_subject": row["raw_subject"],
        }
        for row in report["rows"]
        if not row["match"]
    ]
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
