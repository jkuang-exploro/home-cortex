#!/usr/bin/env python3
"""Reproducible semantic-planner probe over the JSON graph."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from scripts import PROJECT_ROOT

from home_cortex.ollama import OllamaService, _semantic_planner_examples, planner_system_prompt
from scripts.benchmarks.semantic_planner_benchmark import (
    DEFAULT_EVAL_PATH,
    build_json_fact_service,
    collect_provenance,
    load_probe_dataset,
    load_semantic_eval_cases,
    run_tier1_probe,
)

ROOT = PROJECT_ROOT


async def run(args: argparse.Namespace) -> dict:
    dataset = load_probe_dataset(args.eval)
    if args.suite:
        dataset = replace(dataset, cases=load_semantic_eval_cases(args.eval))
    ollama = OllamaService(args.ollama_url, args.model)
    service, context = build_json_fact_service(args.data_dir, args.schema_dir, ollama)

    def progress(row):
        if args.progress and args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.with_suffix(".jsonl").open("a") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    try:
        report = await run_tier1_probe(
            service,
            context,
            dataset,
            warmup=args.warmup,
            repeat=args.repeat,
            verified_cold=args.verified_cold,
            on_result=progress,
        )
    finally:
        await ollama.close()
    report["provenance"] = collect_provenance(
        root=ROOT,
        eval_path=args.eval,
        ollama_url=args.ollama_url,
        ollama_model=args.model,
        backend="json",
        frozen_time=dataset.frozen_time,
        warmup=args.warmup,
        repeat=args.repeat,
        verified_cold=args.verified_cold,
        data_dir=args.data_dir,
        schema_dir=args.schema_dir,
    )
    contract = {
        "examples": _semantic_planner_examples(),
        "capabilities": service.engine.schema.planner_capability_payload(),
        "output_schema": service.engine.schema.planner_output_schema(),
        "system_prompt": planner_system_prompt(
            service.engine.schema.planner_capability_payload()
        ),
    }
    report["planner_contract"] = contract
    report["provenance"]["planner_contract_sha256"] = hashlib.sha256(
        json.dumps(contract, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    report["command"] = sys.argv
    report["model"] = args.model
    report["collected_at"] = datetime.now().isoformat()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps({
        "mode": report["mode"],
        "dataset_size": report["dataset_size"],
        "warmup": report["warmup"],
        "repeat": report["repeat"],
        "verified_cold": report["verified_cold"],
        "scores": report["scores"],
        "scores_all_measured_samples": report["scores_all_measured_samples"],
        "planner_latency_ms": report["planner_latency_ms"],
        "end_to_end_latency_ms": report["end_to_end_latency_ms"],
        "accuracy_by_capability": report["accuracy_by_capability"],
        "failure_table": report["failure_table"],
        "provenance": report["provenance"],
        "output": str(args.output) if args.output else None,
    }, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--schema-dir", type=Path, default=ROOT / "schemas" / "edge")
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Append each completed row to an adjacent .jsonl checkpoint.",
    )
    parser.add_argument(
        "--suite",
        action="store_true",
        help="Measure every fixed suite case on each pass.",
    )
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Requests excluded from reported latency (first is first_request).",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=5,
        help="Measured full passes over the 20 probe cases after warm-up.",
    )
    parser.add_argument(
        "--verified-cold",
        action="store_true",
        help="Label the first request verified_cold; only set after a recorded unload.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the full per-case JSON report to this path.",
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat < 1:
        parser.error("--warmup must be >= 0 and --repeat must be >= 1")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
