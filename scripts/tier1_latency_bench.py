#!/usr/bin/env python3
"""Reproducible 20-question Tier-1 probe. JSON graph. Tier-0 disabled."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from home_cortex.ollama import OllamaService
from home_cortex.semantic_planner_benchmark import (
    DEFAULT_EVAL_PATH,
    build_json_fact_service,
    collect_provenance,
    load_probe_dataset,
    run_tier1_probe,
)

ROOT = Path(__file__).resolve().parents[1]


async def run(args: argparse.Namespace) -> dict:
    dataset = load_probe_dataset(args.eval)
    ollama = OllamaService(args.ollama_url, args.model)
    service, context = build_json_fact_service(args.data_dir, args.schema_dir, ollama)
    try:
        report = await run_tier1_probe(
            service,
            context,
            dataset,
            warmup=args.warmup,
            repeat=args.repeat,
            verified_cold=args.verified_cold,
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
    )
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
