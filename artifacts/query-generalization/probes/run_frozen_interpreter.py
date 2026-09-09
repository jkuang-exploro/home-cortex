#!/usr/bin/env python3
"""G3: frozen interpreter eval on synthetic graphs. No production records.

Does not feed sequence YAML to load_probe_dataset.
Does not edit prompts, ontology files, or frozen labels during a scored run.
V1 = SemanticOntology.load_default(); V2 = ontology-v2.yaml on planner+executor.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

from home_cortex.composition_eval import (
    HOUSEHOLD_IDS,
    household_engine,
    load_sequences,
    load_standalone_cases,
    request_context,
)
from home_cortex.edge_schema import EdgeSchemaRegistry
from home_cortex.fact_benchmark import _JsonGraphDispatcher
from home_cortex.ollama import OllamaService
from home_cortex.schema_catalog import RuntimeSchemaCatalog
from home_cortex.semantic_facts import (
    HouseholdFactEngine,
    SemanticFactPlanner,
    SemanticFactRequest,
    SemanticPlannerFailure,
    SemanticSchemaRegistry,
)
from home_cortex.semantic_ontology import SemanticOntology as Ontology
from home_cortex.semantic_planner_benchmark import (
    FROZEN_EVAL_TIME,
    normalize_semantic_request,
    primary_entity_ids,
    score_structured_result,
)

ROOT = Path(__file__).resolve().parents[3]
V2_PATH = ROOT / "schemas" / "semantic" / "ontology-v2.yaml"

LOCATION_CASES = (
    {
        "id": "loc-rooms-zh",
        "utterance": "家里有几个房间",
        "household": "alpha",
        "speaker_id": "person:alpha_self",
        "bucket_if_member": "wrong_but_legal",
        "notes": "No room relation. member/adult is wrong-but-legal.",
    },
    {
        "id": "loc-rooms-en",
        "utterance": "How many rooms are in this household?",
        "household": "alpha",
        "speaker_id": "person:alpha_self",
        "bucket_if_member": "wrong_but_legal",
    },
    {
        "id": "loc-milk-zh",
        "utterance": "牛奶在哪里",
        "household": "alpha",
        "speaker_id": "person:alpha_self",
        "bucket_if_member": "wrong_but_legal",
        "notes": "No location relation; named item not in alpha graph.",
    },
    {
        "id": "loc-milk-en",
        "utterance": "Where is the milk?",
        "household": "alpha",
        "speaker_id": "person:alpha_self",
        "bucket_if_member": "wrong_but_legal",
    },
    {
        "id": "loc-unsupported-item-type",
        "utterance": "家里食物类物品有几件",
        "household": "alpha",
        "speaker_id": "person:alpha_self",
        "bucket_if_member": "wrong_but_legal",
    },
)


def classify(plan: SemanticFactRequest | None, validation: str | None, gold, result, *, location: bool) -> str:
    if plan is None:
        return "schema_rejection" if validation else "missing_knowledge"
    if location:
        hops = [step.relation for step in plan.subject.path]
        if plan.subject.kind == "current_household" and hops == ["member"]:
            return "wrong_but_legal"
        if plan.subject.kind == "named_entity":
            return "wrong_but_legal"
        if gold is None:
            return "schema_rejection" if not plan.subject.path else "wrong_but_legal"
    if gold is not None:
        actual = normalize_semantic_request(plan)
        expected = normalize_semantic_request(gold)
        if actual == expected:
            return "match" if (result is None or result.status != "semantic_plan_unsupported") else "execution_failure"
        if validation and validation != "VALID":
            return "schema_rejection"
        return "wrong_but_legal"
    return "wrong_but_legal"


def engine_with_ontology(household: str, ontology):
    baseline, _ = household_engine(household)
    schema = SemanticSchemaRegistry(baseline.schema.catalog, ontology)
    return HouseholdFactEngine(baseline.dispatcher, schema)


async def run_case(planner, engine, context, messages, gold_case=None, location=False):
    started = time.perf_counter()
    diagnostics = None
    request = None
    decoder_error = None
    try:
        outcome = await planner.plan(messages, context)
        diagnostics = outcome.diagnostics
        request = outcome.plan.request
    except SemanticPlannerFailure as error:
        diagnostics = error.diagnostics
        decoder_error = error.diagnostics.failure_detail
    latency_ms = (time.perf_counter() - started) * 1000
    result = None
    if request is not None:
        result, *_ = await engine.execute(request, context)
    gold = gold_case.expected if gold_case is not None else None
    validation = diagnostics.validation_result if diagnostics else None
    bucket = classify(request, validation, gold, result, location=location)
    population_ok = None
    if gold_case is not None and result is not None and bucket == "match":
        population_ok = score_structured_result(result, gold_case.as_eval_case())
        if population_ok is False:
            bucket = "execution_failure"
    return {
        "validation": validation,
        "attempts": diagnostics.attempt_count if diagnostics else 0,
        "decoder_error": decoder_error,
        "llm_calls": diagnostics.attempt_count if diagnostics else 0,
        "planner_latency_ms": round(latency_ms, 2),
        "plan": None if request is None else request.model_dump(mode="json", exclude_none=True),
        "result_status": None if result is None else result.status,
        "entity_ids": None if result is None else list(primary_entity_ids(result)),
        "bucket": bucket,
        "population_ok": population_ok,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--profile", choices=("v1", "v2", "both"), default="both")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/query-generalization/probe-g3-summary.json")
    args = parser.parse_args()
    profiles = ("v1", "v2") if args.profile == "both" else (args.profile,)
    frozen = [c for c in load_standalone_cases() if c.split == "frozen"]
    sequences = [s for s in load_sequences() if s.last.split == "frozen"]
    if args.limit:
        frozen = frozen[: args.limit]
        sequences = sequences[: max(1, args.limit // 8)]
    ollama = OllamaService(args.ollama_url, args.model)
    report = {
        "started": datetime.now().isoformat() + "Z",
        "model": args.model,
        "ollama_url": args.ollama_url,
        "profiles": {},
    }
    try:
        for profile in profiles:
            ontology = (
                Ontology.load_default()
                if profile == "v1"
                else Ontology.from_file(V2_PATH)
            )
            rows = []
            buckets: dict[str, int] = {}
            for case in frozen:
                engine = engine_with_ontology(case.household, ontology)
                planner = SemanticFactPlanner(ollama, engine.schema)
                context = request_context(
                    speaker_id=case.speaker_id,
                    household=case.household,
                    frozen_time=datetime.fromisoformat(FROZEN_EVAL_TIME),
                )
                row = await run_case(
                    planner, engine, context, [{"role": "user", "content": case.utterance}], case
                )
                row.update(id=case.case_id, cell=case.cell, utterance=case.utterance, kind="frozen")
                rows.append(row)
                buckets[row["bucket"]] = buckets.get(row["bucket"], 0) + 1
            for sequence in sequences:
                engine = engine_with_ontology(sequence.household, ontology)
                planner = SemanticFactPlanner(ollama, engine.schema)
                context = request_context(
                    speaker_id=sequence.speaker_id,
                    household=sequence.household,
                    frozen_time=datetime.fromisoformat(FROZEN_EVAL_TIME),
                    conversation_id=sequence.sequence_id,
                )
                messages = [{"role": "user", "content": u} for u in sequence.utterances]
                row = await run_case(
                    planner, engine, context, messages, sequence.last
                )
                row.update(id=sequence.sequence_id, cell=sequence.last.cell, utterance=sequence.utterances[-1], kind="sequence")
                rows.append(row)
                buckets[row["bucket"]] = buckets.get(row["bucket"], 0) + 1
            for loc in LOCATION_CASES:
                engine = engine_with_ontology(loc["household"], ontology)
                planner = SemanticFactPlanner(ollama, engine.schema)
                context = request_context(
                    speaker_id=loc["speaker_id"], household=loc["household"]
                )
                row = await run_case(
                    planner, engine, context, [{"role": "user", "content": loc["utterance"]}], location=True
                )
                row.update(id=loc["id"], cell="location", utterance=loc["utterance"], kind="location")
                rows.append(row)
                buckets[row["bucket"]] = buckets.get(row["bucket"], 0) + 1
            first_pass = sum(1 for r in rows if r.get("attempts", 1) == 1 and r["bucket"] == "match")
            retries = sum(1 for r in rows if (r.get("attempts") or 1) > 1)
            decoder_errors = sum(1 for r in rows if r.get("decoder_error"))
            latencies = [r["planner_latency_ms"] for r in rows if r.get("planner_latency_ms") is not None]
            report["profiles"][profile] = {
                "ontology_version": ontology.version,
                "contracts": ontology.version == 2,
                "n": len(rows),
                "buckets": buckets,
                "first_pass_matches": first_pass,
                "rows_with_retry": retries,
                "decoder_errors": decoder_errors,
                "llm_calls": sum(r.get("llm_calls") or 0 for r in rows),
                "latency_ms_mean": round(sum(latencies) / len(latencies), 2) if latencies else None,
                "rows": rows,
            }
    finally:
        await ollama.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in report["profiles"].items()}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
