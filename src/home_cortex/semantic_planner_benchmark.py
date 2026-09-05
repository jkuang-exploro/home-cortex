"""Planner-only semantic-IR quality benchmark and Tier-0 parity report."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import yaml

from .agents import get_agent
from .config import get_settings
from .edge_schema import EdgeSchemaRegistry
from .fact_benchmark import _JsonGraphDispatcher, _percentile
from .grounding import AgentRequestContext
from .ollama import OllamaService
from .schema_catalog import RuntimeSchemaCatalog
from .semantic_facts import (
    HouseholdFactEngine,
    SemanticFactPlanner,
    SemanticFactRequest,
    SemanticFactService,
    SemanticPlannerFailure,
    SemanticSchemaRegistry,
    TierZeroSemanticParser,
    planner_input_summary,
)

def _default_eval_path() -> Path:
    candidates = (
        Path(__file__).resolve().parents[2]
        / "benchmarks"
        / "semantic_planner_eval.yaml",
        Path("/app/benchmarks/semantic_planner_eval.yaml"),
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


DEFAULT_EVAL_PATH = _default_eval_path()


@dataclass(frozen=True)
class SemanticEvalCase:
    utterance: str
    speaker_id: str
    category: str
    plan_id: str
    expected: SemanticFactRequest
    acceptable_alternatives: tuple[SemanticFactRequest, ...] = ()


def load_semantic_eval_cases(
    path: Path = DEFAULT_EVAL_PATH,
    *,
    default_speaker_id: str = "person:jian_kuang",
) -> tuple[SemanticEvalCase, ...]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("version") != 1:
        raise ValueError("semantic planner evaluation dataset must have version 1")
    plans = raw.get("plans")
    groups = raw.get("groups")
    if not isinstance(plans, Mapping) or not isinstance(groups, list):
        raise ValueError("semantic planner evaluation dataset is malformed")
    cases: list[SemanticEvalCase] = []
    seen: set[str] = set()
    for group in groups:
        if not isinstance(group, Mapping):
            raise ValueError("semantic planner evaluation group must be an object")
        key = group.get("plan")
        category = group.get("category")
        utterances = group.get("utterances")
        expected = plans.get(key)
        alternative_keys = group.get("acceptable_plans", [])
        if (
            not isinstance(key, str)
            or not isinstance(category, str)
            or not isinstance(utterances, list)
            or not isinstance(expected, Mapping)
            or not isinstance(alternative_keys, list)
            or not all(isinstance(item, str) for item in alternative_keys)
        ):
            raise ValueError("semantic planner evaluation group is incomplete")
        request = SemanticFactRequest.model_validate(expected)
        alternatives = tuple(
            SemanticFactRequest.model_validate(plans[item])
            for item in alternative_keys
            if item in plans
        )
        if len(alternatives) != len(alternative_keys):
            raise ValueError("semantic planner evaluation references an unknown plan")
        for utterance in utterances:
            if not isinstance(utterance, str) or not utterance.strip():
                raise ValueError("evaluation utterances must be non-empty strings")
            if utterance in seen:
                raise ValueError(f"duplicate evaluation utterance: {utterance}")
            seen.add(utterance)
            cases.append(
                SemanticEvalCase(
                    utterance,
                    str(group.get("speaker_id") or default_speaker_id),
                    category,
                    key,
                    request,
                    alternatives,
                )
            )
    return tuple(cases)


def normalize_semantic_request(request: SemanticFactRequest) -> dict[str, Any]:
    """Canonical JSON form used only for semantic-plan comparison."""
    normalized = request.model_dump(mode="json", exclude_none=True)
    if (
        normalized.get("operation") == "select"
        and normalized.get("property") == "display_name"
        and normalized.get("property_source") == "entity"
        and not normalized.get("filters")
        and "other" not in normalized
    ):
        normalized["operation"] = "resolve_reference"
        normalized.pop("property", None)
    return normalized


def semantic_mismatch_reason(
    actual: Mapping[str, Any] | None,
    expected: Mapping[str, Any],
) -> str:
    if actual is None:
        return "NO_SEMANTIC_PLAN"
    if actual.get("operation") != expected.get("operation"):
        return "OPERATION_MISMATCH"
    actual_subject = actual.get("subject")
    expected_subject = expected.get("subject")
    if not isinstance(actual_subject, Mapping) or not isinstance(
        expected_subject, Mapping
    ):
        return "REFERENCE_MISMATCH"
    if any(
        actual_subject.get(key) != expected_subject.get(key)
        for key in ("kind", "value", "entity_type")
    ):
        return "REFERENCE_MISMATCH"
    actual_relations = [
        step.get("relation")
        for step in actual_subject.get("path", ())
        if isinstance(step, Mapping)
    ]
    expected_relations = [
        step.get("relation")
        for step in expected_subject.get("path", ())
        if isinstance(step, Mapping)
    ]
    if actual_relations != expected_relations:
        return "RELATIONSHIP_MISMATCH"
    if actual.get("property") != expected.get("property"):
        return "PROPERTY_MISMATCH"
    if actual.get("property_source") != expected.get("property_source"):
        return "PROPERTY_SOURCE_MISMATCH"
    if actual_subject.get("path") != expected_subject.get("path") or actual.get(
        "filters"
    ) != expected.get("filters"):
        return "FILTER_MISMATCH"
    if any(
        actual.get(key) != expected.get(key)
        for key in ("other", "mode", "from_unit", "to_unit")
    ):
        return "PARAMETER_MISMATCH"
    return "PLAN_MISMATCH"


async def run_semantic_planner_benchmark(
    service: SemanticFactService,
    context: AgentRequestContext,
    cases: Sequence[SemanticEvalCase],
) -> dict[str, Any]:
    if service.planner is None:
        raise ValueError("planner-only benchmark requires SemanticFactPlanner")
    parser = TierZeroSemanticParser(service.engine.schema.ontology)
    rows: list[dict[str, Any]] = []
    planner_latencies: list[float] = []
    category_totals: Counter[str] = Counter()
    category_correct: Counter[str] = Counter()
    failure_reasons: Counter[str] = Counter()
    tier0_comparisons = 0
    tier0_equivalent = 0
    capability_summary = planner_input_summary(
        service.engine.schema.capability_payload()
    )

    for case in cases:
        case_context = AgentRequestContext(
            caller_entity_id=case.speaker_id,
            assistant_id=context.assistant_id,
            assistant_display_name=context.assistant_display_name,
            household_id=context.household_id,
            current_time=context.current_time,
            locale=context.locale,
        )
        diagnostics = None
        actual = None
        executor_success = False
        executor_result = "not_run"
        executor_latency_ms: float | None = None
        final_answer: str | None = None
        runtime_failure: str | None = None
        try:
            outcome = await service.planner.plan(
                [{"role": "user", "content": case.utterance}],
                case_context,
            )
            diagnostics = outcome.diagnostics
            if outcome.plan.request is not None:
                request = outcome.plan.request
                actual = normalize_semantic_request(request)
                executor_started = perf_counter()
                result, _, _, _ = await service.engine.execute(request, case_context)
                executor_latency_ms = (perf_counter() - executor_started) * 1000
                executor_result = result.status
                executor_success = result.status == "found"
                final_answer = service.renderer.render(request, result, case_context)
        except SemanticPlannerFailure as error:
            diagnostics = error.diagnostics
        except Exception as error:
            runtime_failure = f"RUNTIME_ERROR:{type(error).__name__}"
        expected = normalize_semantic_request(case.expected)
        acceptable_plans = (
            expected,
            *(
                normalize_semantic_request(item)
                for item in case.acceptable_alternatives
            ),
        )
        planner_correct = actual in acceptable_plans
        category_totals[case.category] += 1
        if planner_correct:
            category_correct[case.category] += 1
        validation_result = (
            diagnostics.validation_result if diagnostics is not None else None
        )
        reason = runtime_failure or (
            semantic_mismatch_reason(actual, expected)
            if validation_result in {"VALID", "NOT_A_FACT"}
            else validation_result or "NO_SEMANTIC_PLAN"
        )
        if not planner_correct:
            failure_reasons[reason] += 1
        if diagnostics is not None:
            planner_latencies.append(diagnostics.latency_ms)

        tier0 = parser.parse(case.utterance)
        tier0_plan = normalize_semantic_request(tier0) if tier0 is not None else None
        tier0_matches_planner: bool | None = None
        if tier0 is not None:
            tier0_comparisons += 1
            tier0_matches_planner = tier0_plan == actual
            tier0_equivalent += int(tier0_matches_planner)

        rows.append(
            {
                "utterance": case.utterance,
                "speaker_id": case.speaker_id,
                "category": case.category,
                "expected_plan_id": case.plan_id,
                "planner_input_capabilities": (
                    dict(diagnostics.input_summary)
                    if diagnostics
                    else capability_summary
                ),
                "planner_output": (
                    dict(diagnostics.output_raw)
                    if diagnostics and diagnostics.output_raw is not None
                    else None
                ),
                "normalized_semantic_plan": actual,
                "expected_semantic_plan": expected,
                "acceptable_semantic_plans": list(acceptable_plans[1:]),
                "planner_latency_ms": (
                    round(diagnostics.latency_ms, 3) if diagnostics else None
                ),
                "planner_attempt_count": (
                    diagnostics.attempt_count if diagnostics else 1
                ),
                "planner_success": planner_correct,
                "planner_failure_reason": None if planner_correct else reason,
                "planner_failure_detail": (
                    diagnostics.failure_detail if diagnostics else runtime_failure
                ),
                "validation_result": (
                    diagnostics.validation_result
                    if diagnostics
                    else runtime_failure or "NOT_A_FACT"
                ),
                "executor_success": executor_success,
                "executor_result": executor_result,
                "executor_latency_ms": (
                    round(executor_latency_ms, 3)
                    if executor_latency_ms is not None
                    else None
                ),
                "final_answer": final_answer,
                "tier": 1,
                "tier0_plan": tier0_plan,
                "tier0_matches_planner": tier0_matches_planner,
            }
        )

    total = len(rows)
    correct = sum(int(row["planner_success"]) for row in rows)
    return {
        "mode": "planner_only",
        "dataset_size": total,
        "queries": rows,
        "accuracy": round(correct / total, 4) if total else 0,
        "accuracy_by_capability": {
            category: {
                "correct": category_correct[category],
                "total": count,
                "accuracy": round(category_correct[category] / count, 4),
            }
            for category, count in sorted(category_totals.items())
        },
        "failure_reasons": dict(sorted(failure_reasons.items())),
        "planner_latency_ms": {
            "p50": round(statistics.median(planner_latencies), 3)
            if planner_latencies
            else None,
            "p95": round(_percentile(planner_latencies, 0.95), 3)
            if planner_latencies
            else None,
        },
        "tier0_parity": {
            "compared": tier0_comparisons,
            "equivalent": tier0_equivalent,
            "accuracy": (
                round(tier0_equivalent / tier0_comparisons, 4)
                if tier0_comparisons
                else None
            ),
        },
    }


async def benchmark_json(
    *,
    data_dir: Path,
    schema_dir: Path,
    eval_path: Path = DEFAULT_EVAL_PATH,
    ollama_url: str,
    ollama_model: str,
) -> dict[str, Any]:
    return await _benchmark_cases(
        data_dir=data_dir,
        schema_dir=schema_dir,
        cases=load_semantic_eval_cases(eval_path),
        ollama_url=ollama_url,
        ollama_model=ollama_model,
    )


async def _benchmark_cases(
    *,
    data_dir: Path,
    schema_dir: Path,
    cases: Sequence[SemanticEvalCase],
    ollama_url: str,
    ollama_model: str,
) -> dict[str, Any]:
    registry = EdgeSchemaRegistry.from_directory(schema_dir)
    catalog = RuntimeSchemaCatalog.from_data_dir(data_dir, registry)
    schema = SemanticSchemaRegistry(catalog)
    dispatcher = _JsonGraphDispatcher(data_dir, registry)
    ollama = OllamaService(ollama_url, ollama_model)
    service = SemanticFactService(
        HouseholdFactEngine(dispatcher, schema),
        planner=SemanticFactPlanner(ollama, schema),
        tier_zero_enabled=False,
    )
    steward = get_agent("steward")
    localized = steward.settings.get("localized_identity", {})
    context = AgentRequestContext(
        caller_entity_id="person:jian_kuang",
        assistant_id=steward.id,
        assistant_display_name=localized.get("zh", steward.display_name),
        household_id=steward.settings.get("home_entity_id"),
        current_time=datetime.now(ZoneInfo("America/Los_Angeles")),
        locale="zh",
    )
    try:
        report = await run_semantic_planner_benchmark(
            service,
            context,
            cases,
        )
        return {"model": ollama_model, **report}
    finally:
        await ollama.close()


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=settings.data_dir)
    parser.add_argument("--schema-dir", type=Path, default=settings.edge_schema_dir)
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--ollama-url", default=settings.ollama_url)
    parser.add_argument("--model", default=settings.ollama_model)
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument(
        "--utterance",
        action="append",
        default=[],
        help="Run only this exact dataset utterance; may be repeated.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--one-per-plan", action="store_true")
    args = parser.parse_args()
    cases = load_semantic_eval_cases(args.eval)
    if args.category:
        cases = tuple(case for case in cases if case.category in args.category)
    if args.utterance:
        requested = set(args.utterance)
        cases = tuple(case for case in cases if case.utterance in requested)
        missing = requested.difference(case.utterance for case in cases)
        if missing:
            parser.error(
                "utterance not found in evaluation dataset: "
                + ", ".join(sorted(missing))
            )
    if args.one_per_plan:
        selected: dict[str, SemanticEvalCase] = {}
        for case in cases:
            selected.setdefault(case.plan_id, case)
        cases = tuple(selected.values())
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be at least 1")
        cases = cases[: args.limit]
    report = asyncio.run(
        _benchmark_cases(
            data_dir=args.data_dir,
            schema_dir=args.schema_dir,
            cases=cases,
            ollama_url=args.ollama_url,
            ollama_model=args.model,
        )
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
