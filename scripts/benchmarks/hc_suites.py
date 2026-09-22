"""Adapters from ``hc-bench`` onto the existing benchmark runners.

The runners in this package remain the scoring authority. Adapters select cases,
call those runners, and copy their scores into the shared record format. They do
not open the household database and they do not dispatch mutations.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import io
import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from home_cortex.benchmark.environment import (
    cached_tree_hash,
    semantic_prompt_fingerprint,
    sha256_file,
    stable_digest,
)
from home_cortex.benchmark.registry import registry
from home_cortex.benchmark.runner import CompositeSuite
from home_cortex.benchmark.stats import token_totals
from home_cortex.benchmark.taxonomy import classify_planner_failure
from home_cortex.benchmark.types import CaseRecord, Metric, RunContext, SuiteResult
from scripts import PROJECT_ROOT

_PLANNER_POLICY = (
    "One scoring pass over benchmarks/semantic_planner_eval.yaml. "
    "Latency percentiles describe that pass and can include the first request. "
    "Warmup and repetitions are not applied."
)
_FACT_POLICY = (
    "One pass of the JSON-graph fact benchmark. It records completion and latency, "
    "not gold plan or answer scores. The household database is not opened."
)
_MUTATION_POLICY = (
    "One scoring pass of the unified planner experiment. Two warmup calls per route "
    "are discarded by that runner before scoring. Writes are compiled and never "
    "dispatched. Safety gates use the unified route. Token totals sum that route's "
    "Ollama-reported counts and are not estimated."
)
_LATENCY_POLICY = (
    "Warmup requests are excluded from percentiles. Measured repetitions are "
    "steady-state. The harness does not unload the model or flush caches. "
    "cold_load_ms is Ollama's load_duration_ms on the first request when reported. "
    "--verified-cold only labels that request; unload the model yourself first."
)
_BILINGUAL_POLICY = (
    "One measured pass of the bilingual planner probe, including its prompt-component "
    "measurement. The probe's semantic-contract fixture is the graph, not --data-dir. "
    "A --limit smoke cap skips discourse and negative turns."
)
_READ_POLICY = (
    "Paired legacy and unified read regression. This repeats the planner, probe, "
    "compression, and bilingual corpora and is much larger than the planner suite."
)


def register() -> None:
    """Register the suites ``hc-bench list`` shows. Safe to call again."""

    planner = PlannerSuite()
    fact = FactSuite()
    mutation = UnifiedSuite(
        "mutation",
        "Mutation classification, payload, preview, commit, rejection, and multi-intent. "
        "Writes are compiled, never dispatched. Gates apply to the unified route.",
        ("intents",),
    )
    unified = UnifiedSuite(
        "unified-planner",
        "Paired legacy and unified planner experiment for intents and reads. "
        "The reads group repeats the regression corpus. Writes are never dispatched.",
        ("intents", "reads"),
    )
    latency = LatencySuite()
    bilingual = BilingualSuite()
    standard = CompositeSuite(
        "standard",
        "Representative model check: planner plan/answer scores plus mutation safety. "
        "One scoring pass. Not the repository test suite.",
        (planner, mutation),
    )
    full = CompositeSuite(
        "full",
        "Deeper model check: planner, mutation, fact pipeline, repeated latency probe, "
        "and bilingual probe. The paired read regression stays in unified-planner.",
        (planner, mutation, fact, latency, bilingual),
    )
    chosen = registry()
    for suite in (standard, fact, planner, unified, mutation, latency, bilingual, full):
        chosen.register(suite)


class PlannerSuite:
    name = "planner"
    description = (
        "Fixed semantic planner evaluation: plan correctness, answer correctness, "
        "and validation failures."
    )
    requires_real_model = True
    requires_gpu_host = True

    def run(self, context: RunContext) -> SuiteResult:
        with _num_ctx(context.num_ctx):
            return asyncio.run(_run_planner(context))


class FactSuite:
    name = "fact"
    description = (
        "Live semantic fact path on the JSON graph. Completion and latency only; "
        "plan and answer gold stays in the planner suite. Does not open the household database."
    )
    requires_real_model = True
    requires_gpu_host = True

    def run(self, context: RunContext) -> SuiteResult:
        with _num_ctx(context.num_ctx):
            return asyncio.run(_run_fact(context))


class LatencySuite:
    name = "latency"
    description = (
        "Probe-set latency. Warmup is excluded from percentiles. "
        "Default warmup 1, repetitions 3. Does not unload the model."
    )
    requires_real_model = True
    requires_gpu_host = True

    def run(self, context: RunContext) -> SuiteResult:
        with _num_ctx(context.num_ctx):
            return asyncio.run(_run_latency(context))


class BilingualSuite:
    name = "bilingual"
    description = (
        "Bilingual, mixed, and stress planner probe on the semantic-contract fixture. "
        "Discourse and negatives are included unless --limit is set."
    )
    requires_real_model = True
    requires_gpu_host = True

    def run(self, context: RunContext) -> SuiteResult:
        with _num_ctx(context.num_ctx):
            return asyncio.run(_run_bilingual(context))


class UnifiedSuite:
    """Intents, and optionally the reads group, of the unified planner experiment."""

    def __init__(self, name: str, description: str, groups: tuple[str, ...]) -> None:
        self.name = name
        self.description = description
        self.groups = groups
        self.requires_real_model = True
        self.requires_gpu_host = True

    def run(self, context: RunContext) -> SuiteResult:
        with _num_ctx(context.num_ctx):
            return asyncio.run(_run_unified(self, context))


def planner_metrics(
    report: Mapping[str, Any],
    *,
    suite: str,
    plan_id: str = "plan_correctness",
    answer_id: str = "answer_correctness",
    plan_label: str = "Plan correctness",
    answer_label: str = "Answer correctness",
) -> list[Metric]:
    """Copy ``summarize_scores`` output. Do not recompute it."""

    scores = report["scores"]
    plan = scores["plan_accuracy"]
    answer = scores["answer_correctness"]
    return [
        Metric(plan_id, plan_label, "ratio", int(plan["correct"]), int(plan["scored"]), suite=suite),
        Metric(
            answer_id,
            answer_label,
            "ratio",
            int(answer["correct"]),
            int(answer["scored"]),
            suite=suite,
        ),
    ]


def planner_cases(
    rows: Sequence[Mapping[str, Any]],
    *,
    suite: str,
    score_sample: int | None = 0,
) -> list[CaseRecord]:
    cases: list[CaseRecord] = []
    for row in rows:
        phase = str(row.get("phase") or "measured")
        sample = int(row.get("sample_index") or 0)
        counts = phase == "measured" and (score_sample is None or sample == score_sample)
        failure = classify_planner_failure(row)
        case_id = str(row.get("case_id"))
        if phase != "measured" or sample != 0:
            case_id = f"{case_id}#{phase}-{sample}"
        diagnostics = row.get("planner_diagnostics") if isinstance(row.get("planner_diagnostics"), Mapping) else {}
        cases.append(
            CaseRecord(
                suite=suite,
                case_id=case_id,
                passed=failure is None,
                latency_ms=_float(row.get("planner_latency_ms")),
                expected=row.get("expected_plan_id"),
                actual=row.get("validation_result"),
                failure_type=failure,
                metrics={
                    "phase": phase,
                    "counts_toward_score": counts,
                    "plan_match": row.get("plan_match"),
                    "answer_correct": row.get("answer_correct"),
                    "validation_result": row.get("validation_result"),
                    "failure_stage": row.get("failure_stage"),
                    "executor_status": row.get("executor_status"),
                    "sample_index": sample,
                    "prompt_eval_count": diagnostics.get("prompt_eval_count"),
                    "eval_count": diagnostics.get("eval_count"),
                    "load_duration_ms": diagnostics.get("load_duration_ms"),
                },
            )
        )
    return cases


def mutation_metrics(rows: Sequence[Mapping[str, Any]], *, suite: str, route: str = "unified") -> list[Metric]:
    """Group the experiment's per-row booleans. The booleans themselves are not rescored."""

    selected = [
        row for row in rows
        if row.get("route") == route and int(row.get("sample") or 0) == 0
    ]
    writes = [row for row in selected if row.get("category") == "write"]
    mixed = [row for row in selected if row.get("category") == "mixed"]
    ambiguous = [row for row in selected if row.get("category") == "ambiguous"]
    classified = [row for row in selected if row.get("classification_correct") is not None]
    payload = [row for row in selected if row.get("payload_correct") is not None]
    preview = [row for row in writes if _mode(row) == "preview"]
    commit = [row for row in writes if _mode(row) == "commit"]
    preview_as_commit = sum(
        1
        for row in writes
        if _mode(row) == "preview" and _actual_mode(row) == "commit"
    )
    partial = sum(1 for row in mixed if row.get("partial_plan") is True)
    metrics = [
        item
        for item in (
            _ratio("mutation_classification", "Mutation classify", classified, lambda row: row.get("classification_correct") is True, suite),
            _ratio("mutation_payload", "Mutation payload", payload, lambda row: row.get("payload_correct") is True, suite),
            _ratio("preview_correctness", "Preview correctness", preview, lambda row: row.get("payload_correct") is True, suite),
            _ratio("commit_correctness", "Commit correctness", commit, lambda row: row.get("payload_correct") is True, suite),
            _ratio("rejection_correctness", "Rejection correctness", ambiguous, lambda row: row.get("decision_kind") != "mutation", suite),
            _ratio(
                "multi_intent_correctness",
                "Multi-intent handling",
                mixed,
                lambda row: row.get("decision_kind") == "multi_intent" and row.get("partial_plan") is not True,
                suite,
            ),
        )
        if item is not None
    ]
    metrics.extend(
        [
            Metric("preview_as_commit", "Preview compiled as commit", "count", value=preview_as_commit, lower_is_better=True, suite=suite),
            Metric("mixed_partial_plans", "Multi-intent partial plans", "count", value=partial, lower_is_better=True, suite=suite),
        ]
    )
    return metrics


def unified_read_metrics(rows: Sequence[Mapping[str, Any]], *, suite: str, route: str = "unified") -> list[Metric]:
    selected = [
        row for row in rows
        if row.get("route") == route and int(row.get("sample") or 0) == 0 and "plan_correct" in row
    ]
    if not selected:
        return []
    return [
        Metric(
            "unified_read_plan_correctness",
            "Unified read plan",
            "ratio",
            sum(row.get("plan_correct") is True for row in selected),
            len(selected),
            suite=suite,
        ),
        Metric(
            "unified_read_answer_correctness",
            "Unified read answer",
            "ratio",
            sum(row.get("answer_correct") is True for row in selected),
            len(selected),
            suite=suite,
        ),
    ]


def mutation_cases(rows: Sequence[Mapping[str, Any]], *, suite: str) -> list[CaseRecord]:
    cases: list[CaseRecord] = []
    for row in rows:
        route = str(row.get("route") or "")
        counts = route == "unified" and int(row.get("sample") or 0) == 0
        failure = _mutation_failure(row)
        expected = row.get("expected_mutation") if isinstance(row.get("expected_mutation"), Mapping) else None
        brief = None
        if expected:
            brief = {"operation": expected.get("operation"), "mode": expected.get("mode")}
        cases.append(
            CaseRecord(
                suite=suite,
                case_id=_mutation_case_id(row),
                passed=failure is None,
                latency_ms=_float(row.get("latency_ms")),
                expected=brief,
                actual={"kind": row.get("decision_kind"), "mode": _actual_mode(row)},
                failure_type=failure,
                metrics={
                    "phase": "measured",
                    "counts_toward_score": counts,
                    "route": route,
                    "category": row.get("category"),
                    "classification_correct": row.get("classification_correct"),
                    "payload_correct": row.get("payload_correct"),
                    "decision_kind": row.get("decision_kind"),
                    "partial_plan": row.get("partial_plan"),
                    "plan_correct": row.get("plan_correct"),
                    "answer_correct": row.get("answer_correct"),
                    "prompt_eval_count": row.get("input_tokens"),
                    "eval_count": row.get("output_tokens"),
                },
            )
        )
    return cases


async def _run_planner(context: RunContext) -> SuiteResult:
    from home_cortex.providers.ollama import OllamaService
    from scripts.benchmarks.semantic_planner_benchmark import (
        SCORING_REVISION,
        build_json_fact_service,
        load_semantic_eval_cases,
        run_semantic_planner_benchmark,
    )

    eval_path = PROJECT_ROOT / "benchmarks" / "semantic_planner_eval.yaml"
    cases = load_semantic_eval_cases(eval_path)
    if context.limit is not None:
        cases = cases[: context.limit]
    client = OllamaService(context.ollama_url, context.model)
    try:
        service, request_context = build_json_fact_service(
            context.data_dir, context.schema_dir, client
        )
        report = await run_semantic_planner_benchmark(service, request_context, cases)
        prompt = semantic_prompt_fingerprint(service.engine.schema)
    finally:
        await client.close()
    rows = report["queries"]
    return _result(
        "planner",
        planner_cases(rows, suite="planner"),
        planner_metrics(report, suite="planner"),
        _fingerprint(context, prompt, {"eval": eval_path}, scoring=SCORING_REVISION, warmup=0, repetitions=1),
        _tokens(rows),
        {"warmup": 0, "repetitions": 1},
        _PLANNER_POLICY,
        cold=_reported_cold_load(rows, warmup=0),
        notes=_ignored_repeat_note(context),
    )


async def _run_fact(context: RunContext) -> SuiteResult:
    from scripts.benchmarks.fact_benchmark import QUESTIONS, SPEAKER_CASES, benchmark_json

    questions = QUESTIONS[: context.limit] if context.limit is not None else QUESTIONS
    report = await benchmark_json(
        "person:jian_kuang",
        1,
        context.data_dir,
        context.schema_dir,
        "semantic",
        questions,
        ollama_url=context.ollama_url,
        model=context.model,
    )
    prompt = _prompt_for(context)
    cases = []
    for row in report["queries"]:
        cases.append(
            CaseRecord(
                suite="fact",
                case_id=_stable_id(row["speaker_id"], row["utterance"]),
                passed=True,
                latency_ms=_float(row.get("total_latency_ms")),
                expected=None,
                actual=row.get("result_status"),
                failure_type=None,
                metrics={
                    "phase": "measured",
                    "counts_toward_score": True,
                    "result_status": row.get("result_status"),
                    "failure_stage": row.get("failure_stage"),
                    "llm_ms": (row.get("stage_latency_ms") or {}).get("llm"),
                },
            )
        )
    completed = len(cases)
    return _result(
        "fact",
        cases,
        [Metric("fact_completed", "Fact pipeline completed", "ratio", completed, completed, suite="fact")],
        _fingerprint(
            context,
            prompt,
            {},
            extra={
                "fact_questions": hashlib.sha256("\n".join(QUESTIONS).encode()).hexdigest(),
                "speaker_cases": hashlib.sha256(
                    json.dumps(
                        [(case.speaker_id, case.utterance) for case in SPEAKER_CASES],
                        ensure_ascii=False,
                    ).encode()
                ).hexdigest(),
            },
            warmup=0,
            repetitions=1,
        ),
        {"prompt": None, "output": None, "total": None, "source": "unavailable", "estimated": False},
        {"warmup": 0, "repetitions": 1},
        _FACT_POLICY,
        notes=_ignored_repeat_note(context),
    )


async def _run_latency(context: RunContext) -> SuiteResult:
    from dataclasses import replace

    from home_cortex.providers.ollama import OllamaService
    from scripts.benchmarks.semantic_planner_benchmark import (
        SCORING_REVISION,
        build_json_fact_service,
        load_probe_dataset,
        run_tier1_probe,
    )

    warmup = 1 if context.warmup is None else context.warmup
    repetitions = 3 if context.repetitions is None else context.repetitions
    eval_path = PROJECT_ROOT / "benchmarks" / "semantic_planner_eval.yaml"
    dataset = load_probe_dataset(eval_path)
    if context.limit is not None:
        dataset = replace(dataset, cases=dataset.cases[: context.limit])
    client = OllamaService(context.ollama_url, context.model)
    try:
        service, request_context = build_json_fact_service(
            context.data_dir, context.schema_dir, client
        )
        report = await run_tier1_probe(
            service,
            request_context,
            dataset,
            warmup=warmup,
            repeat=repetitions,
            verified_cold=context.verified_cold,
        )
        prompt = semantic_prompt_fingerprint(service.engine.schema)
    finally:
        await client.close()
    rows = report["queries"]
    return _result(
        "latency",
        planner_cases(rows, suite="latency"),
        planner_metrics(
            report,
            suite="latency",
            plan_id="latency_sample_plan_correctness",
            answer_id="latency_sample_answer_correctness",
            plan_label="Latency-sample plan correctness",
            answer_label="Latency-sample answer correctness",
        ),
        _fingerprint(
            context,
            prompt,
            {"eval": eval_path},
            scoring=SCORING_REVISION,
            warmup=warmup,
            repetitions=repetitions,
        ),
        _tokens(rows),
        {"warmup": warmup, "repetitions": repetitions},
        _LATENCY_POLICY,
        cold=_reported_cold_load(rows, warmup=warmup),
    )


async def _run_bilingual(context: RunContext) -> SuiteResult:
    from scripts.probes.bilingual_planner_probe import DATASET, FIXTURE, SCHEMA, run

    args = argparse.Namespace(
        ollama_url=context.ollama_url,
        model=context.model,
        warmup=0,
        repeat=1,
        limit=context.limit,
    )
    report = await run(args)
    prompt = _prompt_for(context, data_dir=FIXTURE, schema_dir=SCHEMA)
    rows = [
        row for row in report["queries"]
        if int(row.get("sample_index") or 0) == 0
    ]
    cases = [_bilingual_case(row) for row in rows]
    overall = report["scores"]["overall"]
    metrics = [
        Metric(
            "bilingual_match",
            "Bilingual plan match",
            "ratio",
            int(overall["correct"]),
            int(overall["scored"]),
            suite="bilingual",
        )
    ]
    parity = (report.get("parity") or {}).get("both_correct") or {}
    if parity.get("scored"):
        metrics.append(
            Metric(
                "bilingual_parity",
                "Bilingual parity",
                "ratio",
                int(parity["correct"]),
                int(parity["scored"]),
                suite="bilingual",
            )
        )
    return _result(
        "bilingual",
        cases,
        metrics,
        _fingerprint(
            context,
            prompt,
            {"bilingual": DATASET},
            extra={"fixture": cached_tree_hash(FIXTURE, context.digest_cache)},
            data_dir=FIXTURE,
            schema_dir=SCHEMA,
            warmup=0,
            repetitions=1,
        ),
        _tokens(rows),
        {"warmup": 0, "repetitions": 1},
        _BILINGUAL_POLICY,
        notes=_ignored_repeat_note(context),
    )


async def _run_unified(suite: UnifiedSuite, context: RunContext) -> SuiteResult:
    from scripts.benchmarks.unified_planner_experiment import run

    cases: list[CaseRecord] = []
    metrics: list[Metric] = []
    routes: dict[str, Any] = {}
    token_rows: list[Mapping[str, Any]] = []
    prompt = _prompt_for(context)
    files = {"mutation_routing": PROJECT_ROOT / "benchmarks" / "mutation_routing.yaml"}
    if "reads" in suite.groups:
        files["compression"] = PROJECT_ROOT / "benchmarks" / "planner_prompt_compression.yaml"
        files["bilingual"] = PROJECT_ROOT / "benchmarks" / "semantic_planner_bilingual.yaml"
        files["eval"] = PROJECT_ROOT / "benchmarks" / "semantic_planner_eval.yaml"
    for group in suite.groups:
        report = await _experiment_group(run, context, group)
        routes[group] = report.get("summary")
        rows = report.get("rows") or []
        cases.extend(mutation_cases(rows, suite=suite.name))
        if group == "intents":
            metrics.extend(mutation_metrics(rows, suite=suite.name))
        else:
            metrics.extend(unified_read_metrics(rows, suite=suite.name))
        token_rows.extend(
            row for row in rows if row.get("route") == "unified" and int(row.get("sample") or 0) == 0
        )
    policy = _MUTATION_POLICY if suite.groups == ("intents",) else _MUTATION_POLICY + " " + _READ_POLICY
    return _result(
        suite.name,
        cases,
        metrics,
        _fingerprint(context, prompt, files, warmup=0, repetitions=1),
        _tokens(token_rows),
        {"warmup": 0, "repetitions": 1},
        policy,
        routes=routes,
        notes=_ignored_repeat_note(context),
    )


async def _experiment_group(run: Any, context: RunContext, group: str) -> dict[str, Any]:
    utterances = _limited_utterances(context, group)
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "report.json"
        args = argparse.Namespace(
            group=group,
            routing_eval=PROJECT_ROOT / "benchmarks" / "mutation_routing.yaml",
            data_dir=context.data_dir,
            schema_dir=context.schema_dir,
            ollama_url=context.ollama_url,
            model=context.model,
            repeat=1,
            utterance=utterances,
            output=output,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            await run(args)
        return json.loads(output.read_text(encoding="utf-8"))


def _limited_utterances(context: RunContext, group: str) -> list[str]:
    if context.limit is None:
        return []
    if group == "intents":
        import yaml

        dataset = yaml.safe_load(
            (PROJECT_ROOT / "benchmarks" / "mutation_routing.yaml").read_text(encoding="utf-8")
        )
        utterances: list[str] = []
        for category in ("write", "mixed", "ambiguous"):
            utterances.extend(dataset.get(category) or [])
        return utterances[: context.limit]
    from scripts.benchmarks.planner_prompt_experiment import regression_cases
    from scripts.benchmarks.semantic_planner_benchmark import build_json_fact_service

    service, request_context = build_json_fact_service(
        context.data_dir, context.schema_dir, object()
    )
    cases = regression_cases(service, request_context)
    return [case.utterance for case in cases[: context.limit]]


def _bilingual_case(row: Mapping[str, Any]) -> CaseRecord:
    forbidden = row.get("forbidden") or []
    validation = row.get("validation")
    if forbidden or row.get("match") is not True:
        if validation == "MALFORMED_OUTPUT" or "MALFORMED" in str(validation or ""):
            failure = "malformed_structured_output"
        elif validation not in {None, "VALID", "NOT_A_FACT"}:
            failure = "validation_failure"
        else:
            failure = "semantic_mismatch"
    else:
        failure = None
    return CaseRecord(
        suite="bilingual",
        case_id=str(row.get("id")),
        passed=failure is None,
        latency_ms=_float(row.get("wall_ms") if row.get("wall_ms") is not None else row.get("latency_ms")),
        expected=row.get("kind"),
        actual=validation,
        failure_type=failure,
        metrics={
            "phase": "measured",
            "counts_toward_score": True,
            "match": row.get("match"),
            "forbidden": bool(forbidden),
            "validation_result": validation,
            "prompt_eval_count": row.get("prompt_eval_count"),
            "eval_count": row.get("eval_count"),
        },
    )


def _result(
    name: str,
    cases: list[CaseRecord],
    metrics: list[Metric],
    fingerprints: dict[str, Any],
    tokens: dict[str, Any],
    repetition: dict[str, Any],
    policy: str,
    *,
    routes: dict[str, Any] | None = None,
    cold: float | None = None,
    notes: list[str] | None = None,
) -> SuiteResult:
    return SuiteResult(
        name=name,
        components=(name,),
        cases=cases,
        metrics=metrics,
        fingerprints=fingerprints,
        tokens=tokens,
        repetition=repetition,
        timing_policy=policy,
        routes=routes or {},
        cold_load_ms=cold,
        notes=notes or [],
    )


def _fingerprint(
    context: RunContext,
    prompt: str,
    files: Mapping[str, Path],
    *,
    scoring: str | None = None,
    warmup: int,
    repetitions: int,
    extra: Mapping[str, str] | None = None,
    data_dir: Path | None = None,
    schema_dir: Path | None = None,
) -> dict[str, Any]:
    parts = {
        "data": cached_tree_hash(data_dir or context.data_dir, context.digest_cache),
        "schema": cached_tree_hash(schema_dir or context.schema_dir, context.digest_cache),
    }
    ontology = PROJECT_ROOT / "schemas" / "semantic" / "ontology.yaml"
    parts["ontology"] = sha256_file(ontology) if ontology.is_file() else "missing"
    for label, path in files.items():
        parts[label] = sha256_file(path) if path.is_file() else "missing"
    if extra:
        parts.update(extra)
    if context.limit is not None:
        parts["limit"] = str(context.limit)
    return {
        "prompt": prompt,
        "corpus": stable_digest(parts),
        "scoring_revision": scoring,
        "config_inputs": {"warmup": warmup, "repetitions": repetitions},
    }


def _prompt_for(
    context: RunContext,
    *,
    data_dir: Path | None = None,
    schema_dir: Path | None = None,
) -> str:
    from home_cortex.persistence.edge_schema import EdgeSchemaRegistry
    from home_cortex.persistence.schema_catalog import RuntimeSchemaCatalog
    from home_cortex.semantic.schema import SemanticSchemaRegistry

    registry_ = EdgeSchemaRegistry.from_directory(schema_dir or context.schema_dir)
    catalog = RuntimeSchemaCatalog.from_data_dir(data_dir or context.data_dir, registry_)
    return semantic_prompt_fingerprint(SemanticSchemaRegistry(catalog))


def _tokens(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    normalized = []
    for row in rows:
        diagnostics = row.get("planner_diagnostics") if isinstance(row.get("planner_diagnostics"), Mapping) else {}
        prompt = diagnostics.get("prompt_eval_count", row.get("prompt_eval_count", row.get("input_tokens")))
        output = diagnostics.get("eval_count", row.get("eval_count", row.get("output_tokens")))
        normalized.append({"prompt_eval_count": prompt, "eval_count": output})
    return token_totals(normalized)


def _ignored_repeat_note(context: RunContext) -> list[str]:
    notes = []
    if context.warmup not in (None, 0) or context.repetitions not in (None, 1):
        notes.append(
            "Requested warmup or repetitions apply to the latency suite only. "
            "This component scored once."
        )
    if context.verified_cold:
        notes.append("verified_cold was set and ignored by this component.")
    return notes


def _mutation_failure(row: Mapping[str, Any]) -> str | None:
    validation = row.get("validation_error")
    if validation:
        text = str(validation)
        if "MALFORMED" in text:
            return "malformed_structured_output"
        return "validation_failure"
    if _mutation_passed(row):
        return None
    return "semantic_mismatch"


def _mutation_passed(row: Mapping[str, Any]) -> bool:
    category = row.get("category")
    if category == "write":
        return row.get("classification_correct") is True and row.get("payload_correct") is True
    if category == "mixed":
        return row.get("decision_kind") == "multi_intent" and row.get("partial_plan") is not True
    if category == "ambiguous":
        return row.get("decision_kind") != "mutation"
    if "plan_correct" in row:
        return row.get("plan_correct") is True and row.get("answer_correct") is not False
    return True


def _mutation_case_id(row: Mapping[str, Any]) -> str:
    utterance = str(row.get("utterance") or "")
    digest = hashlib.sha256(utterance.encode("utf-8")).hexdigest()[:12]
    return f"{row.get('route')}:{row.get('category')}:{row.get('sample', 0)}:{digest}"


def _mode(row: Mapping[str, Any]) -> str | None:
    expected = row.get("expected_mutation")
    if isinstance(expected, Mapping):
        mode = expected.get("mode")
        return str(mode) if mode is not None else None
    return None


def _actual_mode(row: Mapping[str, Any]) -> str | None:
    mutation = row.get("mutation")
    if isinstance(mutation, Mapping) and mutation.get("mode") is not None:
        return str(mutation.get("mode"))
    return None


def _ratio(
    metric_id: str,
    label: str,
    rows: Sequence[Mapping[str, Any]],
    predicate: Any,
    suite: str,
) -> Metric | None:
    if not rows:
        return None
    return Metric(metric_id, label, "ratio", sum(1 for row in rows if predicate(row)), len(rows), suite=suite)


def _stable_id(*parts: str) -> str:
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:12]
    return digest


def _float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _reported_cold_load(rows: Sequence[Mapping[str, Any]], *, warmup: int) -> float | None:
    for row in rows:
        if row.get("phase") not in {"verified_cold", "first_request"}:
            continue
        load = _load_duration(row)
        if load:
            return load
    if warmup == 0:
        return _load_duration(rows[0]) if rows else None
    return None


def _load_duration(row: Mapping[str, Any]) -> float | None:
    diagnostics = row.get("planner_diagnostics") if isinstance(row.get("planner_diagnostics"), Mapping) else {}
    load = diagnostics.get("load_duration_ms")
    if isinstance(load, (int, float)) and not isinstance(load, bool) and load > 0:
        return float(load)
    return None


@contextmanager
def _num_ctx(value: int | None) -> Iterator[None]:
    """Apply a process-local context override without changing production defaults."""

    if value is None:
        yield
        return
    import home_cortex.providers.ollama as ollama_mod
    import scripts.benchmarks.semantic_planner_benchmark as planner_bench

    modules = [ollama_mod, planner_bench]
    try:
        import scripts.probes.bilingual_planner_probe as bilingual

        modules.append(bilingual)
    except Exception:
        pass
    saved: list[tuple[Any, str, Any]] = []
    for module in modules:
        for name in ("PLANNER_NUM_CTX", "OLLAMA_NUM_CTX"):
            if hasattr(module, name):
                saved.append((module, name, getattr(module, name)))
                setattr(module, name, value)
    try:
        yield
    finally:
        for module, name, previous in saved:
            setattr(module, name, previous)
