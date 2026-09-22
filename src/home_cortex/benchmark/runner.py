"""Run a registered suite and record one immutable result directory."""

from __future__ import annotations

import socket
import sys
import traceback
from datetime import datetime
from typing import Any, Callable, Mapping

from .environment import (
    CACHE_STATES,
    collect_environment,
    designated_gpu_hosts,
    instruction_prompt_fingerprint,
    is_designated_gpu_host,
    planner_options,
    repo_root,
    stable_digest,
)
from .present import format_run_report
from .records import (
    allocate_run_dir,
    build_summary,
    harness_failure_result,
    merge_suite_results,
    results_root,
    write_run,
)
from .registry import registry
from .types import RunContext, RunRequest, SuiteResult

EnvironmentCollector = Callable[..., dict[str, Any]]


class CompositeSuite:
    """Run several registered suites under one name and merge their results."""

    def __init__(self, name: str, description: str, children: tuple[Any, ...]) -> None:
        self.name = name
        self.description = description
        self.children = children
        self.requires_real_model = any(child.requires_real_model for child in children)
        self.requires_gpu_host = any(child.requires_gpu_host for child in children)

    def run(self, context: RunContext) -> SuiteResult:
        parts: list[SuiteResult] = []
        for child in self.children:
            try:
                parts.append(child.run(context))
            except Exception as error:
                failed = harness_failure_result(child.name, error)
                failed.notes.append(traceback.format_exc()[-4000:])
                parts.append(failed)
        merged = merge_suite_results(self.name, parts)
        merged.components = tuple(
            component
            for part in parts
            for component in (part.components or (part.name,))
        )
        return merged


def execute(
    request: RunRequest,
    *,
    environment_collector: EnvironmentCollector = collect_environment,
) -> int:
    """Run one suite. Returns 0, 1 (harness), 2 (usage or host), or 3 (safety gate)."""

    try:
        suite = registry().get(request.suite)
    except KeyError as error:
        print(str(error), file=sys.stderr)
        return 2
    if request.cache_state not in CACHE_STATES:
        print(f"cache_state must be one of: {', '.join(CACHE_STATES)}", file=sys.stderr)
        return 2
    host_ok = is_designated_gpu_host()
    if suite.requires_gpu_host and not host_ok and not request.allow_nonstandard_host:
        print(
            f"Refusing to run {suite.name!r} on {socket.gethostname()}.\n"
            "Real-model semantic benchmarks belong on the production GPU host "
            f"({', '.join(sorted(designated_gpu_hosts()))}).\n"
            "Re-run there, or pass --allow-nonstandard-host to record a nonstandard run.",
            file=sys.stderr,
        )
        return 2
    root = repo_root()
    started = datetime.now().astimezone()
    run_id, directory = allocate_run_dir(results_root(request.results_dir))
    data_dir = request.data_dir or (root / "data")
    schema_dir = request.schema_dir or (root / "schemas" / "edge")
    try:
        environment = environment_collector(
            ollama_url=request.ollama_url,
            model=request.model,
            num_ctx=request.num_ctx,
            root=root,
        )
    except Exception as error:
        environment = {
            "git": {"git_commit": "unavailable", "git_branch": "unavailable", "git_dirty": None},
            "hardware": {"hostname": "unavailable", "os": "unavailable", "cpu": "unavailable",
                         "ram_bytes": None, "gpu": "unavailable", "gpu_vram": "unavailable",
                         "driver": "unavailable"},
            "ollama": {"version": None, "model": request.model, "available": False},
            "planner_mode": "semantic_interpreter",
            "collector_error": f"{type(error).__name__}: {error}",
        }
    nonstandard = bool(suite.requires_gpu_host and not host_ok)
    context = RunContext(
        model=request.model,
        ollama_url=request.ollama_url,
        label=request.label,
        data_dir=data_dir,
        schema_dir=schema_dir,
        results_dir=directory,
        cache_state=request.cache_state,
        warmup=request.warmup,
        repetitions=request.repetitions,
        num_ctx=request.num_ctx,
        limit=request.limit,
        verified_cold=request.verified_cold,
        allow_nonstandard_host=request.allow_nonstandard_host,
    )
    harness_trace = ""
    try:
        result = suite.run(context)
    except Exception as error:
        result = harness_failure_result(suite.name, error)
        harness_trace = traceback.format_exc()
    finished = datetime.now().astimezone()
    fingerprints = _fingerprints(result, request)
    summary = build_summary(result, cache_state=request.cache_state)
    if harness_trace:
        summary["notes"] = [*summary.get("notes", []), harness_trace[-4000:]]
    ollama = dict(environment.get("ollama") or {})
    options = dict(ollama.get("options") or planner_options(request.num_ctx))
    if request.num_ctx is not None:
        options["num_ctx"] = request.num_ctx
    ollama["options"] = options
    ollama["requested_num_ctx"] = options.get("num_ctx")
    run = {
        "run_id": run_id,
        "label": request.label,
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": finished.isoformat(timespec="seconds"),
        "home_cortex": {
            **(environment.get("git") or {}),
            "planner_mode": environment.get("planner_mode", "semantic_interpreter"),
        },
        "ollama": ollama,
        "environment": environment.get("hardware") or {},
        "suites": [request.suite],
        "components": list(result.components or (request.suite,)),
        "requirements": {
            "requires_real_model": suite.requires_real_model,
            "requires_gpu_host": suite.requires_gpu_host,
        },
        "fingerprints": fingerprints,
        "nonstandard_environment": nonstandard,
        "cache_state": request.cache_state,
        "requested_repetitions": {
            "warmup": request.warmup,
            "repetitions": request.repetitions,
            "verified_cold": request.verified_cold,
            "limit": request.limit,
        },
        "results_path": str(directory),
        "timing_policy": result.timing_policy,
    }
    report = format_run_report(run, summary)
    write_run(directory, run, summary, result.cases, report)
    print(report, end="" if report.endswith("\n") else "\n")
    gate_failed = any(gate.get("status") == "fail" for gate in summary.get("gates") or [])
    harness_failed = (summary.get("failure_counts") or {}).get("benchmark_harness_failure", 0) > 0
    if gate_failed:
        return 3
    if harness_failed:
        return 1
    return 0


def plan_matrix(spec: Mapping[str, Any], base: RunRequest) -> list[RunRequest]:
    """Expand a matrix file into sequential run requests. No scheduler."""

    if not isinstance(spec, Mapping):
        raise ValueError("Matrix file must be a mapping")
    suite = spec.get("suite")
    models = spec.get("models")
    if not isinstance(suite, str) or not suite:
        raise ValueError("Matrix file needs a suite name")
    if not isinstance(models, list) or not models or not all(isinstance(item, str) and item for item in models):
        raise ValueError("Matrix file needs a non-empty models list")
    contexts = spec.get("context_lengths") or [None]
    if not isinstance(contexts, list) or not contexts:
        raise ValueError("context_lengths must be a non-empty list when provided")
    for item in contexts:
        if item is not None and (isinstance(item, bool) or not isinstance(item, int)):
            raise ValueError("context_lengths entries must be integers")
    label = spec.get("label")
    if label is not None and not isinstance(label, str):
        raise ValueError("label must be a string")
    requests: list[RunRequest] = []
    for model in models:
        for num_ctx in contexts:
            bits = [bit for bit in (label, model, f"ctx{num_ctx}" if num_ctx is not None else None) if bit]
            requests.append(
                RunRequest(
                    suite=suite,
                    model=model,
                    ollama_url=base.ollama_url,
                    label="-".join(bits),
                    results_dir=base.results_dir,
                    data_dir=base.data_dir,
                    schema_dir=base.schema_dir,
                    cache_state=str(spec.get("cache_state") or base.cache_state),
                    warmup=base.warmup,
                    repetitions=base.repetitions,
                    num_ctx=num_ctx,
                    limit=base.limit,
                    verified_cold=base.verified_cold,
                    allow_nonstandard_host=base.allow_nonstandard_host,
                )
            )
    return requests


def _fingerprints(result: SuiteResult, request: RunRequest) -> dict[str, Any]:
    prompt = result.fingerprints.get("prompt") or instruction_prompt_fingerprint()
    corpus = result.fingerprints.get("corpus")
    config_payload = {
        "suite": request.suite,
        "limit": request.limit,
        "cache_state": request.cache_state,
        "verified_cold": request.verified_cold,
        "requested_warmup": request.warmup,
        "requested_repetitions": request.repetitions,
        "num_ctx": request.num_ctx or planner_options()["num_ctx"],
        "num_predict": planner_options()["num_predict"],
        "seed": planner_options()["seed"],
        "temperature": 0,
        "scoring_revision": result.fingerprints.get("scoring_revision"),
        "applied": result.fingerprints.get("config_inputs") or result.repetition,
    }
    stored = {
        "prompt": prompt,
        "corpus": corpus,
        "config": stable_digest(config_payload),
        "scoring_revision": result.fingerprints.get("scoring_revision"),
    }
    if result.fingerprints.get("prompt_by_suite"):
        stored["prompt_by_suite"] = result.fingerprints["prompt_by_suite"]
    if result.fingerprints.get("corpus_by_suite"):
        stored["corpus_by_suite"] = result.fingerprints["corpus_by_suite"]
    return stored
