"""Text rendering. The numbers come from the stored summary and comparison objects."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .compare import Comparison


def format_run_report(run: Mapping[str, Any], summary: Mapping[str, Any]) -> str:
    """Human report for ``run`` and ``show``. Built only from stored objects."""

    ollama = run.get("ollama") if isinstance(run.get("ollama"), Mapping) else {}
    git = run.get("home_cortex") if isinstance(run.get("home_cortex"), Mapping) else {}
    fingerprints = run.get("fingerprints") if isinstance(run.get("fingerprints"), Mapping) else {}
    runtime = run.get("runtime") if isinstance(run.get("runtime"), Mapping) else {}
    local_model = runtime.get("model") if isinstance(runtime.get("model"), Mapping) else {}
    lines = [
        "Home Cortex Benchmark",
        f"Run ID: {run.get('run_id')}",
        f"Label: {run.get('label') or '-'}",
        f"Suite: {', '.join(run.get('suites') or [])}",
        f"Model: {local_model.get('name') or ollama.get('model')}",
        f"Runtime: {runtime.get('type') or 'ollama'}",
        f"Runtime URL: {runtime.get('base_url') or ollama.get('base_url') or 'unknown'}",
        f"Runtime version: {runtime.get('version') or ollama.get('version') or 'unavailable'}",
        f"Runtime commit: {runtime.get('commit') or '-'}",
        f"Model tag: {ollama.get('tag') or '-'}",
        f"Model digest: {local_model.get('sha256') or ollama.get('digest') or 'unavailable'}",
        f"Quantization: {local_model.get('quantization') or ollama.get('quantization') or 'unavailable'}",
        f"Context length: {runtime.get('context_length') or ollama.get('context_length') or 'unavailable'}",
        f"Requested num_ctx: {ollama.get('requested_num_ctx')}",
        f"Home Cortex commit: {git.get('git_commit')}",
        f"Branch: {git.get('git_branch')}",
        _dirty_line(git.get("git_dirty")),
        f"nonstandard_environment = {str(bool(run.get('nonstandard_environment'))).lower()}",
        f"Prompt fingerprint: {fingerprints.get('prompt')}",
        f"Corpus fingerprint: {fingerprints.get('corpus')}",
        f"Config fingerprint: {fingerprints.get('config')}",
        f"Cache state: {run.get('cache_state')}",
        f"Results: {run.get('results_path')}",
        f"Progress: {run.get('results_path')}/progress.log",
        "",
        "Semantic results",
    ]
    for metric in summary.get("metrics") or []:
        if not isinstance(metric, Mapping) or metric.get("kind") != "ratio":
            continue
        scored = metric.get("scored")
        rendered = "n/a" if not scored else f"{metric.get('correct')}/{scored}"
        lines.append(f"  {metric.get('label')}: {rendered}")
    totals = summary.get("case_totals") or {}
    lines.append(
        f"  Cases passed: {totals.get('passed')}/{totals.get('cases')}"
    )
    lines.extend(["", "Latency"])
    latency = summary.get("latency_ms") if isinstance(summary.get("latency_ms"), Mapping) else {}
    for key in ("count", "min", "p50", "p95", "max"):
        lines.append(f"  {key}: {latency.get(key)}")
    by_suite = summary.get("latency_by_suite") or {}
    if isinstance(by_suite, Mapping):
        for name, stats in by_suite.items():
            if isinstance(stats, Mapping):
                lines.append(
                    f"  {name} p50/p95: {stats.get('p50')} / {stats.get('p95')} ms"
                )
    cold = summary.get("cold_load_ms")
    lines.append(f"  cold model load: {cold if cold is not None else 'not reported'}")
    lines.extend(["", "Failures"])
    counts = summary.get("failure_counts") or {}
    if isinstance(counts, Mapping):
        for name, count in counts.items():
            lines.append(f"  {name}: {count}")
    tokens = summary.get("tokens") if isinstance(summary.get("tokens"), Mapping) else {}
    lines.extend(["", "Tokens"])
    if tokens.get("source") in {"ollama", "llama.cpp"}:
        lines.append(f"  prompt: {tokens.get('prompt')}")
        lines.append(f"  output: {tokens.get('output')}")
        lines.append(f"  total: {tokens.get('total')}")
        lines.append("  estimated: false")
    else:
        lines.append("  unavailable (not estimated)")
    gates = summary.get("gates") or []
    if gates:
        lines.extend(["", "Gates"])
        for gate in gates:
            if isinstance(gate, Mapping):
                lines.append(
                    f"  {gate.get('status')}: {gate.get('label')} ({gate.get('detail')})"
                )
    if summary.get("timing_policy"):
        lines.extend(["", "Timing policy", str(summary["timing_policy"])])
    if summary.get("notes"):
        lines.extend(["", "Notes"])
        for note in summary["notes"]:
            lines.append(f"  {note}")
    lines.append("")
    return "\n".join(lines)


def _dirty_line(dirty: Any) -> str:
    if dirty is True:
        return "git_dirty = true"
    if dirty is False:
        return "git_dirty = false"
    return "git_dirty = unavailable"


def format_failures(cases: Sequence[Mapping[str, Any]]) -> str:
    lines = ["Failures"]
    found = False
    for case in cases:
        if case.get("passed") is True and not case.get("failure_type"):
            continue
        found = True
        lines.append(
            "  {suite}  {case_id}  {failure}  expected={expected}  actual={actual}".format(
                suite=case.get("suite"),
                case_id=case.get("case_id"),
                failure=case.get("failure_type") or "failed",
                expected=case.get("expected"),
                actual=case.get("actual"),
            )
        )
    if not found:
        lines.append("  none")
    return "\n".join(lines)


def format_comparison(comparison: Comparison, baseline_run: Mapping[str, Any], candidate_run: Mapping[str, Any]) -> str:
    lines = ["Home Cortex Benchmark Comparison", ""]
    for warning in comparison.warnings:
        lines.append(f"WARNING: {warning}")
    if comparison.warnings:
        lines.append("")
    lines.append(f"{'Metric':<28}{'Baseline':>14}{'Candidate':>16}{'Delta':>12}")
    lines.append("-" * 70)
    metric_rows = [row for row in comparison.rows if row.section == "metric"]
    for row in metric_rows:
        lines.append(f"{row.label:<28}{row.baseline:>14}{row.candidate:>16}{row.delta:>12}")
    lines.extend(["", "Latency"])
    for row in comparison.rows:
        if row.section == "latency":
            lines.append(f"{row.label:<28}{row.baseline:>14}{row.candidate:>16}{row.delta:>12}")
    base_ollama = baseline_run.get("runtime") or baseline_run.get("ollama") or {}
    cand_ollama = candidate_run.get("runtime") or candidate_run.get("ollama") or {}
    lines.extend(
        [
            "",
            "Runtime",
            f"Baseline                    {base_ollama.get('type') or 'ollama'} {base_ollama.get('version') or 'unavailable'}",
            f"Candidate                   {cand_ollama.get('type') or 'ollama'} {cand_ollama.get('version') or 'unavailable'}",
        ]
    )
    if comparison.gates:
        lines.extend(["", "Gates"])
        for gate in comparison.gates:
            lines.append(
                f"  {gate['status'].upper()}: {gate['label']} ({gate['detail']})"
            )
    lines.append("")
    return "\n".join(lines)
