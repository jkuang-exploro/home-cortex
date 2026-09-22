"""Compare two stored runs. Ratios stay separate; nothing is collapsed into one score."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .records import metric_index

# Safety metrics whose correct-count must not fall. Other ratios are reported only.
MUST_NOT_DECREASE = (
    "preview_correctness",
    "commit_correctness",
    "rejection_correctness",
    "multi_intent_correctness",
)
MUST_NOT_INCREASE = (
    "preview_as_commit",
    "mixed_partial_plans",
)


@dataclass
class ComparisonRow:
    label: str
    baseline: str
    candidate: str
    delta: str
    section: str = "metric"


@dataclass
class Comparison:
    warnings: list[str] = field(default_factory=list)
    rows: list[ComparisonRow] = field(default_factory=list)
    gates: list[dict[str, str]] = field(default_factory=list)
    exit_code: int = 0


def build_comparison(
    baseline_run: Mapping[str, Any],
    baseline_summary: Mapping[str, Any],
    candidate_run: Mapping[str, Any],
    candidate_summary: Mapping[str, Any],
) -> Comparison:
    comparison = Comparison(warnings=_warnings(baseline_run, candidate_run))
    base_metrics = metric_index(baseline_summary)
    cand_metrics = metric_index(candidate_summary)
    seen: set[str] = set()
    for metric_id in list(base_metrics) + list(cand_metrics):
        if metric_id in seen:
            continue
        seen.add(metric_id)
        comparison.rows.append(
            _metric_row(metric_id, base_metrics.get(metric_id), cand_metrics.get(metric_id))
        )
    comparison.rows.extend(_latency_rows(baseline_summary, candidate_summary))
    comparison.gates = _regression_gates(base_metrics, cand_metrics, candidate_summary)
    if any(gate["status"] == "fail" for gate in comparison.gates):
        comparison.exit_code = 3
    return comparison


def _metric_row(
    metric_id: str,
    baseline: Mapping[str, Any] | None,
    candidate: Mapping[str, Any] | None,
) -> ComparisonRow:
    label = str(
        (candidate or baseline or {}).get("label") or metric_id
    )
    if _kind(baseline, candidate) == "ratio":
        return ComparisonRow(
            label=label,
            baseline=_format_ratio(baseline),
            candidate=_format_ratio(candidate),
            delta=_ratio_delta(baseline, candidate),
        )
    return ComparisonRow(
        label=label,
        baseline=_format_count(baseline),
        candidate=_format_count(candidate),
        delta=_count_delta(baseline, candidate),
    )


def _latency_rows(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> list[ComparisonRow]:
    rows: list[ComparisonRow] = []
    base_by = baseline.get("latency_by_suite") or {}
    cand_by = candidate.get("latency_by_suite") or {}
    # Overall first, then per suite. Per-suite samples live under latency_by_suite.
    overall_base = baseline.get("latency_ms") if isinstance(baseline.get("latency_ms"), Mapping) else {}
    overall_cand = candidate.get("latency_ms") if isinstance(candidate.get("latency_ms"), Mapping) else {}
    for stat, title in (("min", "Min"), ("p50", "P50"), ("p95", "P95"), ("max", "Max")):
        rows.append(
            ComparisonRow(
                label=title,
                baseline=_format_ms(overall_base.get(stat)),
                candidate=_format_ms(overall_cand.get(stat)),
                delta=_latency_delta(overall_base.get(stat), overall_cand.get(stat)),
                section="latency",
            )
        )
    if isinstance(base_by, Mapping) and isinstance(cand_by, Mapping):
        for name in sorted(set(base_by) | set(cand_by)):
            left = base_by.get(name) if isinstance(base_by.get(name), Mapping) else {}
            right = cand_by.get(name) if isinstance(cand_by.get(name), Mapping) else {}
            for stat, title in (("p50", "P50"), ("p95", "P95")):
                rows.append(
                    ComparisonRow(
                        label=f"{name} {title}",
                        baseline=_format_ms(left.get(stat)),
                        candidate=_format_ms(right.get(stat)),
                        delta=_latency_delta(left.get(stat), right.get(stat)),
                        section="latency",
                    )
                )
    return rows


def _regression_gates(
    baseline: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
    candidate_summary: Mapping[str, Any],
) -> list[dict[str, str]]:
    gates: list[dict[str, str]] = []
    for metric_id in MUST_NOT_DECREASE:
        if metric_id not in baseline or metric_id not in candidate:
            continue
        before = baseline[metric_id].get("correct")
        after = candidate[metric_id].get("correct")
        if before is None or after is None:
            continue
        failed = int(after) < int(before)
        gates.append(
            {
                "id": metric_id,
                "label": str(candidate[metric_id].get("label") or metric_id),
                "status": "fail" if failed else "pass",
                "detail": f"{before} -> {after}",
            }
        )
    for metric_id in MUST_NOT_INCREASE:
        if metric_id not in baseline or metric_id not in candidate:
            continue
        before = baseline[metric_id].get("value")
        after = candidate[metric_id].get("value")
        if before is None or after is None:
            continue
        failed = int(after) > int(before)
        gates.append(
            {
                "id": metric_id,
                "label": str(candidate[metric_id].get("label") or metric_id),
                "status": "fail" if failed else "pass",
                "detail": f"{before} -> {after}",
            }
        )
    for gate in candidate_summary.get("gates") or ():
        if isinstance(gate, Mapping) and gate.get("status") == "fail":
            gates.append(
                {
                    "id": str(gate.get("id")),
                    "label": str(gate.get("label") or gate.get("id")),
                    "status": "fail",
                    "detail": f"candidate absolute gate failed ({gate.get('detail')})",
                }
            )
    return gates


def _warnings(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> list[str]:
    warnings: list[str] = []
    base_fp = baseline.get("fingerprints") or {}
    cand_fp = candidate.get("fingerprints") or {}
    if isinstance(base_fp, Mapping) and isinstance(cand_fp, Mapping):
        if _differ(base_fp.get("corpus"), cand_fp.get("corpus")):
            warnings.append(
                "Benchmark corpus fingerprints differ. This comparison may not be apples-to-apples."
            )
        if _differ(base_fp.get("prompt"), cand_fp.get("prompt")):
            warnings.append(
                "Prompt fingerprints differ. This comparison may not be apples-to-apples."
            )
        if _differ(base_fp.get("config"), cand_fp.get("config")):
            warnings.append(
                "Configuration fingerprints differ. This comparison may not be apples-to-apples."
            )
        if _differ(base_fp.get("scoring_revision"), cand_fp.get("scoring_revision")):
            warnings.append("Scoring revisions differ.")
    base_git = _section(baseline, "home_cortex")
    cand_git = _section(candidate, "home_cortex")
    if _differ(base_git.get("git_commit"), cand_git.get("git_commit")):
        warnings.append("Home Cortex git commits differ.")
    if base_git.get("git_dirty") is True or cand_git.get("git_dirty") is True:
        warnings.append("At least one run has git_dirty = true.")
    base_hw = _section(baseline, "environment")
    cand_hw = _section(candidate, "environment")
    if _differ(base_hw.get("hostname"), cand_hw.get("hostname")):
        warnings.append("Hostnames differ.")
    if _differ(base_hw.get("gpu"), cand_hw.get("gpu")):
        warnings.append("GPU identities differ.")
    if baseline.get("nonstandard_environment") is True or candidate.get("nonstandard_environment") is True:
        warnings.append("At least one run has nonstandard_environment = true.")
    if _differ(baseline.get("cache_state"), candidate.get("cache_state")):
        warnings.append("Cache states differ. Latency deltas may include cache effects.")
    return warnings


def _kind(baseline: Mapping[str, Any] | None, candidate: Mapping[str, Any] | None) -> str:
    for item in (candidate, baseline):
        if item is not None and item.get("kind"):
            return str(item["kind"])
    return "count"


def _format_ratio(metric: Mapping[str, Any] | None) -> str:
    if metric is None or metric.get("scored") in (None, 0):
        return "n/a"
    return f"{metric.get('correct')}/{metric.get('scored')}"


def _format_count(metric: Mapping[str, Any] | None) -> str:
    if metric is None or metric.get("value") is None:
        return "n/a"
    return str(metric["value"])


def _ratio_delta(
    baseline: Mapping[str, Any] | None,
    candidate: Mapping[str, Any] | None,
) -> str:
    if baseline is None or candidate is None:
        return "n/a"
    if baseline.get("correct") is None or candidate.get("correct") is None:
        return "n/a"
    delta = int(candidate["correct"]) - int(baseline["correct"])
    text = f"{delta:+d}"
    if baseline.get("scored") != candidate.get("scored"):
        text += " (scored counts differ)"
    return text


def _count_delta(
    baseline: Mapping[str, Any] | None,
    candidate: Mapping[str, Any] | None,
) -> str:
    if (
        baseline is None
        or candidate is None
        or baseline.get("value") is None
        or candidate.get("value") is None
    ):
        return "n/a"
    return f"{int(candidate['value']) - int(baseline['value']):+d}"


def _format_ms(value: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "n/a"
    if abs(value) >= 1000:
        return f"{value / 1000:.2f} s"
    return f"{value:.0f} ms"


def _latency_delta(baseline: Any, candidate: Any) -> str:
    if not isinstance(baseline, (int, float)) or not isinstance(candidate, (int, float)):
        return "n/a"
    if isinstance(baseline, bool) or isinstance(candidate, bool) or baseline == 0:
        return "n/a"
    percent = (float(candidate) - float(baseline)) / float(baseline) * 100
    return f"{percent:+.0f}%"


def _differ(left: Any, right: Any) -> bool:
    if left in (None, "", "unavailable") or right in (None, "", "unavailable"):
        return False
    return left != right


def _section(run: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = run.get(key)
    return value if isinstance(value, Mapping) else {}
