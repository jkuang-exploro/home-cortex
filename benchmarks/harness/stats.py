"""Latency percentiles. The interpolation matches the existing fact benchmark."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def percentile(values: Sequence[float], fraction: float) -> float:
    """Linear percentile. ``fraction`` is 0.50 for the median and 0.95 for p95."""

    ordered = sorted(float(item) for item in values)
    if not ordered:
        raise ValueError("percentile requires at least one sample")
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def latency_summary(samples: Sequence[float]) -> dict[str, Any]:
    """Count, min, median, p95, and max. Average is intentionally omitted."""

    values = [float(item) for item in samples]
    if not values:
        return {"count": 0, "min": None, "p50": None, "p95": None, "max": None}
    return {
        "count": len(values),
        "min": round(min(values), 3),
        "p50": round(percentile(values, 0.50), 3),
        "p95": round(percentile(values, 0.95), 3),
        "max": round(max(values), 3),
    }


def token_totals(rows: Sequence[Mapping[str, Any]], *, source: str = "ollama") -> dict[str, Any]:
    """Sum runtime-reported token counts. Missing counts stay unavailable.

    Zero is treated as "the runtime did not report a count", because the
    adapter stores ``0`` when the response omits the field. Nothing is estimated.
    """

    prompts: list[int] = []
    outputs: list[int] = []
    for row in rows:
        diagnostics = row.get("planner_diagnostics")
        counts_from = diagnostics if isinstance(diagnostics, Mapping) else row
        prompt = counts_from.get("prompt_eval_count", row.get("prompt_eval_count"))
        output = counts_from.get("eval_count", row.get("eval_count"))
        if isinstance(prompt, (int, float)) and not isinstance(prompt, bool) and prompt > 0:
            prompts.append(int(prompt))
        if isinstance(output, (int, float)) and not isinstance(output, bool) and output > 0:
            outputs.append(int(output))
    if not prompts and not outputs:
        return {
            "prompt": None,
            "output": None,
            "total": None,
            "source": "unavailable",
            "estimated": False,
        }
    prompt_total = sum(prompts) if prompts else 0
    output_total = sum(outputs) if outputs else 0
    return {
        "prompt": prompt_total,
        "output": output_total,
        "total": prompt_total + output_total,
        "source": source,
        "estimated": False,
        "observed_calls": max(len(prompts), len(outputs)),
    }
