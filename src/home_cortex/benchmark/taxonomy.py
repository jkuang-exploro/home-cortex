"""Failure classes for benchmark cases.

Harness and provider failures stay out of the semantic-mismatch bucket.
A case has one primary class so the counts are not double-reported.
"""

from __future__ import annotations

from typing import Mapping

FAILURE_TYPES = (
    "semantic_mismatch",
    "validation_failure",
    "malformed_structured_output",
    "timeout",
    "context_overflow",
    "ollama_runtime_error",
    "provider_error",
    "tool_failure",
    "benchmark_harness_failure",
)

_ACCEPTABLE_VALIDATION = frozenset({"VALID", "NOT_A_FACT", None, ""})
_OLLAMA_RUNTIME = (
    "Ollama",
    "URLError",
    "ConnectionError",
    "ConnectError",
    "ResponseError",
)


def classify_planner_failure(row: Mapping[str, object]) -> str | None:
    """Classify one planner or probe row. ``None`` means the case passed."""

    runtime = str(row.get("runtime_failure") or "")
    validation = row.get("validation_result")
    diagnostics = row.get("planner_diagnostics")
    done = ""
    if isinstance(diagnostics, Mapping):
        done = str(diagnostics.get("done_reason") or "")
    plan_match = row.get("plan_match")
    answer = row.get("answer_correct")
    if plan_match is True and answer is not False and not runtime:
        return None
    if "Timeout" in runtime or "timeout" in runtime.lower():
        return "timeout"
    if done == "length":
        return "context_overflow"
    if validation == "MALFORMED_OUTPUT" or "MALFORMED" in str(validation or ""):
        return "malformed_structured_output"
    if runtime:
        name = runtime.split(":", 1)[-1]
        if "Tool" in name:
            return "tool_failure"
        if "Provider" in name or "OpenRouter" in name:
            return "provider_error"
        if any(token in name for token in _OLLAMA_RUNTIME):
            return "ollama_runtime_error"
        return "benchmark_harness_failure"
    if validation not in _ACCEPTABLE_VALIDATION:
        return "validation_failure"
    if plan_match is False or answer is False:
        return "semantic_mismatch"
    return None
