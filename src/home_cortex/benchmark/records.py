"""Immutable run directories and the one summary builder the CLI prints from."""

from __future__ import annotations

import json
import secrets
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from .stats import latency_summary
from .taxonomy import FAILURE_TYPES
from .types import CaseRecord, Metric, SuiteResult

_SECRET_MARKERS = ("api_key", "secret", "password", "token", "authorization")

# Counts that must be zero. A missing metric means the suite did not measure it.
ABSOLUTE_GATES = {
    "preview_as_commit": "Preview was compiled as commit",
    "mixed_partial_plans": "Multi-intent turn produced a partial plan",
}


def make_run_id(now: datetime | None = None, suffix: str | None = None) -> str:
    """``YYYYMMDD-HHMMSS`` plus four hex characters. The clock is the machine's."""

    moment = now or datetime.now().astimezone()
    token = suffix if suffix is not None else secrets.token_hex(2)
    return f"{moment.strftime('%Y%m%d-%H%M%S')}-{token}"


def results_root(path: Path | None = None) -> Path:
    if path is not None:
        return path
    from .environment import repo_root

    return repo_root() / "benchmarks" / "results"


def allocate_run_dir(root: Path, run_id: str | None = None) -> tuple[str, Path]:
    """Create a new directory. A collision picks a new id instead of overwriting."""

    root.mkdir(parents=True, exist_ok=True)
    candidate = run_id or make_run_id()
    for _ in range(5):
        path = root / candidate
        try:
            path.mkdir()
        except FileExistsError:
            candidate = make_run_id()
            continue
        return candidate, path
    raise FileExistsError(f"Could not allocate a run directory under {root}")


def harness_failure_result(name: str, error: BaseException) -> SuiteResult:
    """A suite that raised is a harness failure, not a semantic miss."""

    message = f"{type(error).__name__}: {error}"
    return SuiteResult(
        name=name,
        components=(name,),
        cases=[],
        metrics=[],
        fingerprints={},
        tokens={
            "prompt": None,
            "output": None,
            "total": None,
            "source": "unavailable",
            "estimated": False,
        },
        repetition={},
        timing_policy="",
        failure_overrides={"benchmark_harness_failure": 1},
        notes=[message[:500]],
    )


def merge_suite_results(name: str, parts: Sequence[SuiteResult]) -> SuiteResult:
    prompt_by = {
        part.name: part.fingerprints["prompt"]
        for part in parts
        if part.fingerprints.get("prompt")
    }
    corpus_by = {
        part.name: part.fingerprints["corpus"]
        for part in parts
        if part.fingerprints.get("corpus")
    }
    scoring = next(
        (
            part.fingerprints.get("scoring_revision")
            for part in parts
            if part.fingerprints.get("scoring_revision")
        ),
        None,
    )
    components: list[str] = []
    for part in parts:
        components.extend(part.components or (part.name,))
    cold = {
        part.name: part.cold_load_ms
        for part in parts
        if part.cold_load_ms is not None
    }
    notes: list[str] = []
    overrides: Counter[str] = Counter()
    for part in parts:
        notes.extend(part.notes)
        overrides.update(part.failure_overrides)
    return SuiteResult(
        name=name,
        components=tuple(components),
        cases=[case for part in parts for case in part.cases],
        metrics=[metric for part in parts for metric in part.metrics],
        fingerprints={
            "prompt": stable_prompt(prompt_by),
            "prompt_by_suite": prompt_by,
            "corpus": stable_prompt(corpus_by),
            "corpus_by_suite": corpus_by,
            "scoring_revision": scoring,
            "config_inputs": {
                part.name: part.fingerprints.get("config_inputs", {}) for part in parts
            },
        },
        tokens=merge_tokens(parts),
        repetition={part.name: part.repetition for part in parts},
        timing_policy="\n".join(
            f"{part.name}: {part.timing_policy}" for part in parts if part.timing_policy
        ),
        routes={part.name: part.routes for part in parts if part.routes},
        cold_load_ms=next((part.cold_load_ms for part in parts if part.cold_load_ms), None),
        failure_overrides=dict(overrides),
        notes=notes + ([f"cold_load_ms_by_suite={cold}"] if cold else []),
    )


def build_summary(result: SuiteResult, *, cache_state: str) -> dict[str, Any]:
    """Aggregate stored cases and runner-supplied metrics. This is the only scorer."""

    scoring = [
        case for case in result.cases if case.metrics.get("counts_toward_score", True)
    ]
    failure_counts = {name: 0 for name in FAILURE_TYPES}
    for case in scoring:
        if case.failure_type:
            failure_counts[case.failure_type] = failure_counts.get(case.failure_type, 0) + 1
    for name, count in result.failure_overrides.items():
        failure_counts[name] = failure_counts.get(name, 0) + count
    measured = [
        case.latency_ms
        for case in result.cases
        if case.latency_ms is not None and case.metrics.get("phase", "measured") == "measured"
    ]
    by_suite: dict[str, list[float]] = {}
    for case in result.cases:
        if case.latency_ms is None or case.metrics.get("phase", "measured") != "measured":
            continue
        by_suite.setdefault(case.suite, []).append(case.latency_ms)
    metrics = [metric.to_dict() for metric in result.metrics]
    metrics.append(
        {
            "id": "validation_failures",
            "label": "Validation failures",
            "kind": "count",
            "correct": None,
            "scored": None,
            "value": failure_counts.get("validation_failure", 0),
            "lower_is_better": True,
            "suite": result.name,
        }
    )
    return {
        "case_totals": {
            "cases": len(scoring),
            "passed": sum(1 for case in scoring if case.passed),
        },
        "metrics": metrics,
        "failure_counts": failure_counts,
        "latency_ms": latency_summary(measured),
        "latency_by_suite": {
            name: latency_summary(samples) for name, samples in sorted(by_suite.items())
        },
        "tokens": result.tokens,
        "gates": evaluate_absolute_gates(metrics),
        "timing_policy": result.timing_policy,
        "cache_state": cache_state,
        "cache_policy": (
            "The harness records cache_state and does not flush Ollama, GPU, or OS "
            "caches. Compare latency only across runs that record the same state "
            "and the same warmup policy."
        ),
        "repetition": result.repetition,
        "cold_load_ms": result.cold_load_ms,
        "routes": result.routes,
        "notes": result.notes,
        "scoring_note": (
            "Plan and answer ratios are copied from the existing runners. "
            "Harness failures are counted separately and are not semantic mismatches. "
            "There is no single aggregate score."
        ),
    }


def evaluate_absolute_gates(metrics: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    by_id = {str(metric["id"]): metric for metric in metrics}
    gates: list[dict[str, str]] = []
    for metric_id, label in ABSOLUTE_GATES.items():
        metric = by_id.get(metric_id)
        if metric is None or metric.get("value") is None:
            continue
        value = int(metric["value"])
        gates.append(
            {
                "id": metric_id,
                "label": label,
                "status": "pass" if value == 0 else "fail",
                "detail": str(value),
            }
        )
    return gates


def write_run(
    directory: Path,
    run: Mapping[str, Any],
    summary: Mapping[str, Any],
    cases: Sequence[CaseRecord],
    report: str,
) -> None:
    """Write the run files. Refuses to replace an existing ``run.json``."""

    target = directory / "run.json"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")
    _dump(target, run)
    _dump(directory / "summary.json", summary)
    with (directory / "cases.jsonl").open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(
                json.dumps(redact(case.to_dict()), ensure_ascii=False) + "\n"
            )
    (directory / "stdout.log").write_text(report, encoding="utf-8")


def load_run(directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    return _load(directory / "run.json"), _load(directory / "summary.json")


def load_cases(directory: Path) -> list[dict[str, Any]]:
    path = directory / "cases.jsonl"
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_baselines(root: Path) -> dict[str, str]:
    path = root / "baselines.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must be an object of name to run id")
    return {str(key): str(value) for key, value in payload.items()}


def write_baseline(root: Path, name: str, run_id: str) -> None:
    """Explicit promotion. This never replaces a run directory."""

    root.mkdir(parents=True, exist_ok=True)
    current = load_baselines(root)
    current[name] = run_id
    _dump(root / "baselines.json", current)


def resolve_run(root: Path, token: str) -> Path:
    direct = Path(token)
    if direct.is_dir() and (direct / "run.json").is_file():
        return direct
    candidate = root / token
    if candidate.is_dir() and (candidate / "run.json").is_file():
        return candidate
    baselines = load_baselines(root)
    if token in baselines:
        named = root / baselines[token]
        if named.is_dir() and (named / "run.json").is_file():
            return named
        raise FileNotFoundError(
            f"Baseline {token!r} points at missing run {baselines[token]}"
        )
    raise FileNotFoundError(f"No benchmark run named {token!r} under {root}")


def redact(value: Any) -> Any:
    """Drop secret-looking keys before anything is written."""

    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if any(marker in str(key).lower() for marker in _SECRET_MARKERS):
                cleaned[str(key)] = "[redacted]"
            else:
                cleaned[str(key)] = redact(item)
        return cleaned
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, tuple):
        return [redact(item) for item in value]
    return value


def stable_prompt(by_suite: Mapping[str, str]) -> str | None:
    if not by_suite:
        return None
    from .environment import stable_digest

    return stable_digest(dict(by_suite))


def merge_tokens(parts: Sequence[SuiteResult]) -> dict[str, Any]:
    prompts = [
        part.tokens["prompt"]
        for part in parts
        if isinstance(part.tokens.get("prompt"), int)
    ]
    outputs = [
        part.tokens["output"]
        for part in parts
        if isinstance(part.tokens.get("output"), int)
    ]
    if not prompts and not outputs:
        return {
            "prompt": None,
            "output": None,
            "total": None,
            "source": "unavailable",
            "estimated": False,
        }
    prompt = sum(prompts) if prompts else 0
    output = sum(outputs) if outputs else 0
    sources = {str(part.tokens.get("source")) for part in parts if part.tokens.get("source") != "unavailable"}
    return {
        "prompt": prompt,
        "output": output,
        "total": prompt + output,
        "source": next(iter(sources)) if len(sources) == 1 else "mixed",
        "estimated": False,
        "partial": len(prompts) != len(parts) or len(outputs) != len(parts),
    }


def metric_index(summary: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = summary.get("metrics")
    if not isinstance(rows, list):
        return {}
    return {
        str(row["id"]): row
        for row in rows
        if isinstance(row, Mapping) and row.get("id")
    }


def _dump(path: Path, payload: Mapping[str, Any] | list[Any]) -> None:
    path.write_text(
        json.dumps(redact(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain an object")
    return payload
