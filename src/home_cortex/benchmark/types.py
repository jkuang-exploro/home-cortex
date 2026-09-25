"""Shared benchmark records. These types carry results; they do not score them."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Metric:
    """One headline measurement copied from an authoritative runner."""

    id: str
    label: str
    kind: str
    correct: int | None = None
    scored: int | None = None
    value: int | None = None
    lower_is_better: bool = False
    suite: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CaseRecord:
    """Sanitized per-case row. Household answer text does not belong here."""

    suite: str
    case_id: str
    passed: bool
    latency_ms: float | None = None
    expected: Any = None
    actual: Any = None
    failure_type: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "case_id": self.case_id,
            "passed": self.passed,
            "latency_ms": self.latency_ms,
            "expected": self.expected,
            "actual": self.actual,
            "failure_type": self.failure_type,
            "metrics": self.metrics,
        }


@dataclass
class SuiteResult:
    """What one suite, or a composite of suites, returns to the recorder."""

    name: str
    components: tuple[str, ...]
    cases: list[CaseRecord]
    metrics: list[Metric]
    fingerprints: dict[str, Any]
    tokens: dict[str, Any]
    repetition: dict[str, Any]
    timing_policy: str
    routes: dict[str, Any] = field(default_factory=dict)
    cold_load_ms: float | None = None
    failure_overrides: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class RunContext:
    """Inputs a suite adapter may use. Suites choose which fields apply."""

    model: str
    ollama_url: str
    label: str | None
    data_dir: Path
    schema_dir: Path
    results_dir: Path
    cache_state: str
    warmup: int | None
    repetitions: int | None
    num_ctx: int | None
    limit: int | None
    verified_cold: bool
    allow_nonstandard_host: bool
    runtime: str = "ollama"
    digest_cache: dict[str, str] = field(default_factory=dict)
    progress: Any = None


@dataclass
class RunRequest:
    """CLI-level run request before defaults are filled in."""

    suite: str
    model: str
    ollama_url: str | None = None
    ollama_url_explicit: bool = False
    label: str | None = None
    results_dir: Path | None = None
    data_dir: Path | None = None
    schema_dir: Path | None = None
    cache_state: str = "unknown"
    warmup: int | None = None
    repetitions: int | None = None
    num_ctx: int | None = None
    limit: int | None = None
    verified_cold: bool = False
    allow_nonstandard_host: bool = False
    runtime: str | None = None
    base_url: str | None = None
