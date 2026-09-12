"""Planner-only semantic-IR quality benchmark."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import statistics
import subprocess
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence
from zoneinfo import ZoneInfo

import yaml

from .agents import get_agent
from .config import get_settings
from .edge_schema import EdgeSchemaRegistry
from .fact_benchmark import _JsonGraphDispatcher, _percentile
from .ollama import (
    PLANNER_KEEP_ALIVE,
    PLANNER_NUM_CTX,
    PLANNER_NUM_PREDICT,
    PLANNER_SEED,
    OllamaService,
)
from .schema_catalog import RuntimeSchemaCatalog
from .semantic_ir import (
    AgentRequestContext,
    FactEvidence,
    FactResult,
    FactRow,
    FactRelationshipEvidence,
    SemanticFactRequest,
    SemanticPlannerFailure,
)
from .household_fact_engine import HouseholdFactEngine
from .semantic_planner import SemanticFactPlanner, planner_input_summary
from .semantic_facts import SemanticFactService
from .semantic_schema import SemanticSchemaRegistry

FROZEN_EVAL_TIME = "2026-09-03T12:00:00-07:00"
SCORING_REVISION = "2026-09-07.2-composition-shapes"

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
    case_id: str = ""
    expected_status: str | None = None
    expected_entity_ids: tuple[str, ...] | None = None
    expected_value: Any = None
    expected_names: tuple[str, ...] | None = None
    expected_equal: bool | None = None
    notes: str | None = None
    expected_unit: str | None = None
    expected_rows: tuple[Mapping[str, Any], ...] | None = None

    def __post_init__(self) -> None:
        if not self.case_id:
            object.__setattr__(self, "case_id", f"{self.plan_id}::{self.utterance}")


@dataclass(frozen=True)
class ProbeDataset:
    frozen_time: datetime
    default_speaker_id: str
    household_id: str
    cases: tuple[SemanticEvalCase, ...]
    path: Path
    version: int


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
                    case_id=f"{key}::{utterance}",
                )
            )
    return tuple(cases)


def load_bilingual_dataset(
    path: Path | None = None,
) -> Mapping[str, Any]:
    """Load Chinese/English/mixed parity cases. Not a serving phrase table."""
    target = path or (
        Path(__file__).resolve().parents[2]
        / "benchmarks"
        / "semantic_planner_bilingual.yaml"
    )
    raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("version") != 1:
        raise ValueError("bilingual planner dataset must have version 1")
    return raw


def load_probe_dataset(path: Path = DEFAULT_EVAL_PATH) -> ProbeDataset:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("version") != 1:
        raise ValueError("semantic planner evaluation dataset must have version 1")
    plans = raw.get("plans")
    probe = raw.get("probe")
    if not isinstance(plans, Mapping) or not isinstance(probe, Mapping):
        raise ValueError("evaluation dataset is missing plans or probe")
    frozen_raw = probe.get("frozen_time") or FROZEN_EVAL_TIME
    frozen_time = datetime.fromisoformat(str(frozen_raw))
    default_speaker_id = str(probe.get("default_speaker_id") or "person:jian_kuang")
    household_id = str(probe.get("household_id") or "address:fort_cerritos")
    rows = probe.get("cases")
    if not isinstance(rows, list) or not rows:
        raise ValueError("probe.cases must be a non-empty list")
    cases: list[SemanticEvalCase] = []
    seen_ids: set[str] = set()
    for item in rows:
        if not isinstance(item, Mapping):
            raise ValueError("probe case must be an object")
        case_id = item.get("id")
        utterance = item.get("utterance")
        plan_key = item.get("plan")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("probe case is missing id")
        if case_id in seen_ids:
            raise ValueError(f"duplicate probe case id: {case_id}")
        seen_ids.add(case_id)
        if not isinstance(utterance, str) or not utterance.strip():
            raise ValueError(f"probe case {case_id} is missing utterance")
        if not isinstance(plan_key, str) or plan_key not in plans:
            raise ValueError(f"probe case {case_id} references unknown plan {plan_key!r}")
        alternative_keys = item.get("acceptable_plans", [])
        if not isinstance(alternative_keys, list) or not all(
            isinstance(key, str) for key in alternative_keys
        ):
            raise ValueError(f"probe case {case_id} has invalid acceptable_plans")
        missing = [key for key in alternative_keys if key not in plans]
        if missing:
            raise ValueError(
                f"probe case {case_id} references unknown plans: {', '.join(missing)}"
            )
        entity_ids = item.get("expected_entity_ids")
        expected_entity_ids = (
            tuple(str(value) for value in entity_ids)
            if isinstance(entity_ids, list)
            else None
        )
        raw_names = item.get("expected_names")
        if raw_names is None:
            expected_names = None
        elif isinstance(raw_names, list) and all(
            isinstance(value, (str, int)) for value in raw_names
        ):
            expected_names = tuple(str(value) for value in raw_names)
        else:
            raise ValueError(f"probe case {case_id} has invalid expected_names")
        cases.append(
            SemanticEvalCase(
                utterance,
                str(item.get("speaker_id") or default_speaker_id),
                str(item.get("category") or _plan_category(raw, plan_key)),
                plan_key,
                SemanticFactRequest.model_validate(plans[plan_key]),
                tuple(
                    SemanticFactRequest.model_validate(plans[key])
                    for key in alternative_keys
                ),
                case_id=case_id,
                expected_unit=item.get("expected_unit"),
                expected_rows=(tuple(item["expected_rows"]) if item.get("expected_rows") is not None else None),
                expected_status=(
                    str(item["expected_status"])
                    if item.get("expected_status") is not None
                    else None
                ),
                expected_entity_ids=expected_entity_ids,
                expected_value=item.get("expected_value"),
                expected_names=expected_names,
                expected_equal=item.get("expected_equal"),
                notes=str(item["notes"]) if item.get("notes") is not None else None,
            )
        )
    return ProbeDataset(
        frozen_time=frozen_time,
        default_speaker_id=default_speaker_id,
        household_id=household_id,
        cases=tuple(cases),
        path=path,
        version=int(raw["version"]),
    )


def _plan_category(raw: Mapping[str, Any], plan_key: str) -> str:
    for group in raw.get("groups") or ():
        if isinstance(group, Mapping) and group.get("plan") == plan_key:
            category = group.get("category")
            if isinstance(category, str):
                return category
    return "unspecified"


def normalize_semantic_request(request: SemanticFactRequest) -> dict[str, Any]:
    """Canonical JSON form used only for semantic-plan comparison."""
    normalized = request.model_dump(mode="json", exclude_none=True)
    if (
        normalized.get("operation") == "select"
        and normalized.get("property") == "display_name"
        and normalized.get("property_source") == "entity"
        and normalized.get("projection", "scalar") == "scalar"
        and not normalized.get("exclude")
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
        for key in ("kind", "value", "entity_type", "turn_offset", "cardinality")
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
        for key in ("other", "mode", "from_unit", "to_unit", "projection", "exclude", "amount")
    ):
        return "PARAMETER_MISMATCH"
    return "PLAN_MISMATCH"


def classify_failure_stage(
    *,
    runtime_failure: str | None,
    validation_result: str | None,
    plan_match: bool | None,
    executor_status: str | None,
    answer_correct: bool | None = None,
) -> str | None:
    if runtime_failure:
        return "transport"
    if validation_result not in {None, "VALID", "NOT_A_FACT"}:
        return "planner_validation"
    if plan_match is False:
        return "plan_mismatch"
    if executor_status in {
        "entity_not_found",
        "ambiguous",
        "relationship_not_found",
        "caller_context_missing",
        "discourse_context_missing",
    }:
        return "entity_resolution"
    if executor_status not in {None, "found", "not_run"}:
        return "execution"
    if answer_correct is False:
        return "answer_mismatch"
    return None


def primary_entity_ids(result: FactResult) -> tuple[str, ...]:
    if result.shape == "rows":
        return tuple(dict.fromkeys(str(row.entity["id"]) for row in result.rows))
    value = result.value
    if isinstance(value, Mapping):
        selected = value.get("selected")
        if isinstance(selected, Mapping) and selected.get("id"):
            return (str(selected["id"]),)
        if value.get("id"):
            return (str(value["id"]),)
    if isinstance(value, list):
        ids = tuple(
            str(item["id"])
            for item in value
            if isinstance(item, Mapping) and item.get("id")
        )
        if ids:
            return ids
    return tuple(str(item) for item in result.evidence.entity_ids if item)


def values_match(actual: Any, expected: Any) -> bool:
    if expected is None:
        return True
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        return all(actual.get(key) == value for key, value in expected.items())
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return actual == expected
    return actual == expected


def extract_result_names(value: Any) -> tuple[str, ...]:
    if isinstance(value, str) and value.strip():
        return (value,)
    if isinstance(value, Mapping):
        names: list[str] = []
        for key in ("name", "display_name", "given_name", "first_name"):
            if key in value:
                names.extend(extract_result_names(value[key]))
        return tuple(names)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        names = []
        for item in value:
            names.extend(extract_result_names(item))
        return tuple(names)
    return ()


def names_match(actual: Any, expected_names: Sequence[str]) -> bool:
    allowed = {name for name in expected_names if name}
    extracted = extract_result_names(actual)
    return bool(allowed) and bool(extracted) and all(name in allowed for name in extracted)


def score_structured_result(result: FactResult, case: SemanticEvalCase) -> bool | None:
    has_expectation = any(
        item is not None
        for item in (
            case.expected_status,
            case.expected_entity_ids,
            case.expected_value,
            case.expected_names,
            case.expected_equal,
            case.expected_unit,
            case.expected_rows,
        )
    )
    if not has_expectation:
        return None
    if case.expected_status is not None and result.status != case.expected_status:
        return False
    if case.expected_rows is not None:
        if result.shape != "rows" or len(result.rows) != len(case.expected_rows):
            return False
        actual_rows = [{
            "entity_id": row.entity["id"], "value": row.value, "unit": row.unit,
            "status": row.status, "evidence": jsonable(asdict(row.evidence)),
            "missing_requirements": list(row.missing_requirements),
        } for row in result.rows]
        for expected in case.expected_rows:
            # Membership alone or `found` alone is not a per-row answer oracle.
            if not {"entity_id", "value", "unit", "status"}.issubset(expected):
                raise ValueError("expected row requires entity_id, value, unit and status")
            index = next((i for i, actual in enumerate(actual_rows) if values_match(actual, expected)), None)
            if index is None:
                return False
            actual_rows.pop(index)
    elif result.shape == "rows":
        return None  # New result shape requires an explicit per-row expectation.
    if case.expected_unit is not None and result.unit != case.expected_unit:
        return False
    if case.expected_entity_ids is not None:
        if set(primary_entity_ids(result)) != set(case.expected_entity_ids):
            return False
    value_ok = True
    if case.expected_value is not None or case.expected_names:
        exact = case.expected_value is not None and values_match(
            result.value, case.expected_value
        )
        names = case.expected_names is not None and names_match(
            result.value, case.expected_names
        )
        if case.expected_value is not None and case.expected_names is not None:
            value_ok = exact or names
        elif case.expected_names is not None:
            value_ok = names
        else:
            value_ok = exact
    if not value_ok:
        return False
    if case.expected_equal is not None:
        if not isinstance(result.value, Mapping):
            return False
        if bool(result.value.get("equal")) != bool(case.expected_equal):
            return False
    return True


def fact_result_from_serialized(payload: Mapping[str, Any] | None) -> FactResult | None:
    if not payload:
        return None
    status = payload.get("status")
    if not isinstance(status, str) or not status:
        return None
    candidates = tuple(
        item
        for item in (payload.get("candidates") or ())
        if isinstance(item, Mapping)
    )
    return FactResult(
        status,  # type: ignore[arg-type]
        payload.get("value"),
        FactEvidence(
            entity_ids=tuple(
                str(item) for item in (payload.get("entity_ids") or ()) if item
            ),
            relationship=payload.get("relationship"),
            semantic_property=payload.get("semantic_property"),
        ),
        missing_requirements=tuple(
            str(item) for item in (payload.get("missing_requirements") or ())
        ),
        candidates=candidates,
        unit=payload.get("unit"),
        shape=payload.get("shape", "scalar"),
        focus_entity_ids=tuple(payload.get("focus_entity_ids", ())),
        rows=tuple(FactRow(
            entity=row["entity"], status=row["status"], value=row.get("value"), unit=row.get("unit"),
            evidence=FactEvidence(
                entity_ids=tuple(row["evidence"].get("entity_ids", ())),
                relationship=row["evidence"].get("relationship"),
                semantic_property=row["evidence"].get("semantic_property"),
                relationships=tuple(FactRelationshipEvidence(**edge) for edge in row["evidence"].get("relationships", ())),
            ), missing_requirements=tuple(row.get("missing_requirements", ())),
        ) for row in payload.get("rows", ())),
    )


def rescore_saved_row(row: Mapping[str, Any], case: SemanticEvalCase) -> dict[str, Any]:
    updated = dict(row)
    result = fact_result_from_serialized(
        row.get("executor") if isinstance(row.get("executor"), Mapping) else None
    )
    if result is not None:
        answer_correct = score_structured_result(result, case)
    else:
        answer_correct = False if case.expected_status is not None else None
    updated["answer_correct"] = answer_correct
    updated["failure_stage"] = classify_failure_stage(
        runtime_failure=row.get("runtime_failure")
        if isinstance(row.get("runtime_failure"), str)
        else None,
        validation_result=row.get("validation_result")
        if isinstance(row.get("validation_result"), str)
        else None,
        plan_match=row.get("plan_match")
        if isinstance(row.get("plan_match"), bool)
        else None,
        executor_status=row.get("executor_status")
        if isinstance(row.get("executor_status"), str)
        else None,
        answer_correct=answer_correct,
    )
    updated["scoring_revision"] = SCORING_REVISION
    return updated


def rescore_probe_report(
    report: Mapping[str, Any],
    dataset: ProbeDataset,
) -> dict[str, Any]:
    cases = {case.case_id: case for case in dataset.cases}
    updated = dict(report)
    queries = [
        rescore_saved_row(row, cases[row["case_id"]])
        if isinstance(row, Mapping) and row.get("case_id") in cases
        else dict(row) if isinstance(row, Mapping) else row
        for row in report.get("queries") or ()
    ]
    updated["queries"] = queries
    measured = [row for row in queries if row.get("phase") == "measured"]
    first_pass = [row for row in measured if row.get("sample_index") == 0]
    updated["measured"] = measured
    updated["scores_original"] = report.get("scores")
    updated["scores_all_measured_samples_original"] = report.get(
        "scores_all_measured_samples"
    )
    updated["scores"] = summarize_scores(first_pass)
    updated["scores_all_measured_samples"] = summarize_scores(measured)
    updated["failure_table"] = [
        {
            "case_id": row["case_id"],
            "utterance": row["utterance"],
            "failure_stage": row["failure_stage"],
            "validation_result": row.get("validation_result"),
            "plan_match": row.get("plan_match"),
            "answer_correct": row.get("answer_correct"),
            "executor_status": row.get("executor_status"),
            "expected_semantic_plan": row.get("expected_semantic_plan"),
            "normalized_semantic_plan": row.get("normalized_semantic_plan"),
            "planner_output": row.get("planner_output"),
            "executor": row.get("executor"),
        }
        for row in first_pass
        if not row.get("plan_match") or row.get("answer_correct") is False
    ]
    updated["scoring_revision"] = SCORING_REVISION
    updated["score_change_kind"] = "evaluation_correction"
    return updated


def jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def serialize_fact_result(result: FactResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "status": result.status,
        "value": jsonable(result.value),
        "unit": result.unit,
        "shape": result.shape,
        "rows": [jsonable(asdict(row)) for row in result.rows],
        "focus_entity_ids": list(result.focus_entity_ids),
        "entity_ids": list(result.evidence.entity_ids),
        "primary_entity_ids": list(primary_entity_ids(result)),
        "relationship": result.evidence.relationship,
        "semantic_property": result.evidence.semantic_property,
        "missing_requirements": list(result.missing_requirements),
        "candidates": jsonable(result.candidates),
    }


def serialize_diagnostics(diagnostics: Any | None) -> dict[str, Any] | None:
    if diagnostics is None:
        return None
    return {
        "validation_result": getattr(diagnostics, "validation_result", None),
        "failure_detail": getattr(diagnostics, "failure_detail", None),
        "attempt_count": getattr(diagnostics, "attempt_count", None),
        "latency_ms": getattr(diagnostics, "latency_ms", None),
        "prompt_build_ms": getattr(diagnostics, "prompt_build_ms", None),
        "request_ms": getattr(diagnostics, "request_ms", None),
        "validation_ms": getattr(diagnostics, "validation_ms", None),
        "prompt_eval_count": getattr(diagnostics, "prompt_eval_count", None),
        "prompt_eval_duration_ms": getattr(diagnostics, "prompt_eval_duration_ms", None),
        "eval_count": getattr(diagnostics, "eval_count", None),
        "eval_duration_ms": getattr(diagnostics, "eval_duration_ms", None),
        "load_duration_ms": getattr(diagnostics, "load_duration_ms", None),
        "transport": jsonable(getattr(diagnostics, "transport", None)),
        "output_raw": jsonable(getattr(diagnostics, "output_raw", None)),
        "normalized_plan": jsonable(getattr(diagnostics, "normalized_plan", None)),
        "input_summary": jsonable(getattr(diagnostics, "input_summary", None)),
    }


def request_phase(index: int, *, warmup: int, verified_cold: bool) -> str:
    if index < warmup:
        if index == 0 and verified_cold:
            return "verified_cold"
        if index == 0:
            return "first_request"
        return "warmup"
    return "measured"


def summarize_latencies(samples: Sequence[float]) -> dict[str, Any]:
    values = [float(item) for item in samples]
    return {
        "n": len(values),
        "p50": round(statistics.median(values), 3) if values else None,
        "p95": round(_percentile(values, 0.95), 3) if values else None,
    }


def summarize_scores(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    plan_scored = [row for row in rows if row.get("plan_match") is not None]
    answer_scored = [row for row in rows if row.get("answer_correct") is not None]
    statuses = Counter(
        str(row.get("executor_status") or "not_run") for row in rows
    )
    return {
        "plan_accuracy": {
            "correct": sum(int(row["plan_match"]) for row in plan_scored),
            "scored": len(plan_scored),
            "unscored": len(rows) - len(plan_scored),
            "rate": (
                round(sum(int(row["plan_match"]) for row in plan_scored) / len(plan_scored), 4)
                if plan_scored
                else None
            ),
        },
        "answer_correctness": {
            "correct": sum(int(row["answer_correct"]) for row in answer_scored),
            "scored": len(answer_scored),
            "unscored": len(rows) - len(answer_scored),
            "rate": (
                round(
                    sum(int(row["answer_correct"]) for row in answer_scored)
                    / len(answer_scored),
                    4,
                )
                if answer_scored
                else None
            ),
        },
        "executor_status": dict(sorted(statuses.items())),
        "unscored_or_unverifiable": {
            "plan": [row.get("case_id") for row in rows if row.get("plan_match") is None],
            "answer": [
                row.get("case_id") for row in rows if row.get("answer_correct") is None
            ],
        },
    }


def _sha256_file(path: Path) -> str:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return "unavailable"


def _sha256_tree(root: Path | None) -> str:
    if root is None:
        return "unavailable"
    try:
        digest = hashlib.sha256()
        files = sorted(path for path in root.rglob("*") if path.is_file())
        for path in files:
            digest.update(str(path.relative_to(root)).encode())
            digest.update(b"\0")
            digest.update(_sha256_file(path).encode())
            digest.update(b"\n")
        return digest.hexdigest()
    except OSError:
        return "unavailable"


def collect_provenance(
    *,
    root: Path,
    eval_path: Path,
    ollama_url: str,
    ollama_model: str,
    backend: str,
    frozen_time: datetime,
    warmup: int,
    repeat: int,
    verified_cold: bool,
    data_dir: Path | None = None,
    schema_dir: Path | None = None,
) -> dict[str, Any]:
    git = _git_provenance(root)
    ollama = _ollama_provenance(ollama_url, ollama_model)
    package_path = Path(__file__).resolve()
    return {
        "git_commit": git.get("commit", "unavailable"),
        "git_dirty": git.get("dirty", "unavailable"),
        "host_git_commit": os.environ.get("HOST_GIT_COMMIT", "unavailable"),
        "host_git_dirty": os.environ.get("HOST_GIT_DIRTY", "unavailable"),
        "dataset_path": str(eval_path),
        "dataset_version": 1,
        "eval_sha256": _sha256_file(eval_path),
        "data_tree_sha256": _sha256_tree(data_dir),
        "schema_tree_sha256": _sha256_tree(schema_dir),
        "imported_package_path": str(package_path),
        "copied_package_sha256": os.environ.get(
            "COPIED_PACKAGE_SHA256", _sha256_tree(package_path.parent)
        ),
        "scoring_revision": SCORING_REVISION,
        "model_name": ollama_model,
        "model_digest": ollama.get("digest", "unavailable"),
        "ollama_version": ollama.get("version", "unavailable"),
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or "unavailable",
            "ollama_processor": ollama.get("processor", "unavailable"),
        },
        "request_settings": {
            "think": False,
            "temperature": 0,
            "num_ctx": PLANNER_NUM_CTX,
            "num_predict": PLANNER_NUM_PREDICT,
            "seed": PLANNER_SEED,
            "keep_alive": PLANNER_KEEP_ALIVE,
            "natural_language_path": "semantic_interpreter",
        },
        "backend": backend,
        "frozen_evaluation_time": frozen_time.isoformat(),
        "warmup": warmup,
        "repeat": repeat,
        "verified_cold": verified_cold,
    }


def _git_provenance(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unavailable", "dirty": "unavailable"}


def _ollama_json(url: str, timeout: float = 2.0) -> Any | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None


def _ollama_provenance(ollama_url: str, model: str) -> dict[str, Any]:
    base = ollama_url.rstrip("/")
    version = _ollama_json(f"{base}/api/version")
    tags = _ollama_json(f"{base}/api/tags")
    ps = _ollama_json(f"{base}/api/ps")
    digest = "unavailable"
    processor = "unavailable"
    if isinstance(tags, Mapping):
        for item in tags.get("models") or ():
            if isinstance(item, Mapping) and item.get("name") == model:
                digest = item.get("digest") or item.get("id") or "unavailable"
                break
    if isinstance(ps, Mapping):
        for item in ps.get("models") or ():
            if isinstance(item, Mapping) and item.get("name") == model:
                processor = item.get("processor") or processor
                digest = item.get("digest") or digest
                size = item.get("size")
                size_vram = item.get("size_vram")
                if isinstance(size, int) and isinstance(size_vram, int):
                    if size > 0 and size_vram == size:
                        processor = "100% GPU"
                    elif size_vram:
                        processor = f"GPU {size_vram}/{size}"
                    else:
                        processor = "CPU"
                break
    return {
        "version": (
            version.get("version")
            if isinstance(version, Mapping)
            else "unavailable"
        ),
        "digest": digest,
        "processor": processor,
    }


async def evaluate_planner_case(
    service: SemanticFactService,
    context: AgentRequestContext,
    case: SemanticEvalCase,
    *,
    phase: str = "measured",
    sample_index: int = 0,
) -> dict[str, Any]:
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
    result: FactResult | None = None
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
            final_answer = service.renderer.render(request, result, case_context)
    except SemanticPlannerFailure as error:
        diagnostics = error.diagnostics
        if diagnostics.output_raw and actual is None:
            request_payload = (
                diagnostics.normalized_plan.get("request")
                if isinstance(diagnostics.normalized_plan, Mapping)
                else None
            )
            if isinstance(request_payload, Mapping):
                try:
                    actual = normalize_semantic_request(
                        SemanticFactRequest.model_validate(request_payload)
                    )
                except Exception:
                    actual = None
    except Exception as error:
        runtime_failure = f"RUNTIME_ERROR:{type(error).__name__}"
    expected = normalize_semantic_request(case.expected)
    acceptable_plans = (
        expected,
        *(normalize_semantic_request(item) for item in case.acceptable_alternatives),
    )
    plan_match = None if actual is None and diagnostics is None and runtime_failure else (
        actual in acceptable_plans
    )
    if actual is None and runtime_failure:
        plan_match = False
    if actual is None and diagnostics is not None:
        plan_match = False
    answer_correct = (
        score_structured_result(result, case) if result is not None else (
            False if case.expected_status is not None else None
        )
    )
    validation_result = (
        diagnostics.validation_result if diagnostics is not None else None
    )
    executor_status = result.status if result is not None else "not_run"
    reason = runtime_failure or (
        semantic_mismatch_reason(actual, expected)
        if validation_result in {"VALID", "NOT_A_FACT"}
        else validation_result or "NO_SEMANTIC_PLAN"
    )
    return {
        "case_id": case.case_id,
        "utterance": case.utterance,
        "speaker_id": case.speaker_id,
        "category": case.category,
        "expected_plan_id": case.plan_id,
        "phase": phase,
        "sample_index": sample_index,
        "planner_input_capabilities": (
            dict(diagnostics.input_summary)
            if diagnostics
            else planner_input_summary(service.engine.schema.capability_payload())
        ),
        "planner_output": (
            jsonable(diagnostics.output_raw)
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
        "planner_success": plan_match,
        "plan_match": plan_match,
        "planner_failure_reason": None if plan_match else reason,
        "planner_failure_detail": (
            diagnostics.failure_detail if diagnostics else runtime_failure
        ),
        "validation_result": (
            diagnostics.validation_result
            if diagnostics is not None
            else runtime_failure or "NOT_A_FACT"
        ),
        "planner_diagnostics": serialize_diagnostics(diagnostics),
        "executor_success": bool(result is not None and result.status == "found"),
        "executor_status": executor_status,
        "executor_result": executor_status,
        "executor_latency_ms": (
            round(executor_latency_ms, 3) if executor_latency_ms is not None else None
        ),
        "executor": serialize_fact_result(result),
        "answer_correct": answer_correct,
        "final_answer": final_answer,
        "failure_stage": classify_failure_stage(
            runtime_failure=runtime_failure,
            validation_result=validation_result,
            plan_match=plan_match,
            executor_status=executor_status,
            answer_correct=answer_correct,
        ),
        "scoring_revision": SCORING_REVISION,
        "notes": case.notes,
        "tier": 1,
        "runtime_failure": runtime_failure,
    }


async def run_semantic_planner_benchmark(
    service: SemanticFactService,
    context: AgentRequestContext,
    cases: Sequence[SemanticEvalCase],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    planner_latencies: list[float] = []
    category_totals: Counter[str] = Counter()
    category_correct: Counter[str] = Counter()
    failure_reasons: Counter[str] = Counter()
    capability_summary = planner_input_summary(
        service.engine.schema.capability_payload()
    )

    for case in cases:
        row = await evaluate_planner_case(service, context, case)
        planner_correct = bool(row["plan_match"])
        category_totals[case.category] += 1
        if planner_correct:
            category_correct[case.category] += 1
        if not planner_correct:
            failure_reasons[str(row["planner_failure_reason"] or "NO_SEMANTIC_PLAN")] += 1
        if row["planner_latency_ms"] is not None:
            planner_latencies.append(float(row["planner_latency_ms"]))

        if not row.get("planner_input_capabilities"):
            row["planner_input_capabilities"] = capability_summary
        rows.append(row)

    total = len(rows)
    correct = sum(int(bool(row["planner_success"])) for row in rows)
    scores = summarize_scores(rows)
    return {
        "mode": "planner_only",
        "dataset_size": total,
        "queries": rows,
        "scores": scores,
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
        "planner_latency_ms": summarize_latencies(planner_latencies),
    }


async def run_tier1_probe(
    service: SemanticFactService,
    context: AgentRequestContext,
    dataset: ProbeDataset,
    *,
    warmup: int = 1,
    repeat: int = 5,
    verified_cold: bool = False,
    on_result: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if warmup < 0 or repeat < 1:
        raise ValueError("warmup must be >= 0 and repeat must be >= 1")
    cases = dataset.cases
    frozen_context = AgentRequestContext(
        caller_entity_id=context.caller_entity_id,
        assistant_id=context.assistant_id,
        assistant_display_name=context.assistant_display_name,
        household_id=dataset.household_id or context.household_id,
        current_time=dataset.frozen_time,
        locale=context.locale,
    )
    rows: list[dict[str, Any]] = []
    request_index = 0
    for _ in range(warmup):
        case = cases[request_index % len(cases)]
        row = await evaluate_planner_case(
            service,
            frozen_context,
            case,
            phase=request_phase(
                request_index, warmup=warmup, verified_cold=verified_cold
            ),
            sample_index=-1,
        )
        rows.append(row)
        if on_result is not None:
            on_result(row)
        request_index += 1
    for sample_index in range(repeat):
        for case in cases:
            row = await evaluate_planner_case(
                service,
                frozen_context,
                case,
                phase="measured",
                sample_index=sample_index,
            )
            rows.append(row)
            if on_result is not None:
                on_result(row)
            request_index += 1
    measured = [row for row in rows if row["phase"] == "measured"]
    first_pass = [row for row in measured if row["sample_index"] == 0]
    planner_samples = [
        float(row["planner_latency_ms"])
        for row in measured
        if row["planner_latency_ms"] is not None
    ]
    e2e_samples = [
        float(row["planner_latency_ms"] or 0) + float(row["executor_latency_ms"] or 0)
        for row in measured
        if row["planner_latency_ms"] is not None
    ]
    by_capability: dict[str, list[dict[str, Any]]] = {}
    for row in first_pass:
        by_capability.setdefault(str(row["category"]), []).append(row)
    return {
        "mode": "tier1_probe",
        "dataset_size": len(cases),
        "warmup": warmup,
        "repeat": repeat,
        "verified_cold": verified_cold,
        "queries": rows,
        "measured": measured,
        "scores": summarize_scores(first_pass),
        "scores_all_measured_samples": summarize_scores(measured),
        "accuracy_by_capability": {
            category: {
                "correct": sum(int(bool(row["plan_match"])) for row in group),
                "total": len(group),
                "accuracy": round(
                    sum(int(bool(row["plan_match"])) for row in group) / len(group),
                    4,
                ),
            }
            for category, group in sorted(by_capability.items())
        },
        "planner_latency_ms": summarize_latencies(planner_samples),
        "end_to_end_latency_ms": summarize_latencies(e2e_samples),
        "failure_table": [
            {
                "case_id": row["case_id"],
                "utterance": row["utterance"],
                "failure_stage": row["failure_stage"],
                "validation_result": row["validation_result"],
                "plan_match": row["plan_match"],
                "answer_correct": row["answer_correct"],
                "executor_status": row["executor_status"],
                "expected_semantic_plan": row["expected_semantic_plan"],
                "normalized_semantic_plan": row["normalized_semantic_plan"],
                "planner_output": row["planner_output"],
                "executor": row["executor"],
            }
            for row in first_pass
            if not row["plan_match"] or row["answer_correct"] is False
        ],
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
        eval_path=eval_path,
    )


def build_json_fact_service(
    data_dir: Path,
    schema_dir: Path,
    ollama: Any,
) -> tuple[SemanticFactService, AgentRequestContext]:
    registry = EdgeSchemaRegistry.from_directory(schema_dir)
    catalog = RuntimeSchemaCatalog.from_data_dir(data_dir, registry)
    schema = SemanticSchemaRegistry(catalog)
    service = SemanticFactService(
        HouseholdFactEngine(_JsonGraphDispatcher(data_dir, registry), schema),
        planner=SemanticFactPlanner(ollama, schema),
    )
    steward = get_agent("steward")
    localized = steward.settings.get("localized_identity", {})
    context = AgentRequestContext(
        caller_entity_id="person:jian_kuang",
        assistant_id=steward.id,
        assistant_display_name=localized.get("zh", steward.display_name),
        household_id=steward.settings.get("home_entity_id"),
        current_time=datetime.fromisoformat(FROZEN_EVAL_TIME),
        locale="zh",
    )
    return service, context


async def _benchmark_cases(
    *,
    data_dir: Path,
    schema_dir: Path,
    cases: Sequence[SemanticEvalCase],
    ollama_url: str,
    ollama_model: str,
    eval_path: Path = DEFAULT_EVAL_PATH,
) -> dict[str, Any]:
    ollama = OllamaService(ollama_url, ollama_model)
    service, context = build_json_fact_service(data_dir, schema_dir, ollama)
    try:
        report = await run_semantic_planner_benchmark(
            service,
            context,
            cases,
        )
        report["provenance"] = collect_provenance(
            root=Path(__file__).resolve().parents[2],
            eval_path=eval_path,
            ollama_url=ollama_url,
            ollama_model=ollama_model,
            backend="json",
            frozen_time=context.current_time,
            warmup=0,
            repeat=1,
            verified_cold=False,
            data_dir=data_dir,
            schema_dir=schema_dir,
        )
        report["model"] = ollama_model
        return report
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
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the full JSON report to this path.",
    )
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
            eval_path=args.eval,
        )
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
