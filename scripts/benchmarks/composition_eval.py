"""Ticket 4 compositional evaluation loader. Does not change existing eval paths."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from scripts import PROJECT_ROOT
from typing import Any, Mapping

import yaml

from home_cortex.edge_schema import EdgeSchemaRegistry
from scripts.benchmarks.json_graph import JsonGraphDispatcher
from home_cortex.schema_catalog import RuntimeSchemaCatalog
from home_cortex.semantic_ir import AgentRequestContext, DiscourseContext, SemanticFactRequest
from home_cortex.household_fact_engine import HouseholdFactEngine
from home_cortex.semantic_schema import SemanticSchemaRegistry
from scripts.benchmarks.semantic_planner_benchmark import (
    FROZEN_EVAL_TIME,
    SCORING_REVISION,
    SemanticEvalCase,
    score_structured_result,
)

ROOT = PROJECT_ROOT
COMPOSITION_ROOT = ROOT / "benchmarks" / "composition"
HOUSEHOLD_ROOT = COMPOSITION_ROOT / "households"
SCHEMA_DIR = ROOT / "schemas" / "edge"
ONTOLOGY_PATH = ROOT / "schemas" / "semantic" / "ontology.yaml"

HOUSEHOLD_IDS = {
    "alpha": "address:alpha",
    "beta": "address:beta",
    "gamma": "address:gamma",
}


@dataclass(frozen=True)
class CompositionCase:
    case_id: str
    utterance: str
    speaker_id: str
    household: str
    split: str
    cell: str
    plan_id: str
    expected: SemanticFactRequest
    acceptable_alternatives: tuple[SemanticFactRequest, ...] = ()
    history: tuple[str, ...] = ()
    expected_status: str | None = None
    acceptable_statuses: tuple[str, ...] = ()
    expected_entity_ids: tuple[str, ...] | None = None
    expected_value: Any = None
    expected_unit: str | None = None
    notes: str | None = None
    forbidden_plan_ids: tuple[str, ...] = ()

    def as_eval_case(self) -> SemanticEvalCase:
        return SemanticEvalCase(
            utterance=self.utterance,
            speaker_id=self.speaker_id,
            category=self.cell,
            plan_id=self.plan_id,
            expected=self.expected,
            acceptable_alternatives=self.acceptable_alternatives,
            case_id=self.case_id,
            expected_status=self.expected_status,
            expected_entity_ids=self.expected_entity_ids,
            expected_value=self.expected_value,
            expected_unit=self.expected_unit,
            notes=self.notes,
        )


@dataclass(frozen=True)
class CompositionSequence:
    sequence_id: str
    household: str
    speaker_id: str
    utterances: tuple[str, ...]
    plan_ids: tuple[str, ...]
    last: CompositionCase
    notes: str | None = None


@dataclass(frozen=True)
class CompositionDataset:
    split: str
    household: str | None
    frozen_time: datetime
    plans: Mapping[str, SemanticFactRequest]
    cases: tuple[CompositionCase, ...]
    sequences: tuple[CompositionSequence, ...]
    forbidden_plans: Mapping[str, SemanticFactRequest]
    path: Path


def _request(raw: Mapping[str, Any]) -> SemanticFactRequest:
    return SemanticFactRequest.model_validate(raw)


def _string_tuple(value: Any, *, field: str, case_id: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{case_id} has invalid {field}")
    return tuple(value)


def _optional_ids(value: Any, *, case_id: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, (str, int)) for item in value):
        raise ValueError(f"{case_id} has invalid expected_entity_ids")
    return tuple(str(item) for item in value)


def load_composition_file(path: Path, *, split: str) -> CompositionDataset:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("version") != 1:
        raise ValueError(f"{path} must have version 1")
    plans_raw = raw.get("plans")
    if not isinstance(plans_raw, Mapping):
        raise ValueError(f"{path} is missing plans")
    plans = {key: _request(value) for key, value in plans_raw.items()}
    forbidden_raw = raw.get("forbidden_plans") or {}
    if not isinstance(forbidden_raw, Mapping):
        raise ValueError(f"{path} has invalid forbidden_plans")
    forbidden = {key: _request(value) for key, value in forbidden_raw.items()}
    probe = raw.get("probe")
    sequences_raw = raw.get("sequences") or []
    household = raw.get("household")
    frozen_raw = FROZEN_EVAL_TIME
    cases: list[CompositionCase] = []
    if probe is None and not sequences_raw:
        raise ValueError(f"{path} needs probe.cases or sequences")
    if isinstance(probe, Mapping):
        frozen_raw = str(probe.get("frozen_time") or FROZEN_EVAL_TIME)
        household = str(probe.get("household") or household or "")
        default_speaker = str(probe.get("default_speaker_id") or "")
        rows = probe.get("cases")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{path} probe.cases must be a non-empty list")
        seen: set[str] = set()
        for item in rows:
            if not isinstance(item, Mapping):
                raise ValueError("probe case must be an object")
            case_id = item.get("id")
            utterance = item.get("utterance")
            plan_key = item.get("plan")
            if not isinstance(case_id, str) or not case_id.strip():
                raise ValueError("probe case is missing id")
            if case_id in seen:
                raise ValueError(f"duplicate composition case id: {case_id}")
            seen.add(case_id)
            if not isinstance(utterance, str) or not utterance.strip():
                raise ValueError(f"{case_id} is missing utterance")
            if not isinstance(plan_key, str) or plan_key not in plans:
                raise ValueError(f"{case_id} references unknown plan {plan_key!r}")
            alternative_keys = item.get("acceptable_plans") or []
            if not isinstance(alternative_keys, list) or not all(
                isinstance(key, str) for key in alternative_keys
            ):
                raise ValueError(f"{case_id} has invalid acceptable_plans")
            missing = [key for key in alternative_keys if key not in plans]
            if missing:
                raise ValueError(f"{case_id} references unknown plans: {', '.join(missing)}")
            cases.append(
                CompositionCase(
                    case_id=case_id,
                    utterance=utterance,
                    speaker_id=str(item.get("speaker_id") or default_speaker),
                    household=str(item.get("household") or household),
                    split=split,
                    cell=str(item.get("cell") or "unspecified"),
                    plan_id=plan_key,
                    expected=plans[plan_key],
                    acceptable_alternatives=tuple(plans[key] for key in alternative_keys),
                    history=_string_tuple(item.get("history"), field="history", case_id=case_id),
                    expected_status=(
                        str(item["expected_status"])
                        if item.get("expected_status") is not None
                        else None
                    ),
                    acceptable_statuses=_string_tuple(
                        item.get("acceptable_statuses"),
                        field="acceptable_statuses",
                        case_id=case_id,
                    ),
                    expected_entity_ids=_optional_ids(
                        item.get("expected_entity_ids"), case_id=case_id
                    ),
                    expected_value=item.get("expected_value"),
                    expected_unit=(
                        str(item["expected_unit"])
                        if item.get("expected_unit") is not None
                        else None
                    ),
                    notes=str(item["notes"]) if item.get("notes") is not None else None,
                    forbidden_plan_ids=_string_tuple(
                        item.get("forbidden_plans"),
                        field="forbidden_plans",
                        case_id=case_id,
                    ),
                )
            )
    sequences: list[CompositionSequence] = []
    if not isinstance(sequences_raw, list):
        raise ValueError(f"{path} sequences must be a list")
    for item in sequences_raw:
        if not isinstance(item, Mapping):
            raise ValueError("sequence must be an object")
        sequence_id = item.get("id")
        turns = item.get("turns")
        if not isinstance(sequence_id, str) or not sequence_id.strip():
            raise ValueError("sequence is missing id")
        if not isinstance(turns, list) or len(turns) < 2:
            raise ValueError(f"{sequence_id} needs at least two turns")
        last_raw = turns[-1]
        if not isinstance(last_raw, Mapping):
            raise ValueError(f"{sequence_id} last turn is invalid")
        utterances: list[str] = []
        plan_ids: list[str] = []
        for turn in turns:
            if not isinstance(turn, Mapping):
                raise ValueError(f"{sequence_id} turn is invalid")
            utterance = turn.get("utterance")
            plan_key = turn.get("plan")
            if not isinstance(utterance, str) or not utterance.strip():
                raise ValueError(f"{sequence_id} turn is missing utterance")
            if not isinstance(plan_key, str) or plan_key not in plans:
                raise ValueError(f"{sequence_id} references unknown plan {plan_key!r}")
            utterances.append(utterance)
            plan_ids.append(plan_key)
        seq_household = str(item.get("household") or household or "")
        speaker_id = str(item.get("speaker_id") or "")
        last_plan = plan_ids[-1]
        alternative_keys = last_raw.get("acceptable_plans") or []
        sequences.append(
            CompositionSequence(
                sequence_id=sequence_id,
                household=seq_household,
                speaker_id=speaker_id,
                utterances=tuple(utterances),
                plan_ids=tuple(plan_ids),
                last=CompositionCase(
                    case_id=sequence_id,
                    utterance=utterances[-1],
                    speaker_id=speaker_id,
                    household=seq_household,
                    split=split,
                    cell=str(item.get("cell") or "sequence"),
                    plan_id=last_plan,
                    expected=plans[last_plan],
                    acceptable_alternatives=tuple(
                        plans[key] for key in alternative_keys if key in plans
                    ),
                    history=tuple(utterances[:-1]),
                    expected_status=(
                        str(last_raw["expected_status"])
                        if last_raw.get("expected_status") is not None
                        else None
                    ),
                    acceptable_statuses=_string_tuple(
                        last_raw.get("acceptable_statuses"),
                        field="acceptable_statuses",
                        case_id=sequence_id,
                    ),
                    expected_entity_ids=_optional_ids(
                        last_raw.get("expected_entity_ids"), case_id=sequence_id
                    ),
                    expected_value=last_raw.get("expected_value"),
                    expected_unit=(
                        str(last_raw["expected_unit"])
                        if last_raw.get("expected_unit") is not None
                        else None
                    ),
                    notes=str(item["notes"]) if item.get("notes") is not None else None,
                    forbidden_plan_ids=_string_tuple(
                        last_raw.get("forbidden_plans"),
                        field="forbidden_plans",
                        case_id=sequence_id,
                    ),
                ),
                notes=str(item["notes"]) if item.get("notes") is not None else None,
            )
        )
        if isinstance(probe, Mapping):
            frozen_raw = str(probe.get("frozen_time") or frozen_raw)
        frozen_raw = str(item.get("frozen_time") or frozen_raw)
    return CompositionDataset(
        split=split,
        household=str(household) if household else None,
        frozen_time=datetime.fromisoformat(str(frozen_raw)),
        plans=plans,
        cases=tuple(cases),
        sequences=tuple(sequences),
        forbidden_plans=forbidden,
        path=path,
    )


def iter_composition_files() -> tuple[tuple[str, Path], ...]:
    files = []
    for split in ("development", "frozen"):
        directory = COMPOSITION_ROOT / split
        for path in sorted(directory.glob("*.yaml")):
            if path.name == "MANIFEST.yaml":
                continue
            files.append((split, path))
    return tuple(files)


def load_all_composition_datasets() -> tuple[CompositionDataset, ...]:
    return tuple(
        load_composition_file(path, split=split) for split, path in iter_composition_files()
    )


def load_standalone_cases() -> tuple[CompositionCase, ...]:
    cases: list[CompositionCase] = []
    for dataset in load_all_composition_datasets():
        cases.extend(dataset.cases)
    return tuple(cases)


def load_sequences() -> tuple[CompositionSequence, ...]:
    sequences: list[CompositionSequence] = []
    for dataset in load_all_composition_datasets():
        sequences.extend(dataset.sequences)
    return tuple(sequences)


def household_engine(household: str) -> tuple[HouseholdFactEngine, Path]:
    data_dir = HOUSEHOLD_ROOT / household
    registry = EdgeSchemaRegistry.from_directory(SCHEMA_DIR)
    schema = SemanticSchemaRegistry(RuntimeSchemaCatalog.from_data_dir(data_dir, registry))
    engine = HouseholdFactEngine(JsonGraphDispatcher(data_dir, registry), schema)
    return engine, data_dir


def request_context(
    *,
    speaker_id: str,
    household: str,
    frozen_time: datetime | None = None,
    conversation_id: str | None = None,
    discourse: DiscourseContext | None = None,
) -> AgentRequestContext:
    return AgentRequestContext(
        caller_entity_id=speaker_id,
        assistant_id="assistant:composition",
        assistant_display_name="Helper",
        household_id=HOUSEHOLD_IDS[household],
        current_time=frozen_time or datetime.fromisoformat(FROZEN_EVAL_TIME),
        locale="zh",
        conversation_id=conversation_id,
        discourse=discourse,
    )


def statuses_for(case: CompositionCase) -> frozenset[str]:
    allowed = set(case.acceptable_statuses)
    if case.expected_status is not None:
        allowed.add(case.expected_status)
    return frozenset(allowed)


def gold_matches(result, case: CompositionCase) -> bool:
    eval_case = case.as_eval_case()
    if case.acceptable_statuses and result.status in case.acceptable_statuses:
        if case.expected_entity_ids is None and case.expected_value is None:
            return True
    scored = score_structured_result(result, eval_case)
    if scored is True:
        return True
    if result.status in statuses_for(case) and case.expected_status is not None:
        if result.status != case.expected_status and result.status in case.acceptable_statuses:
            return case.expected_entity_ids is None and case.expected_value is None
    return bool(scored)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(root: Path, *, exclude_names: frozenset[str] = frozenset()) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name in exclude_names:
            continue
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(sha256_file(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def composition_fingerprint_payload() -> dict[str, Any]:
    files = {
        "annotation-guide.md": sha256_file(COMPOSITION_ROOT / "annotation-guide.md"),
        "coverage-matrix.md": sha256_file(COMPOSITION_ROOT / "coverage-matrix.md"),
        "codex-approval.md": sha256_file(COMPOSITION_ROOT / "codex-approval.md"),
        "README.md": sha256_file(COMPOSITION_ROOT / "README.md"),
        "ontology.yaml": sha256_file(ONTOLOGY_PATH),
    }
    test_path = ROOT / "tests" / "test_composition_eval.py"
    if test_path.is_file():
        files["test_composition_eval.py"] = sha256_file(test_path)
    for split, path in iter_composition_files():
        files[str(path.relative_to(ROOT))] = sha256_file(path)
    for household in HOUSEHOLD_IDS:
        files[f"households/{household}"] = sha256_tree(HOUSEHOLD_ROOT / household)
    standalone = load_standalone_cases()
    sequences = load_sequences()
    return {
        "scoring_revision": SCORING_REVISION,
        "frozen_evaluation_time": FROZEN_EVAL_TIME,
        "default_eval_path": "benchmarks/semantic_planner_eval.yaml",
        "counts": {
            "development_standalone": sum(1 for case in standalone if case.split == "development"),
            "frozen_standalone": sum(1 for case in standalone if case.split == "frozen"),
            "development_sequences": sum(1 for item in sequences if item.last.split == "development"),
            "frozen_sequences": sum(1 for item in sequences if item.last.split == "frozen"),
        },
        "files": files,
        "household_tree_sha256": sha256_tree(HOUSEHOLD_ROOT),
        "composition_tree_sha256": sha256_tree(
            COMPOSITION_ROOT, exclude_names=frozenset({"fingerprints.json", "_emit.py"})
        ),
    }


def dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
