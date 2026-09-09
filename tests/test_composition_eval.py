"""Deterministic checks for the Ticket 4 compositional evaluation set."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from home_cortex.composition_eval import (
    COMPOSITION_ROOT,
    HOUSEHOLD_IDS,
    composition_fingerprint_payload,
    gold_matches,
    household_engine,
    load_all_composition_datasets,
    load_sequences,
    load_standalone_cases,
    request_context,
    sha256_file,
)
from home_cortex.ollama import _PLANNER_INSTRUCTIONS, _semantic_planner_examples
from home_cortex.semantic_facts import DiscourseContext, SemanticFactRequest
from home_cortex.semantic_planner_benchmark import (
    DEFAULT_EVAL_PATH,
    SCORING_REVISION,
    load_probe_dataset,
    load_semantic_eval_cases,
    normalize_semantic_request,
    primary_entity_ids,
)

ROOT = Path(__file__).parents[1]
FROZEN_UTTERANCE_SOURCES = (
    ROOT / "benchmarks" / "semantic_planner_eval.yaml",
    ROOT / "benchmarks" / "semantic_planner_heldout.yaml",
    ROOT / "benchmarks" / "semantic_planner_synthetic.yaml",
    ROOT / "benchmarks" / "semantic_planner_age_filters.yaml",
    ROOT / "benchmarks" / "semantic_planner_date_intervals.yaml",
)


def _example_utterances() -> set[str]:
    return {
        message["content"]
        for message in _semantic_planner_examples()
        if message["role"] == "user"
    }


def test_existing_eval_path_and_scoring_revision_are_unchanged() -> None:
    assert DEFAULT_EVAL_PATH == ROOT / "benchmarks" / "semantic_planner_eval.yaml"
    assert DEFAULT_EVAL_PATH.is_file()
    assert SCORING_REVISION == "2026-09-07.2-composition-shapes"
    cases = load_semantic_eval_cases()
    assert len(cases) >= 100


def test_composition_size_matches_approved_coverage() -> None:
    standalone = load_standalone_cases()
    sequences = load_sequences()
    development = [case for case in standalone if case.split == "development"]
    frozen = [case for case in standalone if case.split == "frozen"]
    assert len(development) == 42
    assert len(frozen) == 36
    assert len([item for item in sequences if item.last.split == "development"]) == 8
    assert len([item for item in sequences if item.last.split == "frozen"]) == 8
    assert {case.case_id for case in standalone} == {
        case.case_id for case in standalone
    }
    assert len({case.case_id for case in standalone}) == 78


def test_development_and_frozen_utterances_are_disjoint() -> None:
    standalone = load_standalone_cases()
    development = {case.utterance for case in standalone if case.split == "development"}
    frozen = {case.utterance for case in standalone if case.split == "frozen"}
    assert development.isdisjoint(frozen)


def test_frozen_utterances_are_absent_from_examples_and_existing_eval() -> None:
    frozen = [case for case in load_standalone_cases() if case.split == "frozen"]
    utterances = {case.utterance for case in frozen}
    for sequence in load_sequences():
        if sequence.last.split == "frozen":
            utterances.update(sequence.utterances)
    resources = _PLANNER_INSTRUCTIONS + json.dumps(
        _semantic_planner_examples(), ensure_ascii=False
    )
    for utterance in utterances:
        assert utterance not in _example_utterances()
        assert utterance not in resources
    known = {case.utterance for case in load_semantic_eval_cases()}
    for path in FROZEN_UTTERANCE_SOURCES:
        known.update(case.utterance for case in load_probe_dataset(path).cases)
    assert utterances.isdisjoint(known)


def test_composition_files_are_not_the_default_eval_loader_path() -> None:
    assert DEFAULT_EVAL_PATH.name == "semantic_planner_eval.yaml"
    for dataset in load_all_composition_datasets():
        assert dataset.path != DEFAULT_EVAL_PATH
    for split in ("development", "frozen"):
        with pytest.raises(ValueError):
            load_probe_dataset(COMPOSITION_ROOT / split / "sequences.yaml")


@pytest.mark.asyncio
async def test_gold_plans_validate_and_execute() -> None:
    for case in load_standalone_cases():
        engine, _ = household_engine(case.household)
        assert engine.schema.validation_code(case.expected) == "VALID"
        for alternative in case.acceptable_alternatives:
            assert engine.schema.validation_code(alternative) == "VALID"
        context = request_context(speaker_id=case.speaker_id, household=case.household)
        result, *_ = await engine.execute(case.expected, context)
        assert gold_matches(result, case), (
            case.case_id,
            result.status,
            result.value,
            primary_entity_ids(result),
            result.unit,
        )


@pytest.mark.asyncio
async def test_forbidden_plans_are_not_gold_equivalent() -> None:
    datasets = {dataset.path: dataset for dataset in load_all_composition_datasets()}
    for case in load_standalone_cases():
        dataset = next(
            item
            for item in datasets.values()
            if any(row.case_id == case.case_id for row in item.cases)
        )
        engine, _ = household_engine(case.household)
        context = request_context(speaker_id=case.speaker_id, household=case.household)
        gold, *_ = await engine.execute(case.expected, context)
        gold_ids = set(primary_entity_ids(gold))
        gold_value = gold.value
        for key in case.forbidden_plan_ids:
            forbidden = dataset.forbidden_plans[key]
            assert normalize_semantic_request(forbidden) != normalize_semantic_request(
                case.expected
            )
            code = engine.schema.validation_code(forbidden)
            if code != "VALID":
                continue
            result, *_ = await engine.execute(forbidden, context)
            if result.status == "semantic_plan_unsupported":
                continue
            same_ids = set(primary_entity_ids(result)) == gold_ids
            same_value = result.value == gold_value
            assert not (
                result.status == gold.status and same_ids and same_value
            ), (case.case_id, key, result.status, result.value, primary_entity_ids(result))


@pytest.mark.asyncio
async def test_self_member_is_unsupported_on_every_household() -> None:
    request = SemanticFactRequest.model_validate(
        {
            "operation": "select",
            "subject": {
                "kind": "self",
                "entity_type": "person",
                "path": [{"relation": "member"}],
            },
            "property_source": "entity",
        }
    )
    for household, speaker in (
        ("alpha", "person:alpha_self"),
        ("beta", "person:beta_self"),
        ("gamma", "person:gamma_self"),
    ):
        engine, _ = household_engine(household)
        assert engine.schema.validation_code(request) == "INVALID_PLAN"
        result, *_ = await engine.execute(
            request, request_context(speaker_id=speaker, household=household)
        )
        assert result.status == "semantic_plan_unsupported"


@pytest.mark.asyncio
async def test_critical_path_contrasts_change_person_or_count() -> None:
    engine, _ = household_engine("alpha")
    context = request_context(speaker_id="person:alpha_self", household="alpha")

    async def run(raw: dict):
        request = SemanticFactRequest.model_validate(raw)
        result, *_ = await engine.execute(request, context)
        return result.status, primary_entity_ids(result), result.value

    fil = {
        "operation": "resolve_reference",
        "subject": {
            "kind": "self",
            "entity_type": "person",
            "path": [
                {"relation": "spouse"},
                {"relation": "parent", "filters": [{"property": "gender", "value": "male"}]},
            ],
        },
        "property_source": "entity",
    }
    wife_father = {
        "operation": "resolve_reference",
        "subject": {
            "kind": "self",
            "entity_type": "person",
            "path": [
                {
                    "relation": "spouse",
                    "filters": [{"property": "gender", "value": "female"}],
                },
                {"relation": "parent", "filters": [{"property": "gender", "value": "male"}]},
            ],
        },
        "property_source": "entity",
    }
    spouse_fil = {
        "operation": "resolve_reference",
        "subject": {
            "kind": "self",
            "entity_type": "person",
            "path": [
                {"relation": "spouse"},
                {"relation": "spouse"},
                {"relation": "parent", "filters": [{"property": "gender", "value": "male"}]},
            ],
        },
        "property_source": "entity",
    }
    father = {
        "operation": "resolve_reference",
        "subject": {
            "kind": "self",
            "entity_type": "person",
            "path": [
                {"relation": "parent", "filters": [{"property": "gender", "value": "male"}]}
            ],
        },
        "property_source": "entity",
    }
    fil_status, fil_ids, _ = await run(fil)
    wife_status, wife_ids, _ = await run(wife_father)
    spouse_status, spouse_ids, _ = await run(spouse_fil)
    father_status, father_ids, _ = await run(father)
    assert fil_status == wife_status == "found"
    assert fil_ids == wife_ids == ("person:alpha_fil",)
    assert spouse_ids == father_ids == ("person:alpha_father",)
    assert fil_ids != spouse_ids
    assert normalize_semantic_request(SemanticFactRequest.model_validate(fil)) != (
        normalize_semantic_request(SemanticFactRequest.model_validate(wife_father))
    )

    female = {
        "operation": "count",
        "subject": {
            "kind": "current_household",
            "entity_type": "address",
            "path": [{"relation": "member"}],
        },
        "property_source": "entity",
        "filters": [{"property": "gender", "value": "female"}],
    }
    adult_female = {
        **female,
        "filters": [
            {"predicate": "adult"},
            {"property": "gender", "value": "female"},
        ],
    }
    adult = {
        "operation": "count",
        "subject": female["subject"],
        "property_source": "entity",
        "filters": [{"predicate": "adult"}],
    }
    _, female_ids, female_value = await run(female)
    _, adult_female_ids, adult_female_value = await run(adult_female)
    _, adult_ids, adult_value = await run(adult)
    assert female_value == 5
    assert adult_female_value == 3
    assert adult_value == 7
    assert set(female_ids) != set(adult_female_ids)

    children = {
        "operation": "select",
        "subject": {
            "kind": "self",
            "entity_type": "person",
            "path": [{"relation": "child"}],
        },
        "property_source": "entity",
    }
    minors = {
        "operation": "select",
        "subject": female["subject"],
        "property_source": "entity",
        "filters": [{"predicate": "minor"}],
    }
    _, child_ids, _ = await run(children)
    _, minor_ids, _ = await run(minors)
    assert set(child_ids) == {
        "person:alpha_son_adult",
        "person:alpha_son_minor",
        "person:alpha_daughter",
    }
    assert set(minor_ids) == {
        "person:alpha_son_minor",
        "person:alpha_daughter",
        "person:alpha_niece",
    }

    beta_engine, _ = household_engine("beta")
    beta_ctx = request_context(speaker_id="person:beta_self", household="beta")
    beta_fil, *_ = await beta_engine.execute(
        SemanticFactRequest.model_validate(fil), beta_ctx
    )
    beta_wife, *_ = await beta_engine.execute(
        SemanticFactRequest.model_validate(wife_father), beta_ctx
    )
    assert primary_entity_ids(beta_fil) == ("person:beta_fil",)
    assert beta_wife.status == "relationship_not_found"

    gamma_engine, _ = household_engine("gamma")
    adult_req = SemanticFactRequest.model_validate(adult)
    minor_req = SemanticFactRequest.model_validate(minors)
    gamma_adult, *_ = await gamma_engine.execute(
        adult_req, request_context(speaker_id="person:gamma_self", household="gamma")
    )
    gamma_minor, *_ = await gamma_engine.execute(
        minor_req, request_context(speaker_id="person:gamma_self", household="gamma")
    )
    assert "person:gamma_just_adult" in primary_entity_ids(gamma_adult)
    assert "person:gamma_still_minor" not in primary_entity_ids(gamma_adult)
    assert "person:gamma_still_minor" in primary_entity_ids(gamma_minor)
    assert "person:gamma_just_adult" not in primary_entity_ids(gamma_minor)


@pytest.mark.asyncio
async def test_sequences_score_last_turn_and_lock_discourse() -> None:
    for sequence in load_sequences():
        engine, _ = household_engine(sequence.household)
        assert engine.schema.validation_code(sequence.last.expected) == "VALID"
        turns: list[tuple[str, ...]] = []
        conversation_id = sequence.sequence_id
        last_result = None
        for index, plan_id in enumerate(sequence.plan_ids):
            dataset = next(
                item
                for item in load_all_composition_datasets()
                if any(row.sequence_id == sequence.sequence_id for row in item.sequences)
            )
            request = dataset.plans[plan_id]
            discourse = None
            if request.subject.kind == "discourse":
                discourse = DiscourseContext(
                    conversation_id,
                    sequence.speaker_id,
                    HOUSEHOLD_IDS[sequence.household],
                    "assistant:composition",
                    tuple(turns[-8:]),
                )
            context = request_context(
                speaker_id=sequence.speaker_id,
                household=sequence.household,
                conversation_id=conversation_id,
                discourse=discourse,
            )
            result, *_ = await engine.execute(request, context)
            turns.append(result.focus_entity_ids if result.status == "found" else ())
            last_result = result
            if index == len(sequence.plan_ids) - 1:
                assert gold_matches(result, sequence.last), (
                    sequence.sequence_id,
                    result.status,
                    result.value,
                    primary_entity_ids(result),
                )
                if sequence.last.cell in {"G7", "G8"}:
                    assert request.subject.kind == "discourse"
        assert last_result is not None
        assert sequence.last.history == sequence.utterances[:-1]


def test_kinship_acceptable_plans_are_not_collapsed() -> None:
    cases = {case.case_id: case for case in load_standalone_cases()}
    for case_id in ("dev-alpha-K1-1", "dev-alpha-K2-1", "dev-alpha-K3-1", "dev-beta-K5-1"):
        assert cases[case_id].acceptable_alternatives == ()
    k1 = normalize_semantic_request(cases["dev-alpha-K1-1"].expected)
    k2 = normalize_semantic_request(cases["dev-alpha-K2-1"].expected)
    k3 = normalize_semantic_request(cases["dev-alpha-K3-1"].expected)
    assert k1 != k2 != k3
    assert k1 != k3


def test_fingerprints_match_current_tree() -> None:
    payload = composition_fingerprint_payload()
    recorded = json.loads((COMPOSITION_ROOT / "fingerprints.json").read_text())
    assert recorded["scoring_revision"] == SCORING_REVISION
    assert recorded["counts"] == payload["counts"]
    assert recorded["files"] == payload["files"]
    assert recorded["household_tree_sha256"] == payload["household_tree_sha256"]
    manifest = yaml.safe_load(
        (COMPOSITION_ROOT / "frozen" / "MANIFEST.yaml").read_text(encoding="utf-8")
    )
    assert manifest["size"]["frozen_standalone"] == 36
    assert manifest["size"]["development_standalone"] == 42
    assert len(manifest["cases"]) == 36
    assert sha256_file(COMPOSITION_ROOT / "frozen" / "MANIFEST.yaml")


def test_count_collision_cells_record_entity_ids() -> None:
    needed = {
        "dev-alpha-F1-1",
        "dev-alpha-F7-1",
        "dev-alpha-S2-1",
        "frz-beta-F7-1",
        "frz-beta-F1-1",
    }
    cases = {case.case_id: case for case in load_standalone_cases()}
    for case_id in needed:
        assert cases[case_id].expected_entity_ids, case_id
        assert cases[case_id].expected_value is not None
