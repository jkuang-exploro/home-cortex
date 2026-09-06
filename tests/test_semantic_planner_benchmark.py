from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from home_cortex.edge_schema import EdgeSchemaRegistry
from home_cortex.fact_benchmark import _JsonGraphDispatcher
from home_cortex.schema_catalog import RuntimeSchemaCatalog
from home_cortex.semantic_facts import (
    AgentRequestContext,
    FactEvidence,
    FactResult,
    HouseholdFactEngine,
    SemanticFactPlanner,
    SemanticFactService,
    SemanticSchemaRegistry,
    TierZeroSemanticParser,
)
from home_cortex.semantic_planner_benchmark import (
    DEFAULT_EVAL_PATH,
    classify_failure_stage,
    evaluate_planner_case,
    load_probe_dataset,
    load_semantic_eval_cases,
    request_phase,
    run_semantic_planner_benchmark,
    run_tier1_probe,
    score_structured_result,
    summarize_latencies,
    summarize_scores,
)

ROOT = Path(__file__).parents[1]


def test_semantic_planner_evaluation_is_large_and_adversarial() -> None:
    cases = load_semantic_eval_cases()
    utterances = {case.utterance for case in cases}
    categories = {case.category for case in cases}

    assert DEFAULT_EVAL_PATH.is_file()
    assert len(cases) >= 100
    assert len(categories) >= 7
    assert {
        "我是谁",
        "你是谁",
        "家里都有谁",
        "家里有几个人",
        "谁最年长",
        "谁最年幼",
        "谁年纪最小",
        "有几个成年人",
        "有几个孩子",
        "我老婆是谁",
        "我老婆生日是哪天",
        "我们什么时候结婚的",
        "我们结婚多久了",
        "我儿子是谁",
        "我儿子哪天出生",
        "我儿子的生日还有多少天",
        "我岳父是谁",
    }.issubset(utterances)
    assert {
        "咱家岁数最小的是哪一位",
        "最晚出生的是谁",
        "家里已经成年的一共有多少位",
        "现在未成年的有几个",
        "我和爱人是哪一年开始做夫妻的",
        "德伦再过多久过生日",
        "德伦再过多久过生日？",
        "我爱人的爸爸是谁",
        "匡德伦是谁",
        "我和我老婆谁年龄大",
    }.issubset(utterances)
    parser = TierZeroSemanticParser()
    assert {
        "我是谁",
        "Who am I?",
        "你是谁",
        "Who are you?",
        "家里有几个人",
        "家里都有谁",
    }.issubset(utterances)
    assert all(
        parser.parse(utterance) is None
        for utterance in (
            "咱家岁数最小的是哪一位",
            "我和爱人是哪一年开始做夫妻的",
            "德伦再过多久过生日",
        )
    )


@pytest.mark.asyncio
async def test_planner_only_oracle_executes_full_eval_and_checks_tier0_parity() -> None:
    cases = load_semantic_eval_cases()
    expected = {case.utterance: case.expected for case in cases}

    class OracleInterpreter:
        async def plan_semantic_fact(
            self,
            messages: list[dict[str, Any]],
            *_: Any,
            **__: Any,
        ) -> dict[str, Any]:
            request = expected[messages[-1]["content"]]
            return {
                "requires_fact": True,
                "request": request.model_dump(mode="json"),
            }

    registry = EdgeSchemaRegistry.from_directory(ROOT / "schemas" / "edge")
    catalog = RuntimeSchemaCatalog.from_data_dir(ROOT / "data", registry)
    schema = SemanticSchemaRegistry(catalog)
    service = SemanticFactService(
        HouseholdFactEngine(_JsonGraphDispatcher(ROOT / "data", registry), schema),
        planner=SemanticFactPlanner(OracleInterpreter(), schema),
        tier_zero_enabled=False,
    )
    report = await run_semantic_planner_benchmark(
        service,
        AgentRequestContext(
            caller_entity_id="person:jian_kuang",
            assistant_id="steward",
            assistant_display_name="老管家",
            household_id="address:fort_cerritos",
            current_time=datetime.fromisoformat("2026-09-03T12:00:00-07:00"),
            locale="zh",
        ),
        cases,
    )

    assert report["mode"] == "planner_only"
    assert report["dataset_size"] >= 100
    assert report["accuracy"] == 1
    assert report["failure_reasons"] == {}
    assert report["tier0_parity"] == {
        "compared": 6,
        "equivalent": 6,
        "accuracy": 1,
    }
    assert all(
        row["tier"] == 1
        and row["validation_result"] == "VALID"
        and row["planner_attempt_count"] == 1
        for row in report["queries"]
    )
    assert all(
        row["planner_input_capabilities"]
        and row["planner_output"]
        and row["normalized_semantic_plan"]
        and row["executor_success"]
        and row["final_answer"]
        for row in report["queries"]
    )
    assert all(
        item["accuracy"] == 1
        for item in report["accuracy_by_capability"].values()
    )
    assert report["scores"]["plan_accuracy"]["scored"] == report["dataset_size"]
    assert report["scores"]["plan_accuracy"]["unscored"] == 0
    assert report["scores"]["plan_accuracy"]["correct"] == report["dataset_size"]


def test_probe_dataset_scores_all_twenty_questions() -> None:
    dataset = load_probe_dataset()
    ids = [case.case_id for case in dataset.cases]
    required = {
        "youngest_za_paraphrase",
        "adult_count_paraphrase",
        "father_in_law_lover_dad",
        "named_dylan_birthday_countdown",
        "marriage_duration_wife",
        "wife_birth_date",
        "son_birth_date",
        "oldest_member",
        "minor_count",
        "father_in_law",
        "residence_address",
        "pairwise_older_wife",
        "named_dylan_identity",
        "daughter_given_name_jian",
        "marriage_start",
        "household_list",
        "youngest_member",
        "minor_count_under_eighteen",
        "wife_father_given_name",
        "son_birthday_countdown",
    }

    assert len(dataset.cases) == 20
    assert set(ids) == required
    assert dataset.frozen_time.isoformat() == "2026-09-03T12:00:00-07:00"
    assert all(case.expected_status == "found" for case in dataset.cases)
    assert all(
        case.expected_entity_ids is not None or case.expected_value is not None
        for case in dataset.cases
    )
    assert {case.speaker_id for case in dataset.cases} == {"person:jian_kuang"}


def test_score_denominators_reconcile_plan_and_answer() -> None:
    rows = [
        {"case_id": "a", "plan_match": True, "answer_correct": True, "executor_status": "found"},
        {"case_id": "b", "plan_match": True, "answer_correct": False, "executor_status": "entity_not_found"},
        {"case_id": "c", "plan_match": False, "answer_correct": False, "executor_status": "not_run"},
        {"case_id": "d", "plan_match": None, "answer_correct": None, "executor_status": "not_run"},
    ]
    scores = summarize_scores(rows)

    assert scores["plan_accuracy"] == {
        "correct": 2,
        "scored": 3,
        "unscored": 1,
        "rate": round(2 / 3, 4),
    }
    assert scores["answer_correctness"]["scored"] == 3
    assert scores["answer_correctness"]["unscored"] == 1
    assert scores["answer_correctness"]["correct"] == 1
    assert scores["executor_status"]["entity_not_found"] == 1
    assert scores["unscored_or_unverifiable"]["plan"] == ["d"]
    assert classify_failure_stage(
        runtime_failure=None,
        validation_result="VALID",
        plan_match=True,
        executor_status="entity_not_found",
    ) == "entity_resolution"
    assert classify_failure_stage(
        runtime_failure=None,
        validation_result="UNKNOWN_PROPERTY",
        plan_match=False,
        executor_status="not_run",
    ) == "planner_validation"


def test_structured_result_scoring_uses_canonical_ids_not_status_alone() -> None:
    dataset = load_probe_dataset()
    youngest = next(case for case in dataset.cases if case.case_id == "youngest_member")
    found_wrong = FactResult(
        "found",
        {"id": "person:jian_kuang"},
        FactEvidence(entity_ids=("person:jian_kuang",)),
    )
    found_right = FactResult(
        "found",
        {"id": "person:evelyn_kuang"},
        FactEvidence(entity_ids=("person:evelyn_kuang",)),
    )

    assert score_structured_result(found_wrong, youngest) is False
    assert score_structured_result(found_right, youngest) is True


def test_request_phase_labels_first_request_unless_verified_cold() -> None:
    assert request_phase(0, warmup=1, verified_cold=False) == "first_request"
    assert request_phase(0, warmup=1, verified_cold=True) == "verified_cold"
    assert request_phase(1, warmup=2, verified_cold=False) == "warmup"
    assert request_phase(2, warmup=2, verified_cold=False) == "measured"


def test_warmup_is_excluded_from_latency_aggregation() -> None:
    samples = summarize_latencies([10.0, 20.0, 30.0, 40.0])
    assert samples["n"] == 4
    assert samples["p50"] == 25.0


@pytest.mark.asyncio
async def test_evaluate_records_planner_failure_without_aborting() -> None:
    dataset = load_probe_dataset()
    case = dataset.cases[0]
    remaining = dataset.cases[1]

    class FailingThenOracle:
        def __init__(self) -> None:
            self.calls = 0

        async def plan_semantic_fact(self, *_: Any, **__: Any) -> dict[str, Any]:
            self.calls += 1
            if self.calls == 1:
                return {
                    "requires_fact": True,
                    "request": {
                        "operation": "select",
                        "subject": {"kind": "self", "entity_type": "person"},
                        "property": "invented_private_fact",
                    },
                }
            return {
                "requires_fact": True,
                "request": remaining.expected.model_dump(mode="json"),
            }

    registry = EdgeSchemaRegistry.from_directory(ROOT / "schemas" / "edge")
    catalog = RuntimeSchemaCatalog.from_data_dir(ROOT / "data", registry)
    schema = SemanticSchemaRegistry(catalog)
    service = SemanticFactService(
        HouseholdFactEngine(_JsonGraphDispatcher(ROOT / "data", registry), schema),
        planner=SemanticFactPlanner(FailingThenOracle(), schema),
        tier_zero_enabled=False,
    )
    context = AgentRequestContext(
        caller_entity_id="person:jian_kuang",
        assistant_id="steward",
        assistant_display_name="老管家",
        household_id="address:fort_cerritos",
        current_time=datetime.fromisoformat("2026-09-03T12:00:00-07:00"),
        locale="zh",
    )
    failed = await evaluate_planner_case(service, context, case)
    recovered = await evaluate_planner_case(service, context, remaining)

    assert failed["planner_output"] is not None
    assert failed["validation_result"] == "UNKNOWN_PROPERTY"
    assert failed["plan_match"] is False
    assert failed["failure_stage"] == "planner_validation"
    assert failed["executor_status"] == "not_run"
    assert recovered["plan_match"] is True
    assert recovered["executor_status"] == "found"


@pytest.mark.asyncio
async def test_probe_warmup_and_repeats_are_counted_separately() -> None:
    dataset = load_probe_dataset()
    expected = {case.utterance: case.expected for case in dataset.cases}

    class Oracle:
        async def plan_semantic_fact(
            self,
            messages: list[dict[str, Any]],
            *_: Any,
            **__: Any,
        ) -> dict[str, Any]:
            request = expected[messages[-1]["content"]]
            return {"requires_fact": True, "request": request.model_dump(mode="json")}

    registry = EdgeSchemaRegistry.from_directory(ROOT / "schemas" / "edge")
    catalog = RuntimeSchemaCatalog.from_data_dir(ROOT / "data", registry)
    schema = SemanticSchemaRegistry(catalog)
    service = SemanticFactService(
        HouseholdFactEngine(_JsonGraphDispatcher(ROOT / "data", registry), schema),
        planner=SemanticFactPlanner(Oracle(), schema),
        tier_zero_enabled=False,
    )
    context = AgentRequestContext(
        caller_entity_id="person:jian_kuang",
        assistant_id="steward",
        assistant_display_name="老管家",
        household_id="address:fort_cerritos",
        current_time=dataset.frozen_time,
        locale="zh",
    )
    report = await run_tier1_probe(
        service,
        context,
        dataset,
        warmup=1,
        repeat=2,
        verified_cold=False,
    )

    phases = [row["phase"] for row in report["queries"]]
    measured = report["measured"]
    assert phases[0] == "first_request"
    assert phases.count("measured") == 40
    assert report["planner_latency_ms"]["n"] == 40
    assert report["scores"]["plan_accuracy"]["scored"] == 20
    assert report["scores"]["plan_accuracy"]["unscored"] == 0
    assert report["scores"]["answer_correctness"]["scored"] == 20
    assert report["scores"]["plan_accuracy"]["correct"] == 20
    assert report["scores"]["answer_correctness"]["correct"] == 20
    assert all(row["phase"] == "measured" for row in measured)
    assert {row["sample_index"] for row in measured} == {0, 1}
