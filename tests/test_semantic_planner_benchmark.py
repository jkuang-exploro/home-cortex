from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from home_cortex.edge_schema import EdgeSchemaRegistry
from home_cortex.fact_benchmark import _JsonGraphDispatcher
from home_cortex.grounding import AgentRequestContext
from home_cortex.schema_catalog import RuntimeSchemaCatalog
from home_cortex.semantic_facts import (
    HouseholdFactEngine,
    SemanticFactPlanner,
    SemanticFactService,
    SemanticSchemaRegistry,
    TierZeroSemanticParser,
)
from home_cortex.semantic_planner_benchmark import (
    DEFAULT_EVAL_PATH,
    load_semantic_eval_cases,
    run_semantic_planner_benchmark,
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
        "我爱人的爸爸是谁",
    }.issubset(utterances)
    parser = TierZeroSemanticParser()
    assert set(parser.canonical_utterances()).issubset(utterances)
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
