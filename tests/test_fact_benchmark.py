from pathlib import Path
from datetime import datetime
from typing import Any

import pytest

from home_cortex.fact_benchmark import _run_suite, benchmark_json
from home_cortex.grounding import AgentRequestContext
from home_cortex.semantic_facts import (
    FactAnswer,
    FactEvidence,
    FactResult,
    FactTimings,
    SemanticFactRequest,
    SemanticReference,
)


ROOT = Path(__file__).parents[1]


@pytest.mark.asyncio
async def test_benchmark_reports_mode_speaker_path_and_canonical_ids() -> None:
    async def plan_semantic_fact(
        _self: Any,
        messages: Any,
        _capabilities: Any,
        _output_schema: Any,
        **_: Any,
    ) -> dict[str, Any]:
        text = messages[-1]["content"]
        members = {
            "kind": "current_household",
            "entity_type": "address",
            "path": [{"relation": "member"}],
        }
        if text in {"谁最年幼", "谁年纪最小"}:
            request = {
                "operation": "argmax",
                "subject": members,
                "property": "birth_date",
            }
        elif text == "有几个成年人":
            request = {
                "operation": "count",
                "subject": members,
                "filters": [{"predicate": "adult"}],
            }
        elif text == "有几个孩子":
            request = {
                "operation": "count",
                "subject": members,
                "filters": [{"predicate": "minor"}],
            }
        else:
            request = {
                "operation": "select",
                "subject": {
                    "kind": "self",
                    "entity_type": "person",
                    "path": [{"relation": "spouse"}],
                },
                "property": "start_date",
                "property_source": "relationship",
            }
        return {"requires_fact": True, "request": request}

    from home_cortex.ollama import OllamaService

    original = OllamaService.plan_semantic_fact
    OllamaService.plan_semantic_fact = plan_semantic_fact  # type: ignore[method-assign]
    try:
        result = await benchmark_json(
            "person:jian_kuang",
            1,
            ROOT / "data",
            ROOT / "schemas" / "edge",
            "tier0_enabled",
        )
    finally:
        OllamaService.plan_semantic_fact = original  # type: ignore[method-assign]

    assert result["mode"] == "tier0_enabled"
    assert result["aggregate"]["llm_call_count"] == 2
    assert result["diagnostic_comparisons"]["age_extrema"]["谁最年长"][
        "operation"
    ] == "argmin"
    assert result["diagnostic_comparisons"]["age_extrema"]["谁最年幼"][
        "operation"
    ] == "argmax"
    adult = next(
        row for row in result["queries"] if row["utterance"] == "有几个成年人"
    )
    assert adult["operators"] == ["traverse", "filter", "count"]
    assert adult["filters"][0]["predicate"] == "adult"
    assert adult["relationship_properties"] == ["household_role"]
    marriage = next(
        row
        for row in result["queries"]
        if row["utterance"] == "我们什么时候结婚的"
    )
    assert marriage["relationship_properties"] == ["start_date"]
    assert marriage["failure_stage"] is None
    rows = {
        (row["speaker_id"], row["utterance"]): row
        for row in result["queries"]
    }
    perspectives = (
        ("person:jian_kuang", "我儿子是谁"),
        ("person:pu_ba", "我儿子是谁"),
        ("person:guiqiu_wang", "我孙子是谁"),
        ("person:zhigang_ba", "我外孙是谁"),
        ("person:evelyn_kuang", "我哥哥是谁"),
    )
    for key in perspectives:
        row = rows[key]
        assert row["resolved_entities"] == ["person:dylan_kuang"]
        assert row["resolution_path"]
        assert row["tier"] == 0
        assert row["db_query_count"] >= 1


@pytest.mark.asyncio
async def test_disabled_mode_benchmark_does_not_require_tier_zero() -> None:
    class TierOneOnlyService:
        async def try_answer(self, *_: Any, **__: Any) -> FactAnswer:
            request = SemanticFactRequest(
                operation="resolve_reference",
                subject=SemanticReference(kind="self", entity_type="person"),
            )
            return FactAnswer(
                request,
                FactResult(
                    "found",
                    {"id": "person:dylan_kuang"},
                    FactEvidence(entity_ids=("person:dylan_kuang",)),
                ),
                "resolved",
                FactTimings(tier=1, llm_call_count=1, total_ms=10),
            )

    result = await _run_suite(
        TierOneOnlyService(),  # type: ignore[arg-type]
        AgentRequestContext(
            caller_entity_id="person:jian_kuang",
            assistant_id="steward",
            assistant_display_name="老管家",
            household_id="address:fort_cerritos",
            current_time=datetime.fromisoformat("2026-09-03T12:00:00-07:00"),
            locale="zh",
        ),
        1,
        backend="fake",
        mode="tier0_disabled",
    )

    assert result["mode"] == "tier0_disabled"
    assert result["aggregate"]["llm_call_count"] == len(result["queries"])
    assert all(row["tier"] == 1 for row in result["queries"])
