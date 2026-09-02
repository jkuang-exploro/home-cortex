"""Opt-in behavioral evaluations against a real Ollama planner model."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import pytest

from home_cortex.grounding import GroundingPlanner
from home_cortex.ollama import OllamaService
from home_cortex.schema_catalog import (
    EntityTypeSchema,
    RelationTypeSchema,
    RuntimeSchemaCatalog,
)

RUN_MODEL_EVALS = os.getenv("RUN_GROUNDING_MODEL_EVALS") == "1"
MODEL = os.getenv("OLLAMA_MODEL")

pytestmark = pytest.mark.skipif(
    not RUN_MODEL_EVALS or not MODEL,
    reason=(
        "set RUN_GROUNDING_MODEL_EVALS=1 and OLLAMA_MODEL to run real-model evals"
    ),
)

EVAL_CATALOG = RuntimeSchemaCatalog(
    {
        "person": EntityTypeSchema(
            "person",
            (
                "id",
                "name",
                "dob",
                "occupation",
                "income",
                "shoe_size_us",
                "favorite_temperature_c",
            ),
        ),
        "measurement": EntityTypeSchema(
            "measurement",
            ("id", "temperature_c", "observed_at"),
        ),
    },
    {
        "has_measurement": RelationTypeSchema(
            "has_measurement",
            ("person",),
            ("measurement",),
            (),
            False,
            False,
            None,
        ),
    },
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("question", "expected"),
    (
        ("What is a normal body temperature?", {"grounded": False}),
        (
            "What is Test Person's shoe size?",
            {"grounded": True, "field": "shoe_size_us"},
        ),
        (
            "What does Test Person do for work?",
            {"grounded": True, "field": "occupation"},
        ),
        (
            "What temperature does Test Person prefer?",
            {"grounded": True, "field": "favorite_temperature_c"},
        ),
        (
            "How old is Test Person?",
            {"grounded": True, "field": "dob", "operator": "completed_years"},
        ),
        (
            "What is my latest body temperature?",
            {
                "grounded": True,
                "field": "temperature_c",
                "operator": "latest",
                "reference_type": "speaker",
            },
        ),
        (
            "Who are you?",
            {
                "grounded": True,
                "field": "display_name",
                "reference_type": "assistant",
                "domain": "runtime",
            },
        ),
        (
            "我是谁？",
            {
                "grounded": True,
                "field": "name",
                "reference_type": "speaker",
                "domain": "household",
            },
        ),
        (
            "Who am I?",
            {
                "grounded": True,
                "field": "name",
                "reference_type": "speaker",
                "domain": "household",
            },
        ),
        (
            "你是谁？",
            {
                "grounded": True,
                "field": "display_name",
                "reference_type": "assistant",
                "domain": "runtime",
            },
        ),
    ),
)
async def test_actual_planner_model_behavior(
    question: str,
    expected: dict[str, Any],
) -> None:
    service = OllamaService(
        os.getenv("OLLAMA_URL", "http://localhost:11434"),
        MODEL or "",
    )
    try:
        plan = await GroundingPlanner(service, EVAL_CATALOG).plan(
            [{"role": "user", "content": question}],
            household_now=datetime.fromisoformat("2026-08-31T12:00:00-07:00"),
        )
    finally:
        await service.close()

    assert plan.requires_grounding is expected["grounded"]
    if field := expected.get("field"):
        required_fields = {
            item.field for item in plan.required_evidence if item.field is not None
        }
        assert field in required_fields
    if operator := expected.get("operator"):
        assert plan.transform is not None
        assert plan.transform.operator == operator
    if reference_type := expected.get("reference_type"):
        assert plan.subject is not None
        assert plan.subject.reference_type == reference_type
    if domain := expected.get("domain"):
        assert plan.grounding_domain == domain
