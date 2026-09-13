import json
from datetime import datetime
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JSONSchemaValidationError
from ollama import ChatResponse

from home_cortex.ollama import OllamaService
from home_cortex.semantic_ir import SemanticPlan
from scripts.benchmarks.semantic_planner_benchmark import build_json_fact_service
from scripts.probes.unified_semantic_planner import (
    UnifiedShadowPlanner,
    unified_chat_messages,
    unified_output_schema,
    validate_unified_payload,
)
from scripts.profiling.summarize_unified_experiment import summarize


ROOT = Path(__file__).parents[1]


def _schema():
    service, _ = build_json_fact_service(
        ROOT / "benchmarks/fixtures/semantic-contract",
        ROOT / "schemas/edge",
        None,
    )
    return service.engine.schema


@pytest.mark.parametrize(
    "payload,kind",
    [
        (
            {
                "requires_fact": True,
                "request": {
                    "operation": "resolve_reference",
                    "subject": {"kind": "self", "entity_type": "person"},
                    "property": None,
                    "property_source": "entity",
                },
                "mutation": None,
                "multi_intent": False,
            },
            "fact",
        ),
        (
            {
                "requires_fact": False,
                "request": None,
                "mutation": {
                    "operation": "delete",
                    "item_name": "lamp",
                    "mode": "commit",
                },
                "multi_intent": False,
            },
            "mutation",
        ),
        (
            {
                "requires_fact": False,
                "request": None,
                "mutation": None,
                "multi_intent": False,
            },
            "conversation",
        ),
        (
            {
                "requires_fact": False,
                "request": None,
                "mutation": None,
                "multi_intent": True,
            },
            "multi_intent",
        ),
    ],
)
def test_unified_contract_reuses_semantic_plan_branches(payload, kind):
    schema = _schema()
    Draft202012Validator(unified_output_schema(schema)).validate(payload)
    plan = validate_unified_payload(schema, payload, "ordinary input")
    assert isinstance(plan, SemanticPlan)
    actual = (
        "multi_intent" if plan.multi_intent
        else "mutation" if plan.mutation
        else "fact" if plan.request
        else "conversation"
    )
    assert actual == kind


def test_unified_schema_rejects_parallel_branches_and_unknown_write_attributes():
    schema = _schema()
    validator = Draft202012Validator(unified_output_schema(schema))
    both = {
        "requires_fact": True,
        "request": {
            "operation": "resolve_reference",
            "subject": {"kind": "self"},
            "property": None,
            "property_source": "entity",
        },
        "mutation": {
            "operation": "delete",
            "item_name": "lamp",
            "mode": "commit",
        },
        "multi_intent": False,
    }
    with pytest.raises(JSONSchemaValidationError):
        validate_unified_payload(schema, both, "mixed")
    with pytest.raises(JSONSchemaValidationError):
        validate_unified_payload(
            schema,
            {**both, "mutation": None, "multi_intent": True},
            "mixed",
        )
    invalid_attribute = {
        "requires_fact": False,
        "request": None,
        "mutation": {
            "operation": "update_attributes",
            "item_name": "lamp",
            "attributes": {"storage_record_id": "item:secret"},
            "mode": "commit",
        },
        "multi_intent": False,
    }
    assert list(validator.iter_errors(invalid_attribute))


def test_unified_prompt_transforms_existing_mutation_examples_into_semantic_plan():
    schema = _schema()
    messages = unified_chat_messages(
        [{"role": "user", "content": "Who am I? Delete the lamp record."}],
        schema,
        household_now="2026-09-13T00:00:00-07:00",
    )
    content = "\n".join(message["content"] for message in messages)
    assert "Emit no partial request or mutation" in content
    assert '"multi_intent":true' in content
    assert '"mode":"preview"' in content
    assert '"mutation":{"operation":"create"' in content
    assert '"requires_mutation":true' not in "\n".join(
        message["content"] for message in messages if message["role"] == "assistant"
    )


@pytest.mark.asyncio
async def test_shadow_planner_returns_intent_without_execution():
    schema = _schema()
    response = ChatResponse.model_validate({
        "model": "qwen3.5:9b",
        "created_at": "2026-09-13T00:00:00Z",
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 100,
        "eval_count": 20,
        "message": {
            "role": "assistant",
            "content": json.dumps({
                "requires_fact": False,
                "request": None,
                "mutation": {
                    "operation": "delete",
                    "item_name": "lamp",
                    "mode": "commit",
                },
                "multi_intent": False,
            }),
        },
    })

    class Client:
        calls = []

        async def chat(self, **kwargs):
            self.calls.append(kwargs)
            return response

        async def close(self):
            pass

    client = Client()
    ollama = OllamaService("http://unused", "qwen3.5:9b", client=client)
    result = await UnifiedShadowPlanner(ollama, schema).plan(
        [{"role": "user", "content": "Delete the lamp record."}],
        household_now=datetime(2026, 9, 13),
    )
    assert result.plan is not None and result.plan.mutation is not None
    assert result.attempts == 1
    assert len(client.calls) == 1
    assert client.calls[0]["format"] == unified_output_schema(schema)


def test_shadow_summary_reports_paired_safety_regression():
    base = {
        "sample": 0,
        "category": "mixed",
        "utterance": "question plus delete",
        "llm_calls": 1,
        "input_tokens": 10,
        "output_tokens": 2,
        "latency_ms": 1,
        "validation_error": None,
        "length_stops": 0,
        "classification_correct": True,
        "payload_correct": True,
        "decision_kind": "mutation",
        "mutation": {"operation": "delete"},
        "partial_plan": True,
    }
    candidate = {
        **base,
        "route": "unified",
        "classification_correct": False,
        "payload_correct": False,
        "decision_kind": "fact",
        "mutation": None,
    }
    report = summarize({
        "summary": {},
        "rows": [{**base, "route": "existing"}, candidate],
    }, only_deltas=True)
    assert report["paired_deltas"]["classification_correct"] == {
        "regressions": 1,
        "improvements": 0,
    }
    assert len(report["cases"]) == 1
