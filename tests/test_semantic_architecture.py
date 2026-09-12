"""Small ownership contracts for agents changing the factual pipeline."""
import json
import subprocess
import sys
from unittest.mock import AsyncMock

import pytest

from home_cortex.semantic_ir import FactEvidence, FactResult, SemanticFactRequest, SemanticReference
from test_semantic_contract import household


@pytest.mark.parametrize("module,forbidden", [
    ("semantic_ir", {"writing", "db", "retrieval", "tools", "semantic_facts"}),
    ("household_fact_engine", {"semantic_planner", "fact_renderer", "semantic_facts", "ollama"}),
])
def test_semantic_layers_do_not_import_their_callers(module, forbidden):
    script = (
        f"import home_cortex.{module}; import json, sys; "
        "print(json.dumps([name.removeprefix('home_cortex.') "
        "for name in sys.modules if name.startswith('home_cortex.')]))"
    )
    loaded = json.loads(subprocess.check_output([sys.executable, "-c", script], text=True))
    assert forbidden.isdisjoint(loaded)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [
    "entity_not_found", "caller_context_missing", "discourse_context_missing",
    "ambiguous", "relationship_not_found", "filter_input_missing", "collection_incomplete",
])
async def test_resolution_failure_keeps_canonical_result_and_evidence(household, monkeypatch, status):
    engine, context, _ = household
    failure = FactResult(
        status, evidence=FactEvidence(entity_ids=("person:unresolved",)),
        missing_requirements=("required_fact",), candidates=({"id": "person:candidate"},),
    )
    monkeypatch.setattr(engine.resolver, "resolve", AsyncMock(return_value=failure))
    result, *_ = await engine.execute(SemanticFactRequest(
        operation="resolve_reference", subject=SemanticReference(kind="self"),
    ), context)
    # Grounding already produced the public failure; execution must not rename
    # its status or rebuild a narrower result that drops evidence or candidates.
    assert result is failure
