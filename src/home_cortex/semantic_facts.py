"""Coordinate interpretation, deterministic execution, rendering, and diagnostics.

This is the utterance-to-answer entry point. Structured callers may invoke
HouseholdFactEngine directly; semantic types live in semantic_ir.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from time import perf_counter
from typing import Any

from .semantic_ir import (
    AgentRequestContext, FactAnswer, FactResult, FactStatus, FactTimings,
    PlannerDiagnostics, SemanticFactRequest, SemanticMutationIntent,
    SemanticPlannerFailure, SemanticReference,
)
from .household_fact_engine import HouseholdFactEngine
from .semantic_planner import SemanticFactPlanner
from .fact_renderer import FactRenderer
from .text import safe_log_token

logger = logging.getLogger("uvicorn.error.home_cortex.semantic_facts")

class SemanticFactService:
    def __init__(
        self,
        engine: HouseholdFactEngine,
        planner: SemanticFactPlanner,
        renderer: FactRenderer | None = None,
    ) -> None:
        self.engine = engine
        self.renderer = renderer or FactRenderer(engine.schema.ontology)
        self.planner = planner

    async def try_answer(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        context: AgentRequestContext,
        request_id: str = "-",
    ) -> FactAnswer | SemanticMutationIntent | None:
        started = perf_counter()
        routing_started = perf_counter()
        llm_ms = 0.0
        llm_call_count = 0
        planner_diagnostics: PlannerDiagnostics | None = None
        llm_started = perf_counter()
        try:
            outcome = await self.planner.plan(messages, context)
            plan = outcome.plan
            llm_ms = outcome.latency_ms
            planner_diagnostics = outcome.diagnostics
            llm_call_count = outcome.diagnostics.attempt_count
        except SemanticPlannerFailure as error:
            planner_diagnostics = error.diagnostics
            llm_ms = error.diagnostics.latency_ms
            llm_call_count = error.diagnostics.attempt_count
            logger.warning(
                "semantic_plan_invalid request_id=%s validation=%s attempts=%d",
                safe_log_token(request_id),
                safe_log_token(error.diagnostics.validation_result),
                error.diagnostics.attempt_count,
            )
            return self._failure_answer(
                context,
                started,
                request_id=request_id,
                llm_ms=llm_ms,
                llm_call_count=llm_call_count,
                planner_diagnostics=planner_diagnostics,
            )
        except Exception as error:
            llm_ms = (perf_counter() - llm_started) * 1000
            llm_call_count = max(llm_call_count, 1)
            logger.warning(
                "semantic_plan_invalid request_id=%s error=%s",
                safe_log_token(request_id),
                safe_log_token(type(error).__name__),
            )
            return self._failure_answer(
                context,
                started,
                request_id=request_id,
                llm_ms=llm_ms,
                llm_call_count=llm_call_count,
                planner_diagnostics=planner_diagnostics,
            )
        if plan.mutation is not None:
            # Return intent only: discourse replay must never execute writes.
            return SemanticMutationIntent(
                plan.mutation, FactTimings(tier=1, llm_ms=llm_ms, llm_call_count=llm_call_count,
                                          total_ms=(perf_counter() - started) * 1000),
                outcome.diagnostics,
            )
        if not plan.requires_fact:
            return None
        request = plan.request
        if request is None or not self.engine.schema.validates(request):
            return self._failure_answer(
                context,
                started,
                request=request,
                request_id=request_id,
                llm_ms=llm_ms,
                llm_call_count=llm_call_count,
                planner_diagnostics=planner_diagnostics,
            )
        return await self.answer_request(
            request,
            context=context,
            request_id=request_id,
            started=started,
            routing_started=routing_started,
            llm_ms=llm_ms,
            llm_call_count=llm_call_count,
            planner_diagnostics=planner_diagnostics,
        )

    async def answer_request(
        self,
        request: SemanticFactRequest,
        *,
        context: AgentRequestContext,
        request_id: str = "-",
        started: float | None = None,
        routing_started: float | None = None,
        llm_ms: float = 0.0,
        llm_call_count: int = 0,
        planner_diagnostics: PlannerDiagnostics | None = None,
    ) -> FactAnswer:
        """Execute a previously validated semantic request without another LLM call."""
        started = perf_counter() if started is None else started
        routing_started = started if routing_started is None else routing_started
        if not self.engine.schema.validates(request):
            return self._failure_answer(
                context,
                started,
                request=request,
                request_id=request_id,
                llm_ms=llm_ms,
                llm_call_count=llm_call_count,
                planner_diagnostics=planner_diagnostics,
            )
        routing_ms = (perf_counter() - routing_started) * 1000
        query_started = perf_counter()
        result, query_count, resolution_ms, computation_ms = await self.engine.execute(
            request,
            context,
        )
        fact_query_ms = (perf_counter() - query_started) * 1000
        render_started = perf_counter()
        rendered = self.renderer.render(request, result, context)
        render_ms = (perf_counter() - render_started) * 1000
        total_ms = (perf_counter() - started) * 1000
        timings = FactTimings(
            tier=1,
            routing_ms=routing_ms,
            entity_resolution_ms=resolution_ms,
            fact_query_ms=fact_query_ms,
            computation_ms=computation_ms,
            render_ms=render_ms,
            llm_ms=llm_ms,
            total_ms=total_ms,
            llm_call_count=llm_call_count,
            db_query_count=query_count,
        )
        _log_fact_query(
            request_id,
            request,
            result,
            timings,
            planner_diagnostics=planner_diagnostics,
        )
        return FactAnswer(request, result, rendered, timings, planner_diagnostics)

    def _failure_answer(
        self,
        context: AgentRequestContext,
        started: float,
        *,
        request: SemanticFactRequest | None = None,
        request_id: str,
        llm_ms: float,
        llm_call_count: int,
        planner_diagnostics: PlannerDiagnostics | None = None,
        status: FactStatus = "semantic_plan_unsupported",
    ) -> FactAnswer:
        safe_request = request or SemanticFactRequest(
            operation="resolve_reference",
            subject=SemanticReference(kind="self", entity_type="person"),
        )
        result = FactResult(status)
        render_started = perf_counter()
        rendered = self.renderer.render(safe_request, result, context)
        render_ms = (perf_counter() - render_started) * 1000
        timings = FactTimings(
            tier=1,
            render_ms=render_ms,
            llm_ms=llm_ms,
            llm_call_count=llm_call_count,
            total_ms=(perf_counter() - started) * 1000,
        )
        _log_fact_query(
            request_id,
            safe_request,
            result,
            timings,
            planner_diagnostics=planner_diagnostics,
        )
        return FactAnswer(
            safe_request,
            result,
            rendered,
            timings,
            planner_diagnostics,
        )


def _log_fact_query(
    request_id: str,
    request: SemanticFactRequest,
    result: FactResult,
    timings: FactTimings,
    *,
    planner_diagnostics: PlannerDiagnostics | None = None,
) -> None:
    logger.info(
        "fact_query request_id=%s tier=%d operation=%s semantic_plan=%s "
        "db_queries=%d llm_calls=%d routing_ms=%.2f "
        "entity_resolution_ms=%.2f fact_query_ms=%.2f computation_ms=%.2f "
        "render_ms=%.2f llm_ms=%.2f total_ms=%.2f status=%s failure_stage=%s "
        "planner_validation=%s planner_attempts=%d "
        "planner_prompt_build_ms=%.2f planner_request_ms=%.2f "
        "planner_validation_ms=%.2f prompt_eval_count=%d "
        "prompt_eval_ms=%.2f eval_count=%d eval_ms=%.2f load_ms=%.2f",
        safe_log_token(request_id),
        timings.tier,
        safe_log_token(request.operation),
        safe_log_token(
            json.dumps(_plan_for_log(request), separators=(",", ":"))
        ),
        timings.db_query_count,
        timings.llm_call_count,
        timings.routing_ms,
        timings.entity_resolution_ms,
        timings.fact_query_ms,
        timings.computation_ms,
        timings.render_ms,
        timings.llm_ms,
        timings.total_ms,
        safe_log_token(result.status),
        safe_log_token(_failure_stage(result.status) or "none"),
        safe_log_token(
            planner_diagnostics.validation_result
            if planner_diagnostics is not None
            else "not_run"
        ),
        planner_diagnostics.attempt_count if planner_diagnostics is not None else 0,
        planner_diagnostics.prompt_build_ms if planner_diagnostics is not None else 0,
        planner_diagnostics.request_ms if planner_diagnostics is not None else 0,
        planner_diagnostics.validation_ms if planner_diagnostics is not None else 0,
        planner_diagnostics.prompt_eval_count if planner_diagnostics is not None else 0,
        planner_diagnostics.prompt_eval_duration_ms
        if planner_diagnostics is not None
        else 0,
        planner_diagnostics.eval_count if planner_diagnostics is not None else 0,
        planner_diagnostics.eval_duration_ms if planner_diagnostics is not None else 0,
        planner_diagnostics.load_duration_ms if planner_diagnostics is not None else 0,
    )


def _plan_for_log(request: SemanticFactRequest) -> dict[str, Any]:
    payload = request.model_dump(mode="json")
    for key in ("subject", "other"):
        reference = payload.get(key)
        if not isinstance(reference, dict):
            continue
        if reference.get("value") is not None:
            reference["value"] = "[redacted]"
        for step in reference.get("path", []):
            if not isinstance(step, dict):
                continue
            for item in step.get("filters", []):
                if isinstance(item, dict) and "value" in item:
                    item["value"] = "[redacted]"
    for item in payload.get("filters", []):
        if isinstance(item, dict) and item.get("value") is not None:
            item["value"] = "[redacted]"
    return payload


def _failure_stage(status: FactStatus) -> str | None:
    return {
        "found": None,
        "caller_context_missing": "context",
        "discourse_context_missing": "context",
        "entity_not_found": "entity_resolution",
        "relationship_not_found": "relationship_resolution",
        "property_unavailable": "entity_property",
        "relation_property_unavailable": "relationship_property",
        "ambiguous": "entity_resolution",
        "filter_input_missing": "filter",
        "filter_unsupported": "filter_validation",
        "operator_unsupported": "operator_validation",
        "computation_input_missing": "computation_input",
        "computation_impossible": "computation",
        "collection_incomplete": "collection_completeness",
        "semantic_plan_unsupported": "semantic_plan_validation",
    }[status]
