"""Historical two-call planner used only to reproduce pre-unification baselines."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from time import perf_counter
from typing import Any

from home_cortex.semantic.ir import (
    AgentRequestContext,
    PlannerDiagnostics,
    SemanticPlan,
    SemanticPlannerOutcome,
)
from home_cortex.semantic.planner import (
    SemanticFactPlanner,
    _planner_runtime_fields,
    planner_input_summary,
)
from home_cortex.semantic.schema import SemanticSchemaRegistry


class LegacyTwoStagePlanner:
    """Preserve the rejected mutation-first route for controlled comparisons."""

    def __init__(self, ollama: Any, schema: SemanticSchemaRegistry) -> None:
        self.ollama = ollama
        self.schema = schema
        self.read_planner = SemanticFactPlanner(ollama, schema)

    async def plan(
        self,
        messages: Sequence[Mapping[str, Any]],
        context: AgentRequestContext,
    ) -> SemanticPlannerOutcome:
        started = perf_counter()
        decision, mutation_runtime = await self.ollama.plan_item_mutation(messages)
        mutation_ms = (perf_counter() - started) * 1000
        if decision.requires_mutation:
            plan = SemanticPlan(requires_fact=False, mutation=decision.mutation)
            latency_ms = (perf_counter() - started) * 1000
            diagnostics = PlannerDiagnostics(
                input_summary=planner_input_summary(self.schema.capability_payload()),
                output_raw=decision.model_dump(mode="json"),
                normalized_plan=plan.model_dump(mode="json"),
                validation_result="VALID",
                attempt_count=1,
                latency_ms=latency_ms,
                request_ms=mutation_ms,
                **_planner_runtime_fields(mutation_runtime),
            )
            return SemanticPlannerOutcome(plan, latency_ms, diagnostics)
        outcome = await self.read_planner.plan(messages, context)
        latency_ms = (perf_counter() - started) * 1000
        read = outcome.diagnostics
        combined = {
            key: getattr(read, key) + float(mutation_runtime.get(key, 0) or 0)
            for key in (
                "prompt_eval_count",
                "prompt_eval_duration_ms",
                "eval_count",
                "eval_duration_ms",
                "load_duration_ms",
            )
        }
        diagnostics = replace(
            read,
            attempt_count=read.attempt_count + 1,
            latency_ms=latency_ms,
            request_ms=read.request_ms + mutation_ms,
            prompt_eval_count=int(combined["prompt_eval_count"]),
            prompt_eval_duration_ms=combined["prompt_eval_duration_ms"],
            eval_count=int(combined["eval_count"]),
            eval_duration_ms=combined["eval_duration_ms"],
            load_duration_ms=combined["load_duration_ms"],
        )
        return SemanticPlannerOutcome(outcome.plan, latency_ms, diagnostics)
