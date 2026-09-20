"""Planning-only adapter for benchmarking the integrated unified planner."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from home_cortex.semantic.ir import (
    AgentRequestContext,
    SemanticPlan,
    SemanticPlannerFailure,
)
from home_cortex.semantic.schema import SemanticSchemaRegistry
from home_cortex.semantic.unified_planner import (
    UnifiedSemanticPlanner,
    unified_chat_messages,
    unified_output_schema,
    validate_unified_payload,
)


@dataclass(frozen=True)
class UnifiedShadowResult:
    plan: SemanticPlan | None
    attempts: int
    latency_ms: float
    calls: tuple[Mapping[str, Any], ...]
    output_raw: Mapping[str, Any] | None
    error: str | None


class _ObservedInterpreter:
    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.calls: list[Mapping[str, Any]] = []
        self.last_planner_runtime: Mapping[str, Any] = {}

    async def plan_unified_semantic(self, messages, output_schema):
        try:
            return await self.delegate.plan_unified_semantic(
                messages, output_schema
            )
        finally:
            self.last_planner_runtime = dict(
                getattr(self.delegate, "last_planner_runtime", {}) or {}
            )
            self.calls.append(self.last_planner_runtime)


class UnifiedShadowPlanner:
    """Expose production planning decisions without executing either branch."""

    def __init__(self, ollama: Any, schema: SemanticSchemaRegistry) -> None:
        self.interpreter = _ObservedInterpreter(ollama)
        self.planner = UnifiedSemanticPlanner(self.interpreter, schema)

    async def plan(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        household_now: datetime,
    ) -> UnifiedShadowResult:
        self.interpreter.calls.clear()
        context = AgentRequestContext(
            caller_entity_id=None,
            assistant_id="steward",
            assistant_display_name="Steward",
            household_id=None,
            current_time=household_now,
        )
        try:
            outcome = await self.planner.plan(messages, context)
        except SemanticPlannerFailure as error:
            diagnostics = error.diagnostics
            return UnifiedShadowResult(
                plan=None,
                attempts=diagnostics.attempt_count,
                latency_ms=diagnostics.latency_ms,
                calls=tuple(self.interpreter.calls),
                output_raw=diagnostics.output_raw,
                error=diagnostics.validation_result,
            )
        return UnifiedShadowResult(
            plan=outcome.plan,
            attempts=outcome.diagnostics.attempt_count,
            latency_ms=outcome.latency_ms,
            calls=tuple(self.interpreter.calls),
            output_raw=outcome.diagnostics.output_raw,
            error=None,
        )


__all__ = [
    "UnifiedShadowPlanner",
    "unified_chat_messages",
    "unified_output_schema",
    "validate_unified_payload",
]
