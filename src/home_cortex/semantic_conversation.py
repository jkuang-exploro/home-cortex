"""Bounded trusted discourse bindings; assistant prose never becomes graph truth."""
from __future__ import annotations

import asyncio

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Any
from uuid import uuid4

from .request_tracing import stage
from .semantic_ir import (
    AgentRequestContext,
    DiscourseContext,
    FactAnswer,
    SemanticMutationIntent,
    SemanticFactRequest,
)
from .semantic_facts import SemanticFactService

MAX_DISCOURSE_TURNS = 8


@dataclass
class _Conversation:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0
    messages: list[dict[str, str]] = field(default_factory=list)
    turns: tuple[tuple[str, ...], ...] = ()


class SemanticConversationService:
    def __init__(self, facts: SemanticFactService, maximum: int = 1000) -> None:
        self.facts = facts
        self.maximum = maximum
        self._states: OrderedDict[tuple[str, str, str | None, str], _Conversation] = OrderedDict()

    @stage("conversation.total")
    async def try_answer(
        self, messages: Sequence[Mapping[str, Any]], *, context: AgentRequestContext,
        request_id: str = "-",
    ) -> FactAnswer | SemanticMutationIntent | None:
        users = [{"role": "user", "content": str(message.get("content", ""))}
                 for message in messages if message.get("role") == "user"]
        if not users:
            raise ValueError("At least one user message is required")
        if context.conversation_id and context.caller_entity_id:
            key = (context.conversation_id, context.caller_entity_id,
                   context.household_id, context.assistant_id)
            state = self._states.get(key)
            if state is None:
                if len(self._states) >= self.maximum:
                    idle = next((key for key, value in self._states.items() if value.users == 0), None)
                    if idle is not None:
                        del self._states[idle]
                    else:
                        # No state capacity: do not import caller history into a session.
                        return await self._replay(users[-1:], context, request_id)
                state = _Conversation()
                self._states[key] = state
            self._states.move_to_end(key)
            state.users += 1
            try:
                async with state.lock:
                    history = [*state.messages, users[-1]]
                    trusted = self._context(context, state.turns)
                    try:
                        answer = await self.facts.try_answer(history, context=trusted, request_id=request_id)
                    except BaseException:
                        # Interrupted turns cannot leave a stale focus current.
                        state.messages = history[-MAX_DISCOURSE_TURNS:]
                        state.turns = (*state.turns, ())[-MAX_DISCOURSE_TURNS:]
                        raise
                    state.messages = history[-MAX_DISCOURSE_TURNS:]
                    state.turns = (*state.turns, self._focus(answer))[-MAX_DISCOURSE_TURNS:]
                    return answer
            finally:
                state.users -= 1
        return await self._replay(users[-(MAX_DISCOURSE_TURNS + 1):], context, request_id)

    async def _replay(
        self, users: Sequence[Mapping[str, str]], context: AgentRequestContext, request_id: str,
    ) -> FactAnswer | SemanticMutationIntent | None:
        started = perf_counter()
        context = replace(context, conversation_id=uuid4().hex, discourse=None)
        return await self._answer_prefix(
            users,
            len(users),
            context,
            request_id,
            {},
            {},
            started,
        )

    async def _answer_prefix(
        self,
        users: Sequence[Mapping[str, str]],
        end: int,
        context: AgentRequestContext,
        request_id: str,
        memo: dict[int, FactAnswer | SemanticMutationIntent | None],
        planner_metrics: dict[int, tuple[float, int]],
        started: float,
    ) -> FactAnswer | SemanticMutationIntent | None:
        """Interpret one turn, resolving only antecedents its plan actually uses."""
        if end in memo:
            return memo[end]
        empty_turns = ((),) * max(0, end - 1)
        answer = await self.facts.try_answer(
            users[:end],
            context=self._context(context, empty_turns),
            request_id=request_id,
        )
        memo[end] = answer
        if answer is not None:
            planner_metrics[end] = (
                answer.timings.llm_ms,
                answer.timings.llm_call_count,
            )
        offsets = _discourse_offsets(answer.request) if isinstance(answer, FactAnswer) else ()
        if not offsets:
            return answer

        turns: list[tuple[str, ...]] = list(empty_turns)
        for offset in offsets:
            target = end - 1 - offset
            if target < 0:
                continue
            antecedent = await self._answer_prefix(
                users,
                target + 1,
                context,
                request_id,
                memo,
                planner_metrics,
                started,
            )
            turns[target] = self._focus(antecedent)
        if answer is None:
            return None
        trusted = self._context(context, tuple(turns[-MAX_DISCOURSE_TURNS:]))
        resolved = await self.facts.answer_request(
            answer.request,
            context=trusted,
            request_id=request_id,
            started=started,
            llm_ms=sum(item[0] for item in planner_metrics.values()),
            llm_call_count=sum(item[1] for item in planner_metrics.values()),
            planner_diagnostics=answer.planner_diagnostics,
        )
        memo[end] = resolved
        return resolved

    @staticmethod
    def _context(context: AgentRequestContext, turns: tuple[tuple[str, ...], ...]) -> AgentRequestContext:
        assert context.conversation_id is not None
        return replace(context, discourse=DiscourseContext(
            context.conversation_id, context.caller_entity_id, context.household_id,
            context.assistant_id, turns,
        ))

    @staticmethod
    def _focus(answer: FactAnswer | SemanticMutationIntent | None) -> tuple[str, ...]:
        if not isinstance(answer, FactAnswer) or answer.result.status != "found":
            return ()
        return answer.result.focus_entity_ids


def _discourse_offsets(request: SemanticFactRequest) -> tuple[int, ...]:
    references = (request.subject, *request.exclude)
    if request.other is not None:
        references = (*references, request.other)
    return tuple(sorted({
        reference.turn_offset
        for reference in references
        if reference.kind == "discourse" and reference.turn_offset is not None
    }))
