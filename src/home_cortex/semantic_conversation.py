"""Bounded trusted discourse bindings; assistant prose never becomes graph truth."""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any
from uuid import uuid4

from .semantic_facts import AgentRequestContext, DiscourseContext, FactAnswer, SemanticFactService

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

    async def try_answer(
        self, messages: Sequence[Mapping[str, Any]], *, context: AgentRequestContext,
        request_id: str = "-",
    ) -> FactAnswer | None:
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
    ) -> FactAnswer | None:
        context = replace(context, conversation_id=uuid4().hex, discourse=None)
        turns: tuple[tuple[str, ...], ...] = ()
        answer = None
        for index in range(len(users)):
            answer = await self.facts.try_answer(users[:index + 1],
                                                context=self._context(context, turns), request_id=request_id)
            turns = (*turns, self._focus(answer))[-MAX_DISCOURSE_TURNS:]
        return answer

    @staticmethod
    def _context(context: AgentRequestContext, turns: tuple[tuple[str, ...], ...]) -> AgentRequestContext:
        assert context.conversation_id is not None
        return replace(context, discourse=DiscourseContext(
            context.conversation_id, context.caller_entity_id, context.household_id,
            context.assistant_id, turns,
        ))

    @staticmethod
    def _focus(answer: FactAnswer | None) -> tuple[str, ...]:
        if answer is None or answer.result.status != "found":
            return ()
        return answer.result.focus_entity_ids
