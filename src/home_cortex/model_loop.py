"""Bounded Ollama tool loop for open-ended, graph-grounded conversation."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Literal

from .display import (
    DisplayNameResolver,
    DisplayTextStream,
    resolve_display_name,
    resolve_person_reference,
)
from .fallbacks import grounding_fallback, no_records_fallback
from .memorable_dates import (
    MemorableDateRegistry,
    default_memorable_date_registry,
)
from .ollama import OllamaService
from .request_analysis import (
    EvidenceRequirements,
    PRIVATE_TOOL_FIELDS,
    RequestAnalysis,
    analyze_household_request,
)
from .text import normalize_language_code, safe_log_token
from .tools import GRAPH_TOOL_NAMES, ToolDispatcher

MAX_AGENT_STEPS = 4
MAX_TOOL_CALLS_PER_STEP = 4
MAX_TOOL_RECORDS = 25
MAX_TOOL_RESULT_BYTES = 16_384
TOOL_EXECUTION_TIMEOUT_SECONDS = 5.0
MODEL_HIDDEN_FIELDS = frozenset(
    {"address_as", "first_name", "gender", "last_name"}
)

logger = logging.getLogger("uvicorn.error.home_cortex.agent_service")
StopReason = Literal["answer", "step_limit", "tool_error", "timeout"]


class AgentLimitError(RuntimeError):
    """Raised when the model exceeds a hard agent-loop safety limit."""

    stop_reason: StopReason = "step_limit"


class AgentStreamingError(RuntimeError):
    """Raised when a streamed model response cannot be handled safely."""

    stop_reason: StopReason = "tool_error"


@dataclass(frozen=True)
class AgentResult:
    answer: str
    steps: int
    tool_calls: int
    stop_reason: StopReason
    messages: tuple[dict[str, Any], ...]


@dataclass
class _LoopState:
    conversation: list[dict[str, Any]]
    request_id: str
    presentation_language: str
    expose_internal_ids: bool
    presentation_values: Sequence[Any]
    trusted_user_entity_id: str | None
    analysis: RequestAnalysis
    successful_tools: set[str] = field(default_factory=set)
    nonempty_tools: set[str] = field(default_factory=set)
    evidence_fields: set[str] = field(default_factory=set)
    total_tool_calls: int = 0
    failure_reason: StopReason | None = None
    grounding_retry: bool = False

    @property
    def evidence_required(self) -> bool:
        return self.analysis.evidence_required

    @property
    def requirements(self) -> EvidenceRequirements:
        return self.analysis.evidence

    @property
    def allowed_private_fields(self) -> frozenset[str]:
        return self.analysis.private_fields

    def graph_evidence_complete(self) -> bool:
        return _has_required_evidence(
            self.evidence_required,
            self.requirements,
            self.successful_tools,
        ) and (
            not self.evidence_required
            or _has_nonempty_evidence(
                self.requirements,
                self.nonempty_tools,
                self.evidence_fields,
            )
        )


class _SelfVocativeStream:
    """Repair an agent role accidentally emitted as the user's salutation."""

    def __init__(
        self,
        agent_name: str | None,
        speaker_address: str | None,
        language: str,
    ) -> None:
        self.agent_name = agent_name
        self.speaker_address = speaker_address
        self.language = language
        self._pending = ""
        self._resolved = agent_name is None

    def feed(self, text: str) -> str:
        if self._resolved:
            return text
        self._pending += text
        prefixes = _self_vocative_prefixes(self.agent_name)
        folded = self._pending.casefold()
        possible = False
        for prefix in prefixes:
            folded_prefix = prefix.casefold()
            if folded.startswith(folded_prefix):
                remainder = self._pending[len(prefix) :].lstrip()
                self._pending = ""
                self._resolved = True
                return (
                    _address_prefix(self.speaker_address, self.language)
                    + remainder
                )
            if folded_prefix.startswith(folded):
                possible = True
        if possible:
            return ""
        self._resolved = True
        content = self._pending
        self._pending = ""
        return content

    def finish(self) -> str:
        content = self._pending
        self._pending = ""
        self._resolved = True
        return content


class ModelLoop:
    """Run a bounded Ollama tool-calling loop."""

    def __init__(
        self,
        ollama: OllamaService,
        dispatcher: ToolDispatcher,
        *,
        system_prompt: str,
        tools: Sequence[Mapping[str, Any]],
        max_steps: int = MAX_AGENT_STEPS,
        max_tool_calls_per_step: int = MAX_TOOL_CALLS_PER_STEP,
        max_tool_records: int = MAX_TOOL_RECORDS,
        max_tool_result_bytes: int = MAX_TOOL_RESULT_BYTES,
        tool_timeout_seconds: float = TOOL_EXECUTION_TIMEOUT_SECONDS,
        localized_identity: Mapping[str, str] | None = None,
        memorable_dates: MemorableDateRegistry | None = None,
    ) -> None:
        self.ollama = ollama
        self.dispatcher = dispatcher
        self.system_prompt = system_prompt.strip()
        if not self.system_prompt:
            raise ValueError("system_prompt cannot be empty")
        self.tools = tuple(dict(tool) for tool in tools)
        if not self.tools:
            raise ValueError("At least one tool definition is required")
        self._post_graph_tools = tuple(
            tool
            for tool in self.tools
            if tool["function"]["name"] not in GRAPH_TOOL_NAMES
        )
        self.localized_identity = {
            normalize_language_code(str(language)): name.strip()
            for language, name in (localized_identity or {}).items()
            if isinstance(name, str) and name.strip()
        }
        self.memorable_dates = memorable_dates or default_memorable_date_registry()
        self._tools_with_limit = frozenset(
            tool["function"]["name"]
            for tool in self.tools
            if "limit" in tool["function"]["parameters"].get("properties", {})
        )
        self._tool_limit_maximums = {
            tool["function"]["name"]: tool["function"]["parameters"][
                "properties"
            ]["limit"].get("maximum")
            for tool in self.tools
            if tool["function"]["name"] in self._tools_with_limit
        }
        self.max_steps = _bounded_limit("max_steps", max_steps, MAX_AGENT_STEPS)
        self.max_tool_calls_per_step = _bounded_limit(
            "max_tool_calls_per_step",
            max_tool_calls_per_step,
            MAX_TOOL_CALLS_PER_STEP,
        )
        self.max_tool_records = _bounded_limit(
            "max_tool_records",
            max_tool_records,
            MAX_TOOL_RECORDS,
        )
        self.max_tool_result_bytes = _bounded_limit(
            "max_tool_result_bytes",
            max_tool_result_bytes,
            MAX_TOOL_RESULT_BYTES,
            minimum=256,
        )
        if not 0 < tool_timeout_seconds <= TOOL_EXECUTION_TIMEOUT_SECONDS:
            raise ValueError(
                "tool_timeout_seconds must be greater than zero and no more than "
                f"{TOOL_EXECUTION_TIMEOUT_SECONDS}"
            )
        self.tool_timeout_seconds = tool_timeout_seconds

    async def stream(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        request_id: str = "-",
        presentation_language: str = "en",
        expose_internal_ids: bool = False,
        presentation_values: Sequence[Any] = (),
        trusted_user_entity_id: str | None = None,
        analysis: RequestAnalysis | None = None,
    ) -> AsyncIterator[str]:
        """Run the tool loop and stream chunks from the final Ollama answer."""
        state = await self._begin_loop(
            messages,
            request_id=request_id,
            presentation_language=presentation_language,
            expose_internal_ids=expose_internal_ids,
            presentation_values=presentation_values,
            trusted_user_entity_id=trusted_user_entity_id,
            analysis=analysis,
        )
        for step in range(1, self.max_steps + 1):
            display_stream = DisplayTextStream(
                DisplayNameResolver.from_messages(
                    state.conversation,
                    state.presentation_values,
                ),
                state.presentation_language,
                expose_internal_ids=state.expose_internal_ids,
            )
            vocative_stream = _SelfVocativeStream(
                self.localized_identity.get(state.presentation_language),
                _speaker_address(
                    state.presentation_values,
                    state.presentation_language,
                ),
                state.presentation_language,
            )
            content_parts: list[str] = []
            tool_calls: list[Any] = []
            emitted_content = False
            can_emit = state.graph_evidence_complete()
            available_tools = self._available_tools(
                state.evidence_required,
                can_emit,
            )

            async for response in self.ollama.stream_chat_with_tools(
                state.conversation,
                available_tools,
            ):
                chunk_tool_calls = list(response.message.tool_calls or [])
                if chunk_tool_calls and emitted_content:
                    logger.info(
                        "agent_stop request_id=%s reason=tool_error steps=%d "
                        "tool_calls=%d",
                        safe_log_token(state.request_id),
                        step,
                        state.total_tool_calls,
                    )
                    raise AgentStreamingError(
                        "Ollama emitted tool calls after final-answer content"
                    )
                tool_calls.extend(chunk_tool_calls)

                content = response.message.content or ""
                if content:
                    content_parts.append(content)
                    if not tool_calls:
                        if can_emit:
                            rendered = display_stream.feed(content)
                            if rendered:
                                addressed = vocative_stream.feed(rendered)
                                if addressed:
                                    emitted_content = True
                                    yield addressed

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": "".join(content_parts),
            }
            if tool_calls:
                assistant_message["tool_calls"] = [
                    tool_call.model_dump(exclude_none=True)
                    for tool_call in tool_calls
                ]
            state.conversation.append(assistant_message)
            self._log_step(state, step, len(tool_calls))

            if not tool_calls:
                decision = self._decide_empty_tools(state, step)
                if decision == "retry":
                    continue
                if decision == "grounding_fallback":
                    yield grounding_fallback(state.presentation_language)
                    self._log_stop(
                        state,
                        state.failure_reason or "tool_error",
                        step,
                    )
                    return
                if decision == "no_records":
                    yield no_records_fallback(state.presentation_language)
                    self._log_stop(state, "answer", step)
                    return
                rendered = display_stream.finish()
                if rendered:
                    addressed = vocative_stream.feed(rendered)
                    if addressed:
                        yield addressed
                trailing = vocative_stream.finish()
                if trailing:
                    yield trailing
                self._log_stop(state, state.failure_reason or "answer", step)
                return

            self._check_tool_call_limits(
                tool_calls,
                step,
                state.total_tool_calls,
                state.request_id,
            )
            await self._apply_tool_calls(state, tool_calls, step)

        self._raise_step_limit(state.request_id, state.total_tool_calls)

    async def run(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        request_id: str = "-",
        presentation_language: str = "en",
        expose_internal_ids: bool = False,
        presentation_values: Sequence[Any] = (),
        trusted_user_entity_id: str | None = None,
        analysis: RequestAnalysis | None = None,
    ) -> AgentResult:
        state = await self._begin_loop(
            messages,
            request_id=request_id,
            presentation_language=presentation_language,
            expose_internal_ids=expose_internal_ids,
            presentation_values=presentation_values,
            trusted_user_entity_id=trusted_user_entity_id,
            analysis=analysis,
        )
        for step in range(1, self.max_steps + 1):
            available_tools = self._available_tools(
                state.evidence_required,
                state.graph_evidence_complete(),
            )
            response = await self.ollama.chat_with_tools(
                state.conversation,
                available_tools,
            )
            assistant_message = response.message.model_dump(exclude_none=True)
            state.conversation.append(assistant_message)
            tool_calls = list(response.message.tool_calls or [])
            self._log_step(state, step, len(tool_calls))

            if not tool_calls:
                decision = self._decide_empty_tools(state, step)
                if decision == "retry":
                    continue
                if decision == "grounding_fallback":
                    answer = grounding_fallback(state.presentation_language)
                    stop_reason = state.failure_reason or "tool_error"
                    self._log_stop(state, stop_reason, step)
                    return self._agent_result(state, answer, step, stop_reason)
                if decision == "no_records":
                    answer = no_records_fallback(state.presentation_language)
                    self._log_stop(state, "answer", step)
                    return self._agent_result(state, answer, step, "answer")
                resolver = DisplayNameResolver.from_messages(
                    state.conversation,
                    state.presentation_values,
                )
                answer = resolver.render(
                    response.message.content or "",
                    state.presentation_language,
                    expose_internal_ids=state.expose_internal_ids,
                )
                answer = _repair_self_vocative(
                    answer,
                    self.localized_identity.get(state.presentation_language),
                    _speaker_address(
                        state.presentation_values,
                        state.presentation_language,
                    ),
                    state.presentation_language,
                )
                stop_reason = state.failure_reason or "answer"
                self._log_stop(state, stop_reason, step)
                return self._agent_result(state, answer, step, stop_reason)

            self._check_tool_call_limits(
                tool_calls,
                step,
                state.total_tool_calls,
                state.request_id,
            )
            await self._apply_tool_calls(state, tool_calls, step)

        self._raise_step_limit(state.request_id, state.total_tool_calls)

    async def _begin_loop(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        request_id: str,
        presentation_language: str,
        expose_internal_ids: bool,
        presentation_values: Sequence[Any],
        trusted_user_entity_id: str | None,
        analysis: RequestAnalysis | None,
    ) -> _LoopState:
        conversation = [dict(message) for message in messages]
        if not conversation:
            raise ValueError("At least one message is required")
        if analysis is None:
            analysis = analyze_household_request(
                conversation,
                memorable_dates=self.memorable_dates,
            )
        state = _LoopState(
            conversation=conversation,
            request_id=request_id,
            presentation_language=presentation_language,
            expose_internal_ids=expose_internal_ids,
            presentation_values=presentation_values,
            trusted_user_entity_id=trusted_user_entity_id,
            analysis=analysis,
        )
        (
            failure_reason,
            prefetched_successful,
            prefetched_nonempty,
            prefetched_fields,
            prefetched_tool_calls,
        ) = await self._prefetch_related_entity_facts(
            state.conversation,
            requirements=state.requirements,
            trusted_user_entity_id=state.trusted_user_entity_id,
            request_id=state.request_id,
            presentation_language=state.presentation_language,
            allowed_private_fields=state.allowed_private_fields,
        )
        state.failure_reason = failure_reason
        state.successful_tools.update(prefetched_successful)
        state.nonempty_tools.update(prefetched_nonempty)
        state.evidence_fields.update(prefetched_fields)
        state.total_tool_calls += prefetched_tool_calls
        return state

    def _decide_empty_tools(self, state: _LoopState, step: int) -> str:
        if not _has_required_evidence(
            state.evidence_required,
            state.requirements,
            state.successful_tools,
        ):
            if (
                state.failure_reason is None
                and not state.grounding_retry
                and step < self.max_steps
            ):
                state.conversation.append(
                    _grounding_retry_message(state.requirements)
                )
                state.grounding_retry = True
                return "retry"
            return "grounding_fallback"
        if state.evidence_required and not _has_nonempty_evidence(
            state.requirements,
            state.nonempty_tools,
            state.evidence_fields,
        ):
            if step < self.max_steps and _should_retry_incomplete_evidence(
                state.requirements,
                state.evidence_fields,
            ):
                state.conversation.append(
                    _grounding_retry_message(state.requirements)
                )
                return "retry"
            return "no_records"
        return "answer"

    async def _apply_tool_calls(
        self,
        state: _LoopState,
        tool_calls: Sequence[Any],
        step: int,
    ) -> None:
        failure_reason, successful, nonempty, fields = await self._append_tool_results(
            state.conversation,
            tool_calls,
            step=step,
            request_id=state.request_id,
            failure_reason=state.failure_reason,
            presentation_language=state.presentation_language,
            allowed_private_fields=state.allowed_private_fields,
            requirements=state.requirements,
            caller_entity_id=state.trusted_user_entity_id,
        )
        state.failure_reason = failure_reason
        state.successful_tools.update(successful)
        state.nonempty_tools.update(nonempty)
        state.evidence_fields.update(fields)
        state.total_tool_calls += len(tool_calls)

    def _log_step(self, state: _LoopState, step: int, tool_calls: int) -> None:
        logger.info(
            "agent_step request_id=%s step=%d tool_calls=%d",
            safe_log_token(state.request_id),
            step,
            tool_calls,
        )

    def _log_stop(self, state: _LoopState, reason: str, step: int) -> None:
        logger.info(
            "agent_stop request_id=%s reason=%s steps=%d tool_calls=%d",
            safe_log_token(state.request_id),
            reason,
            step,
            state.total_tool_calls,
        )

    def _agent_result(
        self,
        state: _LoopState,
        answer: str,
        step: int,
        stop_reason: StopReason,
    ) -> AgentResult:
        return AgentResult(
            answer=answer,
            steps=step,
            tool_calls=state.total_tool_calls,
            stop_reason=stop_reason,
            messages=tuple(state.conversation),
        )

    def _check_tool_call_limits(
        self,
        tool_calls: Sequence[Any],
        step: int,
        total_tool_calls: int,
        request_id: str,
    ) -> None:
        if len(tool_calls) > self.max_tool_calls_per_step:
            logger.info(
                "agent_stop request_id=%s reason=step_limit steps=%d tool_calls=%d",
                safe_log_token(request_id),
                step,
                total_tool_calls,
            )
            raise AgentLimitError(
                "Ollama requested "
                f"{len(tool_calls)} tools in one step; the limit is "
                f"{self.max_tool_calls_per_step}"
            )
        if step == self.max_steps:
            logger.info(
                "agent_stop request_id=%s reason=step_limit steps=%d tool_calls=%d",
                safe_log_token(request_id),
                step,
                total_tool_calls,
            )
            raise AgentLimitError(
                f"Agent did not produce a final answer within {self.max_steps} steps"
            )

    async def _prefetch_related_entity_facts(
        self,
        conversation: list[dict[str, Any]],
        *,
        requirements: EvidenceRequirements,
        trusted_user_entity_id: str | None,
        request_id: str,
        presentation_language: str,
        allowed_private_fields: frozenset[str],
    ) -> tuple[StopReason | None, set[str], set[str], set[str], int]:
        """Resolve a known speaker's relationship + person fact deterministically.

        Asking the model to discover a relationship and then dereference the
        related person consumes most of the bounded loop and is needlessly
        nondeterministic. Cortex owns that graph traversal; the model only
        turns the resulting, privacy-filtered evidence into natural language.
        """
        if (
            trusted_user_entity_id is None
            or len(requirements.relations) != 1
            or ("get_entity", "dob") not in requirements.fields
        ):
            return None, set(), set(), set(), 0

        relation = next(iter(requirements.relations))
        relationship_arguments: dict[str, Any] = {
            "entity_id": trusted_user_entity_id,
            "relation": relation,
            "limit": self.max_tool_records,
        }
        if requirements.relationship_direction is not None:
            relationship_arguments["direction"] = (
                requirements.relationship_direction
            )

        successful_tools: set[str] = set()
        nonempty_tools: set[str] = set()
        evidence_fields: set[str] = set()
        evidence_payloads: list[str] = []
        failure_reason: StopReason | None = None
        tool_calls = 0

        relationship_result = await self._execute_planned_tool(
            "get_relationships",
            relationship_arguments,
            requirements=requirements,
            request_id=request_id,
            caller_entity_id=trusted_user_entity_id,
        )
        tool_calls += 1
        scoped_relationship_result = _scope_tool_result(
            "get_relationships",
            relationship_result,
            requirements,
        )
        successful, nonempty, fields = _tool_evidence(
            "get_relationships",
            relationship_arguments,
            scoped_relationship_result,
        )
        successful_tools.update(successful)
        nonempty_tools.update(nonempty)
        evidence_fields.update(fields)
        evidence_payloads.append(
            self._serialize_tool_result(
                "get_relationships",
                relationship_result,
                presentation_language=presentation_language,
                allowed_private_fields=allowed_private_fields,
                requirements=requirements,
            )
        )
        failure_reason = _tool_failure_reason(relationship_result, failure_reason)

        related_ids: list[str] = []
        relationship_records = scoped_relationship_result.get("result")
        if isinstance(relationship_records, list):
            for record in relationship_records:
                related = (
                    record.get("related_entity")
                    if isinstance(record, Mapping)
                    else None
                )
                related_id = related.get("id") if isinstance(related, Mapping) else None
                if (
                    isinstance(related_id, str)
                    and related_id not in related_ids
                ):
                    related_ids.append(related_id)

        maximum_entity_fetches = max(0, self.max_tool_calls_per_step - 1)
        for related_id in related_ids[
            : min(self.max_tool_records, maximum_entity_fetches)
        ]:
            entity_arguments = {"entity_id": related_id}
            entity_result = await self._execute_planned_tool(
                "get_entity",
                entity_arguments,
                requirements=requirements,
                request_id=request_id,
                caller_entity_id=trusted_user_entity_id,
            )
            tool_calls += 1
            successful, nonempty, fields = _tool_evidence(
                "get_entity",
                entity_arguments,
                entity_result,
            )
            successful_tools.update(successful)
            nonempty_tools.update(nonempty)
            evidence_fields.update(fields)
            evidence_payloads.append(
                self._serialize_tool_result(
                    "get_entity",
                    entity_result,
                    presentation_language=presentation_language,
                    allowed_private_fields=allowed_private_fields,
                    requirements=requirements,
                )
            )
            failure_reason = _tool_failure_reason(entity_result, failure_reason)

        conversation.append(
            {
                "role": "system",
                "content": (
                    "Trusted Home Cortex evidence was retrieved deterministically "
                    "for the latest request. Answer only from this evidence. Do "
                    "not call another tool when the requested relationship and "
                    "fact are present.\n"
                    + "\n".join(evidence_payloads)
                ),
            }
        )
        return (
            failure_reason,
            successful_tools,
            nonempty_tools,
            evidence_fields,
            tool_calls,
        )

    async def _execute_planned_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        requirements: EvidenceRequirements,
        request_id: str,
        caller_entity_id: str | None,
    ) -> dict[str, Any]:
        started = perf_counter()
        tool_result = await self._dispatch(
            tool_name,
            arguments,
            caller_entity_id=caller_entity_id,
        )
        duration_ms = (perf_counter() - started) * 1_000
        scoped_result = _scope_tool_result(tool_name, tool_result, requirements)
        records = scoped_result.get("result")
        record_count = len(records) if isinstance(records, list) else 0
        error = tool_result.get("error")
        error_code = error.get("code") if isinstance(error, Mapping) else "none"
        logger.info(
            "tool_execution request_id=%s step=0 tool=%s success=%s "
            "relation=%s direction=%s record_count=%d duration_ms=%.2f "
            "error_code=%s planned=true",
            safe_log_token(request_id),
            safe_log_token(tool_name),
            str(tool_result.get("ok") is True).lower(),
            safe_log_token(str(arguments.get("relation") or "none")),
            safe_log_token(str(arguments.get("direction") or "none")),
            record_count,
            duration_ms,
            safe_log_token(str(error_code)),
        )
        return tool_result

    async def _append_tool_results(
        self,
        conversation: list[dict[str, Any]],
        tool_calls: Sequence[Any],
        *,
        step: int,
        request_id: str,
        failure_reason: StopReason | None,
        presentation_language: str,
        allowed_private_fields: frozenset[str],
        requirements: EvidenceRequirements,
        caller_entity_id: str | None,
    ) -> tuple[StopReason | None, set[str], set[str], set[str]]:
        successful_tools: set[str] = set()
        nonempty_tools: set[str] = set()
        evidence_fields: set[str] = set()
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = self._bounded_arguments(
                tool_name,
                tool_call.function.arguments,
            )
            arguments = _constrain_tool_arguments(
                tool_name,
                arguments,
                requirements,
            )
            started = perf_counter()
            tool_result = await self._dispatch(
                tool_name,
                arguments,
                caller_entity_id=caller_entity_id,
            )
            duration_ms = (perf_counter() - started) * 1_000
            success = tool_result.get("ok") is True
            scoped_result = _scope_tool_result(
                tool_name,
                tool_result,
                requirements,
            )
            result_records = scoped_result.get("result")
            record_count = (
                len(result_records) if isinstance(result_records, list) else 0
            )
            error = tool_result.get("error")
            error_code = error.get("code") if isinstance(error, Mapping) else "none"
            relation = (
                arguments.get("relation")
                if isinstance(arguments, Mapping)
                else None
            )
            direction = (
                arguments.get("direction")
                if isinstance(arguments, Mapping)
                else None
            )
            failure_reason = _tool_failure_reason(tool_result, failure_reason)
            successful, nonempty, fields = _tool_evidence(
                tool_name,
                arguments if isinstance(arguments, Mapping) else {},
                scoped_result,
            )
            successful_tools.update(successful)
            nonempty_tools.update(nonempty)
            evidence_fields.update(fields)
            logger.info(
                "tool_execution request_id=%s step=%d tool=%s success=%s "
                "relation=%s direction=%s record_count=%d duration_ms=%.2f "
                "error_code=%s",
                safe_log_token(request_id),
                step,
                safe_log_token(tool_name),
                str(success).lower(),
                safe_log_token(str(relation or "none")),
                safe_log_token(str(direction or "none")),
                record_count,
                duration_ms,
                safe_log_token(str(error_code)),
            )
            conversation.append(
                {
                    "role": "tool",
                    "tool_name": tool_name,
                    "content": self._serialize_tool_result(
                        tool_name,
                        tool_result,
                        presentation_language=presentation_language,
                        allowed_private_fields=allowed_private_fields,
                        requirements=requirements,
                    ),
                }
            )
        return failure_reason, successful_tools, nonempty_tools, evidence_fields

    def _raise_step_limit(self, request_id: str, total_tool_calls: int) -> None:
        logger.info(
            "agent_stop request_id=%s reason=step_limit steps=%d tool_calls=%d",
            safe_log_token(request_id),
            self.max_steps,
            total_tool_calls,
        )
        raise AgentLimitError(
            f"Agent did not produce a final answer within {self.max_steps} steps"
        )

    def _available_tools(
        self,
        evidence_required: bool,
        graph_evidence_complete: bool,
    ) -> tuple[dict[str, Any], ...]:
        if evidence_required and graph_evidence_complete:
            return self._post_graph_tools
        return self.tools

    async def _dispatch(
        self,
        tool_name: str,
        arguments: Any,
        *,
        caller_entity_id: str | None,
    ) -> dict[str, Any]:
        try:
            return await asyncio.wait_for(
                self.dispatcher.dispatch(
                    tool_name,
                    arguments,
                    caller_entity_id=caller_entity_id,
                ),
                timeout=self.tool_timeout_seconds,
            )
        except TimeoutError:
            return {
                "ok": False,
                "tool": tool_name,
                "error": {
                    "code": "tool_timeout",
                    "message": (
                        "Tool execution exceeded the "
                        f"{self.tool_timeout_seconds:g} second limit"
                    ),
                },
            }

    def _bounded_arguments(self, tool_name: str, arguments: Any) -> Any:
        if (
            not isinstance(arguments, Mapping)
            or tool_name not in self._tools_with_limit
        ):
            return arguments

        bounded = dict(arguments)
        requested_limit = bounded.get("limit")
        if requested_limit is None:
            bounded["limit"] = self.max_tool_records
        elif (
            isinstance(requested_limit, int)
            and not isinstance(requested_limit, bool)
            and requested_limit <= self._tool_limit_maximums[tool_name]
        ):
            bounded["limit"] = min(requested_limit, self.max_tool_records)
        return bounded

    def _serialize_tool_result(
        self,
        tool_name: str,
        tool_result: dict[str, Any],
        *,
        presentation_language: str,
        allowed_private_fields: frozenset[str],
        requirements: EvidenceRequirements,
    ) -> str:
        scoped_result = _scope_tool_result(tool_name, tool_result, requirements)
        bounded = _prepare_tool_value(
            scoped_result,
            presentation_language,
            allowed_private_fields,
            tool_name=tool_name,
        )
        result = bounded.get("result")
        items, item_key = _truncatable_items(result)
        available = len(items) if items is not None else 0
        if bounded.get("ok") is True and items is not None:
            bounded = _with_truncated_items(
                bounded,
                item_key,
                items[: self.max_tool_records],
                available,
            )
            items, item_key = _truncatable_items(bounded.get("result"))

        serialized = _json(bounded)
        if _byte_length(serialized) <= self.max_tool_result_bytes:
            return serialized

        if bounded.get("ok") is True and items is not None:
            for count in range(len(items) - 1, -1, -1):
                candidate = _with_truncated_items(
                    bounded,
                    item_key,
                    items[:count],
                    available,
                )
                serialized = _json(candidate)
                if _byte_length(serialized) <= self.max_tool_result_bytes:
                    return serialized

        fallback = _json(
            {
                "ok": False,
                "tool": tool_name,
                "error": {
                    "code": "tool_result_too_large",
                    "message": (
                        "Tool result exceeded the "
                        f"{self.max_tool_result_bytes} byte limit"
                    ),
                },
            }
        )
        if _byte_length(fallback) <= self.max_tool_result_bytes:
            return fallback
        return _json({"ok": False, "error": {"code": "tool_result_too_large"}})


def _speaker_address(
    presentation_values: Sequence[Any],
    language: str,
) -> str | None:
    for value in presentation_values:
        if (
            isinstance(value, Mapping)
            and str(value.get("id", "")).startswith("person:")
        ):
            resolved = resolve_person_reference(value, language, mode="address")
            return resolved if resolved else None
    return None


def _self_vocative_prefixes(agent_name: str | None) -> tuple[str, ...]:
    if not agent_name:
        return ()
    return tuple(
        f"{agent_name}{punctuation}"
        for punctuation in ("，", ",", "：", ":")
    )


def _address_prefix(speaker_address: str | None, language: str) -> str:
    if not speaker_address:
        return ""
    punctuation = "，" if language == "zh" else ", "
    return f"{speaker_address}{punctuation}"


def _repair_self_vocative(
    text: str,
    agent_name: str | None,
    speaker_address: str | None,
    language: str,
) -> str:
    stream = _SelfVocativeStream(agent_name, speaker_address, language)
    return stream.feed(text) + stream.finish()


def _has_required_evidence(
    evidence_required: bool,
    requirements: EvidenceRequirements,
    successful_tools: set[str],
) -> bool:
    if not evidence_required:
        return True
    if not requirements.tools.issubset(successful_tools):
        return False
    return all(
        f"get_relationships:{relation}" in successful_tools
        for relation in requirements.relations
    )


def _has_nonempty_evidence(
    requirements: EvidenceRequirements,
    nonempty_tools: set[str],
    evidence_fields: set[str],
) -> bool:
    if not requirements.tools.issubset(nonempty_tools):
        return False
    if not all(
        f"{tool_name}.{field}" in evidence_fields
        for tool_name, field in requirements.fields
    ):
        return False

    related_ids: set[str] = set()
    for relation in requirements.relations:
        prefix = f"get_relationships:{relation}.related_id="
        if requirements.related_gender is not None:
            prefix = (
                f"get_relationships:{relation}.related_gender="
                f"{requirements.related_gender}.id="
            )
        related_ids.update(
            item.removeprefix(prefix)
            for item in evidence_fields
            if item.startswith(prefix)
        )
    if requirements.relations and requirements.related_gender and not related_ids:
        return False

    if ("get_entity", "dob") in requirements.fields and requirements.relations:
        entity_prefix = "get_entity.field=dob.id="
        entity_ids = {
            item.removeprefix(entity_prefix)
            for item in evidence_fields
            if item.startswith(entity_prefix)
        }
        if not related_ids.intersection(entity_ids):
            return False
    for tool_name, field in requirements.fields:
        if tool_name != "get_entity":
            continue
        prefix = f"{tool_name}.field={field}.id="
        matching_ids = {
            item.removeprefix(prefix)
            for item in evidence_fields
            if item.startswith(prefix)
        }
        if len(matching_ids) < requirements.minimum_entity_records:
            return False
    return True


def _should_retry_incomplete_evidence(
    requirements: EvidenceRequirements,
    evidence_fields: set[str],
) -> bool:
    for tool_name, field in requirements.fields:
        if tool_name != "get_entity":
            continue
        prefix = f"{tool_name}.field={field}.id="
        matching_ids = {
            item.removeprefix(prefix)
            for item in evidence_fields
            if item.startswith(prefix)
        }
        if 0 < len(matching_ids) < requirements.minimum_entity_records:
            return True

    if ("get_entity", "dob") in requirements.fields and requirements.relations:
        entity_prefix = "get_entity.field=dob.id="
        entity_ids = {
            item.removeprefix(entity_prefix)
            for item in evidence_fields
            if item.startswith(entity_prefix)
        }
        related_ids = {
            item.rpartition(".id=")[2]
            for item in evidence_fields
            if item.startswith("get_relationships:") and ".related_" in item
        }
        return bool(entity_ids and related_ids and not entity_ids.intersection(related_ids))
    return False


def _constrain_tool_arguments(
    tool_name: str,
    arguments: Any,
    requirements: EvidenceRequirements,
) -> Any:
    if tool_name != "get_relationships" or not isinstance(arguments, Mapping):
        return arguments

    constrained = dict(arguments)
    if len(requirements.relations) == 1:
        constrained["relation"] = next(iter(requirements.relations))
    if requirements.relationship_direction is not None:
        constrained["direction"] = requirements.relationship_direction
    return constrained


def _scope_tool_result(
    tool_name: str,
    tool_result: dict[str, Any],
    requirements: EvidenceRequirements,
) -> dict[str, Any]:
    if tool_name != "get_relationships" or not requirements.relations:
        return tool_result
    records = tool_result.get("result")
    if not isinstance(records, list):
        return tool_result

    scoped = dict(tool_result)
    scoped["result"] = [
        record
        for record in records
        if isinstance(record, Mapping)
        and record.get("relation") in requirements.relations
        and (
            requirements.related_gender is None
            or (
                isinstance(record.get("related_entity"), Mapping)
                and record["related_entity"].get("gender")
                == requirements.related_gender
            )
        )
    ]
    return scoped


def _tool_evidence(
    tool_name: str,
    arguments: Mapping[str, Any],
    tool_result: Mapping[str, Any],
) -> tuple[set[str], set[str], set[str]]:
    """Extract evidence markers without retaining or logging private values."""
    if tool_result.get("ok") is not True:
        return set(), set(), set()

    successful_tools = {tool_name}
    nonempty_tools: set[str] = set()
    evidence_fields: set[str] = set()
    requested_relation = arguments.get("relation")
    if isinstance(requested_relation, str):
        successful_tools.add(f"{tool_name}:{requested_relation}")

    records = tool_result.get("result")
    if not isinstance(records, list) or not records:
        return successful_tools, nonempty_tools, evidence_fields

    nonempty_tools.add(tool_name)
    for record in records:
        if not isinstance(record, Mapping):
            continue
        evidence_fields.update(f"{tool_name}.{field}" for field in record)
        relation = record.get("relation")
        if isinstance(relation, str):
            successful_tools.add(f"{tool_name}:{relation}")
            evidence_fields.add(f"{tool_name}.relation={relation}")
            related = record.get("related_entity")
            if isinstance(related, Mapping):
                related_id = related.get("id")
                if isinstance(related_id, str):
                    evidence_fields.add(
                        f"{tool_name}:{relation}.related_id={related_id}"
                    )
                    gender = related.get("gender")
                    if isinstance(gender, str):
                        evidence_fields.add(
                            f"{tool_name}:{relation}.related_gender={gender}.id="
                            f"{related_id}"
                        )
        record_id = record.get("id")
        if isinstance(record_id, str):
            evidence_fields.add(f"{tool_name}.id={record_id}")
            evidence_fields.update(
                f"{tool_name}.field={field}.id={record_id}"
                for field in record
            )
    return successful_tools, nonempty_tools, evidence_fields


def _tool_failure_reason(
    tool_result: Mapping[str, Any],
    current: StopReason | None,
) -> StopReason | None:
    if tool_result.get("ok") is True:
        return current
    error = tool_result.get("error")
    error_code = error.get("code") if isinstance(error, Mapping) else None
    if error_code == "tool_timeout":
        return "timeout"
    return current if current == "timeout" else "tool_error"


def _truncatable_items(result: Any) -> tuple[list[Any] | None, str | None]:
    if isinstance(result, list):
        return result, None
    if isinstance(result, Mapping):
        for key in ("events", "conflicts"):
            value = result.get(key)
            if isinstance(value, list):
                return value, key
    return None, None


def _with_truncated_items(
    bounded: Mapping[str, Any],
    item_key: str | None,
    items: list[Any],
    available: int,
) -> dict[str, Any]:
    updated = dict(bounded)
    if item_key is None:
        updated["result"] = items
    else:
        payload = dict(bounded.get("result") or {})
        payload[item_key] = items
        if len(items) < available:
            if "complete" in payload:
                payload["complete"] = False
            if "checked" in payload:
                payload["checked"] = False
            if "available" in payload:
                payload["available"] = False
        updated["result"] = payload
    if len(items) < available:
        updated["meta"] = {
            "truncated": True,
            "records_available": available,
            "records_returned": len(items),
        }
    return updated


def _prepare_tool_value(
    value: Any,
    language: str,
    allowed_private_fields: frozenset[str],
    *,
    tool_name: str | None = None,
) -> Any:
    private_fields = (
        PRIVATE_TOOL_FIELDS if tool_name in GRAPH_TOOL_NAMES else {}
    )
    if isinstance(value, Mapping):
        prepared = {
            str(key): _prepare_tool_value(
                item,
                language,
                allowed_private_fields,
                tool_name=tool_name,
            )
            for key, item in value.items()
            if str(key) not in MODEL_HIDDEN_FIELDS
            and (
                private_fields.get(str(key)) in allowed_private_fields
                or str(key) not in private_fields
            )
        }
        record_id = prepared.get("id")
        if "name" in prepared:
            display_name = resolve_display_name(value, language)
            if display_name and display_name != str(record_id):
                prepared["name"] = display_name
        return prepared
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [
            _prepare_tool_value(
                item,
                language,
                allowed_private_fields,
                tool_name=tool_name,
            )
            for item in value
        ]
    return value


def _grounding_retry_message(
    requirements: EvidenceRequirements,
) -> dict[str, str]:
    steps: list[str] = []
    for relation in sorted(requirements.relations):
        direction = (
            f" in semantic direction {requirements.relationship_direction}"
            if requirements.relationship_direction is not None
            else ""
        )
        steps.append(
            f"query the {relation} relationship{direction} with get_relationships"
        )
    for tool_name, field in sorted(requirements.fields):
        count = (
            f" for at least {requirements.minimum_entity_records} distinct entities"
            if tool_name == "get_entity"
            and requirements.minimum_entity_records > 1
            else " for the resolved entity"
        )
        steps.append(f"retrieve {field} with {tool_name}{count}")
    covered_tools = {
        "get_relationships" if requirements.relations else "",
        *(tool_name for tool_name, _ in requirements.fields),
    }
    for tool_name in sorted(requirements.tools - covered_tools):
        steps.append(f"call {tool_name}")
    requirement = ""
    if steps:
        requirement = (
            " Required evidence steps: "
            + "; then ".join(steps)
            + ". Follow related_entity IDs between steps."
        )
    return {
        "role": "system",
        "content": (
            "Grounding check failed: this household-fact request has no successful "
            "Home Cortex tool evidence in the current turn. Use the provided tools "
            "now. Do not answer from memory and do not repeat the unsupported answer."
            f"{requirement}"
        ),
    }


def _bounded_limit(
    name: str,
    value: int,
    hard_limit: int,
    *,
    minimum: int = 1,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not minimum <= value <= hard_limit:
        raise ValueError(f"{name} must be between {minimum} and {hard_limit}")
    return value


def _json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


def _byte_length(value: str) -> int:
    return len(value.encode("utf-8"))


