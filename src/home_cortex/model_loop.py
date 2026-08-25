"""Bounded Ollama tool loop for open-ended, graph-grounded conversation."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

from .display import (
    DisplayNameResolver,
    DisplayTextStream,
    resolve_display_name,
    resolve_person_reference,
)
from .memorable_dates import (
    MemorableDateRegistry,
    default_memorable_date_registry,
)
from .ollama import OllamaService
from .tools import GRAPH_TOOL_NAMES, ToolDispatcher

MAX_AGENT_STEPS = 4
MAX_TOOL_CALLS_PER_STEP = 4
MAX_TOOL_RECORDS = 25
MAX_TOOL_RESULT_BYTES = 16_384
TOOL_EXECUTION_TIMEOUT_SECONDS = 5.0
PRIVATE_TOOL_FIELDS = {
    "dob": "dob",
    "address": "address",
    "start": "relationship_dates",
    "end": "relationship_dates",
    "email": "contact",
    "phone": "contact",
    "phone_number": "contact",
}
MODEL_HIDDEN_FIELDS = frozenset(
    {"address_as", "first_name", "gender", "last_name"}
)

logger = logging.getLogger("uvicorn.error.home_cortex.agent_service")
StopReason = Literal["answer", "step_limit", "tool_error", "timeout"]


@dataclass(frozen=True)
class EvidenceRequirements:
    """Independent tool, relationship, and field evidence needed for an answer."""

    tools: frozenset[str] = frozenset()
    relations: frozenset[str] = frozenset()
    fields: frozenset[tuple[str, str]] = frozenset()
    related_gender: str | None = None
    relationship_direction: Literal["out", "in"] | None = None
    minimum_entity_records: int = 1


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
            str(language).casefold().split("-", 1)[0]: name.strip()
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
    ) -> AsyncIterator[str]:
        """Run the tool loop and stream chunks from the final Ollama answer."""
        conversation = [dict(message) for message in messages]
        if not conversation:
            raise ValueError("At least one message is required")

        total_tool_calls = 0
        successful_tools: set[str] = set()
        nonempty_tools: set[str] = set()
        evidence_fields: set[str] = set()
        failure_reason: StopReason | None = None
        grounding_retry = False
        evidence_required = _requires_graph_evidence(
            conversation,
            self.memorable_dates,
        )
        requirements = _evidence_requirements(
            conversation,
            evidence_required,
            self.memorable_dates,
        )
        allowed_private_fields = _requested_private_fields(
            conversation,
            self.memorable_dates,
        )
        (
            failure_reason,
            prefetched_successful,
            prefetched_nonempty,
            prefetched_fields,
            prefetched_tool_calls,
        ) = await self._prefetch_related_entity_facts(
            conversation,
            requirements=requirements,
            trusted_user_entity_id=trusted_user_entity_id,
            request_id=request_id,
            presentation_language=presentation_language,
            allowed_private_fields=allowed_private_fields,
        )
        successful_tools.update(prefetched_successful)
        nonempty_tools.update(prefetched_nonempty)
        evidence_fields.update(prefetched_fields)
        total_tool_calls += prefetched_tool_calls
        for step in range(1, self.max_steps + 1):
            display_stream = DisplayTextStream(
                DisplayNameResolver.from_messages(
                    conversation,
                    presentation_values,
                ),
                presentation_language,
                expose_internal_ids=expose_internal_ids,
            )
            vocative_stream = _SelfVocativeStream(
                self.localized_identity.get(presentation_language),
                _speaker_address(presentation_values, presentation_language),
                presentation_language,
            )
            content_parts: list[str] = []
            tool_calls: list[Any] = []
            emitted_content = False
            can_emit = _has_required_evidence(
                evidence_required,
                requirements,
                successful_tools,
            ) and (
                not evidence_required
                or _has_nonempty_evidence(
                    requirements,
                    nonempty_tools,
                    evidence_fields,
                )
            )
            available_tools = self._available_tools(
                evidence_required,
                can_emit,
            )

            async for response in self.ollama.stream_chat_with_tools(
                conversation,
                available_tools,
            ):
                chunk_tool_calls = list(response.message.tool_calls or [])
                if chunk_tool_calls and emitted_content:
                    logger.info(
                        "agent_stop request_id=%s reason=tool_error steps=%d "
                        "tool_calls=%d",
                        _safe_log_token(request_id),
                        step,
                        total_tool_calls,
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
            conversation.append(assistant_message)
            logger.info(
                "agent_step request_id=%s step=%d tool_calls=%d",
                _safe_log_token(request_id),
                step,
                len(tool_calls),
            )

            if not tool_calls:
                if not _has_required_evidence(
                    evidence_required,
                    requirements,
                    successful_tools,
                ):
                    if failure_reason is None and not grounding_retry and step < self.max_steps:
                        conversation.append(
                            _grounding_retry_message(
                                requirements,
                            )
                        )
                        grounding_retry = True
                        continue
                    fallback = _grounding_fallback(presentation_language)
                    yield fallback
                    stop_reason = failure_reason or "tool_error"
                    logger.info(
                        "agent_stop request_id=%s reason=%s steps=%d tool_calls=%d",
                        _safe_log_token(request_id),
                        stop_reason,
                        step,
                        total_tool_calls,
                    )
                    return
                if evidence_required and not _has_nonempty_evidence(
                    requirements,
                    nonempty_tools,
                    evidence_fields,
                ):
                    if step < self.max_steps and _should_retry_incomplete_evidence(
                        requirements,
                        evidence_fields,
                    ):
                        conversation.append(_grounding_retry_message(requirements))
                        continue
                    yield _no_records_fallback(presentation_language)
                    logger.info(
                        "agent_stop request_id=%s reason=answer steps=%d tool_calls=%d",
                        _safe_log_token(request_id),
                        step,
                        total_tool_calls,
                    )
                    return
                rendered = display_stream.finish()
                if rendered:
                    addressed = vocative_stream.feed(rendered)
                    if addressed:
                        yield addressed
                trailing = vocative_stream.finish()
                if trailing:
                    yield trailing
                stop_reason = failure_reason or "answer"
                logger.info(
                    "agent_stop request_id=%s reason=%s steps=%d tool_calls=%d",
                    _safe_log_token(request_id),
                    stop_reason,
                    step,
                    total_tool_calls,
                )
                return

            self._check_tool_call_limits(tool_calls, step, total_tool_calls, request_id)
            failure_reason, successful, nonempty, fields = await self._append_tool_results(
                conversation,
                tool_calls,
                step=step,
                request_id=request_id,
                failure_reason=failure_reason,
                presentation_language=presentation_language,
                allowed_private_fields=allowed_private_fields,
                requirements=requirements,
                caller_entity_id=trusted_user_entity_id,
            )
            successful_tools.update(successful)
            nonempty_tools.update(nonempty)
            evidence_fields.update(fields)
            total_tool_calls += len(tool_calls)

        self._raise_step_limit(request_id, total_tool_calls)

    async def run(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        request_id: str = "-",
        presentation_language: str = "en",
        expose_internal_ids: bool = False,
        presentation_values: Sequence[Any] = (),
        trusted_user_entity_id: str | None = None,
    ) -> AgentResult:
        conversation = [dict(message) for message in messages]
        if not conversation:
            raise ValueError("At least one message is required")

        total_tool_calls = 0
        successful_tools: set[str] = set()
        nonempty_tools: set[str] = set()
        evidence_fields: set[str] = set()
        failure_reason: StopReason | None = None
        grounding_retry = False
        evidence_required = _requires_graph_evidence(
            conversation,
            self.memorable_dates,
        )
        requirements = _evidence_requirements(
            conversation,
            evidence_required,
            self.memorable_dates,
        )
        allowed_private_fields = _requested_private_fields(
            conversation,
            self.memorable_dates,
        )
        (
            failure_reason,
            prefetched_successful,
            prefetched_nonempty,
            prefetched_fields,
            prefetched_tool_calls,
        ) = await self._prefetch_related_entity_facts(
            conversation,
            requirements=requirements,
            trusted_user_entity_id=trusted_user_entity_id,
            request_id=request_id,
            presentation_language=presentation_language,
            allowed_private_fields=allowed_private_fields,
        )
        successful_tools.update(prefetched_successful)
        nonempty_tools.update(prefetched_nonempty)
        evidence_fields.update(prefetched_fields)
        total_tool_calls += prefetched_tool_calls
        for step in range(1, self.max_steps + 1):
            evidence_complete = _has_required_evidence(
                evidence_required,
                requirements,
                successful_tools,
            ) and (
                not evidence_required
                or _has_nonempty_evidence(
                    requirements,
                    nonempty_tools,
                    evidence_fields,
                )
            )
            available_tools = self._available_tools(
                evidence_required,
                evidence_complete,
            )
            response = await self.ollama.chat_with_tools(
                conversation,
                available_tools,
            )
            assistant_message = response.message.model_dump(exclude_none=True)
            conversation.append(assistant_message)
            tool_calls = list(response.message.tool_calls or [])
            logger.info(
                "agent_step request_id=%s step=%d tool_calls=%d",
                _safe_log_token(request_id),
                step,
                len(tool_calls),
            )

            if not tool_calls:
                if not _has_required_evidence(
                    evidence_required,
                    requirements,
                    successful_tools,
                ):
                    if failure_reason is None and not grounding_retry and step < self.max_steps:
                        conversation.append(
                            _grounding_retry_message(
                                requirements,
                            )
                        )
                        grounding_retry = True
                        continue
                    answer = _grounding_fallback(presentation_language)
                    stop_reason = failure_reason or "tool_error"
                    logger.info(
                        "agent_stop request_id=%s reason=%s steps=%d tool_calls=%d",
                        _safe_log_token(request_id),
                        stop_reason,
                        step,
                        total_tool_calls,
                    )
                    return AgentResult(
                        answer=answer,
                        steps=step,
                        tool_calls=total_tool_calls,
                        stop_reason=stop_reason,
                        messages=tuple(conversation),
                    )
                if evidence_required and not _has_nonempty_evidence(
                    requirements,
                    nonempty_tools,
                    evidence_fields,
                ):
                    if step < self.max_steps and _should_retry_incomplete_evidence(
                        requirements,
                        evidence_fields,
                    ):
                        conversation.append(_grounding_retry_message(requirements))
                        continue
                    answer = _no_records_fallback(presentation_language)
                    logger.info(
                        "agent_stop request_id=%s reason=answer steps=%d tool_calls=%d",
                        _safe_log_token(request_id),
                        step,
                        total_tool_calls,
                    )
                    return AgentResult(
                        answer=answer,
                        steps=step,
                        tool_calls=total_tool_calls,
                        stop_reason="answer",
                        messages=tuple(conversation),
                    )
                resolver = DisplayNameResolver.from_messages(
                    conversation,
                    presentation_values,
                )
                answer = resolver.render(
                    response.message.content or "",
                    presentation_language,
                    expose_internal_ids=expose_internal_ids,
                )
                answer = _repair_self_vocative(
                    answer,
                    self.localized_identity.get(presentation_language),
                    _speaker_address(
                        presentation_values,
                        presentation_language,
                    ),
                    presentation_language,
                )
                stop_reason = failure_reason or "answer"
                logger.info(
                    "agent_stop request_id=%s reason=%s steps=%d tool_calls=%d",
                    _safe_log_token(request_id),
                    stop_reason,
                    step,
                    total_tool_calls,
                )
                return AgentResult(
                    answer=answer,
                    steps=step,
                    tool_calls=total_tool_calls,
                    stop_reason=stop_reason,
                    messages=tuple(conversation),
                )

            self._check_tool_call_limits(tool_calls, step, total_tool_calls, request_id)
            failure_reason, successful, nonempty, fields = await self._append_tool_results(
                conversation,
                tool_calls,
                step=step,
                request_id=request_id,
                failure_reason=failure_reason,
                presentation_language=presentation_language,
                allowed_private_fields=allowed_private_fields,
                requirements=requirements,
                caller_entity_id=trusted_user_entity_id,
            )
            successful_tools.update(successful)
            nonempty_tools.update(nonempty)
            evidence_fields.update(fields)
            total_tool_calls += len(tool_calls)

        self._raise_step_limit(request_id, total_tool_calls)

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
                _safe_log_token(request_id),
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
                _safe_log_token(request_id),
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
            _safe_log_token(request_id),
            _safe_log_token(tool_name),
            str(tool_result.get("ok") is True).lower(),
            _safe_log_token(str(arguments.get("relation") or "none")),
            _safe_log_token(str(arguments.get("direction") or "none")),
            record_count,
            duration_ms,
            _safe_log_token(str(error_code)),
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
                _safe_log_token(request_id),
                step,
                _safe_log_token(tool_name),
                str(success).lower(),
                _safe_log_token(str(relation or "none")),
                _safe_log_token(str(direction or "none")),
                record_count,
                duration_ms,
                _safe_log_token(str(error_code)),
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
            _safe_log_token(request_id),
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


def _is_household_roster_request(
    messages: Sequence[Mapping[str, Any]],
    memorable_dates: MemorableDateRegistry,
) -> bool:
    """Recognize resident-roster requests for generic evidence enforcement."""
    latest_user = next(
        (
            str(message.get("content", "")).casefold()
            for message in reversed(messages)
            if message.get("role") == "user"
        ),
        "",
    )
    if _contains_any(
        latest_user,
        memorable_dates.aliases
        + (
            "parent",
            "child",
            "daughter",
            "son",
            "spouse",
            "wife",
            "husband",
            "父母",
            "父亲",
            "母亲",
            "孩子",
            "儿子",
            "女儿",
            "配偶",
            "妻子",
            "丈夫",
        ),
    ):
        return False

    chinese_home = r"(?:家里|家中|家里面|家里边|这个家|这里)"
    chinese_people = r"(?:谁|哪些人|什么人|成员|住户)"
    if re.search(
        rf"(?:{chinese_home}.*{chinese_people}|"
        rf"{chinese_people}.*(?:住|居住|待在).*{chinese_home})",
        latest_user,
    ):
        return True
    if re.search(
        rf"{chinese_home}.*(?:多少|几)(?:个|位)?(?:人|住户|居民)",
        latest_user,
    ):
        return True

    patterns = (
        r"\bwho\b.*\b(?:live|lives|living|reside|resides|stays?)\b.*"
        r"\b(?:home|house|household|here)\b",
        r"\b(?:household|home|house)\s+(?:members|residents|occupants)\b",
        r"\bwho\b.*\b(?:in|at)\b.*\b(?:my|our|the)\s+household\b",
        r"\bhow many\b.*\b(?:people|residents|occupants)\b.*"
        r"\b(?:my|our|the)\s+(?:home|house|household)\b",
    )
    return any(re.search(pattern, latest_user) for pattern in patterns)


def _is_household_space_request(
    messages: Sequence[Mapping[str, Any]],
) -> bool:
    """Recognize home room/space lookups for generic evidence enforcement."""
    latest_user = next(
        (
            str(message.get("content", "")).casefold()
            for message in reversed(messages)
            if message.get("role") == "user"
        ),
        "",
    )
    chinese = re.search(
        r"(?:家里|家中|家里面|家里边|这个家).*(?:房间|空间|区域)|"
        r"(?:房间|空间|区域).*(?:家里|家中|这个家)",
        latest_user,
    )
    english = re.search(
        r"\b(?:room|rooms|space|spaces)\b.*\b(?:my|our|the)\s+"
        r"(?:home|house)\b|"
        r"\b(?:my|our|the)\s+(?:home|house)\b.*"
        r"\b(?:room|rooms|space|spaces)\b",
        latest_user,
    )
    return chinese is not None or english is not None


def _is_item_location_request(
    messages: Sequence[Mapping[str, Any]],
) -> bool:
    """Recognize named-entity location lookups for evidence enforcement."""
    latest_user = next(
        (
            str(message.get("content", "")).casefold().strip()
            for message in reversed(messages)
            if message.get("role") == "user"
        ),
        "",
    )
    if re.fullmatch(
        r"(?:where is|where's)\s+(?:(?:my|our|the)\s+)?"
        r"(?:home|house|household|here)[?]?",
        latest_user,
    ) or re.fullmatch(
        r"(?:我家|家里|家中|这里|这儿)(?:在|位于)?"
        r"(?:哪里|哪儿|什么地方)[?？。！!]?",
        latest_user,
    ):
        return False
    chinese = re.fullmatch(
        r"(?:请问|麻烦告诉我)?\s*.+?\s*(?:现在)?(?:在|位于)"
        r"(?:哪里|哪儿|什么地方|哪个房间|哪间房间?)[?？。！!]?",
        latest_user,
    )
    english = re.fullmatch(
        r"(?:(?:where is|where's)\s+.+?|"
        r"(?:which|what) room is\s+.+?\s+in)[?]?",
        latest_user,
    )
    return chinese is not None or english is not None


def _requires_graph_evidence(
    messages: Sequence[Mapping[str, Any]],
    memorable_dates: MemorableDateRegistry,
) -> bool:
    latest_user = next(
        (
            str(message.get("content", ""))
            for message in reversed(messages)
            if message.get("role") == "user"
        ),
        "",
    )
    normalized = latest_user.casefold()
    if _required_evidence_tool(messages, memorable_dates) is None:
        return False

    # Relationship words can appear in ordinary conversation without asking
    # for a stored household fact. Require graph evidence only for lookup intent.
    non_lookup_intent = (
        r"\b(?:advice|chat|feel|feeling|gift|joke|opinion|recommend|story|"
        r"suggest|talk|think|add|decorate|design|remodel|should)\b|"
        r"建议|推荐|礼物|聊|笑话|故事|觉得|认为|心情|装修|设计|改造|增加"
    )
    if re.search(non_lookup_intent, normalized):
        return False

    lookup_intent = (
        r"\b(?:find|identify|list|search|show|tell me|what|when|where|which|"
        r"who|whose|how many|how old)\b|谁|什么|哪|何时|什么时候|多少|几岁|"
        r"几个|几间|是否|查|找|告诉我|列出|显示"
    )
    if re.search(lookup_intent, normalized):
        return True

    # Direct yes/no requests for stored relationship predicates also need
    # evidence, while plain statements mentioning those predicates do not.
    yes_no_predicate = (
        r"(?:\b(?:is|are|was|were|do|does|did)\b.*\b(?:live|lives|living|"
        r"reside|resides|married)\b)|(?:住|居住|结婚|已婚).*吗[？?]?$|"
        r"是否.*(?:住|居住|结婚|已婚)"
    )
    return re.search(yes_no_predicate, normalized.strip()) is not None


def _required_evidence_tool(
    messages: Sequence[Mapping[str, Any]],
    memorable_dates: MemorableDateRegistry,
) -> str | None:
    if _is_household_roster_request(messages, memorable_dates):
        return "get_relationships"
    if _is_household_space_request(messages):
        return "get_relationships"
    if _is_item_location_request(messages):
        return "get_relationships"
    latest_user = next(
        (
            str(message.get("content", "")).casefold()
            for message in reversed(messages)
            if message.get("role") == "user"
        ),
        "",
    )
    date_schema = memorable_dates.match(latest_user)
    if date_schema is not None:
        return (
            "get_entity"
            if date_schema.source_kind == "node"
            else "get_relationships"
        )
    relationship_fields = (
        "live",
        "lives",
        "living",
        "reside",
        "resides",
        "spouse",
        "wife",
        "husband",
        "married",
        "parent",
        "child",
        "daughter",
        "son",
        "household",
        "住",
        "家里有谁",
        "家中有谁",
        "配偶",
        "妻子",
        "丈夫",
        "老婆",
        "老公",
        "父母",
        "孩子",
        "女儿",
        "儿子",
        "结婚",
    )
    if _contains_any(latest_user, relationship_fields):
        return "get_relationships"
    return None


def _required_evidence_relation(
    messages: Sequence[Mapping[str, Any]],
    memorable_dates: MemorableDateRegistry,
) -> str | None:
    if _is_household_roster_request(messages, memorable_dates):
        return "lives_in"
    if _is_household_space_request(messages):
        return "hosts_space"
    if _is_item_location_request(messages):
        return "located_in"
    latest_user = next(
        (
            str(message.get("content", "")).casefold()
            for message in reversed(messages)
            if message.get("role") == "user"
        ),
        "",
    )
    date_schema = memorable_dates.match(latest_user)
    if date_schema is not None and date_schema.source_kind == "edge":
        return date_schema.source_type
    parent_terms = (
        "parent",
        "child",
        "daughter",
        "son",
        "父母",
        "父亲",
        "母亲",
        "爸爸",
        "妈妈",
        "孩子",
        "女儿",
        "儿子",
    )
    spouse_terms = (
        "spouse",
        "wife",
        "husband",
        "married",
        "anniversary",
        "配偶",
        "妻子",
        "丈夫",
        "太太",
        "老婆",
        "老公",
        "结婚",
        "纪念日",
    )
    residence_terms = (
        "live",
        "lives",
        "living",
        "reside",
        "resides",
        "household",
        "住",
        "家里有谁",
        "家中有谁",
    )
    if _contains_any(latest_user, parent_terms):
        return "parent_of"
    if _contains_any(latest_user, spouse_terms):
        return "spouse_of"
    if _contains_any(latest_user, residence_terms):
        return "lives_in"
    return None


def _evidence_requirements(
    messages: Sequence[Mapping[str, Any]],
    evidence_required: bool,
    memorable_dates: MemorableDateRegistry,
) -> EvidenceRequirements:
    if not evidence_required:
        return EvidenceRequirements()

    primary_tool = _required_evidence_tool(messages, memorable_dates)
    relation = _required_evidence_relation(messages, memorable_dates)
    field = _required_evidence_field(messages, memorable_dates)
    tools: set[str] = set()
    relations: set[str] = set()
    fields: set[tuple[str, str]] = set()

    if relation is not None:
        tools.add("get_relationships")
        relations.add(relation)
    if field is not None and primary_tool is not None:
        tools.add(primary_tool)
        fields.add((primary_tool, field))
    if not tools and primary_tool is not None:
        tools.add(primary_tool)

    return EvidenceRequirements(
        tools=frozenset(tools),
        relations=frozenset(relations),
        fields=frozenset(fields),
        related_gender=_required_related_gender(messages),
        relationship_direction=_required_relationship_direction(messages),
        minimum_entity_records=_required_entity_record_count(messages),
    )


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


def _required_related_gender(
    messages: Sequence[Mapping[str, Any]],
) -> str | None:
    latest_user = next(
        (
            str(message.get("content", "")).casefold()
            for message in reversed(messages)
            if message.get("role") == "user"
        ),
        "",
    )
    if _contains_any(
        latest_user,
        (
            "daughter",
            "mother",
            "wife",
            "女儿",
            "母亲",
            "妈妈",
            "妻子",
            "太太",
            "老婆",
        ),
    ):
        return "female"
    if _contains_any(
        latest_user,
        (
            "son",
            "father",
            "husband",
            "儿子",
            "父亲",
            "爸爸",
            "丈夫",
            "老公",
        ),
    ):
        return "male"
    return None


def _required_relationship_direction(
    messages: Sequence[Mapping[str, Any]],
) -> Literal["out", "in"] | None:
    latest_user = next(
        (
            str(message.get("content", "")).casefold()
            for message in reversed(messages)
            if message.get("role") == "user"
        ),
        "",
    )
    if _contains_any(latest_user, ("child", "daughter", "son", "孩子", "女儿", "儿子")):
        return "out"
    if _contains_any(
        latest_user,
        ("parent", "father", "mother", "父母", "父亲", "母亲", "爸爸", "妈妈"),
    ):
        return "in"
    return None


def _required_entity_record_count(
    messages: Sequence[Mapping[str, Any]],
) -> int:
    latest_user = next(
        (
            str(message.get("content", "")).casefold()
            for message in reversed(messages)
            if message.get("role") == "user"
        ),
        "",
    )
    if _contains_any(
        latest_user,
        ("both", "children", "their", "them", "they", "他们", "她们", "孩子们"),
    ):
        return 2
    return 1


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


def _required_evidence_field(
    messages: Sequence[Mapping[str, Any]],
    memorable_dates: MemorableDateRegistry,
) -> str | None:
    latest_user = next(
        (
            str(message.get("content", "")).casefold()
            for message in reversed(messages)
            if message.get("role") == "user"
        ),
        "",
    )
    schema = memorable_dates.match(latest_user)
    return schema.source_field if schema is not None else None


def _requested_private_fields(
    messages: Sequence[Mapping[str, Any]],
    memorable_dates: MemorableDateRegistry,
) -> frozenset[str]:
    latest_user = next(
        (
            str(message.get("content", "")).casefold()
            for message in reversed(messages)
            if message.get("role") == "user"
        ),
        "",
    )
    allowed: set[str] = set()
    schema = memorable_dates.match(latest_user)
    if schema is not None:
        private_field = PRIVATE_TOOL_FIELDS.get(schema.source_field)
        if private_field is not None:
            allowed.add(private_field)
    if _contains_any(
        latest_user,
        ("address", "street address", "地址", "住址"),
    ):
        allowed.add("address")
    if _contains_any(
        latest_user,
        ("move-in date", "when did", "什么时候搬"),
    ):
        allowed.add("relationship_dates")
    if _contains_any(
        latest_user,
        ("email", "phone", "telephone", "邮箱", "电话"),
    ):
        allowed.add("contact")
    return frozenset(allowed)


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


def _contains_any(text: str, terms: Sequence[str]) -> bool:
    for term in terms:
        if term.isascii():
            if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text):
                return True
        elif term in text:
            return True
    return False


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


def _grounding_fallback(language: str) -> str:
    if language == "zh":
        return "老管家目前无法从家庭资料中核实这项信息。"
    return "I could not verify that information from the home graph."


def _no_records_fallback(language: str) -> str:
    if language == "zh":
        return "家庭资料中没有找到与这个问题匹配的信息。"
    return "The home graph does not contain matching information for that request."


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


def _safe_log_token(value: str, maximum_length: int = 128) -> str:
    """Keep model- or client-supplied identifiers on one safe log line."""
    sanitized = "".join(
        character
        if character.isascii() and (character.isalnum() or character in "._-")
        else "_"
        for character in value
    )
    return sanitized[:maximum_length] or "-"
