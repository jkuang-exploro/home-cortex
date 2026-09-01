"""Bounded Ollama tool loop for requests that do not need household facts.

Household graph reads are intentionally absent from this loop. They are planned,
executed, and evidence-gated by :mod:`home_cortex.grounding`; this loop handles
ordinary conversation and non-graph tools such as calendar and calculation.
"""

from __future__ import annotations

import asyncio
import json
import logging
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
from .ollama import OllamaService
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
    total_tool_calls: int = 0
    failure_reason: StopReason | None = None


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
                return _address_prefix(self.speaker_address, self.language) + remainder
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
    """Run a bounded tool-calling loop without direct household-graph access."""

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
    ) -> None:
        self.ollama = ollama
        self.dispatcher = dispatcher
        self.system_prompt = system_prompt.strip()
        if not self.system_prompt:
            raise ValueError("system_prompt cannot be empty")
        supplied_tools = tuple(dict(tool) for tool in tools)
        if not supplied_tools:
            raise ValueError("At least one tool definition is required")
        self.tools = tuple(
            tool
            for tool in supplied_tools
            if tool["function"]["name"] not in GRAPH_TOOL_NAMES
        )
        self.localized_identity = {
            normalize_language_code(str(language)): name.strip()
            for language, name in (localized_identity or {}).items()
            if isinstance(name, str) and name.strip()
        }
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
            "max_tool_calls_per_step", max_tool_calls_per_step, MAX_TOOL_CALLS_PER_STEP
        )
        self.max_tool_records = _bounded_limit(
            "max_tool_records", max_tool_records, MAX_TOOL_RECORDS
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
        state = self._begin_loop(
            messages,
            request_id=request_id,
            presentation_language=presentation_language,
            expose_internal_ids=expose_internal_ids,
            presentation_values=presentation_values,
            trusted_user_entity_id=trusted_user_entity_id,
        )
        for step in range(1, self.max_steps + 1):
            display_stream = DisplayTextStream(
                DisplayNameResolver.from_messages(
                    state.conversation, state.presentation_values
                ),
                state.presentation_language,
                expose_internal_ids=state.expose_internal_ids,
            )
            vocative_stream = _SelfVocativeStream(
                self.localized_identity.get(state.presentation_language),
                _speaker_address(state.presentation_values, state.presentation_language),
                state.presentation_language,
            )
            content_parts: list[str] = []
            tool_calls: list[Any] = []
            emitted_content = False
            async for response in self.ollama.stream_chat_with_tools(
                state.conversation, self.tools
            ):
                chunk_tool_calls = list(response.message.tool_calls or [])
                if chunk_tool_calls and emitted_content:
                    self._log_stop(state, "tool_error", step)
                    raise AgentStreamingError(
                        "Ollama emitted tool calls after final-answer content"
                    )
                tool_calls.extend(chunk_tool_calls)
                content = response.message.content or ""
                if content:
                    content_parts.append(content)
                    if not tool_calls:
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
                    tool_call.model_dump(exclude_none=True) for tool_call in tool_calls
                ]
            state.conversation.append(assistant_message)
            self._log_step(state, step, len(tool_calls))
            if not tool_calls:
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
            self._check_tool_call_limits(tool_calls, step, state)
            await self._append_tool_results(state, tool_calls, step)
        self._raise_step_limit(state)

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
        state = self._begin_loop(
            messages,
            request_id=request_id,
            presentation_language=presentation_language,
            expose_internal_ids=expose_internal_ids,
            presentation_values=presentation_values,
            trusted_user_entity_id=trusted_user_entity_id,
        )
        for step in range(1, self.max_steps + 1):
            response = await self.ollama.chat_with_tools(state.conversation, self.tools)
            state.conversation.append(response.message.model_dump(exclude_none=True))
            tool_calls = list(response.message.tool_calls or [])
            self._log_step(state, step, len(tool_calls))
            if not tool_calls:
                resolver = DisplayNameResolver.from_messages(
                    state.conversation, state.presentation_values
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
                        state.presentation_values, state.presentation_language
                    ),
                    state.presentation_language,
                )
                stop_reason = state.failure_reason or "answer"
                self._log_stop(state, stop_reason, step)
                return AgentResult(
                    answer,
                    step,
                    state.total_tool_calls,
                    stop_reason,
                    tuple(state.conversation),
                )
            self._check_tool_call_limits(tool_calls, step, state)
            await self._append_tool_results(state, tool_calls, step)
        self._raise_step_limit(state)

    def _begin_loop(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        request_id: str,
        presentation_language: str,
        expose_internal_ids: bool,
        presentation_values: Sequence[Any],
        trusted_user_entity_id: str | None,
    ) -> _LoopState:
        conversation = [dict(message) for message in messages]
        if not conversation:
            raise ValueError("At least one message is required")
        return _LoopState(
            conversation,
            request_id,
            presentation_language,
            expose_internal_ids,
            presentation_values,
            trusted_user_entity_id,
        )

    async def _append_tool_results(
        self, state: _LoopState, tool_calls: Sequence[Any], step: int
    ) -> None:
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = self._bounded_arguments(tool_name, tool_call.function.arguments)
            started = perf_counter()
            tool_result = await self._dispatch(
                tool_name,
                arguments,
                caller_entity_id=state.trusted_user_entity_id,
            )
            duration_ms = (perf_counter() - started) * 1_000
            state.failure_reason = _tool_failure_reason(
                tool_result, state.failure_reason
            )
            result_records = tool_result.get("result")
            record_count = (
                len(result_records) if isinstance(result_records, list) else 0
            )
            error = tool_result.get("error")
            error_code = error.get("code") if isinstance(error, Mapping) else "none"
            logger.info(
                "tool_execution request_id=%s step=%d tool=%s success=%s "
                "record_count=%d duration_ms=%.2f error_code=%s",
                safe_log_token(state.request_id),
                step,
                safe_log_token(tool_name),
                str(tool_result.get("ok") is True).lower(),
                record_count,
                duration_ms,
                safe_log_token(str(error_code)),
            )
            state.conversation.append(
                {
                    "role": "tool",
                    "tool_name": tool_name,
                    "content": self._serialize_tool_result(
                        tool_name,
                        tool_result,
                        presentation_language=state.presentation_language,
                    ),
                }
            )
        state.total_tool_calls += len(tool_calls)

    async def _dispatch(
        self,
        tool_name: str,
        arguments: Any,
        *,
        caller_entity_id: str | None,
    ) -> dict[str, Any]:
        if tool_name in GRAPH_TOOL_NAMES:
            return {
                "ok": False,
                "tool": tool_name,
                "error": {
                    "code": "tool_not_available",
                    "message": "Household graph tools require a grounding plan",
                },
            }
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
    ) -> str:
        bounded = _prepare_tool_value(tool_result, presentation_language)
        result = bounded.get("result")
        items, item_key = _truncatable_items(result)
        available = len(items) if items is not None else 0
        if bounded.get("ok") is True and items is not None:
            bounded = _with_truncated_items(
                bounded, item_key, items[: self.max_tool_records], available
            )
            items, item_key = _truncatable_items(bounded.get("result"))
        serialized = _json(bounded)
        if _byte_length(serialized) <= self.max_tool_result_bytes:
            return serialized
        if bounded.get("ok") is True and items is not None:
            for count in range(len(items) - 1, -1, -1):
                candidate = _with_truncated_items(
                    bounded, item_key, items[:count], available
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

    def _check_tool_call_limits(
        self, tool_calls: Sequence[Any], step: int, state: _LoopState
    ) -> None:
        if len(tool_calls) > self.max_tool_calls_per_step:
            self._log_stop(state, "step_limit", step)
            raise AgentLimitError(
                "Ollama requested "
                f"{len(tool_calls)} tools in one step; the limit is "
                f"{self.max_tool_calls_per_step}"
            )
        if step == self.max_steps:
            self._raise_step_limit(state)

    def _raise_step_limit(self, state: _LoopState) -> None:
        self._log_stop(state, "step_limit", self.max_steps)
        raise AgentLimitError(
            f"Agent did not produce a final answer within {self.max_steps} steps"
        )

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


def _speaker_address(
    presentation_values: Sequence[Any], language: str
) -> str | None:
    for value in presentation_values:
        if isinstance(value, Mapping) and str(value.get("id", "")).startswith(
            "person:"
        ):
            resolved = resolve_person_reference(value, language, mode="address")
            return resolved if resolved else None
    return None


def _self_vocative_prefixes(agent_name: str | None) -> tuple[str, ...]:
    if not agent_name:
        return ()
    return tuple(
        f"{agent_name}{punctuation}" for punctuation in ("，", ",", "：", ":")
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


def _tool_failure_reason(
    tool_result: Mapping[str, Any], current: StopReason | None
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
            for key in ("complete", "checked", "available"):
                if key in payload:
                    payload[key] = False
        updated["result"] = payload
    if len(items) < available:
        updated["meta"] = {
            "truncated": True,
            "records_available": available,
            "records_returned": len(items),
        }
    return updated


def _prepare_tool_value(value: Any, language: str) -> Any:
    if isinstance(value, Mapping):
        prepared = {
            str(key): _prepare_tool_value(item, language)
            for key, item in value.items()
            if str(key) not in MODEL_HIDDEN_FIELDS
        }
        record_id = prepared.get("id")
        if "name" in prepared:
            display_name = resolve_display_name(value, language)
            if display_name and display_name != str(record_id):
                prepared["name"] = display_name
        return prepared
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_prepare_tool_value(item, language) for item in value]
    return value


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
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _byte_length(value: str) -> int:
    return len(value.encode("utf-8"))
