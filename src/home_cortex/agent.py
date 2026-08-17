import asyncio
import json
import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

from .ollama import OllamaService
from .tools import TOOLS, ToolDispatcher

MAX_AGENT_STEPS = 4
MAX_TOOL_CALLS_PER_STEP = 4
MAX_TOOL_RECORDS = 25
MAX_TOOL_RESULT_BYTES = 16_384
TOOL_EXECUTION_TIMEOUT_SECONDS = 5.0

logger = logging.getLogger("uvicorn.error.home_cortex.agent")
StopReason = Literal["answer", "step_limit", "tool_error", "timeout"]

SYSTEM_PROMPT = """You are the Home Cortex assistant.
Use the provided read-only tools when home-graph facts are needed.
Base factual claims about the home on tool results, and say when no matching
fact is available. Never invent entity IDs or relationships.

Answer in the language explicitly requested by the user. Otherwise, answer in
the language of the latest user message. An entity's name field is an ordered
list of multilingual aliases for the same entity, not a list of different
entities. When possible, use the stored name alias matching the answer language.
Use name, rather than assembling a display name from first_name and last_name.
Never invent or translate a name when no matching stored alias is available.

When invoking a tool, always use the native tool-calling mechanism. Never print
or describe a tool call as JSON in message content. Use the argument names from
the provided tool definition exactly; get_relationships uses entity_id.

Answer only what the user requested. Do not include sensitive personal fields,
such as dates of birth or full addresses, unless the user explicitly requests
them. Preserve dates and other factual values exactly rather than replacing
them with approximate summaries.

For relationship questions, follow the complete lookup process before answering:
1. Extract the distinctive entity name or ID from the question.
2. Call search_entities with only that name or ID, never the full question.
3. For each relevant match, call get_relationships with its record ID.
4. Read the linked record from each relationship's related_entity field.
Do not claim relationship information is unavailable after only searching for
the entity. For example, "Who resides at Fort Cerritos?" requires searching for
"Fort Cerritos" and then getting its resides_in relationships.
"""

_TOOLS_WITH_LIMIT = frozenset(
    tool["function"]["name"]
    for tool in TOOLS
    if "limit" in tool["function"]["parameters"].get("properties", {})
)
_TOOL_LIMIT_MAXIMUMS = {
    tool["function"]["name"]: tool["function"]["parameters"]["properties"][
        "limit"
    ].get("maximum")
    for tool in TOOLS
    if tool["function"]["name"] in _TOOLS_WITH_LIMIT
}


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


class AgentService:
    """Run a bounded Ollama tool-calling loop."""

    def __init__(
        self,
        ollama: OllamaService,
        dispatcher: ToolDispatcher,
        *,
        max_steps: int = MAX_AGENT_STEPS,
        max_tool_calls_per_step: int = MAX_TOOL_CALLS_PER_STEP,
        max_tool_records: int = MAX_TOOL_RECORDS,
        max_tool_result_bytes: int = MAX_TOOL_RESULT_BYTES,
        tool_timeout_seconds: float = TOOL_EXECUTION_TIMEOUT_SECONDS,
    ) -> None:
        self.ollama = ollama
        self.dispatcher = dispatcher
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

    async def answer(
        self,
        question: str,
        *,
        request_id: str = "-",
    ) -> AgentResult:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty")
        return await self.answer_messages(
            [{"role": "user", "content": question}],
            request_id=request_id,
        )

    async def answer_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        request_id: str = "-",
    ) -> AgentResult:
        """Answer a conversation while always applying the Cortex system prompt."""
        if not messages:
            raise ValueError("At least one message is required")
        return await self.run(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                *(dict(message) for message in messages),
            ],
            request_id=request_id,
        )

    async def stream_answer_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        request_id: str = "-",
    ) -> AsyncIterator[str]:
        """Yield final-answer tokens while keeping tool steps internal."""
        if not messages:
            raise ValueError("At least one message is required")
        async for token in self.stream(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                *(dict(message) for message in messages),
            ],
            request_id=request_id,
        ):
            yield token

    async def stream(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        request_id: str = "-",
    ) -> AsyncIterator[str]:
        """Run the tool loop and stream chunks from the final Ollama answer."""
        conversation = [dict(message) for message in messages]
        if not conversation:
            raise ValueError("At least one message is required")

        total_tool_calls = 0
        failure_reason: StopReason | None = None
        for step in range(1, self.max_steps + 1):
            content_parts: list[str] = []
            tool_calls: list[Any] = []
            emitted_content = False

            async for response in self.ollama.stream_chat_with_tools(conversation):
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
                        emitted_content = True
                        yield content

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
            failure_reason = await self._append_tool_results(
                conversation,
                tool_calls,
                step=step,
                request_id=request_id,
                failure_reason=failure_reason,
            )
            total_tool_calls += len(tool_calls)

        self._raise_step_limit(request_id, total_tool_calls)

    async def run(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        request_id: str = "-",
    ) -> AgentResult:
        conversation = [dict(message) for message in messages]
        if not conversation:
            raise ValueError("At least one message is required")

        total_tool_calls = 0
        failure_reason: StopReason | None = None
        for step in range(1, self.max_steps + 1):
            response = await self.ollama.chat_with_tools(conversation)
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
                stop_reason = failure_reason or "answer"
                logger.info(
                    "agent_stop request_id=%s reason=%s steps=%d tool_calls=%d",
                    _safe_log_token(request_id),
                    stop_reason,
                    step,
                    total_tool_calls,
                )
                return AgentResult(
                    answer=response.message.content or "",
                    steps=step,
                    tool_calls=total_tool_calls,
                    stop_reason=stop_reason,
                    messages=tuple(conversation),
                )

            self._check_tool_call_limits(tool_calls, step, total_tool_calls, request_id)
            failure_reason = await self._append_tool_results(
                conversation,
                tool_calls,
                step=step,
                request_id=request_id,
                failure_reason=failure_reason,
            )
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

    async def _append_tool_results(
        self,
        conversation: list[dict[str, Any]],
        tool_calls: Sequence[Any],
        *,
        step: int,
        request_id: str,
        failure_reason: StopReason | None,
    ) -> StopReason | None:
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = self._bounded_arguments(
                tool_name,
                tool_call.function.arguments,
            )
            started = perf_counter()
            tool_result = await self._dispatch(tool_name, arguments)
            duration_ms = (perf_counter() - started) * 1_000
            success = tool_result.get("ok") is True
            result_records = tool_result.get("result")
            record_count = (
                len(result_records) if isinstance(result_records, list) else 0
            )
            error = tool_result.get("error")
            error_code = error.get("code") if isinstance(error, Mapping) else "none"
            if not success:
                if error_code == "tool_timeout":
                    failure_reason = "timeout"
                elif failure_reason != "timeout":
                    failure_reason = "tool_error"
            logger.info(
                "tool_execution request_id=%s step=%d tool=%s success=%s "
                "record_count=%d duration_ms=%.2f error_code=%s",
                _safe_log_token(request_id),
                step,
                _safe_log_token(tool_name),
                str(success).lower(),
                record_count,
                duration_ms,
                _safe_log_token(str(error_code)),
            )
            conversation.append(
                {
                    "role": "tool",
                    "tool_name": tool_name,
                    "content": self._serialize_tool_result(tool_name, tool_result),
                }
            )
        return failure_reason

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

    async def _dispatch(self, tool_name: str, arguments: Any) -> dict[str, Any]:
        try:
            return await asyncio.wait_for(
                self.dispatcher.dispatch(tool_name, arguments),
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
        if not isinstance(arguments, Mapping) or tool_name not in _TOOLS_WITH_LIMIT:
            return arguments

        bounded = dict(arguments)
        requested_limit = bounded.get("limit")
        if requested_limit is None:
            bounded["limit"] = self.max_tool_records
        elif (
            isinstance(requested_limit, int)
            and not isinstance(requested_limit, bool)
            and requested_limit <= _TOOL_LIMIT_MAXIMUMS[tool_name]
        ):
            bounded["limit"] = min(requested_limit, self.max_tool_records)
        return bounded

    def _serialize_tool_result(
        self,
        tool_name: str,
        tool_result: dict[str, Any],
    ) -> str:
        bounded = dict(tool_result)
        result = bounded.get("result")
        if bounded.get("ok") is True and isinstance(result, list):
            available = len(result)
            bounded["result"] = result[: self.max_tool_records]
            if len(bounded["result"]) < available:
                bounded["meta"] = {
                    "truncated": True,
                    "records_available": available,
                    "records_returned": len(bounded["result"]),
                }

        serialized = _json(bounded)
        if _byte_length(serialized) <= self.max_tool_result_bytes:
            return serialized

        records = bounded.get("result")
        if bounded.get("ok") is True and isinstance(records, list):
            available = len(result) if isinstance(result, list) else len(records)
            for count in range(len(records) - 1, -1, -1):
                candidate = dict(bounded)
                candidate["result"] = records[:count]
                candidate["meta"] = {
                    "truncated": True,
                    "records_available": available,
                    "records_returned": count,
                }
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
