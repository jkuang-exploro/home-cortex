import asyncio
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .ollama import OllamaService
from .tools import TOOLS, ToolDispatcher

MAX_AGENT_STEPS = 4
MAX_TOOL_CALLS_PER_STEP = 4
MAX_TOOL_RECORDS = 25
MAX_TOOL_RESULT_BYTES = 16_384
TOOL_EXECUTION_TIMEOUT_SECONDS = 5.0

SYSTEM_PROMPT = """You are the Home Cortex assistant.
Use the provided read-only tools when home-graph facts are needed.
Base factual claims about the home on tool results, and say when no matching
fact is available. Never invent entity IDs or relationships.
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


@dataclass(frozen=True)
class AgentResult:
    answer: str
    steps: int
    tool_calls: int
    messages: tuple[dict[str, Any], ...]


class AgentService:
    """Run a bounded, non-streaming Ollama tool-calling loop."""

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

    async def answer(self, question: str) -> AgentResult:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty")
        return await self.run(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ]
        )

    async def run(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> AgentResult:
        conversation = [dict(message) for message in messages]
        if not conversation:
            raise ValueError("At least one message is required")

        total_tool_calls = 0
        for step in range(1, self.max_steps + 1):
            response = await self.ollama.chat_with_tools(conversation)
            assistant_message = response.message.model_dump(exclude_none=True)
            conversation.append(assistant_message)
            tool_calls = list(response.message.tool_calls or [])

            if not tool_calls:
                return AgentResult(
                    answer=response.message.content or "",
                    steps=step,
                    tool_calls=total_tool_calls,
                    messages=tuple(conversation),
                )

            if len(tool_calls) > self.max_tool_calls_per_step:
                raise AgentLimitError(
                    "Ollama requested "
                    f"{len(tool_calls)} tools in one step; the limit is "
                    f"{self.max_tool_calls_per_step}"
                )
            if step == self.max_steps:
                raise AgentLimitError(
                    f"Agent did not produce a final answer within {self.max_steps} steps"
                )

            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                arguments = self._bounded_arguments(
                    tool_name,
                    tool_call.function.arguments,
                )
                tool_result = await self._dispatch(tool_name, arguments)
                conversation.append(
                    {
                        "role": "tool",
                        "tool_name": tool_name,
                        "content": self._serialize_tool_result(
                            tool_name,
                            tool_result,
                        ),
                    }
                )
                total_tool_calls += 1

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
