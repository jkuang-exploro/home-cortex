"""OpenAI-compatible OpenRouter client with the same Cortex LLM surface as Ollama."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

import httpx
from ollama import ChatResponse

from .request_tracing import model_call, observe_usage, stream_model_call
from .ollama import PLANNER_NUM_PREDICT, PLANNER_SEED, planner_chat_messages
from .mutation_ir import MutationDecision, mutation_messages, read_plan_schema, attribute_output_schema

DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1"
_CHAT_TIMEOUT = httpx.Timeout(120.0, connect=10.0)


class OpenRouterService:
    """Chat, tools, streaming, and structured planner calls via OpenRouter."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str,
        http_referer: str | None = None,
        app_title: str = "home-cortex",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("OpenRouter API key cannot be empty")
        if not model.strip():
            raise ValueError("OpenRouter model cannot be empty")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._api_key = api_key
        self._http_referer = http_referer
        self._app_title = app_title
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(timeout=_CHAT_TIMEOUT)
        self.last_planner_runtime: dict[str, Any] = {}

    async def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> ChatResponse:
        return await self._complete(messages, tools=None, stream=False)

    async def chat_with_tools(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> ChatResponse:
        return await self._complete(messages, tools=tools, stream=False)

    async def plan_item_mutation(self, messages):
        payload = await self._post({
            "model": self.model, "messages": mutation_messages(messages),
            "temperature": 0, "max_tokens": PLANNER_NUM_PREDICT, "seed": PLANNER_SEED,
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "MutationDecision", "strict": False,
                "schema": attribute_output_schema(MutationDecision.model_json_schema()),
            }},
            "provider": {"require_parameters": True},
        })
        return (MutationDecision.model_validate(_parse_json_object(_message_content(payload))),
                _openrouter_runtime_metrics(payload))

    async def plan_semantic_fact(
        self,
        messages: Sequence[Mapping[str, Any]],
        capabilities: Mapping[str, Any],
        output_schema: Mapping[str, Any],
        *,
        household_now: str,
    ) -> Mapping[str, Any]:
        # Keep serving on the expanded contract; compact transport is offline-only.
        payload = await self._post(
            {
                "model": self.model,
                "messages": planner_chat_messages(
                    messages, capabilities, household_now=household_now
                ),
                "temperature": 0,
                "max_tokens": PLANNER_NUM_PREDICT,
                "seed": PLANNER_SEED,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "SemanticPlan",
                        "strict": False,
                        "schema": read_plan_schema(output_schema),
                    },
                },
                "provider": {"require_parameters": True},
            }
        )
        self.last_planner_runtime = _openrouter_runtime_metrics(payload)
        content = _message_content(payload)
        parsed = _parse_json_object(content)
        if not isinstance(parsed, Mapping):
            raise ValueError("Semantic fact planner returned a non-object")
        return parsed

    @stream_model_call("openrouter")
    async def stream_chat_with_tools(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> AsyncIterator[ChatResponse]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": _openai_messages(messages),
            "stream": True,
            "tools": list(tools),
        }
        headers = self._headers()
        async with self.client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=body,
        ) as response:
            await _raise_for_status(response)
            tool_fragments: dict[int, dict[str, Any]] = {}
            async for event in _iter_sse_json(response):
                observe_usage(event)
                delta = _choice_delta(event)
                if delta is None:
                    continue
                content = delta.get("content") or ""
                for item in delta.get("tool_calls") or []:
                    if not isinstance(item, Mapping):
                        continue
                    index = int(item.get("index") or 0)
                    fragment = tool_fragments.setdefault(
                        index,
                        {"id": "", "function": {"name": "", "arguments": ""}},
                    )
                    if item.get("id"):
                        fragment["id"] = str(item["id"])
                    function = item.get("function")
                    if isinstance(function, Mapping):
                        if function.get("name"):
                            fragment["function"]["name"] = str(function["name"])
                        if function.get("arguments"):
                            fragment["function"]["arguments"] += str(
                                function["arguments"]
                            )
                if content:
                    yield _chat_response(self.model, content, tool_calls=None)
            if tool_fragments:
                yield _chat_response(
                    self.model,
                    "",
                    tool_calls=[
                        tool_fragments[index] for index in sorted(tool_fragments)
                    ],
                )

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    async def _complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None,
        stream: bool,
    ) -> ChatResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _openai_messages(messages),
            "stream": stream,
        }
        if tools is not None:
            payload["tools"] = list(tools)
        body = await self._post(payload)
        self.last_planner_runtime = _openrouter_runtime_metrics(body)
        message = _choice_message(body)
        return _chat_response(
            self.model,
            message.get("content") or "",
            tool_calls=message.get("tool_calls"),
        )

    @model_call("openrouter")
    async def _post(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        response = await self.client.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=dict(payload),
        )
        await _raise_for_status(response)
        body = response.json()
        if not isinstance(body, dict):
            raise ValueError("OpenRouter returned a non-object response")
        return body

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "X-Title": self._app_title,
        }
        if self._http_referer:
            headers["HTTP-Referer"] = self._http_referer
        return headers


def _openai_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Translate Cortex/Ollama conversation records to OpenAI chat messages."""
    converted: list[dict[str, Any]] = []
    pending_tool_ids: list[str] = []
    for index, message in enumerate(messages):
        role = str(message.get("role") or "user")
        if role == "tool":
            tool_call_id = str(message.get("tool_call_id") or "")
            if not tool_call_id and pending_tool_ids:
                tool_call_id = pending_tool_ids.pop(0)
            converted.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id or f"call_{index}",
                    "content": str(message.get("content") or ""),
                }
            )
            continue
        item: dict[str, Any] = {
            "role": role,
            "content": message.get("content") or "",
        }
        raw_calls = message.get("tool_calls")
        if role == "assistant" and isinstance(raw_calls, Sequence) and raw_calls:
            openai_calls = []
            pending_tool_ids = []
            for call_index, call in enumerate(raw_calls):
                if not isinstance(call, Mapping):
                    continue
                function = call.get("function")
                if not isinstance(function, Mapping):
                    continue
                arguments = function.get("arguments")
                if isinstance(arguments, Mapping):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                tool_call_id = str(call.get("id") or f"call_{call_index}")
                pending_tool_ids.append(tool_call_id)
                openai_calls.append(
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": str(function.get("name") or ""),
                            "arguments": str(arguments or "{}"),
                        },
                    }
                )
            item["tool_calls"] = openai_calls
            if not item["content"]:
                item["content"] = None
        converted.append(item)
    return converted


def _chat_response(
    model: str,
    content: str,
    *,
    tool_calls: Sequence[Mapping[str, Any]] | None,
) -> ChatResponse:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = [
            {
                "function": {
                    "name": str((call.get("function") or {}).get("name") or ""),
                    "arguments": _json_arguments(
                        (call.get("function") or {}).get("arguments")
                    ),
                }
            }
            for call in tool_calls
            if isinstance(call, Mapping)
        ]
    return ChatResponse.model_validate(
        {
            "model": model,
            "created_at": "1970-01-01T00:00:00Z",
            "done": True,
            "message": message,
        }
    )


def _json_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip():
        parsed = json.loads(value)
        if isinstance(parsed, Mapping):
            return dict(parsed)
    return {}


def _parse_json_object(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped[:4].casefold() == "json":
            stripped = stripped[4:]
        stripped = stripped.strip()
    return json.loads(stripped)


def _choice_message(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("OpenRouter response did not include a choice")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise ValueError("OpenRouter choice was not an object")
    message = first.get("message")
    if not isinstance(message, Mapping):
        raise ValueError("OpenRouter choice did not include a message")
    return message


def _message_content(payload: Mapping[str, Any]) -> str:
    content = _choice_message(payload).get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(item.get("text") or "")
            for item in content
            if isinstance(item, Mapping)
        ]
        return "".join(parts)
    return ""


def _choice_delta(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        return None
    delta = choices[0].get("delta")
    return delta if isinstance(delta, Mapping) else None


def _openrouter_runtime_metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
    usage = payload.get("usage")
    if not isinstance(usage, Mapping):
        return {}
    return {
        "prompt_eval_count": int(usage.get("prompt_tokens") or 0),
        "prompt_eval_duration_ms": 0.0,
        "eval_count": int(usage.get("completion_tokens") or 0),
        "eval_duration_ms": 0.0,
        "load_duration_ms": 0.0,
    }


async def _raise_for_status(response: httpx.Response) -> None:
    if response.is_success:
        return
    await response.aread()
    raise RuntimeError(
        f"OpenRouter request failed with HTTP {response.status_code}"
    )


async def _iter_sse_json(response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
    async for line in response.aiter_lines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            yield payload
