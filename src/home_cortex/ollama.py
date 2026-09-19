import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, cast

from ollama import AsyncClient, ChatResponse

from .request_tracing import model_call
from .mutation_ir import MutationDecision, mutation_messages, read_plan_schema, attribute_output_schema
from .semantic_prompt import (
    PLANNER_NUM_PREDICT,
    PLANNER_SEED,
    _PLANNER_HISTORY_BOUNDARY,
    _PLANNER_INSTRUCTIONS,
    _semantic_planner_examples,
    planner_chat_messages,
    planner_system_prompt,
)


# Keep the same resident runner configuration across ordinary chat and planning.
# Different context sizes cause Ollama to restart the runner between paths.
# The planner prompt is the largest input: few-shot grammar, the capability
# payload, and up to MAX_DISCOURSE_TURNS prior turns already reach ~8.6K tokens
# before the plan is generated. At 8192 Ollama truncated the prompt and stopped
# generation mid-JSON (`done_reason=length`), so every plan that needed more than
# a handful of tokens came back unterminated.
OLLAMA_KEEP_ALIVE = "24h"
OLLAMA_NUM_CTX = 16384
# Retain the planner names used by benchmark fingerprinting and probe scripts.
PLANNER_KEEP_ALIVE = OLLAMA_KEEP_ALIVE
PLANNER_NUM_CTX = OLLAMA_NUM_CTX
class OllamaService:
    """Make individual Ollama chat calls for the Cortex agent."""

    def __init__(
        self,
        base_url: str,
        model: str,
        client: AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._owns_client = client is None
        self.client = client or AsyncClient(host=self.base_url)
        self.last_planner_runtime: dict[str, Any] = {}

    @model_call("ollama")
    async def _chat(self, **kwargs):
        return await self.client.chat(**kwargs)

    async def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> ChatResponse:
        """Send one ordinary chat request without exposing tools."""
        return await self._chat(
            model=self.model,
            messages=messages,
            stream=False,
            think=False,
            keep_alive=OLLAMA_KEEP_ALIVE,
            options={"num_ctx": OLLAMA_NUM_CTX},
        )

    async def stream_chat(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> AsyncIterator[str]:
        """Stream one ordinary chat response without tools or Cortex agents."""
        response = await self._chat(
            model=self.model,
            messages=messages,
            stream=True,
            think=False,
            keep_alive=OLLAMA_KEEP_ALIVE,
            options={"num_ctx": OLLAMA_NUM_CTX},
        )
        stream = cast(AsyncIterator[ChatResponse], response)
        try:
            async for chunk in stream:
                content = getattr(getattr(chunk, "message", None), "content", None)
                if content:
                    yield content
        finally:
            close = getattr(stream, "aclose", None)
            if close is not None:
                await close()

    async def chat_with_tools(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> ChatResponse:
        """Send one request that lets the model choose a read-only Cortex tool."""
        return await self._chat(
            model=self.model,
            messages=messages,
            tools=tools,
            stream=False,
            think=False,
            keep_alive=OLLAMA_KEEP_ALIVE,
            options={"num_ctx": OLLAMA_NUM_CTX},
        )

    async def plan_item_mutation(self, messages):
        response = await self._chat(
            model=self.model, messages=mutation_messages(messages),
            stream=False, think=False, keep_alive=PLANNER_KEEP_ALIVE,
            format=attribute_output_schema(MutationDecision.model_json_schema()),
            options={"temperature": 0, "num_ctx": PLANNER_NUM_CTX,
                     "num_predict": PLANNER_NUM_PREDICT, "seed": PLANNER_SEED},
        )
        return (MutationDecision.model_validate_json(response.message.content or ""),
                _ollama_runtime_metrics(response))

    async def plan_semantic_fact(
        self,
        messages: Sequence[Mapping[str, Any]],
        capabilities: Mapping[str, Any],
        output_schema: Mapping[str, Any],
        *,
        household_now: str,
    ) -> Mapping[str, Any]:
        """Interpret using the expanded contract validated by the serving path.

        The canonical expanded semantic schema is the only serving format.
        Never infer a negative fact from a decoding failure.
        """
        response = await self._chat(
            model=self.model,
            messages=planner_chat_messages(
                messages, capabilities, household_now=household_now
            ),
            stream=False,
            think=False,
            keep_alive=PLANNER_KEEP_ALIVE,
            format=read_plan_schema(output_schema),
            options={
                "temperature": 0,
                "num_ctx": PLANNER_NUM_CTX,
                "num_predict": PLANNER_NUM_PREDICT,
                "seed": PLANNER_SEED,
            },
        )
        self.last_planner_runtime = {
            **_ollama_runtime_metrics(response),
            "done_reason": response.done_reason,
        }
        parsed = json.loads(response.message.content or "")
        if not isinstance(parsed, Mapping):
            raise ValueError("Semantic fact planner returned a non-object")
        return parsed

    async def plan_unified_semantic(
        self,
        messages: Sequence[Mapping[str, Any]],
        output_schema: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Run the unified interpreter using its already-built canonical prompt."""
        response = await self._chat(
            model=self.model,
            messages=messages,
            stream=False,
            think=False,
            keep_alive=PLANNER_KEEP_ALIVE,
            format=output_schema,
            options={
                "temperature": 0,
                "num_ctx": PLANNER_NUM_CTX,
                "num_predict": PLANNER_NUM_PREDICT,
                "seed": PLANNER_SEED,
            },
        )
        self.last_planner_runtime = {
            **_ollama_runtime_metrics(response),
            "done_reason": response.done_reason,
        }
        parsed = json.loads(response.message.content or "")
        if not isinstance(parsed, Mapping):
            raise ValueError("Unified semantic planner returned a non-object")
        return parsed

    async def stream_chat_with_tools(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> AsyncIterator[ChatResponse]:
        """Stream one response while allowing read-only Cortex tool calls."""
        response = await self._chat(
            model=self.model,
            messages=messages,
            tools=tools,
            stream=True,
            think=False,
            keep_alive=OLLAMA_KEEP_ALIVE,
            options={"num_ctx": OLLAMA_NUM_CTX},
        )
        stream = cast(AsyncIterator[ChatResponse], response)
        try:
            async for chunk in stream:
                yield chunk
        finally:
            close = getattr(stream, "aclose", None)
            if close is not None:
                await close()

    async def close(self) -> None:
        if self._owns_client:
            await self.client.close()


def _ns_to_ms(value: int | float | None) -> float:
    return 0.0 if not value else float(value) / 1_000_000


def _ollama_runtime_metrics(response: ChatResponse) -> dict[str, Any]:
    return {
        "prompt_eval_count": int(getattr(response, "prompt_eval_count", 0) or 0),
        "prompt_eval_duration_ms": _ns_to_ms(
            getattr(response, "prompt_eval_duration", None)
        ),
        "eval_count": int(getattr(response, "eval_count", 0) or 0),
        "eval_duration_ms": _ns_to_ms(getattr(response, "eval_duration", None)),
        "load_duration_ms": _ns_to_ms(getattr(response, "load_duration", None)),
    }
