import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, cast

from ollama import AsyncClient, ChatResponse

from .operator_registry import operator_prompt_payload


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

    async def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> ChatResponse:
        """Send one ordinary chat request without exposing tools."""
        return await self.client.chat(
            model=self.model,
            messages=messages,
            stream=False,
            think=False,
        )

    async def chat_with_tools(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> ChatResponse:
        """Send one request that lets the model choose a read-only Cortex tool."""
        return await self.client.chat(
            model=self.model,
            messages=messages,
            tools=tools,
            stream=False,
            think=False,
        )

    async def plan_grounding(
        self,
        messages: Sequence[Mapping[str, Any]],
        schema_catalog: Mapping[str, Any],
        output_schema: Mapping[str, Any],
        *,
        household_now: str,
    ) -> Mapping[str, Any]:
        """Classify and plan household evidence using strict structured output."""
        planner_prompt = (
            "You are the Home Cortex evidence planner. Decide whether answering "
            "the latest request requires facts about this specific household, "
            "its people, possessions, finances, health, devices, locations, or "
            "history. General knowledge, advice, creative requests, and purely "
            "hypothetical questions do not require household grounding. Requests "
            "for calendar or calculation tools use grounding_domain=external_tool "
            "with requires_grounding=false; requests needing neither graph nor an "
            "external tool use grounding_domain=none.\n\n"
            "For grounded requests, plan only against the supplied runtime schema. "
            "Never invent a relation or claim that an unlisted property exists. "
            "A property explicitly requested by the user may still appear in "
            "required_evidence when it is absent from the catalog, so the gate can "
            "return FIELD_NOT_AVAILABLE. Represent references canonically: use "
            "reference_type=speaker for first-person references to the current "
            "caller, reference_type=assistant for second-person references to this "
            "assistant, reference_type=configured_home for the current home, "
            "reference_type=entity_id for an explicit canonical record ID, and "
            "reference_type=named_entity only for a genuine named subject. Never "
            "put a pronoun in a named_entity reference. Speaker properties use "
            "grounding_domain=household and the person schema. Assistant metadata "
            "uses grounding_domain=runtime, requires_grounding=true, field "
            "display_name, and no household traversal. An assistant reference must "
            "not be sent to the household graph. Use traversal steps for graph "
            "relationships. Put every field or relation necessary for the answer "
            "in required_evidence, but do not add evidence fields that are not used "
            "by projection, filtering, sorting, transformation, or freshness. The "
            "executor already excludes ended relationships unless include_ended is "
            "true, so do not request a nullable end field merely to establish that "
            "a relationship is current. Use source=edge and edge_fields for properties "
            "stored on a relationship, such as start or end; otherwise use "
            "source=entity. Traversal direction out follows the schema's from-to "
            "orientation and in follows it in reverse; omit direction for a "
            "symmetric relation. Set include_ended only for an explicitly historical "
            "request, and use filters over temporal fields to bound its period. Use "
            "edge filters, sorts, and transforms for the final traversal step only. "
            "Use freshness only when the wording requires a current/recent "
            "observation. "
            "Derived values must use one of the bounded operators in the output "
            "schema; never invent an operator. Operator families map to distinct IR "
            "slots: express select with fields/edge_fields, traverse with traversal, "
            "resolve_reference with subject, predicates with filters, and sort with "
            "sort. Never put those structural operations in transform; "
            "transform.operator accepts only the enum exposed for that slot. "
            "Respect each field's property_types "
            "entry: numeric aggregation requires number/integer fields, temporal "
            "operators require date/datetime fields, and incompatible operations "
            "will be rejected deterministically. Use completed_years for age and "
            "date_difference for elapsed "
            "days or seconds. Use filters and "
            "sum for bounded expense periods. Use annual_occurrence with mode=days "
            "for days until the next birthday or anniversary, or omit mode to "
            "return the next occurrence date. If the "
            "runtime schema lacks a requested property, still return a grounded "
            "plan with that canonical snake_case property in required_evidence; "
            "the deterministic gate will report it as unavailable.\n\n"
            f"Household now: {household_now}\n"
            "Allowed generic operator contracts:\n"
            + json.dumps(
                operator_prompt_payload(),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
            "Runtime schema:\n"
            + json.dumps(schema_catalog, ensure_ascii=False, separators=(",", ":"))
        )
        classifier_messages = [
            {"role": "system", "content": planner_prompt},
            *[dict(message) for message in messages],
        ]
        response = await self.client.chat(
            model=self.model,
            messages=classifier_messages,
            stream=False,
            think=False,
            format=dict(output_schema),
        )
        content = response.message.content or ""
        parsed = json.loads(content)
        if not isinstance(parsed, Mapping):
            raise ValueError("Grounding planner returned a non-object")
        return parsed

    async def stream_chat_with_tools(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> AsyncIterator[ChatResponse]:
        """Stream one response while allowing read-only Cortex tool calls."""
        response = await self.client.chat(
            model=self.model,
            messages=messages,
            tools=tools,
            stream=True,
            think=False,
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
