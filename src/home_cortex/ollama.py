import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, cast

from ollama import AsyncClient, ChatResponse

def _semantic_planner_examples() -> list[dict[str, str]]:
    examples: tuple[tuple[str, dict[str, Any]], ...] = (
        (
            "家里有几个人",
            {
                "operation": "count",
                "subject": {
                    "kind": "current_household",
                    "entity_type": "address",
                    "path": [{"relation": "member"}],
                },
            },
        ),
        (
            "谁最年长",
            {
                "operation": "argmin",
                "subject": {
                    "kind": "current_household",
                    "entity_type": "address",
                    "path": [{"relation": "member"}],
                },
                "property": "birth_date",
            },
        ),
        (
            "谁最年幼",
            {
                "operation": "argmax",
                "subject": {
                    "kind": "current_household",
                    "entity_type": "address",
                    "path": [{"relation": "member"}],
                },
                "property": "birth_date",
            },
        ),
        (
            "有几个成年人",
            {
                "operation": "count",
                "subject": {
                    "kind": "current_household",
                    "entity_type": "address",
                    "path": [{"relation": "member"}],
                },
                "filters": [{"predicate": "adult"}],
            },
        ),
        (
            "我老婆是谁",
            {
                "operation": "resolve_reference",
                "subject": {
                    "kind": "self",
                    "entity_type": "person",
                    "path": [
                        {
                            "relation": "spouse",
                            "filters": [{"property": "gender", "value": "female"}],
                        }
                    ],
                },
            },
        ),
        (
            "我家住哪里",
            {
                "operation": "select",
                "subject": {
                    "kind": "self",
                    "entity_type": "person",
                    "path": [{"relation": "residence"}],
                },
                "property": "full_address",
            },
        ),
        (
            "德伦再过多久过生日",
            {
                "operation": "annual_occurrence",
                "subject": {
                    "kind": "named_entity",
                    "value": "德伦",
                    "entity_type": "person",
                },
                "property": "birth_date",
                "mode": "days",
            },
        ),
    )
    messages: list[dict[str, str]] = []
    for utterance, request in examples:
        messages.extend(
            (
                {"role": "user", "content": utterance},
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {"requires_fact": True, "request": request},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            )
        )
    return messages


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

    async def plan_semantic_fact(
        self,
        messages: Sequence[Mapping[str, Any]],
        capabilities: Mapping[str, Any],
        output_schema: Mapping[str, Any],
        *,
        household_now: str,
    ) -> Mapping[str, Any]:
        """Interpret an open-ended request without exposing physical storage."""
        prompt = (
            "You are Home Cortex's semantic interpreter. Map the latest user "
            "request to exactly one supported semantic request. Household member "
            "identity, the authenticated speaker's identity, and this assistant's "
            "identity are facts in this domain. Only ordinary conversation emits "
            "requires_fact=false and request=null. Return only "
            "the strict structured output; never calculate or phrase the answer.\n\n"
            "Identity and name questions are always facts here: compile 'who am I' "
            "as resolve_reference(self), 'who are you' as "
            "resolve_reference(assistant), and 'who is my son/spouse' as "
            "resolve_reference(self followed by the declared relation path). In "
            "every such case requires_fact=true and property=null. Questions for "
            "the assistant's name are also resolve_reference(assistant), never "
            "ordinary conversation.\n\n"
            "References: self is always the authenticated speaker, assistant is "
            "this assistant, current_household is the configured home, and "
            "named_entity contains only a literal stored name or appellation from "
            "the user. "
            "Never emit entity_id or put pronouns/relationship phrases in a "
            "named_entity. For speaker-relative phrases choose kind=self. Expand each "
            "declared reference concept by copying its complete ontology path and "
            "filters; compose multi-hop relationships by appending paths.\n\n"
            "Canonical disambiguation: household member list/count uses "
            "current_household->member with no filters. Unqualified household "
            "adults/minors use exactly one adult/minor request filter. Unqualified "
            "adult/minor predicate filters contain only predicate=adult/minor; do "
            "not attach an age value or property. "
            "'有几个孩子' means household minors; '我有几个孩子' means self->child. "
            "Chinese 最年长/最老/最早出生 = argmin(birth_date); "
            "最年幼/最年轻/年纪最小/最晚出生 = argmax(birth_date). "
            "老婆/妻子 means self->spouse[gender=female]. A literal personal name "
            "such as 德伦 must be named_entity using exactly that text; do not infer "
            "a family relationship for a name. Address questions use "
            "self->residence and property=full_address.\n\n"
            "Use only advertised semantic relations, properties, predicates and "
            "operations—never storage fields, SQL, file names, or code. Use "
            "resolve_reference for identity and leave property=null; use select for "
            "a property or list, count "
            "for size, completed_years for age, and argmin/argmax for extrema. "
            "Earlier birth_date is older: oldest=argmin, youngest=argmax. Collection "
            "predicates adult/minor belong in request.filters after traversing "
            "current_household->member; a speaker's children instead traverse "
            "self->child.\n\n"
            "A birthday date is select(birth_date); its next occurrence is "
            "annual_occurrence(birth_date); a birthday countdown adds mode=days. "
            "Phrasing such as 'from today until the birthday' is still the next "
            "annual occurrence, not date_difference from the original birth date. "
            "A person's home address is self->residence then full_address. Metadata "
            "owned by the final edge uses property_source=relationship. Marriage "
            "start is select(self->spouse, start_date, relationship); marriage "
            "duration uses duration on that same relationship property with a mode. "
            "A spouse's birth_date is an entity property and must keep "
            "property_source=entity; only relationship metadata such as start_date "
            "or end_date uses relationship. "
            f"Household now: {household_now}\n"
            "Semantic capabilities:\n"
            + json.dumps(capabilities, ensure_ascii=False, separators=(",", ":"))
        )
        response = await self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt},
                *_semantic_planner_examples(),
                *[dict(message) for message in messages],
            ],
            stream=False,
            think=False,
            format=dict(output_schema),
            options={"temperature": 0},
        )
        parsed = json.loads(response.message.content or "")
        if not isinstance(parsed, Mapping):
            raise ValueError("Semantic fact planner returned a non-object")
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
