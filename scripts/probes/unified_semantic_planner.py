"""Shadow-only one-call read/write semantic planner candidate.

This module composes existing canonical prompts and IR. It never executes a
fact request or dispatches a mutation.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from time import perf_counter
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator
from pydantic import Field, model_validator

from home_cortex import ollama as prompts
from home_cortex.mutation_ir import _mutation_prefix, attribute_output_schema
from home_cortex.semantic_ir import SemanticPlan
from home_cortex.semantic_planner import (
    _identity_person_mismatch,
    _object_location_mismatch,
    _planner_clock,
)
from home_cortex.semantic_schema import SemanticSchemaRegistry


class UnifiedSemanticDecision(SemanticPlan):
    """Shadow extension of the canonical plan with a non-executable branch."""

    multi_intent: bool = Field(
        default=False,
        description=(
            "True only when the latest user turn contains more than one "
            "independent actionable fact query or mutation."
        ),
    )

    @model_validator(mode="after")
    def validate_multi_intent(self) -> "UnifiedSemanticDecision":
        if self.multi_intent and (self.request is not None or self.mutation is not None):
            raise ValueError("A multi-intent plan cannot carry an executable branch")
        return self


_UNIFIED_RULES = """
Unified output envelope:
- Return the existing SemanticPlan shape only: requires_fact, request, mutation,
  and multi_intent.
- Fact: requires_fact=true, request=<SemanticFactRequest>, mutation=null,
  multi_intent=false.
- Mutation: requires_fact=false, request=null, mutation=<one existing named item
  mutation>, multi_intent=false.
- Ordinary conversation/unsupported: requires_fact=false, request=null,
  mutation=null, multi_intent=false.
- Multiple independent actionable objectives: requires_fact=false, request=null,
  mutation=null, multi_intent=true.
- Exactly one fact, mutation, conversation, or multi-intent branch may be present.
- One user turn supports one semantic objective. Two independent fact questions,
  a fact question plus a mutation, or two mutations are multi-intent even when
  joined in one sentence. Emit no partial request or mutation for the turn.
- Clauses that together specify one fact query or one mutation remain one intent;
  destinations, requested attributes, filters, and qualifiers are not extra intents.
- An explicit preview of a supported mutation is that mutation with mode=preview,
  including "what would happen if I move..." phrasing. Explicit execution uses
  mode=commit. Never turn an explicit preview into commit.
- The mutation instructions' requires_mutation field belongs to the old standalone
  classifier. Do not emit that field in SemanticPlan. A non-mutation must still be
  interpreted using the fact grammar before choosing the none branch.
""".strip()


_MULTI_INTENT_EXAMPLES = (
    (
        "How old am I? Then remove the kettle record.",
        {"requires_fact": False, "request": None, "mutation": None, "multi_intent": True},
    ),
    (
        "我儿子是谁，再把画册移到书房。",
        {"requires_fact": False, "request": None, "mutation": None, "multi_intent": True},
    ),
    (
        "Which room contains the vase, and record the keys in the entry drawer.",
        {"requires_fact": False, "request": None, "mutation": None, "multi_intent": True},
    ),
    (
        "Where is the notebook? Move it to the desk.",
        {"requires_fact": False, "request": None, "mutation": None, "multi_intent": True},
    ),
    (
        "Move the notebook to the study and remove the vase record.",
        {"requires_fact": False, "request": None, "mutation": None, "multi_intent": True},
    ),
    (
        "把雨伞放进门厅柜，然后把画册颜色改为红色。",
        {"requires_fact": False, "request": None, "mutation": None, "multi_intent": True},
    ),
    (
        "我丈夫是谁，茶几里有什么？",
        {"requires_fact": False, "request": None, "mutation": None, "multi_intent": True},
    ),
    (
        "How many rooms are here, and who is my daughter?",
        {"requires_fact": False, "request": None, "mutation": None, "multi_intent": True},
    ),
)


_MUTATION_MODE_EXAMPLES = (
    (
        "Preview recording a notebook in the desk drawer.",
        {
            "requires_fact": False,
            "request": None,
            "mutation": {
                "operation": "create",
                "item_name": "notebook",
                "location_name": "desk drawer",
                "name_en": "notebook",
                "name_zh": "笔记本",
                "item_key": "notebook",
                "attributes": {"item_type": "book"},
                "mode": "preview",
            },
            "multi_intent": False,
        },
    ),
    (
        "先预览把雨伞记录在门厅柜里。",
        {
            "requires_fact": False,
            "request": None,
            "mutation": {
                "operation": "create",
                "item_name": "雨伞",
                "location_name": "门厅柜",
                "name_en": "umbrella",
                "name_zh": "雨伞",
                "item_key": "yusan",
                "attributes": {"item_type": "unknown"},
                "mode": "preview",
            },
            "multi_intent": False,
        },
    ),
)


_UNIFIED_READ_REPAIR_EXAMPLES = (
    (
        "How many children are there?",
        {
            "requires_fact": True,
            "request": {
                "operation": "count",
                "subject": {
                    "kind": "current_household",
                    "value": None,
                    "entity_type": "address",
                    "path": [{"concept": "member"}],
                },
                "property": None,
                "property_source": "entity",
                "filters": [{"predicate": "minor"}],
            },
            "mutation": None,
            "multi_intent": False,
        },
    ),
    (
        "我和丈夫的婚姻持续多久了？",
        {
            "requires_fact": True,
            "request": {
                "operation": "date_difference",
                "subject": {
                    "kind": "self",
                    "value": None,
                    "entity_type": "person",
                    "path": [{"concept": "husband"}],
                },
                "property": "start_date",
                "property_source": "relationship",
                "mode": "days",
            },
            "mutation": None,
            "multi_intent": False,
        },
    ),
    (
        "请按出生日期选出最年轻的家庭成员。",
        {
            "requires_fact": True,
            "request": {
                "operation": "argmax",
                "subject": {
                    "kind": "current_household",
                    "value": None,
                    "entity_type": "address",
                    "path": [{"concept": "member"}],
                },
                "property": "birth_date",
                "property_source": "entity",
            },
            "mutation": None,
            "multi_intent": False,
        },
    ),
    (
        "我的丈夫叫什么名字？",
        {
            "requires_fact": True,
            "request": {
                "operation": "resolve_reference",
                "subject": {
                    "kind": "self",
                    "value": None,
                    "entity_type": "person",
                    "path": [{"concept": "husband"}],
                },
                "property": None,
                "property_source": "entity",
            },
            "mutation": None,
            "multi_intent": False,
        },
    ),
    (
        "哪一位男性是我的孩子？",
        {
            "requires_fact": True,
            "request": {
                "operation": "resolve_reference",
                "subject": {
                    "kind": "self",
                    "value": None,
                    "entity_type": "person",
                    "path": [{"concept": "son"}],
                },
                "property": None,
                "property_source": "entity",
            },
            "mutation": None,
            "multi_intent": False,
        },
    ),
    (
        "列出我的女性子女。",
        {
            "requires_fact": True,
            "request": {
                "operation": "select",
                "subject": {
                    "kind": "self",
                    "value": None,
                    "entity_type": "person",
                    "path": [{"concept": "daughter"}],
                },
                "property": None,
                "property_source": "entity",
            },
            "mutation": None,
            "multi_intent": False,
        },
    ),
    (
        "我的配偶的女性家长是谁？",
        {
            "requires_fact": True,
            "request": {
                "operation": "resolve_reference",
                "subject": {
                    "kind": "self",
                    "value": None,
                    "entity_type": "person",
                    "path": [
                        {"concept": "spouse"},
                        {"concept": "mother"},
                    ],
                },
                "property": None,
                "property_source": "entity",
            },
            "mutation": None,
            "multi_intent": False,
        },
    ),
)


def unified_output_schema(schema: SemanticSchemaRegistry) -> dict[str, Any]:
    """Reuse the canonical one-of plan and declared writable-property constraints."""
    result = schema.planner_output_schema()
    result["properties"]["multi_intent"] = (
        UnifiedSemanticDecision.model_json_schema()["properties"]["multi_intent"]
    )
    result["required"] = [
        "requires_fact", "request", "mutation", "multi_intent"
    ]
    for name, definition in result.get("$defs", {}).items():
        properties = definition.get("properties", {})
        if name.startswith("Named") and "mode" in properties:
            definition.setdefault("required", []).append("mode")
            properties["mode"]["description"] = (
                "Choose preview for an explicit preview, proposed-change, or "
                "what-would-happen request; choose commit only for execution."
            )
    fact_request = result["$defs"]["SemanticFactRequest"]
    fact_request.setdefault("allOf", []).append({
        "if": {
            "properties": {"operation": {"const": "resolve_reference"}},
            "required": ["operation"],
        },
        "then": {
            "properties": {
                "property": {"type": "null"},
                "property_source": {"const": "entity"},
            },
        },
    })
    result["oneOf"] = [
        {
            "properties": {
                "requires_fact": {"const": True},
                "request": {"not": {"type": "null"}},
                "mutation": {"type": "null"},
                "multi_intent": {"const": False},
            },
        },
        {
            "properties": {
                "requires_fact": {"const": False},
                "request": {"type": "null"},
                "mutation": {"not": {"type": "null"}},
                "multi_intent": {"const": False},
            },
        },
        {
            "properties": {
                "requires_fact": {"const": False},
                "request": {"type": "null"},
                "mutation": {"type": "null"},
                "multi_intent": {"const": False},
            },
        },
        {
            "properties": {
                "requires_fact": {"const": False},
                "request": {"type": "null"},
                "mutation": {"type": "null"},
                "multi_intent": {"const": True},
            },
        },
    ]
    return attribute_output_schema(result)


def unified_chat_messages(
    messages: Sequence[Mapping[str, Any]],
    schema: SemanticSchemaRegistry,
    *,
    household_now: str,
) -> list[dict[str, Any]]:
    """Compose semantic and mutation contracts without another vocabulary."""
    capabilities = schema.planner_capability_payload()
    built = prompts.planner_chat_messages(
        messages,
        capabilities,
        household_now=household_now,
    )
    old_opening = (
        "You are Home Cortex's semantic interpreter. Compile the latest user "
        "message into one JSON semantic request. Do not compute or state the "
        "answer. Household facts, identity, and names: requires_fact=true. "
        "Ordinary chat only: requires_fact=false and request=null."
    )
    new_opening = (
        "You are Home Cortex's unified semantic interpreter. First classify the "
        "latest user turn as exactly one fact, mutation, conversation, or "
        "multi-intent decision, then compile its single objective. Do not compute "
        "or state an answer. Two or more independent actionable objectives always "
        "use the non-executable multi-intent branch. A fact-only turn uses "
        "requires_fact=true; ordinary chat uses the conversation branch."
    )
    if old_opening not in built[0]["content"]:
        raise RuntimeError("semantic planner opening changed")
    built[0]["content"] = built[0]["content"].replace(
        old_opening, new_opening, 1
    )
    for message in built:
        if message.get("role") != "assistant":
            continue
        try:
            example = json.loads(message["content"])
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(example, dict):
            example.setdefault("request", None)
            example.setdefault("mutation", None)
            example["multi_intent"] = False
            message["content"] = json.dumps(
                example, ensure_ascii=False, separators=(",", ":")
            )
    mutation_prefix = _mutation_prefix()
    built[0]["content"] += (
        "\n\nSupported item mutations (canonical mutation contract):\n"
        + mutation_prefix[0][1]
        + "\n\n"
        + _UNIFIED_RULES
    )
    mutation_examples: list[dict[str, str]] = []
    for index in range(1, len(mutation_prefix), 2):
        user = mutation_prefix[index]
        assistant = mutation_prefix[index + 1]
        decision = json.loads(assistant[1])
        if not decision.get("requires_mutation") or decision.get("mutation") is None:
            continue
        mutation_examples.extend((
            {"role": user[0], "content": user[1]},
            {
                "role": assistant[0],
                "content": json.dumps(
                    {
                        "requires_fact": False,
                        "request": None,
                        "mutation": decision["mutation"],
                        "multi_intent": False,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        ))
    multi_intent_examples: list[dict[str, str]] = []
    for utterance, payload in _MULTI_INTENT_EXAMPLES:
        multi_intent_examples.extend((
            {"role": "user", "content": utterance},
            {
                "role": "assistant",
                "content": json.dumps(
                    payload, ensure_ascii=False, separators=(",", ":")
                ),
            },
        ))
    mutation_mode_examples: list[dict[str, str]] = []
    for utterance, payload in _MUTATION_MODE_EXAMPLES:
        mutation_mode_examples.extend((
            {"role": "user", "content": utterance},
            {
                "role": "assistant",
                "content": json.dumps(
                    payload, ensure_ascii=False, separators=(",", ":")
                ),
            },
        ))
    read_repair_examples: list[dict[str, str]] = []
    for utterance, payload in _UNIFIED_READ_REPAIR_EXAMPLES:
        read_repair_examples.extend((
            {"role": "user", "content": utterance},
            {
                "role": "assistant",
                "content": json.dumps(
                    payload, ensure_ascii=False, separators=(",", ":")
                ),
            },
        ))
    # Keep the established semantic examples closest to the current request.
    # The rejected candidate placed mutation examples after them and introduced
    # repeatable identity and kinship regressions.
    reminder_index = 1 + len(prompts._semantic_planner_examples())
    built[reminder_index]["content"] += (
        "\nFirst count independent actionable objectives in the latest turn. "
        "If there is more than one, set multi_intent=true and emit no request "
        "or mutation. Asking where an item is and also ordering that item moved "
        "are two independent objectives, even when both clauses name the same item. "
        "For an explicit mutation preview, mode=preview is required; "
        "never omit mode or emit commit. Person identity or name wording, including "
        "叫什么, uses resolve_reference with property=null; never attach "
        "display_name. A complete son or daughter concept already carries its "
        "gender constraint; do not repeat that constraint in request.filters."
    )
    return [
        built[0],
        *mutation_examples,
        *mutation_mode_examples,
        *built[1:reminder_index],
        *read_repair_examples,
        *multi_intent_examples,
        *built[reminder_index:],
    ]


def validate_unified_payload(
    schema: SemanticSchemaRegistry,
    payload: Mapping[str, Any],
    utterance: str,
) -> UnifiedSemanticDecision:
    """Apply the existing structural and fact-contract validation boundaries."""
    output_schema = unified_output_schema(schema)
    Draft202012Validator(output_schema).validate(payload)
    plan = UnifiedSemanticDecision.model_validate(
        schema.expand_planner_concepts(payload)
    )
    if plan.request is None:
        return plan
    references = (plan.request.subject, plan.request.other, *plan.request.exclude)
    if any(reference is not None and reference.kind == "entity_id" for reference in references):
        raise ValueError("MODEL_ORIGINATED_ENTITY_ID")
    validation = schema.validation_code(plan.request)
    if validation != "VALID":
        raise ValueError(validation)
    mismatch = (
        _identity_person_mismatch(utterance, plan.request)
        or _object_location_mismatch(utterance, plan.request)
    )
    if mismatch:
        raise ValueError("INVALID_PLAN")
    return plan


@dataclass(frozen=True)
class UnifiedShadowResult:
    plan: UnifiedSemanticDecision | None
    attempts: int
    latency_ms: float
    calls: tuple[Mapping[str, Any], ...]
    output_raw: Mapping[str, Any] | None
    error: str | None


class UnifiedShadowPlanner:
    """One-call common path with one validation retry; shadow use only."""

    def __init__(self, ollama: prompts.OllamaService, schema: SemanticSchemaRegistry):
        self.ollama = ollama
        self.schema = schema

    async def plan(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        household_now: Any,
    ) -> UnifiedShadowResult:
        started = perf_counter()
        output_schema = unified_output_schema(self.schema)
        payload: Mapping[str, Any] | None = None
        error: Exception | None = None
        calls: list[Mapping[str, Any]] = []
        plan: UnifiedSemanticDecision | None = None
        utterance = next(
            (
                str(message.get("content", ""))
                for message in reversed(messages)
                if message.get("role") == "user"
            ),
            "",
        )
        for attempt in (1, 2):
            request_messages = [dict(message) for message in messages]
            if attempt == 2:
                request_messages.append({
                    "role": "system",
                    "content": (
                        "The previous unified result failed strict validation. "
                        "Recompile the original current-turn meaning into exactly one "
                        "fact, mutation, conversation, or multi-intent branch. "
                        "A multi-intent result must contain no executable branch."
                    ),
                })
            response = await self.ollama._chat(
                model=self.ollama.model,
                messages=unified_chat_messages(
                    request_messages,
                    self.schema,
                    household_now=_planner_clock(household_now),
                ),
                stream=False,
                think=False,
                keep_alive=prompts.PLANNER_KEEP_ALIVE,
                format=output_schema,
                options={
                    "temperature": 0,
                    "num_ctx": prompts.PLANNER_NUM_CTX,
                    "num_predict": prompts.PLANNER_NUM_PREDICT,
                    "seed": prompts.PLANNER_SEED,
                },
            )
            calls.append({
                "prompt_tokens": response.prompt_eval_count,
                "output_tokens": response.eval_count,
                "prompt_eval_ms": (
                    response.prompt_eval_duration / 1_000_000
                    if response.prompt_eval_duration is not None else None
                ),
                "eval_ms": (
                    response.eval_duration / 1_000_000
                    if response.eval_duration is not None else None
                ),
                "done_reason": response.done_reason,
            })
            try:
                decoded = json.loads(response.message.content or "")
                if not isinstance(decoded, Mapping):
                    raise ValueError("unified result is not an object")
                payload = decoded
                plan = validate_unified_payload(self.schema, payload, utterance)
                error = None
                break
            except Exception as exc:
                error = exc
        return UnifiedShadowResult(
            plan=plan,
            attempts=len(calls),
            latency_ms=(perf_counter() - started) * 1000,
            calls=tuple(calls),
            output_raw=payload,
            error=(
                f"{type(error).__name__}: {error}"
                if error is not None else None
            ),
        )
