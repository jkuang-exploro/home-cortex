import json
from functools import lru_cache
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, cast

from ollama import AsyncClient, ChatResponse

from .request_tracing import model_call, stage
from .mutation_ir import MutationDecision, mutation_messages, read_plan_schema, attribute_output_schema


# Keep the same resident runner configuration across ordinary chat and planning.
# Different context sizes cause Ollama to restart the runner between paths.
OLLAMA_KEEP_ALIVE = "24h"
OLLAMA_NUM_CTX = 8192
# Retain the planner names used by benchmark fingerprinting and probe scripts.
PLANNER_KEEP_ALIVE = OLLAMA_KEEP_ALIVE
PLANNER_NUM_CTX = OLLAMA_NUM_CTX
PLANNER_NUM_PREDICT = 384
PLANNER_SEED = 0
_PLANNER_HISTORY_BOUNDARY = (
    "[End of earlier user turn. Its assistant answer is omitted. Compile only "
    "the following user message; use this earlier turn solely for discourse "
    "antecedents.]"
)

# Semantic contract is English-primary. Canonical IR identifiers stay English.
# Keep Chinese text only for wording that is itself the rule (deixis, kinship
# terms, age-threshold particles). Chinese, English, and mixed utterances
# compile through this same grammar into one language-neutral IR.
_PLANNER_INSTRUCTIONS = """You are Home Cortex's semantic interpreter. Compile the latest user message into one JSON semantic request. Do not compute or state the answer. Household facts, identity, and names: requires_fact=true. Ordinary chat only: requires_fact=false and request=null.

Language: Interpret Chinese, English, and mixed Chinese-English with the same semantic rules. Surface language must not change the IR. Equivalent utterances compile to equivalent normalized requests. Do not translate the utterance as an intermediate step; compile surface language directly into IR. Canonical identifiers stay English regardless of input language: self, assistant, current_household, member, son, daughter, wife, husband, father_in_law, birth_date, date_difference, annual_occurrence, resolve_reference, same_entity, and other declared names.

Deixis and household scope:
- First person 我 / 我的 / I / me / my → kind=self when referring to the authenticated speaker. Second person 你 / 您 / you / your addressing this assistant → kind=assistant. Identity questions look at person, not the word who/谁. Never compile second person as self or first person as assistant.
- 咱家 / 我家 / 我家里 / my household / our household asking who is in it, how many people, or who is oldest/youngest → current_household then member; never self then member (person cannot traverse member) and never self then residence. 我家的儿子 / my son → self then child. 我家住哪里 / where I live / street address → self then residence.
- Kinship surface forms: 岳父 / 公公 / father-in-law → father_in_law; 丈夫 / husband → husband; 妻子 / 老婆 / wife → wife; 儿子 / son → son; 女儿 / daughter → daughter. Use the most specific declared concept.

Composition and discourse:
- projection=each applies the same scalar operation or select(property) to each collection item; a single person stays scalar. Reductions (count/argmin/argmax) do not use each.
- exclude is a complete list of references to remove from a collection; it is not comparison other. 其他 / the others excludes self or discourse from context; if unknown, kind=unresolved; never guess. A follow-up asking 还有哪些 / what other after a prior entity excludes that typed discourse entity while keeping any explicit named collection root as subject. Item classes (电器/appliance, 家具/furniture, 食物/food) filter the stored item_type field.
- date_add: signed amount with mode=years/months/days on an entity or relationship date. The Nth anniversary adds N years even if past; do not replace with annual_occurrence. Missing target day → first of next month (Feb 29 + 1 year = Mar 1).
- Prior turns explain referring expressions only; they are not a fact source. Pronouns or prior objects: kind=discourse, turn_offset=1..8 (Nth previous user turn), entity_type, cardinality=single|collection, value=null. Trusted context resolves identity; singular cannot pick one person from a multi-person prior. Unclear references: kind=unresolved. path may continue from a discourse entity. Never rewrite a pronoun as a guessed name or ID.
- Compile only the last user message. Earlier user messages are antecedents only. The fixed assistant omission marker delimits user turns and contains no facts. First/second person in the latest message are self/assistant and do not refer to earlier turns. Use discourse only for third person, demonstratives, or explicit prior references that need an earlier user object.
- If the latest sentence is itself a complete fact question (count, list, extremum, gender, birth year, age threshold, kinship, same-entity, lives-here), request.filters and path contain only that sentence's conditions and concepts. Do not copy adult, minor, gender, date_range, or identity resolve_reference from the previous question.

References:
- Spatial structure and inventory are different collections: use concept hosted_space for the spaces/areas/compartments a named entity hosts, concept host for what hosts a named space, and concept contents for the items it holds. Hosted spaces exist independently of whether they contain any items. Do not append contents when the requested result is spaces; use an additional contents hop only when the user explicitly asks for their items.
- self is the authenticated speaker; assistant is this helper; both entity_type=person, value=null. Second-person identity about the assistant is still a fact request with subject=assistant. current_household is the configured home, entity_type=address, value=null.
- named_entity.value copies the user's literal person name, item name, or space name. Do not guess IDs, treat kinship phrases as names, or translate/normalize names because surrounding language changed (林青 stays 林青).
- named_entity also covers explicitly named items and spaces. Keep the complete compound name in subject.value, including container/subspace qualifiers; never shorten it to the containing object. entity_type is a semantic constraint, not a guess at how a named object is stored. For a named container or storage subspace whose item/space type is not explicit, use entity_type=null so exact-name resolution determines the type. Preserve an explicitly stated type; never replace the named subject with another entity. 花瓶在哪里 / where is the kettle: named item as subject, entity_type=item, path concept location, resolve_reference, property=null. 在哪里/where is attaches to that named object; it is not a household concept, not adult/minor, and the object is not a person. A named room/area such as 厨房, 车库, kitchen, or garage is named_entity entity_type=space; to list its items, use only path concept contents and select. The room concept is only for an unqualified household room collection. current_household is context, not a substitute for a named subject. Do not drop explicit item or space names. Emit full_address only for an explicit street-address request. Obey relation_signatures: location accepts items, not spaces; do not replace a missing path with a different executable question.
- path is ordered {"concept": name} steps, optional filters. reference_concepts are authoritative. A complete phrase matching an alias uses that most specific concept once; do not split or append near-synonyms. son ≠ child, wife ≠ spouse, husband ≠ spouse, father ≠ parent, father_in_law ≠ parent. The ontology expands the concept's relations and filters.
- Nested kinship adds a hop. Gender and other adjectives constrain the same target and AND with the concept's filters; they do not add a hop. Speaker relatives always start from self; 我家的儿子 / my son does not walk the household first. Do not keep a hop from the previous question and swap in a new concept. A complete kinship phrase in the latest sentence → only that most specific concept.
- Unqualified adults/minors/children: household member plus the matching predicate; 我的孩子 / my children → self then child. Rooms: current_household then room, not member, and not space_type on people. Street/home address: self then residence then full_address. Obey relation_signatures start and end types.

Operations:
- Inspect all recorded attributes/properties of one entity: inspect, property=null, property_source=entity. Asking for one named attribute uses select(property); inspecting attributes never changes them.
- Who/identity/name of a person, or which person is a relative: resolve_reference, property=null. Name parts only when explicitly asked: select(given_name/family_name). List a collection: select, property=null. Count: count, property=null. Same entity: same_entity, property=null, both subject and other; returns equality, not an introduction. Currently lives here: subject is that person's residence (kinship composition allowed), other is current_household; do not resolve_reference the person.
- Distinguish entity sets, raw properties, and computed values. Age is completed calendar years from birth_date to now: date_difference(birth_date), mode=years, integer not a date. Raw birthday: select(birth_date). Next birthday: annual_occurrence(birth_date). Days until that birthday: annual_occurrence, mode=days.
- argmin = smallest property value; argmax = largest. Earlier birth_date is smaller, so oldest/eldest/earliest-born = argmin; youngest/latest-born = argmax. Do not use earliest/latest or numeric min/max to name a person.
- Pairwise comparison needs property and two complete references. Speaker participating → subject=self, the other person in other. Collection extrema have no other.
- All intervals use date_difference with explicit mode=years/months/days/seconds from the user's unit. Years and months are whole calendar periods; never convert years to days. Unspecified duration defaults to days. Spouse identity and spouse properties put concept spouse (or wife/husband) on subject.path. Relationship start and duration use one subject through that relation; do not split the two people into subject and other; do not put person predicates in request.filters.

Ownership:
- Choose property_source from property_ownership. entity = final entity field. relationship = last edge field; requires nonempty path and no other.
- Spouse birth_date is entity. Marriage start_date belongs to the spouse relationship. Duration uses date_difference on that same start_date (years/months/days as asked). property=null → property_source=entity. Never silently switch owners.

Filters:
- Field condition: {"property":name,"operator":cmp,"value":literal}. source=entity constrains people/places; source=relation constrains the edge. value_from=anchor is allowed only on path-step comparisons (traversal-start property), not year extraction; request.filters cannot use dynamic references.
- Matching set: select, property=null, all conditions in request.filters. The filtered property is not the output property. Sets may be empty, singleton, or many; do not substitute resolve_reference. Counting the same set: count.
- Dates follow filter_requirements. date_range value=[inclusive start, exclusive end] as ISO dates. A year is that year's Jan 1 to the next year's Jan 1, not equality with a year number. Do not drop a date bound or project birth_date instead.
- 男/男性/男的/male → {"property":"gender","value":"male"}; 女/女性/女的/female → {"property":"gender","value":"female"}. Gender is not adult, not minor, and not their conjunction. predicate_disjointness forbids adult and minor on the same set.
- Age ≥ N / at least N / 满 N 岁 / N 岁以上: operator=gte, value=N. Age < N / under N / 未满 N 岁 / N 岁以下: operator=lt. 以上 is not 以下; never invert. Shape: {"property":"birth_date","transform":"date_difference","mode":"years","operator":"gte","value":N}. The executor uses Household now. Do not rewrite as birth_date date_range, invent ISO cutoffs, or replace arbitrary ages with adult/minor.
- Predicates: only {"predicate":declared name} in request.filters. definition_only explains meaning and is not emitted. date_difference and other operations are not predicates.
- Use only declared operations, properties, relations, concepts, and predicates. Do not drop unsupported qualifiers, invent vocabulary, guess identity, or patch answers. Return strict structured output only.
"""

def _semantic_planner_examples() -> list[dict[str, str]]:
    # Cache only immutable text; callers still own their message dictionaries.
    return [{"role": role, "content": content} for role, content in _example_text()]


@lru_cache(maxsize=1)
def _example_text() -> tuple[tuple[str, str], ...]:
    """Illustrate reusable grammar without evaluation wording or household facts."""
    def reference(kind: str, *concepts: str) -> dict[str, Any]:
        return {"kind": kind, "value": None,
                "entity_type": "address" if kind == "current_household" else "person",
                **({"path": [{"concept": name} for name in concepts]} if concepts else {})}

    def named(value: str, entity_type: str | None, *concepts: str) -> dict[str, Any]:
        return {"kind": "named_entity", "value": value, "entity_type": entity_type,
                **({"path": [{"concept": name} for name in concepts]} if concepts else {})}

    def dump(payload: dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    examples = (
        ("请介绍一下你自己。", "resolve_reference", reference("assistant"), None, "entity", {}),
        ("Who is the authenticated speaker?", "resolve_reference", reference("self"), None, "entity", {}),
        ("Who belongs to my household?", "select", reference("current_household", "member"), None, "entity", {}),
        ("How many people in my household?", "count", reference("current_household", "member"), None, "entity", {}),
        ("我这个家里出生日期最靠前的是哪一位？", "argmin", reference("current_household", "member"), "birth_date", "entity", {}),
        ("Who here was born most recently?", "argmax", reference("current_household", "member"), "birth_date", "entity", {}),
        ("本户符合成年条件的成员有多少？", "count", reference("current_household", "member"), None, "entity", {"filters": [{"predicate": "adult"}]}),
        ("How many household minors are there?", "count", reference("current_household", "member"), None, "entity", {"filters": [{"predicate": "minor"}]}),
        ("本户男性成员人数是多少？", "count", reference("current_household", "member"), None, "entity", {"filters": [{"property": "gender", "operator": "eq", "value": "male"}]}),
        ("How many female members here?", "count", reference("current_household", "member"), None, "entity", {"filters": [{"property": "gender", "operator": "eq", "value": "female"}]}),
        ("本户出生于1991年的成员有几位？", "count", reference("current_household", "member"), None, "entity", {"filters": [{"property": "birth_date", "operator": "date_range", "value": ["1991-01-01", "1992-01-01"]}]}),
        ("List members aged 40 or older.", "select", reference("current_household", "member"), None, "entity", {"filters": [{"property": "birth_date", "transform": "date_difference", "mode": "years", "operator": "gte", "value": 40}]}),
        ("本户未满三十周岁的成员有多少？", "count", reference("current_household", "member"), None, "entity", {"filters": [{"property": "birth_date", "transform": "date_difference", "mode": "years", "operator": "lt", "value": 30}]}),
        ("请列出本户的房间。", "select", reference("current_household", "room"), None, "entity", {}),
        ("我的男孩后代是哪一位？", "resolve_reference", reference("self", "son"), None, "entity", {}),
        ("List my female descendants.", "select", reference("self", "daughter"), None, "entity", {}),
        ("我母亲的丈夫是哪位？", "resolve_reference", reference("self", "mother", "husband"), None, "entity", {}),
        ("Who is 我岳父?", "resolve_reference", reference("self", "father_in_law"), None, "entity", {}),
        ("我 wife 的 birthday 是哪天？", "select", reference("self", "wife"), "birth_date", "entity", {}),
        ("我 son 几岁了？", "date_difference", reference("self", "son"), "birth_date", "entity", {"mode": "years"}),
        ("List members whose residence started in 2015.", "select", reference("current_household", "member"), None, "entity", {"filters": [{"property": "start_date", "source": "relation", "operator": "date_range", "value": ["2015-01-01", "2016-01-01"]}]}),
        ("Give this home's street address.", "select", reference("self", "residence"), "full_address", "entity", {}),
        ("说话人现居所是否即当前配置家庭？", "same_entity", reference("self", "residence"), None, "entity", {"other": reference("current_household")}),
        ("Does that mother live at the configured home?", "same_entity", reference("self", "mother", "residence"), None, "entity", {"other": reference("current_household")}),
        ("我的居住关系从哪天开始？", "select", reference("self", "residence"), "start_date", "relationship", {}),
        ("How many years has this residence lasted?", "date_difference", reference("self", "residence"), "start_date", "relationship", {"mode": "years"}),
        ("住进现居所至今有多少天？", "date_difference", reference("self", "residence"), "start_date", "relationship", {"mode": "days"}),
        ("Between me and my mother, who was born earlier?", "argmin", reference("self"), "birth_date", "entity", {"other": reference("self", "mother")}),
        ("花瓶在哪里？", "resolve_reference", named("花瓶", "item", "location"), None, "entity", {}),
        ("工作间里还有哪些工具？", "select", named("工作间", "space", "contents"), None, "entity", {"filters": [{"property": "item_type", "operator": "eq", "value": "tool"}], "exclude": [{"kind": "discourse", "entity_type": "item", "turn_offset": 1, "cardinality": "single"}]}),
        ("旅行箱里有哪些东西？", "select", named("旅行箱", None, "contents"), None, "entity", {}),
        ("What does the workbench cubby hold?", "select", named("workbench cubby", None, "contents"), None, "entity", {}),
        ("展示柜有哪些分区？", "select", named("展示柜", None, "hosted_space"), None, "entity", {}),
        ("How many spaces does 写字台 host?", "count", named("写字台", None, "hosted_space"), None, "entity", {}),
        ("Where is the kettle?", "resolve_reference", named("kettle", "item", "location"), None, "entity", {}),
        ("How many days until 林青's next birthday?", "annual_occurrence", named("林青", "person"), "birth_date", "entity", {"mode": "days"}),
    )
    messages: list[dict[str, str]] = []
    for utterance, operation, subject, prop, owner, extra in examples:
        request = {"operation": operation, "subject": subject, "property": prop,
                   "property_source": owner, **extra}
        messages.extend((
            {"role": "user", "content": utterance},
            {"role": "assistant", "content": dump({"requires_fact": True, "request": request})},
        ))
    discourse = {
        "kind": "discourse",
        "entity_type": "person",
        "turn_offset": 1,
        "cardinality": "single",
    }
    messages.extend((
        {"role": "user", "content": "周岚现在多大？"},
        {"role": "assistant", "content": dump({
            "requires_fact": True,
            "request": {
                "operation": "date_difference",
                "subject": named("周岚", "person"),
                "property": "birth_date",
                "property_source": "entity",
                "mode": "years",
            },
        })},
        {"role": "user", "content": "When was his 20th birthday?"},
        {"role": "assistant", "content": dump({
            "requires_fact": True,
            "request": {
                "operation": "date_add",
                "subject": discourse,
                "property": "birth_date",
                "property_source": "entity",
                "amount": 20,
                "mode": "years",
            },
        })},
        {"role": "user", "content": "Count this household's adults."},
        {"role": "assistant", "content": dump({
            "requires_fact": True,
            "request": {
                "operation": "count",
                "subject": reference("current_household", "member"),
                "property": None,
                "property_source": "entity",
                "filters": [{"predicate": "adult"}],
            },
        })},
        {"role": "user", "content": "How old is Zhou Lan?"},
        {"role": "assistant", "content": dump({
            "requires_fact": True,
            "request": {
                "operation": "date_difference",
                "subject": named("Zhou Lan", "person"),
                "property": "birth_date",
                "property_source": "entity",
                "mode": "years",
            },
        })},
        {"role": "user", "content": "他的二十岁生日是哪天？"},
        {"role": "assistant", "content": dump({
            "requires_fact": True,
            "request": {
                "operation": "date_add",
                "subject": discourse,
                "property": "birth_date",
                "property_source": "entity",
                "amount": 20,
                "mode": "years",
            },
        })},
        {"role": "user", "content": "Just chatting, no household question."},
        {"role": "assistant", "content": dump({"requires_fact": False, "request": None})},
    ))
    return tuple((message["role"], message["content"]) for message in messages)


def planner_system_prompt(capabilities: Mapping[str, Any]) -> str:
    # Capability maps may originate from sets or differently ordered registries.
    # Canonicalize object keys only; ordered semantic arrays remain unchanged.
    return (
        _PLANNER_INSTRUCTIONS
        + "\nCapabilities:\n"
        + json.dumps(capabilities, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    )


@stage("planner.messages")
def planner_chat_messages(
    messages: Sequence[Mapping[str, Any]],
    capabilities: Mapping[str, Any],
    *,
    household_now: str,
) -> list[dict[str, Any]]:
    """Build planner messages: examples plus user turns, no assistant answers."""
    forwarded: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") != "user":
            continue
        if forwarded:
            forwarded.append({
                "role": "assistant",
                "content": _PLANNER_HISTORY_BOUNDARY,
            })
        forwarded.append({
            "role": "user",
            "content": str(message.get("content", "")),
        })
    from .semantic_planner import _identity_person_hint, _object_location_hint

    notes = [
        str(message.get("content", "")) for message in messages
        if message.get("role") == "system" and str(message.get("content", "")).strip()
    ]
    last_user = next(
        (item["content"] for item in reversed(forwarded) if item["role"] == "user"),
        "",
    )
    hint = _identity_person_hint(last_user)
    if hint and not any(hint in note for note in notes):
        notes.append(hint)
    location = _object_location_hint(last_user)
    if location and not any(location in note for note in notes):
        notes.append(location)
    reminder = (
        f"Household now: {household_now}\n"
        "Compile only the latest user message. Do not add path, filters, or "
        "amount unless that utterance requires them. Do not copy filters from "
        "earlier turns."
    )
    built = [
        {"role": "system", "content": planner_system_prompt(capabilities)},
        *_semantic_planner_examples(),
        {"role": "system", "content": reminder},
        *forwarded,
    ]
    if notes:
        built.append({"role": "system", "content": "\n".join(notes)})
    return built


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
        self.last_planner_runtime = _ollama_runtime_metrics(response)
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


def language_model_from_settings(
    settings: Any,
    model_name: str | None = None,
) -> OllamaService:
    """Construct the deployment LLM client. OpenRouter is returned duck-typed."""
    if getattr(settings, "llm_provider", "ollama") == "openrouter":
        from .openrouter import OpenRouterService

        name = model_name or settings.openrouter_model
        secret = settings.openrouter_api_key
        if not name or secret is None:
            raise ValueError(
                "OPENROUTER_API_KEY and OPENROUTER_MODEL are required "
                "when LLM_PROVIDER=openrouter"
            )
        return OpenRouterService(  # type: ignore[return-value]
            settings.openrouter_base_url,
            name,
            api_key=secret.get_secret_value(),
            http_referer=settings.openrouter_http_referer,
            app_title=settings.openrouter_app_title,
        )
    name = model_name or settings.ollama_model
    if not name:
        raise ValueError("OLLAMA_MODEL is required when LLM_PROVIDER=ollama")
    return OllamaService(settings.ollama_url, name)
