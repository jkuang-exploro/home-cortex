import json
from functools import lru_cache
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, cast

from ollama import AsyncClient, ChatResponse

from .profiling import model_call, stage


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

_PLANNER_INSTRUCTIONS = """你是 Home Cortex 的语义解释器。把最新用户问题编译成一个 JSON 语义请求，不计算或表述答案。家庭事实、身份和姓名问题 requires_fact=true；只有普通闲聊才是 false 且 request=null。

组合与对话语法：
- projection=each 明确对集合逐项执行同一标量操作或 select(property)，保留每个实体的结果；单个人仍使用默认 scalar。count/argmin 等归约不用 each。
- exclude 是要从集合中排除的完整引用列表，与比较用的 other 不同。“其他”由语境决定排除 self 还是 discourse；无法确定时引用 unresolved，不猜。
- date_add 使用 amount（有符号整数）和 mode=years/months/days 给实体或关系日期加日历偏移。指定第 N 周年直接加 N 年，即使已过去；不能替换成 annual_occurrence。目标月不存在该日时使用下个月第一天：闰日加一年为三月一日。
- 前文仅供理解话语，不是事实来源。代词或前文对象用 kind=discourse、turn_offset=1..8（倒数第几个用户轮次）、entity_type、cardinality=single|collection，value=null。由可信上下文解析身份；单数不能从多人前文中猜选一人。需要澄清的指代用 kind=unresolved。path 可从前文实体继续组合。不要把代词改写成猜测姓名或 ID。
- 只编译最后一条 user 消息。更早的 user 消息仅用于判断最后一条消息中指代语的先行词；固定的 assistant 省略标记只划分用户轮次，不包含事实。最后一条消息中的第一人称使用 self，第二人称使用 assistant；它们不引用更早轮次。只有必须从更早 user 消息取得对象的第三人称、指示词或明确前文引用才使用 discourse。
- 身份问句看人称，不看“谁”。第一人称（我、I、me、my）问身份、姓名或称呼 → kind=self。第二人称（你、您、you、your）问身份、姓名、角色或自我介绍 → kind=assistant。不要把第二人称问句编成 self，也不要把第一人称问句编成 assistant。

引用语法：
- self 是已认证的当前说话人，assistant 是本助手，二者 entity_type=person、value=null。对助手的第二人称身份问题仍是事实请求，subject 必须是 assistant。current_household 是配置的家庭，entity_type=address、value=null。named_entity.value 只能逐字复制用户说出的姓名或称呼；不得猜测 ID 或把亲属短语当姓名。
- path 是概念的有序组合，每步 {"concept":"名称"}，可附加 filters。reference_concepts 是权威定义；完整短语匹配某个 aliases 时，path 只放该最具体概念一次，不拆解或追加近义概念。son 不能简化为 child，wife 不能简化为 spouse，father 不能简化为 parent。本体会完整展开概念的关系和过滤条件。
- 只有嵌套亲属才增加一跳。性别等形容词约束同一个目标，不增加一跳；额外 filters 与概念原有条件取 AND。说话人的亲属始终从 self 开始，“我家的儿子”也不先遍历全家。不得把上一问概念展开后的某一跳保留下来再换上当前问的另一概念：最新一句已有完整亲属短语时，path 只含该短语对应的最具体概念。
- 家庭成员列表、计数和极值是 current_household 后接 member。无所属限定的成年人/未成年人/孩子是家庭 member 加对应 predicate；明确“我的孩子”才是 self 后接 child。任何住址或家庭地址查询都从 self 接 residence，再取 full_address；current_household 只用于成员集合，不直接投影地址。遵守 relation_signatures 的起点与终点类型。

外层操作：
- 问一个人是谁、身份或叫什么，以及“哪个人是我的某亲属”，用 resolve_reference 且 property=null；明确问名或姓的组成部分才用 select(given_name/family_name)。列出集合用 select 且 property=null；计数用 count 且 property=null。
- 先区分返回实体集合、原始属性值还是计算值。年龄是从出生日期到当前日期的日历年数，用 date_difference(birth_date)、mode=years，返回整数而不是日期；原始出生日期才用 select(birth_date)。下一次生日用 annual_occurrence(birth_date)；从今天到生日的天数用 annual_occurrence，mode=days。
- argmin 返回属性值最小者，argmax 返回最大者。birth_date 越早值越小，所以“年龄最大”也和年长、最早出生一样只能用 argmin；年幼、年龄小、最晚出生用 argmax。问人是谁时不用 earliest/latest 或数值 min/max。
- 两人比较必须提供 property 和两个完整引用 subject、other；说话人参与时 subject=self，另一人在 other。集合极值没有 other。
- 所有日期间隔统一用 date_difference，明确 mode=years/months/days/seconds；单位来自用户要求。年份和月份按完整日历周期计算，不把多少年改成天数。未指定单位的持续时间才默认 days。配偶身份和配偶属性都把 spouse 概念放在 subject.path。关系开始日期和持续时间也只有一个经该关系的 subject，不把两端人物拆成 subject 和 other，不在 request.filters 放人物谓词。

属性所有权：
- 必须根据 property_ownership 明确选择 property_source。entity 是最终实体属性；relationship 是最后一条关系边的属性，要求非空 path，且没有 other。
- 配偶的 birth_date 属于 entity。婚姻 start_date 属于 spouse 关系；关系持续时间使用 date_difference 和同一关系 start_date；问多少年用 years，多少个月用 months，多少天用 days。property=null 时 property_source=entity。不得静默改变错误的所有者。

过滤语法：
- 集合字段条件是 {"property":名称,"operator":比较符,"value":字面值}，source=entity 约束人或地点，source=relation 约束边。只有 path 步骤中的比较可以用 value_from=anchor 引用遍历起点的属性；它不是年份提取，request.filters 不允许动态引用。
- 查询符合条件的实体集合用 select、property=null，条件完整放入 request.filters；条件引用的属性不等于要输出的属性。集合可以有零个、一个或多个结果，不能用单人 resolve_reference 代替。统计同一集合则用 count。
- 日期条件遵循 filter_requirements：date_range 的 value=[含起点,不含终点]，用 ISO 日期表达完整区间；年份限定是该年年初至下一年年初，不是日期与年份数字相等。不得把日期限定丢掉或变成出生日期投影。
- 集合谓词只能是 {"predicate":声明名称}，放在 request.filters。definition_only 只解释含义，不输出为附加条件；date_difference 等操作不是谓词。
- 只用已声明的操作、属性、关系、概念和谓词。不得丢弃不支持的限定、发明词汇、猜身份或修补事实答案。只返回严格结构化输出。
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

    examples = (
        ("请介绍一下你自己。", "resolve_reference", reference("assistant"), None, "entity", {}),
        ("当前已认证的说话人是哪一位？", "resolve_reference", reference("self"), None, "entity", {}),
        ("请说明你的身份。", "resolve_reference", reference("assistant"), None, "entity", {}),
        ("应该怎样称呼这位助手？", "resolve_reference", reference("assistant"), None, "entity", {}),
        ("在本户成员中找出出生日期最靠前的人。", "argmin", reference("current_household", "member"), "birth_date", "entity", {}),
        ("本户符合成年条件的成员有多少？", "count", reference("current_household", "member"), None, "entity", {"filters": [{"predicate": "adult"}]}),
        ("本户未成年成员有多少？", "count", reference("current_household", "member"), None, "entity", {"filters": [{"predicate": "minor"}]}),
        ("我的男孩后代是哪一位？", "resolve_reference", reference("self", "son"), None, "entity", {}),
        ("请列出我的女性后代。", "select", reference("self", "daughter"), None, "entity", {}),
        ("我母亲的丈夫是哪位？", "resolve_reference", reference("self", "mother", "husband"), None, "entity", {}),
        ("我的丈夫出生于哪天？", "select", reference("self", "husband"), "birth_date", "entity", {}),
        ("请介绍岳父的身份。", "resolve_reference", reference("self", "father_in_law"), None, "entity", {}),
        ("妻子下个生日距今有多少天？", "annual_occurrence", reference("self", "wife"), "birth_date", "entity", {"mode": "days"}),
        ("我的母亲如今已满多少周岁？", "date_difference", reference("self", "mother"), "birth_date", "entity", {"mode": "years"}),
        ("列出居住关系始于2015年的本户成员。", "select", reference("current_household", "member"), None, "entity", {"filters": [{"property": "start_date", "source": "relation", "operator": "date_range", "value": ["2015-01-01", "2016-01-01"]}]}),
        ("请提供这个家庭的完整地址。", "select", reference("self", "residence"), "full_address", "entity", {}),
        ("我的居住关系从哪天开始？", "select", reference("self", "residence"), "start_date", "relationship", {}),
        ("这段居住关系至今已满多少年？", "date_difference", reference("self", "residence"), "start_date", "relationship", {"mode": "years"}),
        ("住进现居所至今有多少天？", "date_difference", reference("self", "residence"), "start_date", "relationship", {"mode": "days"}),
        ("女儿下个生日距今有多少天？", "annual_occurrence", reference("self", "daughter"), "birth_date", "entity", {"mode": "days"}),
        ("我与母亲相比，出生较早的是谁？", "argmin", reference("self"), "birth_date", "entity", {"other": reference("self", "mother")}),
        ("林青下次生日还要几天？", "annual_occurrence", {"kind": "named_entity", "value": "林青", "entity_type": "person"}, "birth_date", "entity", {"mode": "days"}),
    )
    messages: list[dict[str, str]] = []
    for utterance, operation, subject, prop, owner, extra in examples:
        request = {"operation": operation, "subject": subject, "property": prop,
                   "property_source": owner, **extra}
        messages.extend((
            {"role": "user", "content": utterance},
            {"role": "assistant", "content": json.dumps(
                {"requires_fact": True, "request": request},
                ensure_ascii=False, separators=(",", ":"),
            )},
        ))
    messages.extend((
        {"role": "user", "content": "周岚现在多大？"},
        {"role": "assistant", "content": json.dumps({
            "requires_fact": True,
            "request": {
                "operation": "date_difference",
                "subject": {
                    "kind": "named_entity",
                    "value": "周岚",
                    "entity_type": "person",
                },
                "property": "birth_date",
                "property_source": "entity",
                "mode": "years",
            },
        }, ensure_ascii=False, separators=(",", ":"))},
        {"role": "user", "content": "他的二十岁生日是哪天？"},
        {"role": "assistant", "content": json.dumps({
            "requires_fact": True,
            "request": {
                "operation": "date_add",
                "subject": {
                    "kind": "discourse",
                    "entity_type": "person",
                    "turn_offset": 1,
                    "cardinality": "single",
                },
                "property": "birth_date",
                "property_source": "entity",
                "amount": 20,
                "mode": "years",
            },
        }, ensure_ascii=False, separators=(",", ":"))},
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
    from .semantic_facts import _identity_person_hint

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
    reminder = (
        f"Household now: {household_now}\n"
        "Person deixis for identity: first person → kind=self; "
        "second person addressing this helper → kind=assistant. "
        "Do not add path, filters, or amount unless the utterance "
        "requires them."
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

    async def plan_semantic_fact(
        self,
        messages: Sequence[Mapping[str, Any]],
        capabilities: Mapping[str, Any],
        output_schema: Mapping[str, Any],
        *,
        household_now: str,
    ) -> Mapping[str, Any]:
        """Interpret an open-ended request without exposing physical storage."""
        response = await self._chat(
            model=self.model,
            messages=planner_chat_messages(
                messages, capabilities, household_now=household_now
            ),
            stream=False,
            think=False,
            keep_alive=PLANNER_KEEP_ALIVE,
            format=dict(output_schema),
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
