import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, cast

from ollama import AsyncClient, ChatResponse

from .text import latest_user_message

PLANNER_KEEP_ALIVE = "24h"
PLANNER_NUM_PREDICT = 384
PLANNER_SEED = 0

_PLANNER_INSTRUCTIONS = """你是 Home Cortex 的语义解释器。把最新用户问题编译成一个 JSON 语义请求，不计算或表述答案。家庭事实、身份和姓名问题 requires_fact=true；只有普通闲聊才是 false 且 request=null。

引用语法：
- self 是已认证的当前说话人，assistant 是本助手，二者 entity_type=person、value=null。用户对助手说“你”并询问身份、名字或称呼时必须引用 assistant，仍是事实请求。current_household 是配置的家庭，entity_type=address、value=null。named_entity.value 只能逐字复制用户说出的姓名或称呼；不得猜测 ID 或把亲属短语当姓名。
- path 是概念的有序组合，每步 {"concept":"名称"}，可附加 filters。reference_concepts 是权威定义；完整短语匹配某个 aliases 时，path 只放该最具体概念一次，不拆解或追加近义概念。son 不能简化为 child，wife 不能简化为 spouse，father 不能简化为 parent。本体会完整展开概念的关系和过滤条件。
- 只有嵌套亲属才增加一跳。性别等形容词约束同一个目标，不增加一跳；额外 filters 与概念原有条件取 AND。说话人的亲属始终从 self 开始，“我家的儿子”也不先遍历全家。
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
    """Illustrate reusable grammar without evaluation wording or household facts."""
    def reference(kind: str, *concepts: str) -> dict[str, Any]:
        return {"kind": kind, "value": None,
                "entity_type": "address" if kind == "current_household" else "person",
                **({"path": [{"concept": name} for name in concepts]} if concepts else {})}

    examples = (
        ("应该怎样称呼这位助手？", "resolve_reference", reference("assistant"), None, "entity", {}),
        ("在本户成员中找出出生日期最靠前的人。", "argmin", reference("current_household", "member"), "birth_date", "entity", {}),
        ("本户符合成年条件的成员有多少？", "count", reference("current_household", "member"), None, "entity", {"filters": [{"predicate": "adult"}]}),
        ("本户未成年成员有多少？", "count", reference("current_household", "member"), None, "entity", {"filters": [{"predicate": "minor"}]}),
        ("我的男孩后代是哪一位？", "resolve_reference", reference("self", "son"), None, "entity", {}),
        ("请列出我的女性后代。", "select", reference("self", "daughter"), None, "entity", {}),
        ("我母亲的丈夫是哪位？", "resolve_reference", reference("self", "mother", "husband"), None, "entity", {}),
        ("我的丈夫出生于哪天？", "select", reference("self", "husband"), "birth_date", "entity", {}),
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
        self.last_planner_runtime: dict[str, Any] = {}
        self._cached_planner_system: str | None = None
        self._cached_planner_capabilities_id: int | None = None

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

    def _planner_system_prompt(self, capabilities: Mapping[str, Any]) -> str:
        marker = id(capabilities)
        if (
            self._cached_planner_system is not None
            and self._cached_planner_capabilities_id == marker
        ):
            return self._cached_planner_system
        prompt = (
            _PLANNER_INSTRUCTIONS
            + "\nCapabilities:\n"
            + json.dumps(capabilities, ensure_ascii=False, separators=(",", ":"))
        )
        self._cached_planner_system = prompt
        self._cached_planner_capabilities_id = marker
        return prompt

    async def plan_semantic_fact(
        self,
        messages: Sequence[Mapping[str, Any]],
        capabilities: Mapping[str, Any],
        output_schema: Mapping[str, Any],
        *,
        household_now: str,
    ) -> Mapping[str, Any]:
        """Interpret an open-ended request without exposing physical storage."""
        forwarded: list[dict[str, Any]] = [
            {"role": "user", "content": latest_user_message(messages)}
        ]
        validation_feedback = "\n".join(
            str(message.get("content", "")) for message in messages
            if message.get("role") == "system"
            and "strict structural" in str(message.get("content", ""))
        )
        response = await self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": (
                    self._planner_system_prompt(capabilities)
                    + f"\nHousehold now: {household_now}"
                    + ("\n" + validation_feedback if validation_feedback else "")
                )},
                *_semantic_planner_examples(),
                *forwarded,
            ],
            stream=False,
            think=False,
            keep_alive=PLANNER_KEEP_ALIVE,
            format=dict(output_schema),
            options={
                "temperature": 0,
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
