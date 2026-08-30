"""Shared household-request analysis for facts and the model loop."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from .memorable_dates import MemorableDateRegistry
from .text import latest_user_message

if TYPE_CHECKING:
    from .facts import FactRequest

PRIVATE_TOOL_FIELDS = {
    "dob": "dob",
    "address": "address",
    "start": "relationship_dates",
    "end": "relationship_dates",
    "email": "contact",
    "phone": "contact",
    "phone_number": "contact",
}


@dataclass(frozen=True)
class EvidenceRequirements:
    """Independent tool, relationship, and field evidence needed for an answer."""

    tools: frozenset[str] = frozenset()
    relations: frozenset[str] = frozenset()
    fields: frozenset[tuple[str, str]] = frozenset()
    related_gender: str | None = None
    relationship_direction: Literal["out", "in"] | None = None
    minimum_entity_records: int = 1


@dataclass(frozen=True)
class RequestAnalysis:
    """One request interpretation shared by FactService and ModelLoop."""

    text: str
    private_fields: frozenset[str]
    evidence_required: bool
    evidence: EvidenceRequirements
    fact_request: FactRequest | None


def analyze_household_request(
    messages: Sequence[Mapping[str, Any]],
    *,
    memorable_dates: MemorableDateRegistry,
    identity: Mapping[str, Any] | None = None,
) -> RequestAnalysis:
    """Compute household intent, evidence, and privacy requirements once."""
    from .facts import parse_fact_request

    text = latest_user_message(messages)
    fact_request = parse_fact_request(
        messages,
        identity=identity,
        memorable_dates=memorable_dates,
    )
    evidence_required = requires_graph_evidence(messages, memorable_dates)
    return RequestAnalysis(
        text=text,
        private_fields=requested_private_fields(messages, memorable_dates),
        evidence_required=evidence_required,
        evidence=evidence_requirements(
            messages,
            evidence_required,
            memorable_dates,
        ),
        fact_request=fact_request,
    )


def requires_graph_evidence(
    messages: Sequence[Mapping[str, Any]],
    memorable_dates: MemorableDateRegistry,
) -> bool:
    latest_user = latest_user_message(messages)
    normalized = latest_user.casefold()
    if required_evidence_tool(messages, memorable_dates) is None:
        return False

    # Relationship words can appear in ordinary conversation without asking
    # for a stored household fact. Require graph evidence only for lookup intent.
    non_lookup_intent = (
        r"\b(?:advice|chat|feel|feeling|gift|joke|opinion|recommend|story|"
        r"suggest|talk|think|add|decorate|design|remodel|should|enough|"
        r"adequate|comfortable|crowded|spacious|suitable|fit)\b|"
        r"建议|推荐|礼物|聊|笑话|故事|觉得|认为|心情|装修|设计|改造|增加|"
        r"够|合适|适合|舒服|舒适|拥挤|宽敞|住得下|住不下|住得开|"
        r"住不开|好不好|怎么样|如何"
    )
    if re.search(non_lookup_intent, normalized):
        return False

    lookup_intent = (
        r"\b(?:find|identify|list|search|show|tell me|what|when|where|which|"
        r"who|whose|how many|how old)\b|谁|什么|哪|何时|什么时候|多少|几岁|"
        r"几个|几间|是否|查|找|告诉我|列出|显示"
    )
    if re.search(lookup_intent, normalized):
        return True

    # Direct yes/no requests for stored relationship predicates also need
    # evidence, while plain statements mentioning those predicates do not.
    yes_no_predicate = (
        r"(?:\b(?:is|are|was|were|do|does|did)\b.*\b(?:live|lives|living|"
        r"reside|resides|married)\b)|(?:住|居住|结婚|已婚).*吗[？?]?$|"
        r"是否.*(?:住|居住|结婚|已婚)"
    )
    return re.search(yes_no_predicate, normalized.strip()) is not None


def evidence_requirements(
    messages: Sequence[Mapping[str, Any]],
    evidence_required: bool,
    memorable_dates: MemorableDateRegistry,
) -> EvidenceRequirements:
    if not evidence_required:
        return EvidenceRequirements()

    primary_tool = required_evidence_tool(messages, memorable_dates)
    relation = required_evidence_relation(messages, memorable_dates)
    field = required_evidence_field(messages, memorable_dates)
    tools: set[str] = set()
    relations: set[str] = set()
    fields: set[tuple[str, str]] = set()

    if relation is not None:
        tools.add("get_relationships")
        relations.add(relation)
    if field is not None and primary_tool is not None:
        tools.add(primary_tool)
        fields.add((primary_tool, field))
    if not tools and primary_tool is not None:
        tools.add(primary_tool)

    return EvidenceRequirements(
        tools=frozenset(tools),
        relations=frozenset(relations),
        fields=frozenset(fields),
        related_gender=required_related_gender(messages),
        relationship_direction=required_relationship_direction(messages),
        minimum_entity_records=required_entity_record_count(messages),
    )


def requested_private_fields(
    messages: Sequence[Mapping[str, Any]],
    memorable_dates: MemorableDateRegistry,
) -> frozenset[str]:
    latest_user = latest_user_message(messages).casefold()
    allowed: set[str] = set()
    schema = memorable_dates.match(latest_user)
    if schema is not None:
        private_field = PRIVATE_TOOL_FIELDS.get(schema.source_field)
        if private_field is not None:
            allowed.add(private_field)
    if _contains_any(
        latest_user,
        ("address", "street address", "地址", "住址"),
    ):
        allowed.add("address")
    if _contains_any(
        latest_user,
        ("move-in date", "when did", "什么时候搬"),
    ):
        allowed.add("relationship_dates")
    if _contains_any(
        latest_user,
        ("email", "phone", "telephone", "邮箱", "电话"),
    ):
        allowed.add("contact")
    return frozenset(allowed)


def is_household_roster_request(
    messages: Sequence[Mapping[str, Any]],
    memorable_dates: MemorableDateRegistry,
) -> bool:
    """Recognize resident-roster requests for generic evidence enforcement."""
    latest_user = latest_user_message(messages).casefold()
    if _contains_any(
        latest_user,
        memorable_dates.aliases
        + (
            "parent",
            "child",
            "daughter",
            "son",
            "spouse",
            "wife",
            "husband",
            "父母",
            "父亲",
            "母亲",
            "孩子",
            "儿子",
            "女儿",
            "配偶",
            "妻子",
            "丈夫",
        ),
    ):
        return False

    chinese_home = r"(?:家里|家中|家里面|家里边|这个家|这里)"
    chinese_people = r"(?:谁|哪些人|什么人|成员|住户)"
    if re.search(
        rf"(?:{chinese_home}.*{chinese_people}|"
        rf"{chinese_people}.*(?:住|居住|待在).*{chinese_home})",
        latest_user,
    ):
        return True
    if re.search(
        rf"{chinese_home}.*(?:多少|几)(?:个|位)?(?:人|住户|居民)",
        latest_user,
    ):
        return True

    patterns = (
        r"\bwho\b.*\b(?:live|lives|living|reside|resides|stays?)\b.*"
        r"\b(?:home|house|household|here)\b",
        r"\b(?:household|home|house)\s+(?:members|residents|occupants)\b",
        r"\bwho\b.*\b(?:in|at)\b.*\b(?:my|our|the)\s+household\b",
        r"\bhow many\b.*\b(?:people|residents|occupants)\b.*"
        r"\b(?:my|our|the)\s+(?:home|house|household)\b",
    )
    return any(re.search(pattern, latest_user) for pattern in patterns)


def is_household_space_request(
    messages: Sequence[Mapping[str, Any]],
) -> bool:
    """Recognize home room/space lookups for generic evidence enforcement."""
    latest_user = latest_user_message(messages).casefold()
    chinese = re.search(
        r"(?:家里|家中|家里面|家里边|这个家).*(?:房间|空间|区域)|"
        r"(?:房间|空间|区域).*(?:家里|家中|这个家)",
        latest_user,
    )
    english = re.search(
        r"\b(?:room|rooms|space|spaces)\b.*\b(?:my|our|the)\s+"
        r"(?:home|house)\b|"
        r"\b(?:my|our|the)\s+(?:home|house)\b.*"
        r"\b(?:room|rooms|space|spaces)\b",
        latest_user,
    )
    return chinese is not None or english is not None


def is_item_location_request(
    messages: Sequence[Mapping[str, Any]],
) -> bool:
    """Recognize named-entity location lookups for evidence enforcement."""
    latest_user = latest_user_message(messages).casefold().strip()
    if re.fullmatch(
        r"(?:where is|where's)\s+(?:(?:my|our|the)\s+)?"
        r"(?:home|house|household|here)[?]?",
        latest_user,
    ) or re.fullmatch(
        r"(?:我家|家里|家中|这里|这儿)(?:在|位于)?"
        r"(?:哪里|哪儿|什么地方)[?？。！!]?",
        latest_user,
    ):
        return False
    chinese = re.fullmatch(
        r"(?:请问|麻烦告诉我)?\s*.+?\s*(?:现在)?(?:在|位于)"
        r"(?:哪里|哪儿|什么地方|哪个房间|哪间房间?)[?？。！!]?",
        latest_user,
    )
    english = re.fullmatch(
        r"(?:(?:where is|where's)\s+.+?|"
        r"(?:which|what) room is\s+.+?\s+in)[?]?",
        latest_user,
    )
    return chinese is not None or english is not None


def required_evidence_tool(
    messages: Sequence[Mapping[str, Any]],
    memorable_dates: MemorableDateRegistry,
) -> str | None:
    if is_household_roster_request(messages, memorable_dates):
        return "get_relationships"
    if is_household_space_request(messages):
        return "get_relationships"
    if is_item_location_request(messages):
        return "get_relationships"
    latest_user = latest_user_message(messages).casefold()
    date_schema = memorable_dates.match(latest_user)
    if date_schema is not None:
        return (
            "get_entity"
            if date_schema.source_kind == "node"
            else "get_relationships"
        )
    relationship_fields = (
        "live",
        "lives",
        "living",
        "reside",
        "resides",
        "spouse",
        "wife",
        "husband",
        "married",
        "parent",
        "child",
        "daughter",
        "son",
        "household",
        "住",
        "家里有谁",
        "家中有谁",
        "配偶",
        "妻子",
        "丈夫",
        "老婆",
        "老公",
        "父母",
        "孩子",
        "女儿",
        "儿子",
        "结婚",
    )
    if _contains_any(latest_user, relationship_fields):
        return "get_relationships"
    return None


def required_evidence_relation(
    messages: Sequence[Mapping[str, Any]],
    memorable_dates: MemorableDateRegistry,
) -> str | None:
    if is_household_roster_request(messages, memorable_dates):
        return "lives_in"
    if is_household_space_request(messages):
        return "hosts_space"
    if is_item_location_request(messages):
        return "located_in"
    latest_user = latest_user_message(messages).casefold()
    date_schema = memorable_dates.match(latest_user)
    if date_schema is not None and date_schema.source_kind == "edge":
        return date_schema.source_type
    parent_terms = (
        "parent",
        "child",
        "daughter",
        "son",
        "父母",
        "父亲",
        "母亲",
        "爸爸",
        "妈妈",
        "孩子",
        "女儿",
        "儿子",
    )
    spouse_terms = (
        "spouse",
        "wife",
        "husband",
        "married",
        "anniversary",
        "配偶",
        "妻子",
        "丈夫",
        "太太",
        "老婆",
        "老公",
        "结婚",
        "纪念日",
    )
    residence_terms = (
        "live",
        "lives",
        "living",
        "reside",
        "resides",
        "household",
        "住",
        "家里有谁",
        "家中有谁",
    )
    if _contains_any(latest_user, parent_terms):
        return "parent_of"
    if _contains_any(latest_user, spouse_terms):
        return "spouse_of"
    if _contains_any(latest_user, residence_terms):
        return "lives_in"
    return None


def required_related_gender(
    messages: Sequence[Mapping[str, Any]],
) -> str | None:
    latest_user = latest_user_message(messages).casefold()
    if _contains_any(
        latest_user,
        (
            "daughter",
            "mother",
            "wife",
            "女儿",
            "母亲",
            "妈妈",
            "妻子",
            "太太",
            "老婆",
        ),
    ):
        return "female"
    if _contains_any(
        latest_user,
        (
            "son",
            "father",
            "husband",
            "儿子",
            "父亲",
            "爸爸",
            "丈夫",
            "老公",
        ),
    ):
        return "male"
    return None


def required_relationship_direction(
    messages: Sequence[Mapping[str, Any]],
) -> Literal["out", "in"] | None:
    latest_user = latest_user_message(messages).casefold()
    if _contains_any(latest_user, ("child", "daughter", "son", "孩子", "女儿", "儿子")):
        return "out"
    if _contains_any(
        latest_user,
        ("parent", "father", "mother", "父母", "父亲", "母亲", "爸爸", "妈妈"),
    ):
        return "in"
    return None


def required_entity_record_count(
    messages: Sequence[Mapping[str, Any]],
) -> int:
    latest_user = latest_user_message(messages).casefold()
    if _contains_any(
        latest_user,
        ("both", "children", "their", "them", "they", "他们", "她们", "孩子们"),
    ):
        return 2
    return 1


def required_evidence_field(
    messages: Sequence[Mapping[str, Any]],
    memorable_dates: MemorableDateRegistry,
) -> str | None:
    latest_user = latest_user_message(messages).casefold()
    schema = memorable_dates.match(latest_user)
    return schema.source_field if schema is not None else None


def _contains_any(text: str, terms: Sequence[str]) -> bool:
    for term in terms:
        if term.isascii():
            if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text):
                return True
        elif term in text:
            return True
    return False
