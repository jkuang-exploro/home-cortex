"""Shared household-request analysis for facts and the model loop."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from .memorable_dates import MemorableDateRegistry, default_memorable_date_registry
from .text import latest_user_message

PRIVATE_TOOL_FIELDS = {
    "dob": "dob",
    "address": "address",
    "start": "relationship_dates",
    "end": "relationship_dates",
    "email": "contact",
    "phone": "contact",
    "phone_number": "contact",
}

FactField = Literal[
    "address",
    "identity",
    "items",
    "relationship_to_speaker",
    "relationship_exists",
    "count",
    "memorable_date",
    "location",
    "residents",
    "spaces",
]
SubjectKind = Literal["speaker", "home", "item", "named", "relative", "space"]
DateQuery = Literal["stored", "next"]


@dataclass(frozen=True)
class SubjectReference:
    kind: SubjectKind
    value: str | tuple[str, ...] | None = None


@dataclass(frozen=True)
class FactRequest:
    subject: SubjectReference
    field: FactField
    cardinality: Literal["one", "all"] = "one"
    memorable_date: str | None = None
    date_query: DateQuery = "stored"
    relation: str | None = None
    target: SubjectReference | None = None
    space_type: str | None = None


@dataclass(frozen=True)
class RelationshipStep:
    relation: str
    direction: Literal["out", "in", "both"] | None = None
    gender: Literal["female", "male"] | None = None


@dataclass(frozen=True)
class RelativeDefinition:
    steps: tuple[RelationshipStep, ...]


RELATIVES: Mapping[str, RelativeDefinition] = {
    "spouse": RelativeDefinition((RelationshipStep("spouse_of"),)),
    "wife": RelativeDefinition(
        (RelationshipStep("spouse_of", gender="female"),)
    ),
    "husband": RelativeDefinition(
        (RelationshipStep("spouse_of", gender="male"),)
    ),
    "children": RelativeDefinition(
        (RelationshipStep("parent_of", direction="out"),)
    ),
    "son": RelativeDefinition(
        (RelationshipStep("parent_of", direction="out", gender="male"),)
    ),
    "daughter": RelativeDefinition(
        (RelationshipStep("parent_of", direction="out", gender="female"),)
    ),
    "parents": RelativeDefinition(
        (RelationshipStep("parent_of", direction="in"),)
    ),
    "father": RelativeDefinition(
        (RelationshipStep("parent_of", direction="in", gender="male"),)
    ),
    "mother": RelativeDefinition(
        (RelationshipStep("parent_of", direction="in", gender="female"),)
    ),
    "father_in_law": RelativeDefinition(
        (
            RelationshipStep("spouse_of", gender="female"),
            RelationshipStep("parent_of", direction="in", gender="male"),
        )
    ),
    "mother_in_law": RelativeDefinition(
        (
            RelationshipStep("spouse_of", gender="female"),
            RelationshipStep("parent_of", direction="in", gender="female"),
        )
    ),
}

# These are vocabulary aliases, not memorized questions. The longest terms are
# checked first so a compound relationship such as father-in-law is not reduced
# to the more general father relationship.
_RELATIVE_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("father_in_law", ("father-in-law", "father in law", "岳父")),
    ("mother_in_law", ("mother-in-law", "mother in law", "岳母")),
    ("daughter", ("daughters", "daughter", "女儿")),
    ("son", ("sons", "son", "儿子")),
    ("children", ("children", "child", "孩子们", "孩子")),
    ("mother", ("mothers", "mother", "moms", "mom", "mums", "mum", "母亲", "妈妈")),
    ("father", ("fathers", "father", "dads", "dad", "父亲", "爸爸")),
    ("parents", ("parents", "parent", "父母")),
    ("wife", ("wives", "wife", "太太", "妻子", "老婆")),
    ("husband", ("husbands", "husband", "丈夫", "老公")),
    ("spouse", ("spouses", "spouse", "配偶")),
)
_PLURAL_RELATIVE_TERMS = (
    "daughters",
    "sons",
    "children",
    "mothers",
    "moms",
    "mums",
    "fathers",
    "dads",
    "parents",
    "wives",
    "husbands",
    "spouses",
    "孩子们",
)

_NEXT_OCCURRENCE_PATTERNS = (
    r"\b(?:how many days|days)\s+(?:are\s+)?(?:left\s+)?until\b",
    r"\bhow long until\b",
    r"(?:还有|再过)(?:多少|几)天",
    r"(?:还有多久|多久以后)",
)
_LOOKUP_TERMS = (
    "who",
    "which",
    "list",
    "identify",
    "谁",
    "哪位",
    "哪些",
    "列出",
)
_COUNT_TERMS = ("how many", "count", "几个", "几位", "多少")
_RESIDENCE_TERMS = (
    "live in",
    "lives in",
    "living in",
    "live at",
    "lives at",
    "reside in",
    "resides in",
    "住",
    "住在",
    "居住在",
)
_HOME_REFERENCE_TERMS = (
    "my home",
    "our home",
    "my house",
    "our house",
    "my household",
    "our household",
    "我家",
    "我们家",
    "家里",
    "家中",
)
_ROOM_TERMS = ("room", "rooms", "房间")
_SPACE_TERMS = _ROOM_TERMS + ("space", "spaces", "区域", "空间")
_SPACE_LOOKUP_TERMS = _COUNT_TERMS + (
    "what",
    "which",
    "list",
    "show",
    "哪些",
    "有什么",
    "列出",
    "显示",
)
_SPACE_NON_LOOKUP_TERMS = (
    "add",
    "decorate",
    "design",
    "recommend",
    "remodel",
    "should",
    "建议",
    "推荐",
    "装修",
    "设计",
    "改造",
    "增加",
)


def _contains_any(text: str, terms: Sequence[str]) -> bool:
    return any(term in text for term in terms)


def entity_aliases(entity: Mapping[str, Any]) -> tuple[str, ...]:
    name = entity.get("name")
    if isinstance(name, str):
        return (name.strip(),) if name.strip() else ()
    if isinstance(name, Mapping):
        return tuple(
            value.strip()
            for value in name.values()
            if isinstance(value, str) and value.strip()
        )
    if isinstance(name, Sequence) and not isinstance(
        name,
        (str, bytes, bytearray),
    ):
        return tuple(
            value.strip()
            for value in name
            if isinstance(value, str) and value.strip()
        )
    return ()


def parse_fact_request(
    messages: Sequence[Mapping[str, Any]],
    *,
    identity: Mapping[str, Any] | None,
    memorable_dates: MemorableDateRegistry | None = None,
) -> FactRequest | None:
    """Reduce common fact language to semantics without memorizing sentences."""
    registry = memorable_dates or default_memorable_date_registry()
    text = latest_user_message(messages).strip()
    normalized = text.casefold()
    date_schema = registry.match(normalized)
    date_query: DateQuery = (
        "next" if _is_next_occurrence(normalized) else "stored"
    )

    # Classify the more specific household and kinship concepts before the
    # generic first-person identity concept.  For example, ``我女儿是谁``
    # contains both "my" and "who", but its subject is the daughter.
    space_inventory = _space_inventory_request(normalized)
    if space_inventory is not None:
        space_name, field, space_type = space_inventory
        return FactRequest(
            SubjectReference("space", space_name),
            field,
            "all",
            relation="located_in" if field == "items" else "hosts_space",
            space_type=space_type,
        )

    if _is_home_address_request(normalized) or (
        _contains_any(normalized, ("address", "地址", "住址"))
        and _previous_home_subject(messages, identity, registry)
    ):
        return FactRequest(SubjectReference("home"), "address")

    item_name = _item_location_subject(normalized)
    if item_name is not None:
        return FactRequest(
            SubjectReference("item", item_name),
            "location",
            relation="located_in",
        )

    if _is_home_identity_request(normalized):
        return FactRequest(SubjectReference("home"), "identity")

    is_home_space_request, home_space_type = _home_space_request(normalized)
    if not is_home_space_request:
        is_home_space_request, home_space_type = _elliptical_home_space_request(
            normalized
        )
    if is_home_space_request:
        field: FactField = (
            "count" if _contains_word(normalized, _COUNT_TERMS) else "spaces"
        )
        return FactRequest(
            SubjectReference("home"),
            field,
            "all",
            relation="hosts_space",
            space_type=home_space_type,
        )

    if _is_home_roster_count_request(normalized) or (
        _is_elliptical_roster_count_request(normalized)
        and _previous_home_subject(messages, identity, registry)
    ):
        return FactRequest(
            SubjectReference("home"),
            "count",
            "all",
            relation="lives_in",
        )

    if _is_roster_request(normalized, registry.aliases):
        return FactRequest(SubjectReference("home"), "residents", "all")

    relative = _relative_kind(normalized)
    if relative is not None and identity is not None:
        if date_schema is not None:
            date_relative = (
                "spouse" if date_schema.source_kind == "edge" else relative
            )
            return FactRequest(
                SubjectReference("relative", date_relative),
                "memorable_date",
                memorable_date=date_schema.id,
                date_query=date_query,
            )
        if _is_home_residence_check(normalized):
            return FactRequest(
                SubjectReference("relative", relative),
                "relationship_exists",
                relation="lives_in",
                target=SubjectReference("home"),
            )
        if _contains_word(normalized, _COUNT_TERMS):
            return FactRequest(SubjectReference("relative", relative), "count", "all")
        if _contains_word(normalized, _LOOKUP_TERMS):
            return FactRequest(
                SubjectReference("relative", relative),
                "identity",
                "all"
                if relative in {"children", "parents"}
                or _contains_word(normalized, _PLURAL_RELATIVE_TERMS)
                else "one",
            )

    if (
        identity is not None
        and date_schema is not None
        and date_schema.source_kind == "edge"
    ):
        return FactRequest(
            SubjectReference("relative", "spouse"),
            "memorable_date",
            memorable_date=date_schema.id,
            date_query=date_query,
        )

    if _is_speaker_identity(normalized, identity):
        return FactRequest(SubjectReference("speaker"), "identity")

    # A named-person fact is meaningful here only when it can be related to a
    # trusted caller identity.  Without one, let the ordinary guarded model
    # loop handle the conversation instead of guessing the user's perspective.
    names = _named_subjects(text) if identity is not None else ()
    if names:
        return FactRequest(
            SubjectReference("named", names),
            "relationship_to_speaker",
            "all" if len(names) > 1 else "one",
        )

    # Resolve an elliptical factual follow-up from the last explicit semantic
    # subject in the conversation.  This supports pronouns such as "他们" and
    # short follow-ups such as "生日呢" without asking the model to remember a
    # special sentence.
    if identity is not None and date_schema is not None:
        inherited = _previous_fact_subject(messages, identity, registry)
        if inherited is not None:
            return FactRequest(
                inherited,
                "memorable_date",
                "all",
                memorable_date=date_schema.id,
                date_query=date_query,
            )
    return None


def _is_next_occurrence(text: str) -> bool:
    return any(re.search(pattern, text) for pattern in _NEXT_OCCURRENCE_PATTERNS)


def _is_home_address_request(text: str) -> bool:
    chinese = bool(
        re.fullmatch(
            r"(?:我们?家|这个家|家里|家中|^家的?)"
            r"(?:的)?(?:"
            r"(?:住在|位于|在)?(?:哪里|哪儿|哪|在哪)|"
            r"(?:位置|地址|住址)(?:是|在|位于)?"
            r"(?:哪里|哪儿|哪)?"
            r")"
            r"[?？。！!]?",
            text,
        )
    )
    english = bool(
        re.search(r"\b(?:my|our|the)\s+(?:home|house)\b", text)
        and _contains_any(text, ("address", "street address"))
    )
    return chinese or english


def _is_home_identity_request(text: str) -> bool:
    return bool(
        re.fullmatch(
            r"(?:这|这里|这儿|这个地方)(?:是)?(?:哪里|哪儿|哪)[?？。！!]?",
            text,
        )
    )


def _home_space_request(text: str) -> tuple[bool, str | None]:
    home_terms = _HOME_REFERENCE_TERMS + ("the home", "the house", "这个家")
    if not _contains_word(text, home_terms) or not _contains_word(text, _SPACE_TERMS):
        return False, None
    if _contains_word(text, _SPACE_NON_LOOKUP_TERMS):
        return False, None
    if not _contains_word(text, _SPACE_LOOKUP_TERMS):
        return False, None
    if _contains_word(text, _ROOM_TERMS):
        return True, "room"
    return True, None


def _elliptical_home_space_request(text: str) -> tuple[bool, str | None]:
    """Resolve a bare room/space lookup against the configured home."""
    chinese = re.fullmatch(
        r"(?:一共|总共)?(?:有)?(?:哪些|什么|多少|几(?:个|间)?)"
        r"(?:房间|空间|区域)[?？。！!]?",
        text,
    )
    if chinese is not None:
        field_type = "room" if "房间" in text else None
        return True, field_type
    english = re.fullmatch(
        r"(?:what|which|how many) (?:rooms|spaces)(?: are there)?[?]?",
        text,
    )
    if english is not None:
        return True, "room" if "room" in text else None
    return False, None


def _is_home_roster_count_request(text: str) -> bool:
    chinese_home = r"(?:家里|家中|家里面|家里边|这个家|这里)"
    chinese_count = r"(?:多少|几)(?:个|位)?(?:人|住户|居民)"
    if re.search(rf"{chinese_home}.*{chinese_count}", text):
        return True
    if re.search(
        rf"{chinese_count}.*(?:住|居住|待在).*(?:{chinese_home})",
        text,
    ):
        return True
    return bool(
        re.search(r"\bhow many\b.*\b(?:people|residents|occupants)\b", text)
        and re.search(r"\b(?:my|our|the)\s+(?:home|house|household)\b", text)
    )


def _is_elliptical_roster_count_request(text: str) -> bool:
    return bool(
        re.fullmatch(
            r"(?:一共|总共)?(?:有)?(?:多少|几)(?:个|位)?人[?？。！!]?",
            text,
        )
        or re.fullmatch(
            r"how many (?:people|residents|occupants)(?: are there)?[?]?",
            text,
        )
    )


def _is_home_adequacy_request(text: str) -> bool:
    """Identify home judgments that benefit from facts but need model reasoning."""
    chinese_direct = re.search(
        r"(?:够不够住|够住|住得下|住不下|住得开|住不开|住得舒服|"
        r"住着舒服|住.*(?:拥挤|宽敞|合适|适合))",
        text,
    )
    chinese_home = re.search(
        r"(?:家里|家中|我家|我们家|房子|住宅|房间|空间)",
        text,
    )
    chinese_evaluation = re.search(
        r"(?:够不够|够|合适|适合|舒服|舒适|拥挤|宽敞|"
        r"大不大|大吗|小吗)",
        text,
    )
    english = re.search(
        r"\b(?:enough|adequate|comfortable|crowded|spacious|suitable|fit)\b",
        text,
    ) and re.search(r"\b(?:home|house|room|rooms|live|living)\b", text)
    return (
        chinese_direct is not None
        or (chinese_home is not None and chinese_evaluation is not None)
        or bool(english)
    )


def _item_location_subject(text: str) -> str | None:
    chinese = re.fullmatch(
        r"(?:请问|麻烦告诉我)?\s*(?P<name>.+?)\s*"
        r"(?:现在)?(?:在|位于)(?:哪里|哪儿|什么地方|哪个房间|哪间房间?)"
        r"[?？。！!]?",
        text,
    )
    if chinese is not None:
        name = re.sub(
            r"^(?:我家(?:的)?|家里(?:的)?|我的)",
            "",
            chinese.group("name").strip(),
        )
        if name in {
            "家",
            "家里",
            "家中",
            "这个家",
            "这里",
            "这儿",
            "地址",
            "住址",
            "位置",
        }:
            return None
        if _is_relative_alias(name):
            return None
        return name or None

    patterns = (
        r"(?:where is|where's)\s+(?:the\s+|my\s+|our\s+)?"
        r"(?P<name>.+?)[?]?",
        r"(?:which|what) room is\s+(?:the\s+|my\s+|our\s+)?"
        r"(?P<name>.+?)\s+in[?]?",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, text)
        if match is not None:
            name = match.group("name").strip()
            if name in {"home", "house", "household", "here"}:
                return None
            if _is_relative_alias(name):
                return None
            return name or None
    return None


def _space_inventory_request(
    text: str,
) -> tuple[str, Literal["items", "spaces"], str | None] | None:
    chinese = re.fullmatch(
        r"(?:请问)?\s*(?:家里(?:的)?)?(?P<name>.+?)"
        r"(?:里面|里|内)\s*(?:又)?(?P<query>.+?)[?？。！!]?",
        text,
    )
    if chinese is not None:
        name = chinese.group("name").strip()
        query = chinese.group("query").strip()
        if _contains_any(query, ("空间", "区域", "地方")):
            space_type = (
                "storage"
                if _contains_any(query, ("放东西", "储物", "存储", "收纳"))
                else None
            )
            return name, "spaces", space_type
        if _contains_any(query, ("物品", "东西", "设备", "家电")):
            return name, "items", None
        return None

    english = re.fullmatch(
        r"(?:what|which)\s+(?P<kind>items|things|appliances|"
        r"storage spaces|spaces)\s+(?:are\s+)?(?:in|inside)\s+"
        r"(?:the\s+)?(?P<name>.+?)[?]?",
        text,
    )
    if english is None:
        return None
    kind = english.group("kind")
    field: Literal["items", "spaces"] = (
        "spaces" if "spaces" in kind else "items"
    )
    space_type = "storage" if kind == "storage spaces" else None
    return english.group("name").strip(), field, space_type


def _previous_home_subject(
    messages: Sequence[Mapping[str, Any]],
    identity: Mapping[str, Any] | None,
    memorable_dates: MemorableDateRegistry,
) -> bool:
    latest_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if messages[index].get("role") == "user"
        ),
        -1,
    )
    if latest_index <= 0:
        return False
    previous = parse_fact_request(
        messages[:latest_index],
        identity=identity,
        memorable_dates=memorable_dates,
    )
    return previous is not None and previous.subject.kind == "home"


def _is_home_residence_check(text: str) -> bool:
    if not (
        _contains_any(text, _RESIDENCE_TERMS)
        and _contains_any(text, _HOME_REFERENCE_TERMS)
    ):
        return False
    return bool(
        re.search(r"(?:吗[？?]?$|是否|是不是|有没有)", text)
        or re.match(r"\s*(?:do|does|is|are)\b", text)
    )


def _previous_fact_subject(
    messages: Sequence[Mapping[str, Any]],
    identity: Mapping[str, Any],
    memorable_dates: MemorableDateRegistry,
) -> SubjectReference | None:
    latest_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if messages[index].get("role") == "user"
        ),
        -1,
    )
    if latest_index <= 0:
        return None
    previous = parse_fact_request(
        messages[:latest_index],
        identity=identity,
        memorable_dates=memorable_dates,
    )
    if previous is None or previous.subject.kind not in {"named", "relative"}:
        return None
    return previous.subject


def _relative_kind(text: str) -> str | None:
    for kind, aliases in _RELATIVE_ALIASES:
        if _contains_word(text, aliases):
            return kind
    return None


def _is_relative_alias(text: str) -> bool:
    return _contains_word(
        text,
        tuple(alias for _, aliases in _RELATIVE_ALIASES for alias in aliases),
    )


def _speaker_relative_reference(text: str) -> bool:
    if _relative_kind(text) is None:
        return False
    return (
        re.search(r"(?:^|\W)(?:my|our)(?:\W|$)", text) is not None
        or re.search(r"我(?:的)?", text) is not None
    )


def _is_speaker_identity(
    text: str,
    identity: Mapping[str, Any] | None,
) -> bool:
    if (
        identity is None
        or not entity_aliases(identity)
        or not _contains_word(text, _LOOKUP_TERMS + ("身份", "名字", "叫什么"))
    ):
        return False
    references_self = bool(
        re.search(r"(?:^|\W)(?:i|me|myself)(?:$|\W)", text)
        or re.search(r"我|本人|自己", text)
    )
    if references_self:
        return True
    return any(alias.casefold() in text for alias in entity_aliases(identity))


def _is_roster_request(text: str, date_aliases: Sequence[str]) -> bool:
    if _is_relative_alias(text) or _contains_word(text, tuple(date_aliases)):
        return False
    chinese_home = r"(?:家里|家中|家里面|家里边|这个家|这里)"
    chinese_people = r"(?:谁|哪些人|什么人|成员|住户)"
    if re.search(
        rf"(?:{chinese_home}.*{chinese_people}|"
        rf"{chinese_people}.*(?:住|居住|待在).*{chinese_home})",
        text,
    ):
        return True
    patterns = (
        r"\bwho\b.*\b(?:live|lives|living|reside|resides|stays?)\b.*"
        r"\b(?:home|house|household|here)\b",
        r"\b(?:household|home|house)\s+(?:members|residents|occupants)\b",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def _named_subjects(text: str) -> tuple[str, ...]:
    chinese = re.fullmatch(
        r"(?:请问|麻烦告诉我)?\s*(?P<names>.+?)\s*是谁[？?]?",
        text,
    )
    if chinese is not None:
        raw = re.split(r"\s*(?:和|与|以及|、)\s*", chinese.group("names"))
    else:
        english = re.fullmatch(
            r"(?:please\s+)?(?:tell me\s+)?who (?:is|are)\s+"
            r"(?P<names>.+?)[?]?",
            text,
            re.IGNORECASE,
        )
        if english is None:
            return ()
        raw = re.split(
            r"\s+(?:and|&)\s+",
            english.group("names"),
            flags=re.IGNORECASE,
        )

    excluded = {
        "i",
        "me",
        "you",
        "he",
        "she",
        "this person",
        "that person",
        "someone",
        "我",
        "你",
        "您",
        "他",
        "她",
        "这个人",
        "那个人",
        "某人",
    }
    names: list[str] = []
    seen: set[str] = set()
    for value in raw:
        name = re.sub(
            r"(?:先生|女士|小姐|太太|夫人|mr\.?|mrs\.?|ms\.?)$",
            "",
            value,
            flags=re.IGNORECASE,
        ).strip()
        normalized = name.casefold()
        if name and normalized not in excluded and normalized not in seen:
            seen.add(normalized)
            names.append(name)
    return tuple(names)


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
    text = latest_user_message(messages)
    fact_request = parse_fact_request(
        messages,
        identity=identity,
        memorable_dates=memorable_dates,
    )
    evidence_required = requires_graph_evidence(
        messages,
        memorable_dates,
        fact_request=fact_request,
        identity=identity,
    )
    return RequestAnalysis(
        text=text,
        private_fields=requested_private_fields(
            messages,
            memorable_dates,
            fact_request=fact_request,
        ),
        evidence_required=evidence_required,
        evidence=evidence_requirements(
            messages,
            evidence_required,
            memorable_dates,
            fact_request=fact_request,
            identity=identity,
        ),
        fact_request=fact_request,
    )


def is_home_adequacy_request(text: str) -> bool:
    return _is_home_adequacy_request(text)


def requires_graph_evidence(
    messages: Sequence[Mapping[str, Any]],
    memorable_dates: MemorableDateRegistry,
    fact_request: FactRequest | None = None,
    identity: Mapping[str, Any] | None = None,
) -> bool:
    latest_user = latest_user_message(messages)
    normalized = latest_user.casefold()
    if _is_non_lookup_intent(normalized):
        return False
    if fact_request is not None:
        return _fact_request_needs_graph(fact_request)
    allow_relative = identity is None or _speaker_relative_reference(normalized)
    if allow_relative and _relative_kind(normalized) is not None:
        return _is_lookup_or_yes_no(normalized)
    if required_evidence_tool(
        messages,
        memorable_dates,
        allow_relative=allow_relative,
    ) is None:
        return False
    return _is_lookup_or_yes_no(normalized)


def evidence_requirements(
    messages: Sequence[Mapping[str, Any]],
    evidence_required: bool,
    memorable_dates: MemorableDateRegistry,
    fact_request: FactRequest | None = None,
    identity: Mapping[str, Any] | None = None,
) -> EvidenceRequirements:
    if not evidence_required:
        return EvidenceRequirements()
    derived = _evidence_from_fact_request(fact_request, memorable_dates, messages)
    if derived is not None:
        return derived
    normalized = latest_user_message(messages).casefold()
    allow_relative = identity is None or _speaker_relative_reference(normalized)
    if allow_relative:
        relative = _relative_kind(normalized)
        if relative is not None:
            derived = _evidence_from_relative(
                relative,
                memorable_dates,
                messages,
            )
            if derived is not None:
                return derived
    return _heuristic_evidence(
        messages,
        memorable_dates,
        allow_relative=allow_relative,
    )


def requested_private_fields(
    messages: Sequence[Mapping[str, Any]],
    memorable_dates: MemorableDateRegistry,
    fact_request: FactRequest | None = None,
) -> frozenset[str]:
    allowed: set[str] = set()
    schema = None
    if fact_request is not None and fact_request.memorable_date is not None:
        try:
            schema = memorable_dates.get(fact_request.memorable_date)
        except LookupError:
            schema = None
    if schema is None:
        schema = memorable_dates.match(latest_user_message(messages).casefold())
    if schema is not None:
        private_field = PRIVATE_TOOL_FIELDS.get(schema.source_field)
        if private_field is not None:
            allowed.add(private_field)
    latest_user = latest_user_message(messages).casefold()
    if fact_request is not None and fact_request.field == "address":
        allowed.add("address")
    if _contains_word(latest_user, ("address", "street address", "地址", "住址")):
        allowed.add("address")
    if _contains_word(latest_user, ("move-in date", "when did", "什么时候搬")):
        allowed.add("relationship_dates")
    if _contains_word(latest_user, ("email", "phone", "telephone", "邮箱", "电话")):
        allowed.add("contact")
    return frozenset(allowed)


def _fact_request_needs_graph(request: FactRequest) -> bool:
    if request.field == "identity" and request.subject.kind in {"speaker", "home"}:
        return False
    if request.field == "address" and request.subject.kind == "home":
        return False
    return True


def _evidence_from_fact_request(
    request: FactRequest | None,
    memorable_dates: MemorableDateRegistry,
    messages: Sequence[Mapping[str, Any]],
) -> EvidenceRequirements | None:
    if request is None:
        return None
    tools: set[str] = set()
    relations: set[str] = set()
    fields: set[tuple[str, str]] = set()
    related_gender: str | None = None
    direction: Literal["out", "in"] | None = None

    if request.subject.kind == "relative" and isinstance(request.subject.value, str):
        relative_evidence = _evidence_from_relative(
            request.subject.value,
            memorable_dates,
            messages,
            memorable_date=request.memorable_date,
        )
        if relative_evidence is not None:
            tools.update(relative_evidence.tools)
            relations.update(relative_evidence.relations)
            fields.update(relative_evidence.fields)
            related_gender = relative_evidence.related_gender
            direction = relative_evidence.relationship_direction

    if request.relation:
        tools.add("get_relationships")
        relations.add(request.relation)

    if request.field == "residents":
        tools.add("get_relationships")
        relations.add("lives_in")
    elif request.field == "location":
        tools.add("get_relationships")
        relations.add(request.relation or "located_in")
    elif request.field in {"spaces", "items"}:
        tools.add("get_relationships")
        if request.relation:
            relations.add(request.relation)

    if request.field == "memorable_date" and request.memorable_date is not None:
        try:
            schema = memorable_dates.get(request.memorable_date)
        except LookupError:
            schema = None
        if schema is not None:
            tool_name = (
                "get_entity" if schema.source_kind == "node" else "get_relationships"
            )
            tools.add(tool_name)
            fields.add((tool_name, schema.source_field))
            if schema.source_kind == "edge":
                relations.add(schema.source_type)
                tools.add("get_relationships")

    if request.field == "relationship_to_speaker":
        tools.add("get_relationships")

    if not tools:
        return None
    return EvidenceRequirements(
        tools=frozenset(tools),
        relations=frozenset(relations),
        fields=frozenset(fields),
        related_gender=related_gender,
        relationship_direction=direction,
        minimum_entity_records=required_entity_record_count(messages),
    )


def _evidence_from_relative(
    relative: str,
    memorable_dates: MemorableDateRegistry,
    messages: Sequence[Mapping[str, Any]],
    memorable_date: str | None = None,
) -> EvidenceRequirements | None:
    definition = RELATIVES.get(relative)
    if definition is None:
        return None
    tools: set[str] = {"get_relationships"}
    relations: set[str] = set()
    fields: set[tuple[str, str]] = set()
    related_gender: str | None = None
    direction: Literal["out", "in"] | None = None
    for step in definition.steps:
        relations.add(step.relation)
    terminal = definition.steps[-1]
    if terminal.direction in {"out", "in"}:
        direction = terminal.direction
    related_gender = terminal.gender
    if memorable_date is not None:
        try:
            schema = memorable_dates.get(memorable_date)
        except LookupError:
            schema = None
        if schema is not None:
            tool_name = (
                "get_entity" if schema.source_kind == "node" else "get_relationships"
            )
            tools.add(tool_name)
            fields.add((tool_name, schema.source_field))
            if schema.source_kind == "edge":
                relations.add(schema.source_type)
    latest = latest_user_message(messages).casefold()
    date_schema = memorable_dates.match(latest)
    if memorable_date is None and date_schema is not None:
        tool_name = (
            "get_entity" if date_schema.source_kind == "node" else "get_relationships"
        )
        tools.add(tool_name)
        fields.add((tool_name, date_schema.source_field))
        if date_schema.source_kind == "edge":
            relations.add(date_schema.source_type)
    return EvidenceRequirements(
        tools=frozenset(tools),
        relations=frozenset(relations),
        fields=frozenset(fields),
        related_gender=related_gender,
        relationship_direction=direction,
        minimum_entity_records=required_entity_record_count(messages),
    )


def _heuristic_evidence(
    messages: Sequence[Mapping[str, Any]],
    memorable_dates: MemorableDateRegistry,
    *,
    allow_relative: bool = True,
) -> EvidenceRequirements:
    primary_tool = required_evidence_tool(
        messages,
        memorable_dates,
        allow_relative=allow_relative,
    )
    relation = required_evidence_relation(
        messages,
        memorable_dates,
        allow_relative=allow_relative,
    )
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
        related_gender=None,
        relationship_direction=None,
        minimum_entity_records=required_entity_record_count(messages),
    )


def required_evidence_tool(
    messages: Sequence[Mapping[str, Any]],
    memorable_dates: MemorableDateRegistry,
    *,
    allow_relative: bool = True,
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
    if allow_relative and _relative_kind(latest_user) is not None:
        return "get_relationships"
    if _contains_word(
        latest_user,
        ("live", "lives", "living", "reside", "resides", "married", "household", "住", "结婚"),
    ):
        return "get_relationships"
    return None


def required_evidence_relation(
    messages: Sequence[Mapping[str, Any]],
    memorable_dates: MemorableDateRegistry,
    *,
    allow_relative: bool = True,
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
    relative = _relative_kind(latest_user) if allow_relative else None
    if relative is not None:
        definition = RELATIVES.get(relative)
        if definition is not None and len(definition.steps) == 1:
            return definition.steps[0].relation
        return None
    if _contains_word(
        latest_user,
        ("live", "lives", "living", "reside", "resides", "household", "住"),
    ):
        return "lives_in"
    if _contains_word(latest_user, ("married", "结婚")):
        return "spouse_of"
    return None


def required_evidence_field(
    messages: Sequence[Mapping[str, Any]],
    memorable_dates: MemorableDateRegistry,
) -> str | None:
    latest_user = latest_user_message(messages).casefold()
    schema = memorable_dates.match(latest_user)
    return schema.source_field if schema is not None else None


def required_entity_record_count(
    messages: Sequence[Mapping[str, Any]],
) -> int:
    latest_user = latest_user_message(messages).casefold()
    if _contains_word(
        latest_user,
        ("both", "children", "their", "them", "they", "他们", "她们", "孩子们"),
    ):
        return 2
    return 1


def is_household_roster_request(
    messages: Sequence[Mapping[str, Any]],
    memorable_dates: MemorableDateRegistry,
) -> bool:
    latest_user = latest_user_message(messages).casefold()
    if _is_relative_alias(latest_user) or _contains_word(
        latest_user, memorable_dates.aliases
    ):
        return False
    return _is_roster_request(latest_user, memorable_dates.aliases) or bool(
        re.search(
            r"(?:家里|家中|家里面|家里边|这个家|这里).*(?:多少|几)(?:个|位)?(?:人|住户|居民)",
            latest_user,
        )
    ) or bool(
        re.search(
            r"\bhow many\b.*\b(?:people|residents|occupants)\b.*"
            r"\b(?:my|our|the)\s+(?:home|house|household)\b",
            latest_user,
        )
    )


def is_household_space_request(
    messages: Sequence[Mapping[str, Any]],
) -> bool:
    latest_user = latest_user_message(messages).casefold()
    matched, _ = _home_space_request(latest_user)
    if matched:
        return True
    matched, _ = _elliptical_home_space_request(latest_user)
    return matched


def is_item_location_request(
    messages: Sequence[Mapping[str, Any]],
) -> bool:
    return _item_location_subject(latest_user_message(messages).casefold().strip()) is not None


def _is_non_lookup_intent(normalized: str) -> bool:
    return re.search(
        r"\b(?:advice|chat|feel|feeling|gift|joke|opinion|recommend|story|"
        r"suggest|talk|think|add|decorate|design|remodel|should|enough|"
        r"adequate|comfortable|crowded|spacious|suitable|fit)\b|"
        r"建议|推荐|礼物|聊|笑话|故事|觉得|认为|心情|装修|设计|改造|增加|"
        r"够|合适|适合|舒服|舒适|拥挤|宽敞|住得下|住不下|住得开|"
        r"住不开|好不好|怎么样|如何",
        normalized,
    ) is not None


def _is_lookup_or_yes_no(normalized: str) -> bool:
    lookup_intent = (
        r"\b(?:find|identify|list|search|show|tell me|what|when|where|which|"
        r"who|whose|how many|how old)\b|谁|什么|哪|何时|什么时候|多少|几岁|"
        r"几个|几间|是否|查|找|告诉我|列出|显示"
    )
    if re.search(lookup_intent, normalized):
        return True
    yes_no_predicate = (
        r"(?:\b(?:is|are|was|were|do|does|did)\b.*\b(?:live|lives|living|"
        r"reside|resides|married)\b)|(?:住|居住|结婚|已婚).*吗[？?]?$|"
        r"是否.*(?:住|居住|结婚|已婚)"
    )
    return re.search(yes_no_predicate, normalized.strip()) is not None


def _contains_word(text: str, terms: Sequence[str]) -> bool:
    for term in terms:
        if term.isascii():
            if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text):
                return True
        elif term in text:
            return True
    return False
