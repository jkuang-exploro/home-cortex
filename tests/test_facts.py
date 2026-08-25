from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from home_cortex.facts import (
    FactRequest,
    FactService,
    SubjectReference,
    parse_fact_request,
)
from home_cortex.memorable_dates import (
    MemorableDateRegistry,
    MemorableDateSchema,
)


IDENTITY: dict[str, Any] = {
    "id": "person:jian_kuang",
    "name": ["Jian Kuang", "匡健"],
}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "我岳父生日是哪天？",
            FactRequest(
                SubjectReference("relative", "father_in_law"),
                "memorable_date",
                memorable_date="birthday",
            ),
        ),
        (
            "When is my father-in-law's birthday?",
            FactRequest(
                SubjectReference("relative", "father_in_law"),
                "memorable_date",
                memorable_date="birthday",
            ),
        ),
        (
            "Who are my children?",
            FactRequest(
                SubjectReference("relative", "children"),
                "identity",
                "all",
            ),
        ),
        (
            "我有几个孩子？",
            FactRequest(
                SubjectReference("relative", "children"),
                "count",
                "all",
            ),
        ),
        (
            "我们的结婚纪念日是哪天？",
            FactRequest(
                SubjectReference("relative", "spouse"),
                "memorable_date",
                memorable_date="wedding_anniversary",
            ),
        ),
        (
            "我岳父住在我家吗？",
            FactRequest(
                SubjectReference("relative", "father_in_law"),
                "relationship_exists",
                relation="lives_in",
                target=SubjectReference("home"),
            ),
        ),
        (
            "家里都有谁？",
            FactRequest(SubjectReference("home"), "residents", "all"),
        ),
        (
            "家里有多少人？",
            FactRequest(
                SubjectReference("home"),
                "count",
                "all",
                relation="lives_in",
            ),
        ),
        (
            "How many people are in my home?",
            FactRequest(
                SubjectReference("home"),
                "count",
                "all",
                relation="lives_in",
            ),
        ),
        (
            "家的位置",
            FactRequest(SubjectReference("home"), "address"),
        ),
        (
            "家里地址是哪",
            FactRequest(SubjectReference("home"), "address"),
        ),
        (
            "这是哪里",
            FactRequest(SubjectReference("home"), "identity"),
        ),
        (
            "What is my home address?",
            FactRequest(SubjectReference("home"), "address"),
        ),
        (
            "家里有几个房间？",
            FactRequest(
                SubjectReference("home"),
                "count",
                "all",
                relation="hosts_space",
                space_type="room",
            ),
        ),
        (
            "家里有哪些房间？",
            FactRequest(
                SubjectReference("home"),
                "spaces",
                "all",
                relation="hosts_space",
                space_type="room",
            ),
        ),
        (
            "What rooms are in my house?",
            FactRequest(
                SubjectReference("home"),
                "spaces",
                "all",
                relation="hosts_space",
                space_type="room",
            ),
        ),
        (
            "冰箱在哪里？",
            FactRequest(
                SubjectReference("item", "冰箱"),
                "location",
                relation="located_in",
            ),
        ),
        (
            "冰箱位于哪个房间？",
            FactRequest(
                SubjectReference("item", "冰箱"),
                "location",
                relation="located_in",
            ),
        ),
        (
            "Where is the refrigerator?",
            FactRequest(
                SubjectReference("item", "refrigerator"),
                "location",
                relation="located_in",
            ),
        ),
        (
            "匡德伦和匡悠然是谁？",
            FactRequest(
                SubjectReference("named", ("匡德伦", "匡悠然")),
                "relationship_to_speaker",
                "all",
            ),
        ),
    ],
)
def test_fact_parser_maps_language_to_semantics(
    text: str,
    expected: FactRequest,
) -> None:
    assert parse_fact_request(
        [{"role": "user", "content": text}],
        identity=IDENTITY,
    ) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Tell me a joke.",
        "What gift should I buy for my wife?",
        "我女儿今天开心吗？",
        "What room should I add to my house?",
        "我家房间应该怎么装修？",
    ],
)
def test_fact_parser_leaves_open_ended_conversation_to_model(text: str) -> None:
    assert parse_fact_request(
        [{"role": "user", "content": text}],
        identity=IDENTITY,
    ) is None


def test_relative_request_requires_trusted_identity() -> None:
    assert parse_fact_request(
        [{"role": "user", "content": "Who is my son?"}],
        identity=None,
    ) is None


def test_birthday_followup_inherits_previous_semantic_subject() -> None:
    request = parse_fact_request(
        [
            {"role": "user", "content": "匡德伦和匡悠然是谁？"},
            {"role": "assistant", "content": "他们是您的孩子。"},
            {"role": "user", "content": "他们的生日分别是哪天？"},
        ],
        identity=IDENTITY,
    )

    assert request == FactRequest(
        SubjectReference("named", ("匡德伦", "匡悠然")),
        "memorable_date",
        "all",
        memorable_date="birthday",
    )


def test_birthday_countdown_followup_inherits_previous_semantic_subject() -> None:
    request = parse_fact_request(
        [
            {"role": "user", "content": "我岳父生日哪天？"},
            {"role": "assistant", "content": "您岳父的生日是10月10日。"},
            {"role": "user", "content": "还有多少天过生日？"},
        ],
        identity=IDENTITY,
    )

    assert request == FactRequest(
        SubjectReference("relative", "father_in_law"),
        "memorable_date",
        "all",
        memorable_date="birthday",
        date_query="next",
    )


def test_address_followup_inherits_home_subject() -> None:
    request = parse_fact_request(
        [
            {"role": "user", "content": "这是哪里"},
            {"role": "assistant", "content": "这里是喜瑞匡家。"},
            {"role": "user", "content": "地址在哪里"},
        ],
        identity=IDENTITY,
    )

    assert request == FactRequest(SubjectReference("home"), "address")


def test_resident_count_followup_inherits_home_subject() -> None:
    request = parse_fact_request(
        [
            {"role": "user", "content": "家里有哪些房间？"},
            {"role": "assistant", "content": "家里有厨房和客房。"},
            {"role": "user", "content": "有多少人？"},
        ],
        identity=IDENTITY,
    )

    assert request == FactRequest(
        SubjectReference("home"),
        "count",
        "all",
        relation="lives_in",
    )


@pytest.mark.asyncio
async def test_home_address_is_loaded_from_configured_location() -> None:
    class Dispatcher:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, Any]]] = []

        async def dispatch(
            self,
            tool_name: str,
            arguments: dict[str, Any],
            **_: Any,
        ) -> dict[str, Any]:
            self.calls.append((tool_name, arguments))
            return {
                "ok": True,
                "tool": tool_name,
                "result": [
                    {
                        "id": "location:fort_cerritos",
                        "name": ["Fort Cerritos", "喜瑞匡家"],
                        "address": {
                            "street": "12745 Droxford St",
                            "city": "Cerritos",
                            "state": "CA",
                            "zip": "90703",
                        },
                    }
                ],
            }

    dispatcher = Dispatcher()
    answer = await FactService(
        dispatcher,
        home_entity_id="location:fort_cerritos",
    ).try_answer(
        [{"role": "user", "content": "家的位置"}],
        identity={**IDENTITY, "address_as": {"zh": "先生"}},
        language="zh",
        request_id="test-home-address",
    )

    assert answer is not None
    assert answer.text == (
        "先生，家（喜瑞匡家）的地址是 "
        "12745 Droxford St, Cerritos, CA 90703。"
    )
    assert dispatcher.calls == [
        ("get_entity", {"entity_id": "location:fort_cerritos"})
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("家里有几个房间？", "先生，家里共有 3 个房间。"),
        (
            "家里有哪些房间？",
            "先生，家里的房间有：厨房、主卧室和客房。",
        ),
    ],
)
async def test_home_rooms_follow_house_item_hosted_spaces(
    question: str,
    expected: str,
) -> None:
    class Dispatcher:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, Any]]] = []

        async def dispatch(
            self,
            tool_name: str,
            arguments: dict[str, Any],
            **_: Any,
        ) -> dict[str, Any]:
            self.calls.append((tool_name, arguments))
            if arguments["relation"] == "located_in":
                result = [
                    {
                        "relation": "located_in",
                        "related_entity": {
                            "id": "item:fort_cerritos_house",
                            "item_type": "house",
                            "name": "喜瑞匡家房屋",
                        },
                    }
                ]
            else:
                result = [
                    {
                        "relation": "hosted_by",
                        "related_entity": {
                            "id": "space:kitchen",
                            "space_type": "room",
                            "name": "厨房",
                        },
                    },
                    {
                        "relation": "hosted_by",
                        "related_entity": {
                            "id": "space:master_room",
                            "space_type": "room",
                            "name": "主卧室",
                        },
                    },
                    {
                        "relation": "hosted_by",
                        "related_entity": {
                            "id": "space:guest_room",
                            "space_type": "room",
                            "name": "客房",
                        },
                    },
                    {
                        "relation": "hosted_by",
                        "related_entity": {
                            "id": "space:backyard",
                            "space_type": "outdoor_space",
                            "name": "后院",
                        },
                    },
                ]
            return {"ok": True, "tool": tool_name, "result": result}

    dispatcher = Dispatcher()
    answer = await FactService(
        dispatcher,
        home_entity_id="location:fort_cerritos",
    ).try_answer(
        [{"role": "user", "content": question}],
        identity={**IDENTITY, "address_as": {"zh": "先生"}},
        language="zh",
        request_id="test-home-rooms",
    )

    assert answer is not None
    assert answer.text == expected
    assert answer.tool_calls == 2
    assert dispatcher.calls == [
        (
            "get_relationships",
            {
                "entity_id": "location:fort_cerritos",
                "relation": "located_in",
                "limit": 25,
            },
        ),
        (
            "get_relationships",
            {
                "entity_id": "item:fort_cerritos_house",
                "relation": "hosts_space",
                "limit": 25,
            },
        ),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("question", ["冰箱在哪里？", "冰箱位于哪个房间？"])
async def test_item_location_uses_stored_located_in_relationship(
    question: str,
) -> None:
    class Dispatcher:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, Any]]] = []

        async def dispatch(
            self,
            tool_name: str,
            arguments: dict[str, Any],
            **_: Any,
        ) -> dict[str, Any]:
            self.calls.append((tool_name, arguments))
            if tool_name == "search_entities":
                result = [
                    {
                        "id": "item:fridge_01",
                        "item_type": "refrigerator",
                        "name": {"en": "Kitchen refrigerator", "zh": "厨房冰箱"},
                    }
                ]
            else:
                result = [
                    {
                        "relation": "located_in",
                        "related_entity": {
                            "id": "space:fort_cerritos_kitchen",
                            "space_type": "room",
                            "name": {"en": "Kitchen", "zh": "厨房"},
                        },
                    }
                ]
            return {"ok": True, "tool": tool_name, "result": result}

    dispatcher = Dispatcher()
    answer = await FactService(
        dispatcher,
        home_entity_id="location:fort_cerritos",
    ).try_answer(
        [{"role": "user", "content": question}],
        identity={**IDENTITY, "address_as": {"zh": "先生"}},
        language="zh",
        request_id="test-item-location",
    )

    assert answer is not None
    assert answer.text == "先生，厨房冰箱在厨房。"
    assert answer.tool_calls == 2
    assert dispatcher.calls == [
        (
            "search_entities",
            {"text": "冰箱", "entity_type": "item", "limit": 25},
        ),
        (
            "get_relationships",
            {
                "entity_id": "item:fridge_01",
                "relation": "located_in",
                "limit": 25,
            },
        ),
    ]


@pytest.mark.asyncio
async def test_named_birthday_followup_is_rendered_without_a_model() -> None:
    people = {
        "匡德伦": {
            "id": "person:dylan_kuang",
            "name": ["Dylan Kuang", "匡德伦"],
            "dob": "2016-10-30",
        },
        "匡悠然": {
            "id": "person:evelyn_kuang",
            "name": ["Evelyn Kuang", "匡悠然"],
            "dob": "2019-10-08",
        },
    }

    class Dispatcher:
        async def dispatch(
            self,
            tool_name: str,
            arguments: dict[str, Any],
            **_: Any,
        ) -> dict[str, Any]:
            if tool_name == "search_entities":
                result = [people[arguments["text"]]]
            else:
                result = [
                    person
                    for person in people.values()
                    if person["id"] == arguments["entity_id"]
                ]
            return {"ok": True, "tool": tool_name, "result": result}

    answer = await FactService(
        Dispatcher(),
        home_entity_id=None,
    ).try_answer(
        [
            {"role": "user", "content": "匡德伦和匡悠然是谁？"},
            {"role": "assistant", "content": "他们是您的孩子。"},
            {"role": "user", "content": "他们的生日分别是哪天？"},
        ],
        identity=IDENTITY,
        language="zh",
        request_id="test-followup",
    )

    assert answer is not None
    assert answer.text == (
        "匡德伦的生日是2016年10月30日；"
        "匡悠然的生日是2019年10月8日。"
    )
    assert answer.tool_calls == 4


@pytest.mark.asyncio
async def test_new_edge_date_kind_needs_only_a_registry_entry() -> None:
    registry = MemorableDateRegistry(
        {
            "move_in_anniversary": MemorableDateSchema(
                id="move_in_anniversary",
                aliases=("move-in day",),
                label={"en": "move-in anniversary"},
                recurrence="annual",
                source_kind="edge",
                source_type="lives_in",
                source_field="start",
            )
        }
    )

    class Dispatcher:
        async def dispatch(
            self,
            tool_name: str,
            arguments: dict[str, Any],
            **_: Any,
        ) -> dict[str, Any]:
            assert tool_name == "get_relationships"
            assert arguments["relation"] == "lives_in"
            return {
                "ok": True,
                "tool": tool_name,
                "result": [
                    {
                        "relation": "lives_in",
                        "start": "2020-09-01",
                        "related_entity": {
                            "id": "location:fort_cerritos",
                            "name": "Fort Cerritos",
                        },
                    }
                ],
            }

    answer = await FactService(
        Dispatcher(),
        home_entity_id=None,
        memorable_dates=registry,
    ).try_answer(
        [{"role": "user", "content": "How many days until our move-in day?"}],
        identity=IDENTITY,
        language="en",
        request_id="test-custom-date",
        current_date=date(2026, 8, 22),
    )

    assert answer is not None
    assert answer.text == (
        "Your move-in anniversary with Fort Cerritos is in 10 days."
    )
