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
