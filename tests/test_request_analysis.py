from __future__ import annotations

from typing import Any

import pytest

from home_cortex.memorable_dates import default_memorable_date_registry
from home_cortex.request_analysis import (
    RelationshipStep,
    analyze_household_request,
    parse_fact_request,
)

IDENTITY: dict[str, Any] = {
    "id": "person:jian_kuang",
    "name": ["Jian Kuang", "匡健"],
}
REGISTRY = default_memorable_date_registry()


def _messages(text: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": text}]


def _analyze(text: str, *, identity: dict[str, Any] | None = IDENTITY):
    return analyze_household_request(
        _messages(text),
        identity=identity,
        memorable_dates=REGISTRY,
    )


@pytest.mark.parametrize(
    (
        "text",
        "field",
        "relations",
        "direction",
        "gender",
        "private",
        "requires_evidence",
    ),
    [
        (
            "Who are my children?",
            "identity",
            frozenset({"parent_of"}),
            "out",
            None,
            frozenset(),
            True,
        ),
        (
            "我有几个孩子？",
            "count",
            frozenset({"parent_of"}),
            "out",
            None,
            frozenset(),
            True,
        ),
        (
            "When is my daughter's birthday?",
            "memorable_date",
            frozenset({"parent_of"}),
            "out",
            "female",
            frozenset({"dob"}),
            True,
        ),
        (
            "我岳父生日是哪天？",
            "memorable_date",
            frozenset({"spouse_of", "parent_of"}),
            "in",
            "male",
            frozenset({"dob"}),
            True,
        ),
        (
            "Who is my spouse?",
            "identity",
            frozenset({"spouse_of"}),
            None,
            None,
            frozenset(),
            True,
        ),
        (
            "我们的结婚纪念日是哪天？",
            "memorable_date",
            frozenset({"spouse_of"}),
            None,
            None,
            frozenset({"relationship_dates"}),
            True,
        ),
        (
            "家里都有谁？",
            "residents",
            frozenset({"lives_in"}),
            None,
            None,
            frozenset(),
            True,
        ),
        (
            "Who lives at home?",
            "residents",
            frozenset({"lives_in"}),
            None,
            None,
            frozenset(),
            True,
        ),
        (
            "What is my home address?",
            "address",
            frozenset(),
            None,
            None,
            frozenset({"address"}),
            False,
        ),
        (
            "我们家的地址是哪里？",
            "address",
            frozenset(),
            None,
            None,
            frozenset({"address"}),
            False,
        ),
    ],
)
def test_shared_analysis_for_english_and_chinese_household_requests(
    text: str,
    field: str,
    relations: frozenset[str],
    direction: str | None,
    gender: str | None,
    private: frozenset[str],
    requires_evidence: bool,
) -> None:
    messages = _messages(text)
    analysis = _analyze(text)

    assert analysis.fact_request == parse_fact_request(
        messages,
        identity=IDENTITY,
        memorable_dates=REGISTRY,
    )
    assert analysis.fact_request is not None
    assert analysis.fact_request.field == field
    assert analysis.private_fields == private
    assert analysis.evidence_required is requires_evidence
    assert analysis.evidence.relations == relations
    assert analysis.evidence.related_gender == gender
    assert analysis.evidence.relationship_direction == direction


@pytest.mark.parametrize("text", ["Hello", "你好"])
def test_open_ended_conversation_does_not_require_graph_evidence(text: str) -> None:
    analysis = _analyze(text, identity=None)

    assert analysis.fact_request is None
    assert analysis.evidence_required is False
    assert analysis.evidence.tools == frozenset()
    assert analysis.private_fields == frozenset()


def test_named_birthday_without_identity_still_requires_entity_evidence() -> None:
    analysis = _analyze("When is Alex's birthday?", identity=None)

    assert analysis.fact_request is None
    assert analysis.evidence_required is True
    assert ("get_entity", "dob") in analysis.evidence.fields
    assert "dob" in analysis.private_fields


def test_parent_in_law_analysis_preserves_ordered_relationship_steps() -> None:
    analysis = _analyze("我岳父生日是哪天？")

    assert analysis.evidence.relationship_path == (
        RelationshipStep("spouse_of", gender="female"),
        RelationshipStep("parent_of", direction="in", gender="male"),
    )


def test_person_does_not_parse_as_son_with_trusted_identity() -> None:
    request = parse_fact_request(
        _messages("Who is this person?"),
        identity=IDENTITY,
        memorable_dates=REGISTRY,
    )
    analysis = _analyze("Who is this person?")

    assert request is None or (
        request.subject.kind != "relative" and request.subject.value != "son"
    )
    assert "parent_of" not in analysis.evidence.relations


def test_whose_does_not_count_as_who_lookup_for_a_son_request() -> None:
    request = parse_fact_request(
        _messages("Whose son is that?"),
        identity=IDENTITY,
        memorable_dates=REGISTRY,
    )
    analysis = _analyze("Whose son is that?")

    assert request is None or request.field != "identity"
    assert "parent_of" not in analysis.evidence.relations


def test_english_plural_sons_are_relative_requests() -> None:
    count = _analyze("How many sons do I have?")
    assert count.fact_request is not None
    assert count.fact_request.field == "count"
    assert count.fact_request.subject.value == "son"
    assert count.evidence_required is True
    assert "parent_of" in count.evidence.relations
    assert count.evidence.relationship_direction == "out"

    who = _analyze("Who are my sons?")
    assert who.fact_request is not None
    assert who.fact_request.field == "identity"
    assert who.fact_request.subject.value == "son"
    assert who.fact_request.cardinality == "all"
    assert "parent_of" in who.evidence.relations


def test_english_relative_location_is_not_an_item_lookup() -> None:
    for text in (
        "Where is my son?",
        "Where is my wife?",
        "Where is my father?",
        "Which room is my son in?",
    ):
        request = parse_fact_request(
            _messages(text),
            identity=IDENTITY,
            memorable_dates=REGISTRY,
        )
        assert request is None or request.subject.kind != "item"


def test_phone_number_is_not_a_relative_count() -> None:
    request = parse_fact_request(
        _messages("What is my son's phone number?"),
        identity=IDENTITY,
        memorable_dates=REGISTRY,
    )
    analysis = _analyze("What is my son's phone number?")

    assert request is None or request.field != "count"
    assert "phone" in analysis.private_fields
    assert "email" not in analysis.private_fields
    assert analysis.evidence_required is True
    assert analysis.evidence.tools == frozenset(
        {"get_relationships", "get_entity"}
    )
    assert ("get_entity", "phone") in analysis.evidence.fields


@pytest.mark.parametrize(
    ("text", "private_field", "relation"),
    [
        ("What is my son's address?", "address", "parent_of"),
        ("What is Alex's email address?", "email", None),
    ],
)
def test_private_entity_lookups_require_the_requested_field(
    text: str,
    private_field: str,
    relation: str | None,
) -> None:
    analysis = _analyze(text)

    assert analysis.evidence_required is True
    assert private_field in analysis.private_fields
    assert "get_entity" in analysis.evidence.tools
    assert ("get_entity", private_field) in analysis.evidence.fields
    if relation is not None:
        assert relation in analysis.evidence.relations
    else:
        assert analysis.evidence.tools == frozenset(
            {"search_entities", "get_entity"}
        )
        assert analysis.evidence.required_entity_names == ("alex",)


def test_self_private_lookup_is_bound_to_trusted_identity() -> None:
    analysis = _analyze("What is my phone number?")

    assert analysis.evidence.entity_binding_required is True
    assert analysis.evidence.required_entity_ids == frozenset(
        {"person:jian_kuang"}
    )


@pytest.mark.parametrize(
    ("text", "expected_name"),
    [
        ("Could you tell me Alex's email address?", "alex"),
        ("Please show me Alex's email address", "alex"),
        ("请告诉我 Alex 的电话号码", "alex"),
        ("麻烦告诉我艾力克斯的邮箱", "艾力克斯"),
    ],
)
def test_polite_named_private_lookup_binds_the_named_person(
    text: str,
    expected_name: str,
) -> None:
    analysis = _analyze(text)

    assert analysis.evidence.entity_binding_required is True
    assert analysis.evidence.required_entity_ids == frozenset()
    assert analysis.evidence.required_entity_names == (expected_name,)


@pytest.mark.parametrize("text", ["我的电话号码是多少？", "我生日是哪天？"])
def test_chinese_self_private_lookup_still_binds_trusted_identity(
    text: str,
) -> None:
    analysis = _analyze(text)

    assert analysis.evidence.required_entity_ids == frozenset(
        {"person:jian_kuang"}
    )
    assert analysis.evidence.required_entity_names == ()


def test_named_birthday_requires_verified_name_resolution() -> None:
    analysis = _analyze("When is Alex's birthday?", identity=None)

    assert analysis.evidence.entity_binding_required is True
    assert analysis.evidence.required_entity_names == ("alex",)
    assert analysis.evidence.tools == frozenset(
        {"search_entities", "get_entity"}
    )


def test_item_location_does_not_authorize_contact_fields() -> None:
    analysis = _analyze("Where is my phone?")

    assert analysis.fact_request is not None
    assert analysis.fact_request.subject.kind == "item"
    assert "phone" not in analysis.private_fields
    assert "email" not in analysis.private_fields
