from typing import Any

import pytest

from home_cortex.memorable_dates import default_memorable_date_registry
from home_cortex.model_loop import (
    _constrain_tool_arguments,
    _has_nonempty_evidence,
    _prepare_tool_value,
    _scope_tool_result,
    _should_retry_incomplete_evidence,
    _tool_evidence,
)
from home_cortex.request_analysis import analyze_household_request


IDENTITY: dict[str, Any] = {
    "id": "person:jian_kuang",
    "name": ["Jian Kuang", "匡健"],
}
REGISTRY = default_memorable_date_registry()


def _requirements(text: str, *, identity: dict[str, Any] | None = IDENTITY):
    return analyze_household_request(
        [{"role": "user", "content": text}],
        identity=identity,
        memorable_dates=REGISTRY,
    ).evidence


def _collect_evidence(
    requirements: Any,
    calls: list[tuple[str, dict[str, Any], dict[str, Any]]],
) -> tuple[set[str], set[str], set[Any]]:
    nonempty: set[str] = set()
    fields: set[str] = set()
    relationships: set[Any] = set()
    for tool_name, arguments, result in calls:
        scoped = _scope_tool_result(tool_name, result, requirements)
        _, call_nonempty, call_fields, call_relationships = _tool_evidence(
            tool_name,
            arguments,
            scoped,
        )
        nonempty.update(call_nonempty)
        fields.update(call_fields)
        relationships.update(call_relationships)
    return nonempty, fields, relationships


def test_multihop_relationship_steps_keep_their_own_constraints() -> None:
    requirements = _requirements("When is my father-in-law's birthday?")

    spouse_arguments = _constrain_tool_arguments(
        "get_relationships",
        {
            "entity_id": "person:jian_kuang",
            "relation": "spouse_of",
            "direction": "out",
        },
        requirements,
    )
    parent_arguments = _constrain_tool_arguments(
        "get_relationships",
        {"entity_id": "person:pu_ba", "relation": "parent_of"},
        requirements,
    )
    spouse_result = _scope_tool_result(
        "get_relationships",
        {
            "ok": True,
            "result": [
                {
                    "relation": "spouse_of",
                    "related_entity": {
                        "id": "person:pu_ba",
                        "gender": "female",
                    },
                }
            ],
        },
        requirements,
    )
    parent_result = _scope_tool_result(
        "get_relationships",
        {
            "ok": True,
            "result": [
                {
                    "relation": "parent_of",
                    "related_entity": {
                        "id": "person:zhigang_ba",
                        "gender": "male",
                    },
                }
            ],
        },
        requirements,
    )

    assert spouse_arguments == {
        "entity_id": "person:jian_kuang",
        "relation": "spouse_of",
    }
    assert parent_arguments["direction"] == "in"
    assert spouse_result["result"]
    assert parent_result["result"]


def test_multihop_private_field_must_belong_to_terminal_entity() -> None:
    requirements = _requirements("What is my mother-in-law's birthday?")
    relationship_calls = [
        (
            "get_relationships",
            {"entity_id": "person:jian_kuang", "relation": "spouse_of"},
            {
                "ok": True,
                "result": [
                    {
                        "relation": "spouse_of",
                        "related_entity": {
                            "id": "person:pu_ba",
                            "gender": "female",
                        },
                    }
                ],
            },
        ),
        (
            "get_relationships",
            {"entity_id": "person:pu_ba", "relation": "parent_of"},
            {
                "ok": True,
                "result": [
                    {
                        "relation": "parent_of",
                        "related_entity": {
                            "id": "person:mother_in_law",
                            "gender": "female",
                        },
                    }
                ],
            },
        ),
    ]
    wrong_entity_call = (
        "get_entity",
        {"entity_id": "person:pu_ba"},
        {
            "ok": True,
            "result": [{"id": "person:pu_ba", "dob": "1988-02-26"}],
        },
    )
    correct_entity_call = (
        "get_entity",
        {"entity_id": "person:mother_in_law"},
        {
            "ok": True,
            "result": [
                {"id": "person:mother_in_law", "dob": "1960-01-02"}
            ],
        },
    )

    nonempty, fields, relationships = _collect_evidence(
        requirements,
        [*relationship_calls, wrong_entity_call],
    )
    assert not _has_nonempty_evidence(
        requirements,
        nonempty,
        fields,
        relationships,
        "person:jian_kuang",
    )
    assert _should_retry_incomplete_evidence(
        requirements,
        fields,
        relationships,
        "person:jian_kuang",
    )

    nonempty, fields, relationships = _collect_evidence(
        requirements,
        [*relationship_calls, correct_entity_call],
    )
    assert _has_nonempty_evidence(
        requirements,
        nonempty,
        fields,
        relationships,
        "person:jian_kuang",
    )


def test_multihop_evidence_must_form_a_connected_path() -> None:
    requirements = _requirements("When is my father-in-law's birthday?")
    calls = [
        (
            "get_relationships",
            {"entity_id": "person:jian_kuang", "relation": "spouse_of"},
            {
                "ok": True,
                "result": [
                    {
                        "relation": "spouse_of",
                        "related_entity": {
                            "id": "person:pu_ba",
                            "gender": "female",
                        },
                    }
                ],
            },
        ),
        (
            "get_relationships",
            {"entity_id": "person:someone_else", "relation": "parent_of"},
            {
                "ok": True,
                "result": [
                    {
                        "relation": "parent_of",
                        "related_entity": {
                            "id": "person:unrelated_father",
                            "gender": "male",
                        },
                    }
                ],
            },
        ),
        (
            "get_entity",
            {"entity_id": "person:unrelated_father"},
            {
                "ok": True,
                "result": [
                    {"id": "person:unrelated_father", "dob": "1950-01-01"}
                ],
            },
        ),
    ]

    nonempty, fields, relationships = _collect_evidence(requirements, calls)

    assert not _has_nonempty_evidence(
        requirements,
        nonempty,
        fields,
        relationships,
        "person:jian_kuang",
    )
    assert _should_retry_incomplete_evidence(
        requirements,
        fields,
        relationships,
        "person:jian_kuang",
    )


@pytest.mark.parametrize(
    ("question", "contact_field"),
    [
        ("What is my son's phone number?", "phone"),
        ("What is my son's phone number?", "phone_number"),
        ("What is my son's email?", "email"),
    ],
)
def test_private_contact_evidence_must_come_from_related_entity(
    question: str,
    contact_field: str,
) -> None:
    requirements = _requirements(question)
    relationship_call = (
        "get_relationships",
        {"entity_id": "person:jian_kuang", "relation": "parent_of"},
        {
            "ok": True,
            "result": [
                {
                    "relation": "parent_of",
                    "related_entity": {
                        "id": "person:dylan_kuang",
                        "gender": "male",
                    },
                }
            ],
        },
    )
    unrelated_contact_call = (
        "get_entity",
        {"entity_id": "person:someone_else"},
        {
            "ok": True,
            "result": [
                {"id": "person:someone_else", contact_field: "unrelated"}
            ],
        },
    )
    related_contact_call = (
        "get_entity",
        {"entity_id": "person:dylan_kuang"},
        {
            "ok": True,
            "result": [
                {"id": "person:dylan_kuang", contact_field: "verified"}
            ],
        },
    )

    nonempty, fields, relationships = _collect_evidence(
        requirements,
        [relationship_call],
    )
    assert not _has_nonempty_evidence(
        requirements,
        nonempty,
        fields,
        relationships,
        "person:jian_kuang",
    )

    nonempty, fields, relationships = _collect_evidence(
        requirements,
        [relationship_call, unrelated_contact_call],
    )
    assert not _has_nonempty_evidence(
        requirements,
        nonempty,
        fields,
        relationships,
        "person:jian_kuang",
    )

    nonempty, fields, relationships = _collect_evidence(
        requirements,
        [relationship_call, related_contact_call],
    )
    assert _has_nonempty_evidence(
        requirements,
        nonempty,
        fields,
        relationships,
        "person:jian_kuang",
    )


def test_email_does_not_satisfy_phone_evidence_requirement() -> None:
    requirements = _requirements("What is my son's phone number?")
    calls = [
        (
            "get_relationships",
            {"entity_id": "person:jian_kuang", "relation": "parent_of"},
            {
                "ok": True,
                "result": [
                    {
                        "relation": "parent_of",
                        "related_entity": {
                            "id": "person:dylan_kuang",
                            "gender": "male",
                        },
                    }
                ],
            },
        ),
        (
            "get_entity",
            {"entity_id": "person:dylan_kuang"},
            {
                "ok": True,
                "result": [
                    {"id": "person:dylan_kuang", "email": "dylan@example.com"}
                ],
            },
        ),
    ]

    nonempty, fields, relationships = _collect_evidence(requirements, calls)

    assert not _has_nonempty_evidence(
        requirements,
        nonempty,
        fields,
        relationships,
        "person:jian_kuang",
    )


def test_self_private_evidence_must_match_trusted_identity() -> None:
    requirements = _requirements("What is my phone number?")
    wrong_call = (
        "get_entity",
        {"entity_id": "person:someone_else"},
        {
            "ok": True,
            "result": [
                {"id": "person:someone_else", "phone_number": "555-9999"}
            ],
        },
    )
    correct_call = (
        "get_entity",
        {"entity_id": "person:jian_kuang"},
        {
            "ok": True,
            "result": [
                {"id": "person:jian_kuang", "phone_number": "555-0100"}
            ],
        },
    )

    nonempty, fields, relationships = _collect_evidence(
        requirements,
        [wrong_call],
    )
    assert not _has_nonempty_evidence(
        requirements,
        nonempty,
        fields,
        relationships,
        "person:jian_kuang",
    )
    assert _should_retry_incomplete_evidence(
        requirements,
        fields,
        relationships,
        "person:jian_kuang",
    )

    nonempty, fields, relationships = _collect_evidence(
        requirements,
        [correct_call],
    )
    assert _has_nonempty_evidence(
        requirements,
        nonempty,
        fields,
        relationships,
        "person:jian_kuang",
    )


def test_named_private_evidence_must_match_unique_search_result() -> None:
    requirements = _requirements("What is Alex's email address?")
    search_call = (
        "search_entities",
        {"text": "Alex", "entity_type": "person", "limit": 25},
        {
            "ok": True,
            "result": [{"id": "person:alex", "name": ["Alex", "艾力克斯"]}],
        },
    )
    wrong_call = (
        "get_entity",
        {"entity_id": "person:someone_else"},
        {
            "ok": True,
            "result": [
                {
                    "id": "person:someone_else",
                    "name": ["Someone Else"],
                    "email": "wrong@example.com",
                }
            ],
        },
    )
    correct_call = (
        "get_entity",
        {"entity_id": "person:alex"},
        {
            "ok": True,
            "result": [
                {
                    "id": "person:alex",
                    "name": ["Alex", "艾力克斯"],
                    "email": "alex@example.com",
                }
            ],
        },
    )

    nonempty, fields, relationships = _collect_evidence(
        requirements,
        [search_call, wrong_call],
    )
    assert not _has_nonempty_evidence(
        requirements,
        nonempty,
        fields,
        relationships,
        "person:jian_kuang",
    )
    assert _should_retry_incomplete_evidence(
        requirements,
        fields,
        relationships,
        "person:jian_kuang",
    )

    nonempty, fields, relationships = _collect_evidence(
        requirements,
        [search_call, correct_call],
    )
    assert _has_nonempty_evidence(
        requirements,
        nonempty,
        fields,
        relationships,
        "person:jian_kuang",
    )


def test_duplicate_named_matches_cannot_satisfy_private_evidence() -> None:
    requirements = _requirements("What is Alex's email address?")
    calls = [
        (
            "search_entities",
            {"text": "Alex", "entity_type": "person", "limit": 25},
            {
                "ok": True,
                "result": [
                    {"id": "person:alex_one", "name": ["Alex"]},
                    {"id": "person:alex_two", "name": ["Alex"]},
                ],
            },
        ),
        (
            "get_entity",
            {"entity_id": "person:alex_one"},
            {
                "ok": True,
                "result": [
                    {
                        "id": "person:alex_one",
                        "name": ["Alex"],
                        "email": "one@example.com",
                    }
                ],
            },
        ),
        (
            "get_entity",
            {"entity_id": "person:alex_two"},
            {
                "ok": True,
                "result": [
                    {
                        "id": "person:alex_two",
                        "name": ["Alex"],
                        "email": "two@example.com",
                    }
                ],
            },
        ),
    ]

    nonempty, fields, relationships = _collect_evidence(requirements, calls)

    assert not _has_nonempty_evidence(
        requirements,
        nonempty,
        fields,
        relationships,
        "person:jian_kuang",
    )
    scoped = _scope_tool_result(
        "get_entity",
        calls[1][2],
        requirements,
        bound_entity_ids=set(),
    )
    assert "email" not in scoped["result"][0]


def test_full_named_search_page_cannot_prove_unique_resolution() -> None:
    requirements = _requirements("What is Alex's email address?")
    search_records = [{"id": "person:alex", "name": ["Alex"]}]
    search_records.extend(
        {
            "id": f"person:alexandra_{index}",
            "name": [f"Alexandra {index}"],
        }
        for index in range(24)
    )
    calls = [
        (
            "search_entities",
            {"text": "Alex", "entity_type": "person", "limit": 25},
            {"ok": True, "result": search_records},
        ),
        (
            "get_entity",
            {"entity_id": "person:alex"},
            {
                "ok": True,
                "result": [
                    {
                        "id": "person:alex",
                        "name": ["Alex"],
                        "email": "alex@example.com",
                    }
                ],
            },
        ),
    ]

    nonempty, fields, relationships = _collect_evidence(requirements, calls)

    assert not _has_nonempty_evidence(
        requirements,
        nonempty,
        fields,
        relationships,
        "person:jian_kuang",
    )


def test_relative_private_evidence_requires_a_trusted_root() -> None:
    requirements = _requirements(
        "What is my son's phone number?",
        identity=None,
    )
    calls = [
        (
            "get_relationships",
            {"entity_id": "person:arbitrary_parent", "relation": "parent_of"},
            {
                "ok": True,
                "result": [
                    {
                        "relation": "parent_of",
                        "related_entity": {
                            "id": "person:arbitrary_child",
                            "gender": "male",
                        },
                    }
                ],
            },
        ),
        (
            "get_entity",
            {"entity_id": "person:arbitrary_child"},
            {
                "ok": True,
                "result": [
                    {
                        "id": "person:arbitrary_child",
                        "phone_number": "555-9999",
                    }
                ],
            },
        ),
    ]

    nonempty, fields, relationships = _collect_evidence(requirements, calls)

    assert not _has_nonempty_evidence(
        requirements,
        nonempty,
        fields,
        relationships,
        None,
    )
    assert not _should_retry_incomplete_evidence(
        requirements,
        fields,
        relationships,
        None,
    )


def test_unbound_entity_payload_redacts_requested_private_field() -> None:
    requirements = _requirements("What is Alex's email address?")
    result = {
        "ok": True,
        "result": [
            {
                "id": "person:someone_else",
                "name": ["Someone Else"],
                "email": "wrong@example.com",
            },
            {
                "id": "person:alex",
                "name": ["Alex"],
                "email": "alex@example.com",
            },
        ],
    }

    scoped = _scope_tool_result(
        "get_entity",
        result,
        requirements,
        bound_entity_ids={"person:alex"},
    )
    payload = _prepare_tool_value(
        scoped,
        "en",
        frozenset({"email"}),
        tool_name="get_entity",
    )

    assert payload["result"] == [
        {"id": "person:someone_else", "name": "Someone Else"},
        {
            "id": "person:alex",
            "name": "Alex",
            "email": "alex@example.com",
        },
    ]


def test_private_contact_payload_excludes_unrequested_sibling_fields() -> None:
    phone_analysis = analyze_household_request(
        [{"role": "user", "content": "What is my son's phone number?"}],
        identity=IDENTITY,
        memorable_dates=REGISTRY,
    )
    email_analysis = analyze_household_request(
        [{"role": "user", "content": "What is my son's email?"}],
        identity=IDENTITY,
        memorable_dates=REGISTRY,
    )
    record = {
        "id": "person:dylan_kuang",
        "phone_number": "555-0100",
        "email": "dylan@example.com",
    }

    phone_payload = _prepare_tool_value(
        record,
        "en",
        phone_analysis.private_fields,
        tool_name="get_entity",
    )
    email_payload = _prepare_tool_value(
        record,
        "en",
        email_analysis.private_fields,
        tool_name="get_entity",
    )

    assert phone_payload == {
        "id": "person:dylan_kuang",
        "phone_number": "555-0100",
    }
    assert email_payload == {
        "id": "person:dylan_kuang",
        "email": "dylan@example.com",
    }
