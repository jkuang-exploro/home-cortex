from pathlib import Path

from home_cortex.edge_schema import EdgeSchemaRegistry
from home_cortex.facts import (
    SubjectReference as FactsSubjectReference,
    parse_fact_request as facts_parse_fact_request,
)
from home_cortex.fallbacks import grounding_fallback, no_records_fallback
from home_cortex.request_analysis import SubjectReference, parse_fact_request
from home_cortex.text import latest_user_message, normalize_language_code, safe_log_token


def test_latest_user_message_returns_the_last_user_turn() -> None:
    assert (
        latest_user_message(
            [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "second"},
            ]
        )
        == "second"
    )


def test_latest_user_message_skips_non_string_content() -> None:
    assert (
        latest_user_message(
            [
                {"role": "user", "content": "kept"},
                {"role": "user", "content": None},
            ]
        )
        == "kept"
    )


def test_normalize_language_code_uses_the_primary_subtag() -> None:
    assert normalize_language_code("zh-CN") == "zh"
    assert normalize_language_code("en-US") == "en"
    assert normalize_language_code("EN") == "en"


def test_safe_log_token_keeps_one_ascii_line() -> None:
    assert safe_log_token("person:jian_kuang") == "person_jian_kuang"
    assert safe_log_token("req id\nwith\tnewlines") == "req_id_with_newlines"
    assert safe_log_token("") == "-"
    assert len(safe_log_token("a" * 200)) == 128


def test_fallback_messages_share_one_implementation() -> None:
    assert grounding_fallback("en") == (
        "I could not verify that information from the home graph."
    )
    assert no_records_fallback("en") == (
        "The home graph does not contain matching information for that request."
    )
    assert grounding_fallback("zh") == "老管家目前无法从家庭资料中核实这项信息。"
    assert no_records_fallback("zh") == (
        "家庭资料中没有找到与这个问题匹配的信息。"
    )


def test_edge_schema_load_default_finds_repository_schemas() -> None:
    registry = EdgeSchemaRegistry.load_default(Path(__file__).parents[1] / "data")

    assert "spouse_of" in registry.relationship_names
    assert "lives_in" in registry.relationship_names
    assert registry.resolve("hosts_space").schema.id == "hosted_by"


def test_facts_preserves_request_parser_compatibility_exports() -> None:
    assert facts_parse_fact_request is parse_fact_request
    assert FactsSubjectReference is SubjectReference
